"""
Baseline target acquisition — E1 condition (b), the memoryless comparator.

The baseline agent is FORBIDDEN to act on anything it is not currently
seeing. Mechanism: the freshness gate — a semantic hit counts only if its
last observation is younger than `baseline.freshness_gate_s` (2 s).
Everything older sits invisible in memory. Same RTSM, same perception,
same pose stream, same nav/monitor/safety; the ONLY masked capability is
persistence. (Clock note: `last_seen_wall_utc` is the RTSM server's wall
clock; the gate compares it to our own time.time(), valid because agent
and RTSM run on the same machine — the locked demo topology. The ~0.5 s
pipeline processing lag is inside the 2 s gate by design.)

Search policy (v1, deliberately near-deterministic for the paper — no
seed-sensitivity questions):

    repeat until fresh hit / timeout / interrupt:
        rotate one step (~35°), sweep direction chosen by the seeded RNG
        dwell ~1.2 s (let frames arrive and the pipeline process)
        poll the freshness-gated query
        after a full sweep with no hit: drive forward ~1.2 s (relocate),
        then sweep again

The searcher shares the mission's safety obligations: interrupts every
tick, drive() every tick while moving (watchdog), bounded pose staleness,
frame_epoch guard. It consumes trial budget; the drive phase afterwards
gets the REMAINDER of timeout_baseline_s.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import Config
from rtsm_client import PoseSample, RtsmClient, SemanticHit
from trial_logger import TrialLogger


@dataclass(frozen=True)
class AcquireResult:
    status: str                       # acquired | timeout | stale_stop |
                                      # frame_reset | estopped | preempted |
                                      # cancelled | shutdown
    detail: str = ""
    hit: Optional[SemanticHit] = None
    pose: Optional[PoseSample] = None  # robot pose from the acquiring query
    elapsed_s: float = 0.0
    sweeps: int = 0


def derive_seed(cfg_seed: int, trial_id: str) -> int:
    """0 -> deterministic per-trial seed from the id (logged either way)."""
    if cfg_seed != 0:
        return int(cfg_seed)
    return sum(ord(c) for c in trial_id) * 2654435761 % (2 ** 31)


def fresh_hits(hits, now_wall: float, gate_s: float):
    """The freshness gate: navigable hits observed within gate_s of now.
    A hit without last_seen_wall_utc (old server) is NEVER fresh — the
    baseline must fail closed, not quietly become the memory condition."""
    out = []
    for h in hits:
        if h.xyz_world is None or h.last_seen_wall_utc is None:
            continue
        if (now_wall - h.last_seen_wall_utc) <= gate_s:
            out.append(h)
    return out


class BaselineSearcher:
    def __init__(
        self,
        cfg: Config,
        bridge,
        rtsm: RtsmClient,
        stop_event: threading.Event,
        preempt_event: threading.Event,
        cancel_event: threading.Event,
        shutdown_event: threading.Event,
        logger: Optional[TrialLogger] = None,
        progress: Optional[dict] = None,
        now_wall: Callable[[], float] = time.time,
    ):
        self._cfg = cfg
        self._bridge = bridge
        self._rtsm = rtsm
        self._stop = stop_event
        self._preempt = preempt_event
        self._cancel = cancel_event
        self._shutdown = shutdown_event
        self._logger = logger
        self._progress = progress if progress is not None else {}
        self._now_wall = now_wall

    # ── the acquisition loop ─────────────────────────────────────────────

    def acquire(self, query: str, trial_id: str, budget_s: float) -> AcquireResult:
        b = self._cfg.baseline
        rng = random.Random(derive_seed(b.rng_seed, trial_id))
        sweep_sign = rng.choice((-1.0, 1.0))          # CCW or CW sweep
        t0 = time.monotonic()
        deadline = t0 + budget_s
        last_ts: Optional[float] = None
        ref_epoch: Optional[int] = None
        last_fresh_mono = t0
        sweeps = 0

        # Try before moving at all — the target might already be in view.
        first = self._gated_poll(query)
        if first is not None:
            return AcquireResult(status="acquired", hit=first[0], pose=first[1],
                                 detail="visible at start",
                                 elapsed_s=time.monotonic() - t0, sweeps=0)

        while True:
            step_in_sweep = 0
            while step_in_sweep < b.steps_per_sweep:
                # one rotate step, then dwell + poll
                r = self._move(sweep_sign * -b.sweep_step_turn,
                               sweep_sign * b.sweep_step_turn,
                               b.sweep_step_s, deadline)
                if r is not None:
                    return self._interrupted(r, t0, sweeps)
                r = self._dwell_and_watch_pose(b.dwell_s, deadline,
                                               last_ts, ref_epoch,
                                               last_fresh_mono)
                if isinstance(r, AcquireResult):
                    return self._stamp(r, t0, sweeps)
                last_ts, ref_epoch, last_fresh_mono = r

                found = self._gated_poll(query)
                if found is not None:
                    return AcquireResult(status="acquired", hit=found[0],
                                         pose=found[1],
                                         detail=f"sweep {sweeps} step {step_in_sweep}",
                                         elapsed_s=time.monotonic() - t0,
                                         sweeps=sweeps)
                step_in_sweep += 1
                self._progress["phase"] = "searching"
                self._progress["ticks"] = self._progress.get("ticks", 0) + 1

            sweeps += 1
            # Full sweep, nothing fresh — relocate forward and sweep again.
            r = self._move(b.walk_speed, b.walk_speed, b.walk_s, deadline)
            if r is not None:
                return self._interrupted(r, t0, sweeps)

    # ── helpers ──────────────────────────────────────────────────────────

    def _gated_poll(self, query):
        """One freshness-gated query. Returns (hit, pose) or None."""
        try:
            res = self._rtsm.semantic_query(query,
                                            top_k=self._cfg.baseline.query_top_k)
        except Exception:  # noqa: BLE001 — an RTSM hiccup is a missed poll
            return None
        fresh = fresh_hits(res.results, self._now_wall(),
                           self._cfg.baseline.freshness_gate_s)
        if not fresh:
            return None
        confirmed = [h for h in fresh if h.confirmed]
        best = (confirmed or fresh)[0]                # ranked order preserved
        return best, res.robot_pose

    def _check_interrupts(self) -> Optional[str]:
        if self._shutdown.is_set():
            return "shutdown"
        if self._stop.is_set():
            return "estopped"
        if self._preempt.is_set():
            self._preempt.clear()
            return "preempted"
        if self._cancel.is_set():
            self._cancel.clear()
            return "cancelled"
        return None

    def _move(self, left: float, right: float, duration_s: float,
              deadline: float) -> Optional[str]:
        """Drive (left,right) for duration_s with per-tick interrupt checks
        and watchdog-correct drive() cadence. Returns interrupt/timeout
        name or None on normal completion (car stopped either way)."""
        tick_s = self._cfg.nav.tick_s
        t_end = time.monotonic() + duration_s
        try:
            while time.monotonic() < t_end:
                if time.monotonic() >= deadline:
                    return "timeout"
                r = self._check_interrupts()
                if r is not None:
                    return r
                self._bridge.drive(left, right)
                time.sleep(tick_s)
        finally:
            self._bridge.stop()
        return None

    def _dwell_and_watch_pose(self, dwell_s: float, deadline: float,
                              last_ts, ref_epoch, last_fresh_mono):
        """Stationary observation pause. Watches the pose stream for the
        same faults the drive phase would abort on: frozen feed past
        stale_abort_s -> stale_stop; frame_epoch change -> frame_reset.
        Returns updated (last_ts, ref_epoch, last_fresh_mono) or an
        AcquireResult fault."""
        t_end = time.monotonic() + dwell_s
        while time.monotonic() < t_end:
            if time.monotonic() >= deadline:
                return AcquireResult(status="timeout", detail="budget exhausted in search")
            r = self._check_interrupts()
            if r is not None:
                return AcquireResult(status=r)
            try:
                pose = self._rtsm.get_robot_pose()
            except Exception:  # noqa: BLE001
                pose = None
            now = time.monotonic()
            if pose is not None and (last_ts is None or pose.timestamp != last_ts):
                last_ts = pose.timestamp
                last_fresh_mono = now
                if pose.frame_epoch is not None:
                    if ref_epoch is None:
                        ref_epoch = pose.frame_epoch
                    elif pose.frame_epoch != ref_epoch:
                        return AcquireResult(
                            status="frame_reset",
                            detail=f"frame_epoch {ref_epoch} -> {pose.frame_epoch} during search")
            elif now - last_fresh_mono > self._cfg.nav.stale_abort_s:
                return AcquireResult(status="stale_stop",
                                     detail="pose feed died during search")
            time.sleep(self._cfg.nav.tick_s * 2)
        return last_ts, ref_epoch, last_fresh_mono

    def _interrupted(self, name: str, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=name, elapsed_s=time.monotonic() - t0,
                             sweeps=sweeps)

    @staticmethod
    def _stamp(r: AcquireResult, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=r.status, detail=r.detail, hit=r.hit,
                             pose=r.pose, elapsed_s=time.monotonic() - t0,
                             sweeps=sweeps)
