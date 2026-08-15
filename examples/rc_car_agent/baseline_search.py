"""
Baseline target acquisition — E1 condition (b), the memoryless comparator.

The baseline agent is FORBIDDEN to act on anything it is not currently
seeing. Mechanism: the freshness gate — a semantic hit counts only if its
last observation is younger than `baseline.freshness_gate_s` (2 s).
Everything older sits invisible in memory. Same RTSM, same perception,
same pose stream, same nav/monitor/safety, and (via the server) the SAME
target-selection rule as condition (a); the ONLY masked capability is
persistence.

Gate fine print (audited 2026-08-03):
  * `last_seen_wall_utc` is stamped at pipeline UPSERT time, not frame-
    capture time — the physical age of an observation is stamp age PLUS
    the ~0.5 s processing lag, so the effective window is gate + lag
    (~2.5 s of history, about one sweep step). Conservative for the
    memory-faster claim; the accepted hit's stamp age is logged per
    acquisition so it is auditable.
  * The gate is two-sided: a stamp AHEAD of our clock by more than
    `clock_skew_tol_s` is rejected — a backward wall-clock step (NTP,
    resume from sleep) must never turn stale memory "fresh".
  * Retrieval fetches DEEP (`gate_fetch_k`, default 50) before gating:
    top-5 over ALL of memory would let stale objects crowd a currently
    visible target out of the candidate list entirely.
  * The acquisition poll itself is frame_epoch-guarded: a Lens restart
    between dwells must not let a pre-restart observation (fresh stamp,
    old-frame coordinate) be acquired.
  * Clock note: writer and reader share one machine (locked topology),
    so time.time() comparisons are valid up to step events, which the
    two-sided gate handles.

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

import math

from config import Config
from geometry import (DegenerateHeadingError, camera_to_car,
                      yaw_from_quat_xyzw)
from rtsm_client import PoseSample, RtsmClient, SemanticHit
from trial_logger import TrialLogger


@dataclass(frozen=True)
class AcquireResult:
    status: str                       # acquired | timeout | stale_stop |
                                      # frame_reset | estopped | preempted |
                                      # cancelled | shutdown
    detail: str = ""
    hits: tuple = ()                  # ALL fresh hits, ranked (selection is
                                      # the server's job — same rule as (a))
    pose: Optional[PoseSample] = None  # robot pose from the acquiring query
    hit_age_s: Optional[float] = None  # stamp age of the top fresh hit
    elapsed_s: float = 0.0
    sweeps: int = 0


def derive_seed(cfg_seed: int, trial_id: str) -> int:
    """0 -> deterministic per-trial seed from the id (logged either way)."""
    if cfg_seed != 0:
        return int(cfg_seed)
    return sum(ord(c) for c in trial_id) * 2654435761 % (2 ** 31)


def fresh_hits(hits, now_wall: float, gate_s: float, skew_tol_s: float = 0.5):
    """The freshness gate: navigable hits observed within gate_s of now.
    Fail-closed twice over: a hit without last_seen_wall_utc (old server)
    is NEVER fresh, and a stamp more than skew_tol_s AHEAD of our clock is
    rejected too — a backward wall-clock step must not resurrect stale
    memory as 'fresh'."""
    out = []
    for h in hits:
        if h.xyz_world is None or h.last_seen_wall_utc is None:
            continue
        age = now_wall - h.last_seen_wall_utc
        if -skew_tol_s <= age <= gate_s:
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

    def acquire(self, query: str, trial_id: str, budget_s: float,
                exclude_ids: frozenset = frozenset()) -> AcquireResult:
        """`exclude_ids`: objects the shared selection rule already judged
        no-match THIS trial — invisible to further polls, so the search
        resumes instead of re-acquiring the same wrong object every dwell
        (and burning an LLM call each time)."""
        b = self._cfg.baseline
        self._exclude = exclude_ids
        rng = random.Random(derive_seed(b.rng_seed, trial_id))
        sweep_sign = rng.choice((-1.0, 1.0))          # CCW or CW sweep
        t0 = time.monotonic()
        deadline = t0 + budget_s
        # Shared per-mission pose/epoch state, seen by BOTH the dwell
        # watcher and the acquisition polls (a Lens restart between the
        # two must not slip through).
        self._st = {"last_ts": None, "ref_epoch": None, "last_fresh_mono": t0}
        self._seed_epoch()
        sweeps = 0

        # Try before moving at all — the target might already be in view.
        first = self._gated_poll(query)
        if isinstance(first, AcquireResult):
            return self._stamp(first, t0, 0)
        if first is not None:
            hits, pose, age = first
            return AcquireResult(status="acquired", hits=tuple(hits), pose=pose,
                                 hit_age_s=age, detail="visible at start",
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
                r = self._dwell_and_watch_pose(b.dwell_s, deadline)
                if r is not None:
                    return self._stamp(r, t0, sweeps)

                found = self._gated_poll(query)
                if isinstance(found, AcquireResult):
                    return self._stamp(found, t0, sweeps)
                if found is not None:
                    hits, pose, age = found
                    return AcquireResult(status="acquired", hits=tuple(hits),
                                         pose=pose, hit_age_s=age,
                                         detail=f"sweep {sweeps} step {step_in_sweep}",
                                         elapsed_s=time.monotonic() - t0,
                                         sweeps=sweeps)
                step_in_sweep += 1
                self._progress["phase"] = "searching"
                self._progress["ticks"] = self._progress.get("ticks", 0) + 1

            sweeps += 1
            # Full sweep, nothing fresh — relocate forward and sweep again.
            # Geofence guard (2026-08-15): the car has no obstacle sensors,
            # so the pose is the only wall protection. If the walk's
            # projected endpoint would leave the drivable box, skip the
            # relocation and keep sweeping in place.
            if self._walk_is_safe():
                r = self._move(b.walk_speed, b.walk_speed, b.walk_s, deadline)
                if r is not None:
                    return self._interrupted(r, t0, sweeps)
            else:
                self._progress["geofence_skips"] = (
                    self._progress.get("geofence_skips", 0) + 1)
                if self._logger is not None:
                    self._logger.log_event(
                        "geofence_walk_skipped", time.monotonic() - t0,
                        sweeps=sweeps)

    # ── helpers ──────────────────────────────────────────────────────────

    def _walk_is_safe(self) -> bool:
        """True when the relocate walk's projected endpoint (+ margin)
        stays inside the configured geofence. Fence unset -> always True.
        Pose unavailable/degenerate -> False (can't verify, don't move).
        A car that somehow ends up OUTSIDE the fence facing inward gets
        True (endpoint inside) — the guard is self-recovering."""
        b = self._cfg.baseline
        fence = (b.geofence_x_min, b.geofence_x_max,
                 b.geofence_z_min, b.geofence_z_max)
        if any(v is None for v in fence):
            return True
        try:
            pose = self._rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001
            return False
        if pose is None:
            return False
        try:
            cam_yaw = yaw_from_quat_xyzw(pose.quaternion_xyzw)
        except DegenerateHeadingError:
            return False
        cal = self._cfg.calibration
        x, z, car_yaw = camera_to_car(pose.xyz, cam_yaw,
                                      cal.yaw_offset_rad, cal.lever_arm_rf)
        # Walk distance at the calibrated speed, plus safety margin.
        walk_m = (b.walk_speed * cal.speed_scale_mps * b.walk_s
                  + b.geofence_margin_m)
        ex = x + walk_m * math.sin(car_yaw)
        ez = z + walk_m * math.cos(car_yaw)
        x0, x1, z0, z1 = fence
        return (x0 <= ex <= x1) and (z0 <= ez <= z1)

    def _seed_epoch(self) -> None:
        try:
            pose = self._rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001
            pose = None
        if pose is not None and pose.frame_epoch is not None:
            self._st["ref_epoch"] = pose.frame_epoch

    def _note_epoch(self, epoch: Optional[int]) -> Optional[str]:
        """Int-vs-int equality gate, adopt-first-int, None never aborts —
        the same matrix the monitor uses."""
        if epoch is None:
            return None
        if self._st["ref_epoch"] is None:
            self._st["ref_epoch"] = epoch
            return None
        if epoch != self._st["ref_epoch"]:
            return f"frame_epoch {self._st['ref_epoch']} -> {epoch}"
        return None

    def _gated_poll(self, query):
        """One freshness-gated query, fetched DEEP so stale memory cannot
        crowd a visible target out of the candidate list. Returns:
        (fresh_hits, pose, top_age_s) on success, an AcquireResult(status=
        frame_reset) if the query's pose reveals a new sender session, or
        None for a miss."""
        b = self._cfg.baseline
        try:
            res = self._rtsm.semantic_query(query,
                                            top_k=max(b.gate_fetch_k,
                                                      b.query_top_k))
        except Exception:  # noqa: BLE001 — an RTSM hiccup is a missed poll
            return None
        if res.robot_pose is not None:
            bad = self._note_epoch(res.robot_pose.frame_epoch)
            if bad is not None:
                return AcquireResult(status="frame_reset",
                                     detail=f"{bad} at acquisition poll")
        now = self._now_wall()
        fresh = fresh_hits(res.results, now, b.freshness_gate_s,
                           b.clock_skew_tol_s)
        fresh = [h for h in fresh
                 if h.id not in getattr(self, "_exclude", frozenset())]
        if not fresh:
            return None
        age = now - fresh[0].last_seen_wall_utc
        return fresh, res.robot_pose, age

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

    def _dwell_and_watch_pose(self, dwell_s: float, deadline: float):
        """Stationary observation pause. Watches the pose stream for the
        same faults the drive phase would abort on: frozen feed past
        stale_abort_s -> stale_stop; frame_epoch change -> frame_reset.
        Updates the shared per-mission state; returns None on a normal
        dwell or an AcquireResult fault."""
        st = self._st
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
            if pose is not None and (st["last_ts"] is None
                                     or pose.timestamp != st["last_ts"]):
                st["last_ts"] = pose.timestamp
                st["last_fresh_mono"] = now
                bad = self._note_epoch(pose.frame_epoch)
                if bad is not None:
                    return AcquireResult(status="frame_reset",
                                         detail=f"{bad} during search")
            elif now - st["last_fresh_mono"] > self._cfg.nav.stale_abort_s:
                return AcquireResult(status="stale_stop",
                                     detail="pose feed died during search")
            time.sleep(self._cfg.nav.tick_s * 2)
        return None

    def _interrupted(self, name: str, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=name, elapsed_s=time.monotonic() - t0,
                             sweeps=sweeps)

    @staticmethod
    def _stamp(r: AcquireResult, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=r.status, detail=r.detail, hits=r.hits,
                             pose=r.pose, hit_age_s=r.hit_age_s,
                             elapsed_s=time.monotonic() - t0, sweeps=sweeps)
