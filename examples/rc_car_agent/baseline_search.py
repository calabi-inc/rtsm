"""
Baseline target acquisition — E1 condition (b), the memoryless comparator.

The baseline agent is FORBIDDEN to act on anything it is not currently
observing. Mechanism: the freshness gate — a semantic hit counts only if
it was last observed DURING the current observation round (the in-place
sweeps at the current standpoint; `freshness_gate_s` survives as the
upsert-lag margin on the round window). Everything older sits invisible
in memory: no persistence across standpoints, no persistence across
trials. Same RTSM, same perception, same pose stream, same nav/monitor/
safety, and (via the server) the SAME target-selection rule as condition
(a); the ONLY masked capability is persistence.

Gate fine print (audited 2026-08-03):
  * `last_seen_wall_utc` is stamped at pipeline UPSERT time, not frame-
    capture time — the physical age of an observation is stamp age PLUS
    the ~0.5 s processing lag, so the effective window is gate + lag
    (~2.5 s of history, about one sweep step). Conservative for the
    memory-faster claim; the top candidate's stamp age (hit_age_s) AND
    the picked candidate's stamp age (target_last_seen_age_s — under the
    round window the two can differ by minutes) are logged per
    acquisition so staleness is auditable.
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

Search policy (v2, 2026-08-28 — observe-then-confirm rounds; v1's
per-dwell confirmation is gone):

    round (repeat until candidates / interrupt / caller's budget):
        sweeps_per_round (3) full 360° sweeps in place — rotate ~30°,
        dwell ~1.2 s, NO queries: pure observation. The pipeline
        accumulates every heading's objects over multiple passes
        (multiple views -> better crops) while the sweep doubles as a
        depth survey (pose + clearance recorded per heading).
        then ONE deep query: candidates = hits observed DURING this
        round (round-scoped freshness), masked ids removed, ranked
        top query_top_k -> returned for ONE batched image-verified
        selection call (the server's shared rule).
    no candidates at all -> relocate immediately (no LLM call).
    caller re-enters after a no-match verdict -> relocate FIRST (this
    standpoint is fully judged), then run the next round.
    relocation: shortest rotation to the most open in-leash heading;
    stride aims walk_max_m (~1 m) into measured free depth while
    keeping the wall-guard buffer; leash-trimmed; chunked with live
    clearance re-checks.

Why rounds (measured 2026-08-28, t20260828-160929-001): per-dwell
confirmation cost one 4-12 s LLM call PER visible object (12 calls in
150 s — the car looked stuck spinning), and the single-standpoint score
band is FLAT (top-15 for "tissue box": 0.028-0.045, true target mid-
pack), so no score floor can pre-filter junk — ranking plus one batched
image-verified call per round is the only defensible selection pressure.

The searcher shares the mission's safety obligations: interrupts every
tick, drive() every tick while moving (watchdog), bounded pose staleness,
frame_epoch guard. It consumes trial budget; the server caps the whole
acquisition phase at baseline.search_cap_s (exhaustion = NOT FOUND) and
the drive phase gets the remainder of timeout_baseline_s.
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
    hits: tuple = ()                  # the round's candidates, ranked
                                      # (selection is the server's job —
                                      # same rule as (a))
    pose: Optional[PoseSample] = None  # robot pose from the acquiring query
    hit_age_s: Optional[float] = None  # stamp age of the top fresh hit
    elapsed_s: float = 0.0
    sweeps: int = 0
    # Audit counters for the caller's outcome coding (2026-08-28): a
    # trial may only conclude NOT FOUND if standpoints were actually
    # queried/judged — a retrieval outage that produced zero successful
    # round queries must stay a plain timeout, not a substantive
    # search conclusion.
    rounds_queried: int = 0            # round queries that SUCCEEDED
    query_failures: int = 0            # rounds lost to query errors


def derive_seed(cfg_seed: int, trial_id: str) -> int:
    """0 -> deterministic per-trial seed from the id (logged either way)."""
    if cfg_seed != 0:
        return int(cfg_seed)
    return sum(ord(c) for c in trial_id) * 2654435761 % (2 ** 31)


def rotation_to_step(k: int, steps_per_sweep: int):
    """Shortest rotation from the sweep-end heading to step k's heading:
    (n_steps, sign). sign +1.0 continues the sweep direction."""
    n_total = int(steps_per_sweep)
    fwd = (k + 1) % n_total
    bwd = n_total - fwd
    return (fwd, 1.0) if fwd <= bwd else (bwd, -1.0)


def leash_limited_stride(px, pz, yaw, start_xy, leash_m):
    """Meters the car may travel along heading `yaw` from (px, pz) before
    its distance to start_xy exceeds leash_m. 0.0 when already outside
    and the heading points further out; unbounded (inf) when leash <= 0."""
    if leash_m <= 0:
        return float("inf")
    dx, dz = px - start_xy[0], pz - start_xy[1]
    ux, uz = math.sin(yaw), math.cos(yaw)
    du = dx * ux + dz * uz
    inside = leash_m * leash_m - (dx * dx + dz * dz)
    disc = du * du + inside
    if disc <= 0:
        return 0.0
    s = -du + math.sqrt(disc)
    return max(0.0, s)


def leashed_choice(samples, start_xy, leash_m, min_clear_m,
                   walk_min_m: float, walk_max_m: float):
    """Steered + leashed relocation choice (2026-08-17).

    samples: per sweep step, None or (x, z, yaw, clearance|None) recorded
    at that step's dwell. Returns (step_k, stride_m, mode) or None.

    mode "open": the most-open heading whose leash-limited stride is
    still worth walking (>= walk_min). Ties prefer fewer rotation steps.
    mode "return": every open heading leads out of the leash — walk the
    heading that points most directly back toward the start instead
    (the car turns around at the boundary rather than leaving the
    venue). None: no usable samples (caller keeps legacy behavior)."""
    best = None                    # (clearance, -rot_steps, k, stride)
    n = len(samples)
    for k, s in enumerate(samples):
        if s is None:
            continue
        x, z, yaw, c = s
        if c is None or c < min_clear_m:
            continue
        stride = min(relocation_stride_m(c, walk_min_m, walk_max_m,
                                         keep_clear_m=min_clear_m),
                     leash_limited_stride(x, z, yaw, start_xy, leash_m))
        if stride < walk_min_m:
            continue
        rot, _sign = rotation_to_step(k, n)
        cand = (float(c), -rot, k, stride)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is not None:
        return best[2], best[3], "open"

    # Return-toward-start: pick the sampled heading pointing most nearly
    # back at the start; stride bounded by distance home and clearance.
    back = None                    # (inward_dot, k, stride)
    for k, s in enumerate(samples):
        if s is None:
            continue
        x, z, yaw, c = s
        dx, dz = start_xy[0] - x, start_xy[1] - z
        dist_home = math.hypot(dx, dz)
        if dist_home < walk_min_m:
            continue               # effectively at start already
        if c is not None and c < min_clear_m:
            continue               # measured BLOCKED heading — never the
                                   # return pick (the floored stride rule
                                   # no longer filters these implicitly)
        inward = (math.sin(yaw) * dx + math.cos(yaw) * dz) / dist_home
        stride = min(dist_home, walk_max_m,
                     relocation_stride_m(c, walk_min_m, walk_max_m,
                                         keep_clear_m=min_clear_m))
        if stride < walk_min_m or inward <= 0.0:
            continue
        cand = (inward, k, stride)
        if back is None or cand[0] > back[0]:
            back = cand
    if back is not None:
        return back[1], back[2], "return"
    return None


def relocation_stride_m(clearance_m, walk_min_m: float, walk_max_m: float,
                        keep_clear_m: float = 0.0):
    """Relocation stride (2026-08-28, superseding the half-depth rule):
    aim walk_max_m (~the operator's 1 m hop) into the measured open
    depth, but never consume the last keep_clear_m of it — the stride
    stops at least the wall-guard buffer short of the measured obstacle.
    Floored and capped; overrun is separately prevented by the chunked
    walk's live clearance re-checks. None clearance -> the floor (blind
    minimal hop)."""
    if clearance_m is None:
        return walk_min_m
    return max(walk_min_m,
               min(walk_max_m, float(clearance_m) - float(keep_clear_m)))


def best_relocation_rotation(clearances, steps_per_sweep: int):
    """Steered relocation (2026-08-17): given the clearance sampled at
    each sweep step (list of Optional[float], index = step), return
    (n_steps, sign) — the SHORTEST rotation from the sweep-end heading to
    the most-open recorded heading — or None when no valid samples exist.
    sign is +1.0 to continue in the sweep's rotation direction, -1.0 to
    rotate back the other way. Ties on clearance prefer fewer steps.

    Heading bookkeeping: step k's dwell heading is (k+1) rotation steps
    from the sweep start; a full sweep returns to the start heading, so
    reaching step k again costs (k+1) mod N steps forward or N-that
    backward."""
    best = None                       # (clearance, -steps, n_steps, sign)
    n_total = int(steps_per_sweep)
    for k, c in enumerate(clearances[:n_total]):
        if c is None:
            continue
        fwd = (k + 1) % n_total
        bwd = n_total - fwd
        steps, sign = (fwd, 1.0) if fwd <= bwd else (bwd, -1.0)
        cand = (float(c), -steps, steps, sign)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None
    return best[2], best[3]


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
        # Depth survey from the most recently completed round — steers
        # the relocate-first on a no-match re-entry (the server re-calls
        # acquire() on the SAME instance after its selection verdict).
        self._last_samples = None
        # TRIAL-lifetime leash anchor (2026-08-28 review finding: a
        # per-call anchor re-anchored the leash at every no-match
        # re-entry, so each rejection granted a fresh 2 m radius and the
        # mandatory relocate hop compounded out of the venue). The
        # searcher instance lives exactly one trial (the server builds a
        # new one per mission), so the anchor set on the FIRST call with
        # a live pose confines the WHOLE trial's search to the start.
        self._start_xy = None
        self._rounds_queried = 0
        self._query_failures = 0

    # ── the acquisition loop ─────────────────────────────────────────────

    def acquire(self, query: str, trial_id: str, budget_s: float,
                exclude_ids: frozenset = frozenset()) -> AcquireResult:
        """`exclude_ids`: objects the shared selection rule already judged
        no-match THIS trial — masked from every candidate list. A
        non-empty set also means the CURRENT standpoint was fully
        observed and judged, so this call relocates BEFORE observing
        again (steered by the samples the judged round recorded)."""
        b = self._cfg.baseline
        self._exclude = exclude_ids
        rng = random.Random(derive_seed(b.rng_seed, trial_id))
        sweep_sign = rng.choice((-1.0, 1.0))          # CCW or CW sweep
        t0 = time.monotonic()
        deadline = t0 + budget_s
        # Shared per-mission pose/epoch state, seen by BOTH the dwell
        # watcher and the round query (a Lens restart between the two
        # must not slip through).
        self._st = {"last_ts": None, "ref_epoch": None, "last_fresh_mono": t0}
        self._seed_epoch()
        sweeps = 0
        rounds = 0
        # Leash anchor (2026-08-17): the TRIAL's start position — set once
        # per searcher instance (= once per trial) and reused across
        # no-match re-entries, so cumulative relocation drift stays
        # bounded by the leash around where the trial began. Depth can
        # see past the venue boundary, so steering is confined to a
        # radius around THERE; a pose that is still unavailable leaves
        # the leash off until one arrives (walks are separately gated on
        # a live pose by _require_live_pose).
        if self._start_xy is None:
            p = self._car_xy_yaw()
            if p is not None:
                self._start_xy = (p[0], p[1])
        start_xy = self._start_xy

        if exclude_ids:
            r = self._relocate(self._last_samples, start_xy, sweep_sign,
                               deadline, t0, sweeps)
            if r is not None:
                return r

        while True:
            # ── observe: sweeps_per_round full 360° passes, no queries ──
            round_start_wall = self._now_wall()
            # Pose + clearance sampled at each dwell heading — the sweep
            # doubles as a free 360° depth survey that steers the
            # relocation, leashed to the start position. Later passes
            # overwrite a heading's sample only when they bring a live
            # clearance reading (freshest usable data wins).
            samples = [None] * b.steps_per_sweep
            for _ in range(b.sweeps_per_round):
                step_in_sweep = 0
                while step_in_sweep < b.steps_per_sweep:
                    r = self._move(sweep_sign * -b.sweep_step_turn,
                                   sweep_sign * b.sweep_step_turn,
                                   b.sweep_step_s, deadline)
                    if r is not None:
                        return self._interrupted(r, t0, sweeps)
                    r = self._dwell_and_watch_pose(b.dwell_s, deadline)
                    if r is not None:
                        return self._stamp(r, t0, sweeps)
                    c_now = self._clearance_now()
                    xyz_yaw = self._car_xy_yaw()
                    if xyz_yaw is not None and (
                            c_now is not None
                            or samples[step_in_sweep] is None):
                        samples[step_in_sweep] = (xyz_yaw[0], xyz_yaw[1],
                                                  xyz_yaw[2], c_now)
                    step_in_sweep += 1
                    self._progress["phase"] = "searching"
                    self._progress["ticks"] = self._progress.get("ticks", 0) + 1
                sweeps += 1
            rounds += 1
            self._last_samples = samples

            # ── confirm: ONE deep query over this round's observations ──
            # Retried on transient errors (2026-08-28 review finding: a
            # single dropped HTTP request must not silently discard a
            # ~2 min observation round and walk the car off the
            # standpoint). Each retry waits under the pose watcher.
            found = "error"
            for _attempt in range(3):
                found = self._round_query(query, round_start_wall)
                if found != "error":
                    break
                # Logged at the FIRST failed attempt — the evidence that
                # this standpoint was observed but not judged must reach
                # the trial log even if the budget dies mid-retry (an
                # analysis must never read a query fault as an empty
                # standpoint).
                if _attempt == 0 and self._logger is not None:
                    self._logger.log_event("round_query_failed",
                                           time.monotonic() - t0,
                                           rounds=rounds, sweeps=sweeps)
                r = self._dwell_and_watch_pose(1.0, deadline)
                if r is not None:
                    return self._stamp(r, t0, sweeps)
            if found == "error":
                # Persistent failure: RE-OBSERVE from here instead of
                # relocating off a standpoint that was never judged.
                self._query_failures += 1
                continue
            self._rounds_queried += 1
            if isinstance(found, AcquireResult):
                return self._stamp(found, t0, sweeps)
            if found is not None:
                hits, pose, age = found
                return AcquireResult(status="acquired", hits=tuple(hits),
                                     pose=pose, hit_age_s=age,
                                     detail=f"round {rounds} ({sweeps} sweeps)",
                                     elapsed_s=time.monotonic() - t0,
                                     sweeps=sweeps,
                                     rounds_queried=self._rounds_queried,
                                     query_failures=self._query_failures)
            # Nothing rankable observed from this standpoint — move on
            # without spending any LLM time.
            if self._logger is not None:
                self._logger.log_event("round_no_candidates",
                                       time.monotonic() - t0,
                                       rounds=rounds, sweeps=sweeps)
            r = self._relocate(samples, start_xy, sweep_sign,
                               deadline, t0, sweeps)
            if r is not None:
                return r

    def _relocate(self, samples, start_xy, sweep_sign, deadline,
                  t0: float, sweeps: int) -> Optional[AcquireResult]:
        """Move ~walk_max_m to a new standpoint. STEERED + LEASHED
        (2026-08-17): rotate the shortest way to the most open heading
        whose stride stays within the leash around the start pose; when
        every open heading leads out of bounds, turn back toward the
        start instead. The stride aims walk_max_m into the measured open
        depth, keeping the wall-guard buffer (leash-trimmed, chunked with
        live re-checks). `samples` is the depth survey recorded by the
        judged round's sweeps (None entries where a heading had no data;
        None entirely when no survey exists — falls through to the
        rotate-until-open loop). Returns None when done (walked, or
        blocked in every direction and staying put), or a terminal
        AcquireResult on interrupt/fault."""
        b = self._cfg.baseline
        # No motion without EVIDENCE the pose feed is alive NOW
        # (2026-08-28 review finding: the re-entry relocate is the only
        # walk not preceded by a full dwell history, and the dwell
        # watcher's first-poll rule counts even a FROZEN pose as fresh —
        # a feed that died during the server's selection call could
        # otherwise commit ~25 s of blind driving).
        r = self._require_live_pose(deadline)
        if r is not None:
            return self._stamp(r, t0, sweeps)
        if samples is None:
            samples = [None] * b.steps_per_sweep
        heading_clearance = [(s[3] if s is not None else None)
                             for s in samples]
        chosen = None
        if start_xy is not None:
            chosen = leashed_choice(samples, start_xy,
                                    b.search_leash_m,
                                    b.min_walk_clearance_m,
                                    b.walk_min_m, b.walk_max_m)
        if chosen is not None:
            k, stride, mode = chosen
            n_steps, sign = rotation_to_step(k, b.steps_per_sweep)
        else:
            # No pose data (or nothing eligible): legacy clearance-
            # only steer with the default stride rule.
            steer = best_relocation_rotation(heading_clearance,
                                             b.steps_per_sweep)
            if steer is not None:
                n_steps, sign = steer
                valid = [c for c in heading_clearance if c is not None]
                stride = relocation_stride_m(
                    max(valid), b.walk_min_m, b.walk_max_m,
                    keep_clear_m=b.min_walk_clearance_m)
                mode = "open_unleashed"
                chosen = True
        if chosen is not None:
            if self._logger is not None:
                self._logger.log_event(
                    "relocate_steered", time.monotonic() - t0,
                    rotate_steps=int(n_steps * sign), mode=mode,
                    stride_m=round(stride, 3), sweeps=sweeps)
            for _ in range(int(n_steps)):
                r = self._move(sweep_sign * sign * -b.sweep_step_turn,
                               sweep_sign * sign * b.sweep_step_turn,
                               b.sweep_step_s, deadline)
                if r is not None:
                    return self._interrupted(r, t0, sweeps)
            r = self._dwell_and_watch_pose(min(b.dwell_s, 0.6), deadline)
            if r is not None:
                return self._stamp(r, t0, sweeps)
            if self._walk_is_clear():
                r = self._walk_stride(stride, deadline)
                if r is not None:
                    return self._interrupted(r, t0, sweeps)
                self._progress["phase"] = "searching"
                return None                  # next round from here
        # Wall guard (2026-08-16): the walk needs measured open space
        # ahead (live depth clearance served by RTSM). When blocked
        # (or the steer target went stale), rotate one step at a time
        # and walk the FIRST open direction — the car turns away from
        # walls instead of grinding them. A full circle with no open
        # direction -> stay put, keep observing from here (logged;
        # budget keeps running).
        walked = False
        for _ in range(b.steps_per_sweep):
            c_now = self._clearance_now()
            if (b.min_walk_clearance_m <= 0
                    or (c_now is not None
                        and c_now >= b.min_walk_clearance_m)):
                stride = relocation_stride_m(
                    c_now, b.walk_min_m, b.walk_max_m,
                    keep_clear_m=b.min_walk_clearance_m)
                xyz_yaw = self._car_xy_yaw()
                if start_xy is not None and xyz_yaw is not None:
                    stride = min(stride, leash_limited_stride(
                        xyz_yaw[0], xyz_yaw[1], xyz_yaw[2],
                        start_xy, b.search_leash_m))
                    if stride < b.walk_min_m:
                        # open but out of bounds — keep rotating
                        r = self._move(
                            sweep_sign * -b.sweep_step_turn,
                            sweep_sign * b.sweep_step_turn,
                            b.sweep_step_s, deadline)
                        if r is not None:
                            return self._interrupted(r, t0, sweeps)
                        continue
                r = self._walk_stride(stride, deadline)
                if r is not None:
                    return self._interrupted(r, t0, sweeps)
                walked = True
                break
            r = self._move(sweep_sign * -b.sweep_step_turn,
                           sweep_sign * b.sweep_step_turn,
                           b.sweep_step_s, deadline)
            if r is not None:
                return self._interrupted(r, t0, sweeps)
            # Short settle so the clearance sample refreshes at the
            # new heading before re-checking.
            r = self._dwell_and_watch_pose(min(b.dwell_s, 0.6), deadline)
            if r is not None:
                return self._stamp(r, t0, sweeps)
        if not walked:
            self._progress["walk_blocked_skips"] = (
                self._progress.get("walk_blocked_skips", 0) + 1)
            if self._logger is not None:
                self._logger.log_event(
                    "walk_blocked_all_directions",
                    time.monotonic() - t0, sweeps=sweeps)
        return None

    # ── helpers ──────────────────────────────────────────────────────────

    def _car_xy_yaw(self):
        """Car ground position + heading from the live pose (calibrated
        camera->car transform), or None when pose is unavailable or the
        heading is degenerate."""
        try:
            pose = self._rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001
            return None
        if pose is None:
            return None
        try:
            cam_yaw = yaw_from_quat_xyzw(pose.quaternion_xyzw)
        except DegenerateHeadingError:
            return None
        cal = self._cfg.calibration
        x, z, car_yaw = camera_to_car(pose.xyz, cam_yaw,
                                      cal.yaw_offset_rad, cal.lever_arm_rf)
        return x, z, car_yaw

    def _clearance_now(self):
        """Fresh clearance meters, or None (no data / stale / hiccup)."""
        b = self._cfg.baseline
        try:
            c = self._rtsm.get_forward_clearance()
        except Exception:  # noqa: BLE001
            return None
        if c is None:
            return None
        if self._now_wall() - float(c.get("timestamp", 0)) > b.clearance_max_age_s:
            return None
        return float(c.get("clearance_m", 0.0))

    def _walk_stride(self, stride_m: float, deadline: float):
        """Walk stride_m forward in chunks of walk_chunk_m, re-checking
        the live clearance guard between chunks — a long stride must not
        outrun a stale measurement. Blocked mid-stride stops silently
        (partial progress is progress; the next sweep happens wherever we
        stopped). Returns an interrupt name from _move, or None."""
        b = self._cfg.baseline
        cal = self._cfg.calibration
        speed_mps = max(1e-6, b.walk_speed * cal.speed_scale_mps)
        remaining = float(stride_m)
        while remaining > 1e-3:
            if not self._walk_is_clear():
                return None
            chunk = min(b.walk_chunk_m, remaining)
            r = self._move(b.walk_speed, b.walk_speed,
                           chunk / speed_mps, deadline)
            if r is not None:
                return r
            remaining -= chunk
        return None

    def _walk_is_clear(self) -> bool:
        """True when the live depth stream measures at least
        min_walk_clearance_m of open space ahead, with a fresh sample.
        Fail-closed: no data / stale / hiccup all mean 'do not walk
        blind'. Guard disabled when min_walk_clearance_m <= 0."""
        b = self._cfg.baseline
        if b.min_walk_clearance_m <= 0:
            return True
        c_m = self._clearance_now()
        return c_m is not None and c_m >= b.min_walk_clearance_m

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

    def _round_query(self, query, round_start_wall: float):
        """The round's ONE deep query — fetched DEEP so stale memory
        cannot crowd a visible target out of the candidate list, gated to
        the ROUND WINDOW: candidates are hits last observed during this
        round's sweeps (freshness_gate_s survives as the upsert-lag
        margin — an observation made in the first dwell may carry a stamp
        from just before round start). Masked ids removed, optional tail
        floor applied, ranked top query_top_k returned for the caller's
        single batched image-verified selection call. Returns:
        (candidates, pose, top_age_s), an AcquireResult(status=
        frame_reset) if the query's pose reveals a new sender session,
        None when nothing rankable was observed from this standpoint, or
        the string "error" on a query exception (caller retries)."""
        b = self._cfg.baseline
        try:
            res = self._rtsm.semantic_query(query,
                                            top_k=max(b.gate_fetch_k,
                                                      b.query_top_k))
        except Exception:  # noqa: BLE001 — retried by the caller; a round
            return "error"  # must never be silently coded as "empty"
        if res.robot_pose is not None:
            bad = self._note_epoch(res.robot_pose.frame_epoch)
            if bad is not None:
                return AcquireResult(status="frame_reset",
                                     detail=f"{bad} at acquisition poll")
        now = self._now_wall()
        window_s = (now - round_start_wall) + b.freshness_gate_s
        fresh = fresh_hits(res.results, now, window_s, b.clock_skew_tol_s)
        fresh = [h for h in fresh
                 if h.id not in getattr(self, "_exclude", frozenset())]
        # Tail floor — disabled by default (the measured single-standpoint
        # band is flat; see BaselineCfg.min_candidate_score).
        if b.min_candidate_score > 0:
            fresh = [h for h in fresh if h.score >= b.min_candidate_score]
        fresh = fresh[:b.query_top_k]
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

    def _require_live_pose(self, deadline: float) -> Optional[AcquireResult]:
        """Wait (bounded by nav.stale_abort_s) for a pose TIMESTAMP CHANGE
        before any relocation motion — a frozen feed repeats its last
        stamp, which the dwell watcher's first-poll rule would wrongly
        count as fresh when the per-mission state was just re-seeded.
        Returns None once a live pose is seen (and folds it into the
        shared watcher state), or a terminal AcquireResult."""
        t_end = time.monotonic() + self._cfg.nav.stale_abort_s
        try:
            p0 = self._rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001
            p0 = None
        ts0 = p0.timestamp if p0 is not None else None
        while time.monotonic() < t_end:
            if time.monotonic() >= deadline:
                return AcquireResult(status="timeout",
                                     detail="budget exhausted in search")
            r = self._check_interrupts()
            if r is not None:
                return AcquireResult(status=r)
            try:
                p = self._rtsm.get_robot_pose()
            except Exception:  # noqa: BLE001
                p = None
            if p is not None and p.timestamp != ts0:
                bad = self._note_epoch(p.frame_epoch)
                if bad is not None:
                    return AcquireResult(status="frame_reset",
                                         detail=f"{bad} before relocation")
                self._st["last_ts"] = p.timestamp
                self._st["last_fresh_mono"] = time.monotonic()
                return None
            time.sleep(self._cfg.nav.tick_s * 2)
        return AcquireResult(status="stale_stop",
                             detail="no live pose before relocation walk")

    def _interrupted(self, name: str, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=name, elapsed_s=time.monotonic() - t0,
                             sweeps=sweeps,
                             rounds_queried=self._rounds_queried,
                             query_failures=self._query_failures)

    def _stamp(self, r: AcquireResult, t0: float, sweeps: int) -> AcquireResult:
        return AcquireResult(status=r.status, detail=r.detail, hits=r.hits,
                             pose=r.pose, hit_age_s=r.hit_age_s,
                             elapsed_s=time.monotonic() - t0, sweeps=sweeps,
                             rounds_queried=self._rounds_queried,
                             query_failures=self._query_failures)
