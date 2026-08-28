"""baseline_search.py tests — the freshness gate (fail-closed), seed
derivation, and the acquisition loop's contract: no motion when the target
is already visible, sweep-then-acquire, bounded timeout, and the same
safety aborts the drive phase honors (e-stop, epoch change, dead feed)."""

import threading
import time
from dataclasses import replace

import pytest

from baseline_search import (AcquireResult, BaselineSearcher,
                             best_relocation_rotation, derive_seed,
                             fresh_hits, leash_limited_stride,
                             leashed_choice, relocation_stride_m)
from config import load_config
from rtsm_client import PoseSample, SemanticHit, SemanticResult
from test_nav import FakeBridge

BASE = load_config()
FAST = replace(
    BASE,
    nav=replace(BASE.nav, tick_s=0.01, stale_abort_s=0.5),
    baseline=replace(BASE.baseline, sweep_step_s=0.1, dwell_s=0.15,
                     steps_per_sweep=3, walk_s=0.1, walk_speed=0.2,
                     walk_min_m=0.01, walk_chunk_m=0.02, walk_max_m=0.03),
    # Unit-speed calibration so strides take milliseconds, not the real
    # rig's 25 s/m.
    calibration=replace(BASE.calibration, speed_scale_mps=1.0),
)


def mk_hit(age_s=0.0, hid="mug-1", xyz=(0.0, 0.3, 2.0), confirmed=True):
    return SemanticHit(id=hid, score=0.8, confirmed=confirmed, stability=0.9,
                       xyz_world=list(xyz),
                       last_seen_wall_utc=time.time() - age_s)


class StubRtsm:
    """semantic_query scripted by results_fn(query_number); pose stream
    advances unless frozen; epoch bumps at a scripted pose-call count."""

    def __init__(self, results_fn):
        self._results_fn = results_fn
        self.queries = 0
        self.top_ks = []
        self._pose_calls = 0
        self._ts = 100.0
        self.freeze_at = None
        self.bump_epoch_at = None
        # Depth wall-guard signal; None = no data (fail-closed, no walk).
        self.clearance = None

    def get_forward_clearance(self):
        return self.clearance

    def get_robot_pose(self):
        self._pose_calls += 1
        frozen = self.freeze_at is not None and self._pose_calls >= self.freeze_at
        if not frozen:
            self._ts += 0.05
        epoch = 7
        if self.bump_epoch_at is not None and self._pose_calls >= self.bump_epoch_at:
            epoch = 8
        return PoseSample(xyz=[0.0, 0.3, 0.0], quaternion_xyzw=[0, 0, 0, 1],
                          timestamp=self._ts, fetched_at_mono=0.0,
                          frame_epoch=epoch)

    def semantic_query(self, query, top_k=5):
        self.queries += 1
        self.top_ks.append(top_k)
        return SemanticResult(query=query, robot_pose=self.get_robot_pose(),
                              results=self._results_fn(self.queries))


def mk_searcher(cfg, bridge, rtsm, *, stop=None, preempt=None, cancel=None):
    return BaselineSearcher(
        cfg, bridge, rtsm,
        stop_event=stop or threading.Event(),
        preempt_event=preempt or threading.Event(),
        cancel_event=cancel or threading.Event(),
        shutdown_event=threading.Event(),
    )


# ── the freshness gate (pure) ────────────────────────────────────────────


def test_gate_keeps_only_fresh_navigable_hits():
    now = time.time()
    hits = [
        mk_hit(age_s=0.5, hid="fresh"),
        mk_hit(age_s=10.0, hid="stale"),
        SemanticHit(id="no-ts", score=0.9, confirmed=True, stability=0.9,
                    xyz_world=[0, 0, 1], last_seen_wall_utc=None),
        SemanticHit(id="no-xyz", score=0.9, confirmed=True, stability=0.9,
                    xyz_world=None, last_seen_wall_utc=now),
    ]
    kept = fresh_hits(hits, now, gate_s=2.0)
    assert [h.id for h in kept] == ["fresh"]


def test_gate_fails_closed_without_last_seen_field():
    """An old RTSM server (no last_seen_wall_utc) must yield an ALWAYS-
    empty gate — the baseline degrades to never-acquires, it must never
    silently become the memory condition."""
    hits = [SemanticHit(id="x", score=0.9, confirmed=True, stability=0.9,
                        xyz_world=[0, 0, 1], last_seen_wall_utc=None)]
    assert fresh_hits(hits, time.time(), 2.0) == []


def test_gate_rejects_stamps_from_the_future():
    """A backward wall-clock step (NTP, resume from sleep) puts old stamps
    AHEAD of our clock — they must not become 'fresh'."""
    now = time.time()
    future = [mk_hit(age_s=-90.0, hid="future")]     # stamp 90 s ahead
    assert fresh_hits(future, now, 2.0) == []
    barely = [mk_hit(age_s=-0.3, hid="skew")]        # benign write/read skew
    assert [h.id for h in fresh_hits(barely, now, 2.0)] == ["skew"]


def test_derive_seed_deterministic():
    assert derive_seed(42, "anything") == 42
    assert derive_seed(0, "t20260803-120000-001") == derive_seed(0, "t20260803-120000-001")
    assert derive_seed(0, "t-a") != derive_seed(0, "t-b")


# ── acquisition loop ─────────────────────────────────────────────────────


def test_visible_at_start_acquires_without_motion():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-1", budget_s=5.0)
    assert acq.status == "acquired"
    assert acq.sweeps == 0 and "start" in acq.detail
    assert acq.hits and acq.hits[0].id == "mug-1"
    assert acq.hit_age_s == pytest.approx(0.1, abs=0.05)
    assert acq.pose is not None                      # pose from the same query
    assert bridge.drive_calls == [], "no motion needed when already visible"


def test_gated_poll_fetches_deep():
    """Retrieval must fetch gate_fetch_k deep — top-5 over ALL memory
    would let stale objects crowd a visible target out of the list."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])
    mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-1b", budget_s=5.0)
    assert rtsm.top_ks[0] == max(FAST.baseline.gate_fetch_k,
                                 FAST.baseline.query_top_k)


def test_acquires_after_sweeping():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)] if n >= 4 else [mk_hit(age_s=99)])
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-2", budget_s=10.0)
    assert acq.status == "acquired"
    assert acq.hits
    rotate = [c for c in bridge.drive_calls if (c[1] < 0 < c[2]) or (c[2] < 0 < c[1])]
    assert rotate, "should have rotated while searching"
    assert bridge.stop_calls, "each step must end stopped"
    assert rtsm.queries >= 4


def test_epoch_change_at_acquisition_poll_aborts():
    """The poll the searcher ACTS on is epoch-guarded too: a Lens restart
    between dwells must not let a fresh-stamped, old-frame observation be
    acquired (the drive would then adopt the NEW epoch and its own guard
    could never fire)."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])   # always "fresh"
    rtsm.bump_epoch_at = 2      # call 1 seeds epoch 7; the poll sees 8
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-2b", budget_s=5.0)
    assert acq.status == "frame_reset"
    assert "acquisition poll" in acq.detail


def test_never_fresh_times_out_bounded():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=99)])
    t0 = time.monotonic()
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-3", budget_s=1.2)
    assert acq.status == "timeout"
    assert time.monotonic() - t0 < 3.0               # bounded, promptly
    assert bridge.stop_calls


def test_estop_interrupts_search():
    bridge = FakeBridge()
    stop = threading.Event()
    stop.set()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=99)])
    acq = mk_searcher(FAST, bridge, rtsm, stop=stop).acquire("m", "t-4", 5.0)
    assert acq.status == "estopped"


def test_preempt_interrupts_and_clears():
    bridge = FakeBridge()
    preempt = threading.Event()
    preempt.set()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=99)])
    acq = mk_searcher(FAST, bridge, rtsm, preempt=preempt).acquire("m", "t-5", 5.0)
    assert acq.status == "preempted"
    assert not preempt.is_set()


def test_epoch_change_during_search_aborts():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=99)])
    rtsm.bump_epoch_at = 6                           # Lens "restarts" mid-search
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-6", budget_s=10.0)
    assert acq.status == "frame_reset"
    assert "frame_epoch" in acq.detail


def test_dead_pose_feed_during_search_aborts():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=99)])
    rtsm.freeze_at = 3                               # feed dies mid-search
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-7", budget_s=10.0)
    assert acq.status == "stale_stop"


def _clear_now():
    return {"clearance_m": 2.5, "valid_frac": 0.9, "timestamp": time.time()}


def _searcher_with_progress(cfg, rtsm):
    progress = {}
    s = BaselineSearcher(
        cfg, FakeBridge(), rtsm,
        stop_event=threading.Event(), preempt_event=threading.Event(),
        cancel_event=threading.Event(), shutdown_event=threading.Event(),
        progress=progress,
    )
    return s, progress


# ── steered relocation (2026-08-17: walk toward measured open space) ────


def test_steer_picks_most_open_heading():
    # Step 2 (0-indexed) of 12 is the most open; its heading is 3 steps
    # forward from sweep end -> rotate 3 steps in the sweep direction.
    c = [0.5] * 12
    c[2] = 3.0
    assert best_relocation_rotation(c, 12) == (3, 1.0)


def test_steer_takes_shortest_way_around():
    # Step 9 of 12: 10 forward vs 2 backward -> 2 steps the OTHER way.
    c = [0.5] * 12
    c[9] = 3.0
    assert best_relocation_rotation(c, 12) == (2, -1.0)


def test_steer_tie_prefers_fewer_steps():
    c = [None] * 12
    c[0] = 2.0                                   # 1 step forward
    c[9] = 2.0                                   # 2 steps backward
    assert best_relocation_rotation(c, 12) == (1, 1.0)


def test_steer_none_without_samples():
    assert best_relocation_rotation([None] * 12, 12) is None


def test_steer_full_circle_costs_zero():
    # The last step's heading IS the sweep-end heading: no rotation.
    c = [0.5] * 12
    c[11] = 3.0
    assert best_relocation_rotation(c, 12) == (0, 1.0)


def test_low_score_fresh_hit_never_reaches_selection():
    # Relevance floor (2026-08-28): junk in view (fresh but scoring ~0.02
    # vs the goal) must not trigger acquisition/LLM calls; a real target
    # scoring above the floor still acquires.
    junk = mk_hit(age_s=0.1, hid="wall-1")
    junk = type(junk)(id="wall-1", score=0.02, confirmed=True,
                      stability=0.9, xyz_world=[1.0, 0.3, 1.0],
                      last_seen_wall_utc=time.time() - 0.1)
    rtsm = StubRtsm(lambda n: [junk])
    rtsm.clearance = _clear_now()
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-fl",
                                                        budget_s=1.5)
    assert acq.status == "timeout"          # junk never acquired

    real = mk_hit(age_s=0.1, hid="mug-1")   # score 0.8 >> floor
    rtsm2 = StubRtsm(lambda n: [real])
    acq2 = mk_searcher(FAST, FakeBridge(), rtsm2).acquire("m", "t-fl2",
                                                          budget_s=5.0)
    assert acq2.status == "acquired"


def test_stride_is_half_clearance_capped_and_floored():
    assert relocation_stride_m(2.0, 0.12, 1.2) == 1.0     # half
    assert relocation_stride_m(5.0, 0.12, 1.2) == 1.2     # cap
    assert relocation_stride_m(0.1, 0.12, 1.2) == 0.12    # floor
    assert relocation_stride_m(None, 0.12, 1.2) == 0.12   # no data -> floor


# ── search leash (2026-08-17: depth sees past the venue; stay near start) ─

import math as _math


def test_leash_from_center_allows_full_radius():
    # At the start, any heading may travel exactly the leash length.
    s = leash_limited_stride(0.0, 0.0, 0.0, (0.0, 0.0), 2.0)
    assert abs(s - 2.0) < 1e-9


def test_leash_trims_outbound_heading():
    # 1.5 m from start, heading straight away: only 0.5 m remains.
    s = leash_limited_stride(0.0, 1.5, 0.0, (0.0, 0.0), 2.0)
    assert abs(s - 0.5) < 1e-9


def test_leash_inbound_heading_gets_extra_room():
    # 1.5 m out, heading straight BACK: may cross start and continue to
    # the far side of the leash = 1.5 + 2.0.
    s = leash_limited_stride(0.0, 1.5, _math.pi, (0.0, 0.0), 2.0)
    assert abs(s - 3.5) < 1e-6


def test_leash_disabled_is_unbounded():
    assert leash_limited_stride(9.0, 9.0, 0.0, (0.0, 0.0), 0.0) == float("inf")


def test_leashed_choice_prefers_open_inside():
    # Step 0: hugely open but points out of the leash from a far spot.
    # Step 1: modestly open, points inward. Choice must be step 1.
    samples = [
        (0.0, 1.9, 0.0, 5.0),          # near edge, heading further out
        (0.0, 1.9, _math.pi, 1.6),     # heading back toward start
    ] + [None] * 10
    k, stride, mode = leashed_choice(samples, (0.0, 0.0), 2.0, 0.6,
                                     0.12, 1.0)
    assert (k, mode) == (1, "open")
    assert abs(stride - 0.8) < 1e-9    # half of 1.6, inside leash room


def test_inbound_open_heading_wins_even_from_outside():
    # Drifted outside the leash: a clear heading pointing home is still
    # an ordinary 'open' choice (crossing back in is always allowed).
    samples = [(0.0, 2.5, 0.0, 4.0),               # heading further out
               (0.0, 2.5, _math.pi, 4.0)] + [None] * 10   # heading home
    k, stride, mode = leashed_choice(samples, (0.0, 0.0), 2.0, 0.6,
                                     0.12, 1.0)
    assert (k, mode) == (1, "open")
    assert stride == 1.0               # capped at walk_max


def test_leashed_choice_returns_home_when_no_open_inbound():
    # Outbound headings are leash-blocked and the homeward heading has no
    # depth data -> 'return' mode walks it at the safe floor stride.
    samples = [(0.0, 2.5, 0.0, 4.0),               # out, blocked by leash
               (0.0, 2.5, _math.pi, None)] + [None] * 10  # home, no depth
    k, stride, mode = leashed_choice(samples, (0.0, 0.0), 2.0, 0.6,
                                     0.12, 1.0)
    assert mode == "return"
    assert k == 1
    assert abs(stride - 0.12) < 1e-9   # no depth -> floor stride only


def test_leashed_choice_none_without_samples():
    assert leashed_choice([None] * 12, (0.0, 0.0), 2.0, 0.6,
                          0.12, 1.0) is None


# ── depth wall guard (2026-08-16: no corners, the camera senses walls) ───


def test_no_clearance_data_blocks_walk_fail_closed():
    rtsm = StubRtsm(lambda n: [])                    # clearance stays None
    s, progress = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-9", budget_s=2.0)
    assert acq.status == "timeout"
    assert progress.get("walk_blocked_skips", 0) >= 1
    # never a straight-line walk command (equal positive wheels)
    walks = [c for c in s._bridge.drive_calls
             if c[1] == c[2] and c[1] > 0]
    assert walks == []


def test_stale_clearance_blocks_walk():
    rtsm = StubRtsm(lambda n: [])
    rtsm.clearance = {"clearance_m": 3.0, "valid_frac": 0.9,
                      "timestamp": time.time() - 30.0}   # ancient sample
    s, progress = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-10", budget_s=2.0)
    assert acq.status == "timeout"
    assert progress.get("walk_blocked_skips", 0) >= 1


def test_fresh_clearance_allows_walk():
    rtsm = StubRtsm(lambda n: [])
    rtsm.clearance = _clear_now()
    s, _ = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-11", budget_s=2.5)
    assert acq.status == "timeout"
    walks = [c for c in s._bridge.drive_calls
             if c[1] == c[2] and c[1] > 0]
    assert walks                                     # relocation happened


def test_blocked_then_clear_rotates_and_walks():
    # The "turn away from the wall" behavior: blocked ahead, the searcher
    # rotates a step, re-checks, and walks the first open direction.
    rtsm = StubRtsm(lambda n: [])
    blocked = {"clearance_m": 0.2, "valid_frac": 0.9,
               "timestamp": time.time()}
    calls = {"n": 0}

    def clearance_script():
        calls["n"] += 1
        if calls["n"] <= 2:                          # blocked twice...
            return dict(blocked, timestamp=time.time())
        return _clear_now()                          # ...then open

    rtsm.get_forward_clearance = clearance_script
    s, progress = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-12", budget_s=3.0)
    assert acq.status == "timeout"
    walks = [c for c in s._bridge.drive_calls
             if c[1] == c[2] and c[1] > 0]
    assert walks                                     # eventually walked
    assert progress.get("walk_blocked_skips", 0) == 0  # found a way, no skip
