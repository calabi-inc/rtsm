"""baseline_search.py tests — the freshness gate (fail-closed), seed
derivation, and the acquisition loop's contract: no motion when the target
is already visible, sweep-then-acquire, bounded timeout, and the same
safety aborts the drive phase honors (e-stop, epoch change, dead feed)."""

import threading
import time
from dataclasses import replace

import pytest

from baseline_search import AcquireResult, BaselineSearcher, derive_seed, fresh_hits
from config import load_config
from rtsm_client import PoseSample, SemanticHit, SemanticResult
from test_nav import FakeBridge

BASE = load_config()
FAST = replace(
    BASE,
    nav=replace(BASE.nav, tick_s=0.01, stale_abort_s=0.5),
    baseline=replace(BASE.baseline, sweep_step_s=0.1, dwell_s=0.15,
                     steps_per_sweep=3, walk_s=0.1, walk_speed=0.2),
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
        self._pose_calls = 0
        self._ts = 100.0
        self.freeze_at = None
        self.bump_epoch_at = None

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
    assert acq.pose is not None                      # pose from the same query
    assert bridge.drive_calls == [], "no motion needed when already visible"


def test_acquires_after_sweeping():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)] if n >= 4 else [mk_hit(age_s=99)])
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-2", budget_s=10.0)
    assert acq.status == "acquired"
    rotate = [c for c in bridge.drive_calls if (c[1] < 0 < c[2]) or (c[2] < 0 < c[1])]
    assert rotate, "should have rotated while searching"
    assert bridge.stop_calls, "each step must end stopped"
    assert rtsm.queries >= 4


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
