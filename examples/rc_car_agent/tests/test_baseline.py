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
                     steps_per_sweep=3, sweeps_per_round=2,
                     walk_s=0.1, walk_speed=0.2,
                     walk_min_m=0.01, walk_chunk_m=0.02, walk_max_m=0.03,
                     # Tiny tick margin so the ROUND window (~1.5 s here)
                     # is what admits candidates — distinguishes round-
                     # scoped freshness from the old per-poll gate.
                     freshness_gate_s=0.05),
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
        # Label search: None -> endpoint returns no hits (rounds fall back
        # to semantic, so pre-label tests run unchanged); "raise" -> older
        # server without /search/label; else a results_fn(label_query_no).
        self.label_results_fn = None
        self.label_queries = 0
        self._pose_calls = 0
        self._ts = 100.0
        self.freeze_at = None
        self.bump_epoch_at = None
        self.bump_epoch_at_query = None   # epoch bumps ONLY in the round
                                          # query's pose (dwell polls keep
                                          # the seeded epoch)
        # Depth wall-guard signal; None = no data (fail-closed, no walk).
        self.clearance = None

    def get_forward_clearance(self):
        if self.clearance is None:
            return None
        if getattr(self, "clearance_live", True):
            # Model a LIVE depth stream (re-stamp per read): with a
            # one-shot stamp, every walk check made after a full round
            # sat ~0.15 s from the 2.0 s staleness limit — flaky on a
            # loaded machine. Staleness tests set clearance_live=False.
            return dict(self.clearance, timestamp=time.time())
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

    def label_query(self, query, top_k=5):
        self.label_queries += 1
        if self.label_results_fn == "raise":
            raise RuntimeError("no /search/label on this server")
        results = ([] if self.label_results_fn is None
                   else self.label_results_fn(self.label_queries))
        return SemanticResult(query=query, robot_pose=self.get_robot_pose(),
                              results=results)

    def semantic_query(self, query, top_k=5):
        self.queries += 1
        self.top_ks.append(top_k)
        pose = self.get_robot_pose()
        if (self.bump_epoch_at_query is not None
                and self.queries >= self.bump_epoch_at_query):
            pose = PoseSample(xyz=pose.xyz,
                              quaternion_xyzw=pose.quaternion_xyzw,
                              timestamp=pose.timestamp,
                              fetched_at_mono=0.0, frame_epoch=8)
        return SemanticResult(query=query, robot_pose=pose,
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


def test_full_observation_round_precedes_any_confirm():
    """The operator's rule: >= sweeps_per_round full spins, THEN one
    confirm — a target in view from second one still waits for the whole
    round (multi-view evidence), and exactly ONE query serves the round."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-1", budget_s=8.0)
    assert acq.status == "acquired"
    assert acq.sweeps == FAST.baseline.sweeps_per_round
    assert "round 1" in acq.detail
    assert acq.hits and acq.hits[0].id == "mug-1"
    assert acq.pose is not None                      # pose from the same query
    assert rtsm.queries == 1, "one batched confirm per round, not per dwell"
    rotate = [c for c in bridge.drive_calls if (c[1] < 0 < c[2]) or (c[2] < 0 < c[1])]
    assert len(rotate) > 0, "the round sweeps before confirming"


def test_round_query_fetches_deep():
    """Retrieval must fetch gate_fetch_k deep — top-k over ALL memory
    would let stale objects crowd a visible target out of the list."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])
    mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-1b", budget_s=8.0)
    assert rtsm.top_ks[0] == max(FAST.baseline.gate_fetch_k,
                                 FAST.baseline.query_top_k)


def test_acquires_on_a_later_round_after_relocating():
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)] if n >= 2 else [mk_hit(age_s=99)])
    rtsm.clearance = _clear_now()
    acq = mk_searcher(FAST, bridge, rtsm).acquire("red mug", "t-2", budget_s=10.0)
    assert acq.status == "acquired"
    assert acq.hits
    assert rtsm.queries == 2, "round 1 empty -> relocate -> round 2 acquires"
    assert acq.sweeps == 2 * FAST.baseline.sweeps_per_round
    walks = [c for c in bridge.drive_calls if c[1] == c[2] and c[1] > 0]
    assert walks, "an empty round must relocate before the next round"
    assert bridge.stop_calls, "each step must end stopped"


def test_round_window_admits_hits_seen_early_in_the_round():
    """Round-scoped freshness: an object observed during the round's
    FIRST sweep (stamp ~1 s old at confirm time, far beyond the 0.05 s
    tick margin) is still a candidate — while anything last seen before
    the round started stays invisible."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=1.0, hid="early-sweep"),
                               mk_hit(age_s=99.0, hid="pre-round")])
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-rw", budget_s=8.0)
    assert acq.status == "acquired"
    ids = [h.id for h in acq.hits]
    assert ids == ["early-sweep"]


def test_round_candidates_capped_at_top_k_in_rank_order():
    rtsm = StubRtsm(
        lambda n: [mk_hit(age_s=0.1, hid=f"h-{i:02d}") for i in range(15)])
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-k", budget_s=8.0)
    assert acq.status == "acquired"
    assert len(acq.hits) == FAST.baseline.query_top_k
    assert [h.id for h in acq.hits] == [f"h-{i:02d}"
                                        for i in range(FAST.baseline.query_top_k)]


def test_no_match_reentry_relocates_before_observing():
    """After the caller's no-match verdict, the standpoint is fully
    judged: the re-entered call must WALK first (steered by the judged
    round's depth survey), and the masked id never reappears."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="junk-1"),
                               mk_hit(age_s=0.1, hid="mug-2")])
    rtsm.clearance = _clear_now()
    bridge = FakeBridge()
    s = mk_searcher(FAST, bridge, rtsm)
    acq1 = s.acquire("m", "t-re", budget_s=8.0)
    assert acq1.status == "acquired"
    assert [h.id for h in acq1.hits] == ["junk-1", "mug-2"]

    bridge.drive_calls.clear()
    forward_at_query = []
    orig = rtsm.semantic_query

    def spy(q, top_k=5):
        forward_at_query.append(
            len([c for c in bridge.drive_calls if c[1] == c[2] and c[1] > 0]))
        return orig(q, top_k)

    rtsm.semantic_query = spy
    acq2 = s.acquire("m", "t-re", budget_s=8.0,
                     exclude_ids=frozenset({"junk-1"}))
    assert acq2.status == "acquired"
    assert [h.id for h in acq2.hits] == ["mug-2"], "rejected id stays masked"
    assert forward_at_query and forward_at_query[0] > 0, \
        "relocation walk must precede the re-entered round's query"


def test_union_ranks_label_hits_first_and_dedupes():
    """UNION retrieval (2026-08-28 review): label hits lead the candidate
    list, semantic-only hits follow, shared ids appear once. The served
    path is stamped into the acquisition detail for the audit."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="both-1"),
                               mk_hit(age_s=0.1, hid="sem-1")])
    rtsm.label_results_fn = lambda n: [mk_hit(age_s=0.1, hid="lab-1"),
                                       mk_hit(age_s=0.1, hid="both-1")]
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-lb",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert [h.id for h in acq.hits] == ["lab-1", "both-1", "sem-1"]
    assert "label+semantic" in acq.detail
    assert rtsm.queries >= 1, "semantic is ALWAYS consulted (union)"


def test_masked_label_hits_never_suppress_semantic_candidates():
    """REVIEW 2026-08-28: a fall-back-on-miss predicate decided before
    the gate/mask ran, so one stale or already-rejected label hit could
    starve every later round of its semantic candidates. Under the
    union, a fully-masked label side still leaves semantic serving the
    round."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="mug-2")])
    rtsm.label_results_fn = lambda n: [mk_hit(age_s=0.1, hid="rejected-1")]
    s = mk_searcher(FAST, FakeBridge(), rtsm)
    acq = s.acquire("m", "t-sup", budget_s=8.0,
                    exclude_ids=frozenset({"rejected-1"}))
    assert acq.status == "acquired"
    assert [h.id for h in acq.hits] == ["mug-2"]
    assert "semantic" in acq.detail


def test_label_error_falls_back_to_semantic():
    """An older server without /search/label must degrade gracefully —
    the round is served by semantic search, never lost."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="sem-1")])
    rtsm.label_results_fn = "raise"
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-lb2",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert [h.id for h in acq.hits] == ["sem-1"]
    assert "semantic" in acq.detail


def test_label_miss_falls_back_to_semantic():
    """No label hits (off-vocabulary goal) -> semantic serves the round."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-lb3",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert "semantic" in acq.detail
    assert rtsm.label_queries >= 1, "label search was tried first"


def test_label_hit_mid_sweep_exits_early():
    """Label early-exit (2026-08-28): a goal-labeled hit visible from
    the first dwell must end observation immediately — no waiting out
    the full sweep. (Operator: 'RTSM answers fast; the agent takes too
    long to lock.')"""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="lab-1")])
    rtsm.label_results_fn = lambda n: [mk_hit(age_s=0.1, hid="lab-1")]
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-ee",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert acq.hits[0].id == "lab-1"
    assert "early" in acq.detail
    assert acq.sweeps == 0, "sweep was cut short — never completed"


def test_semantic_only_hits_wait_for_sweep_end():
    """No label hits -> no early exit: semantic-only candidates (the
    flat-band junk class) are still judged once per completed sweep."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="sem-1")])
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-ee2",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert "early" not in acq.detail
    assert acq.sweeps == FAST.baseline.sweeps_per_round


def test_rejected_early_candidate_reobserves_in_place():
    """An early-exit acquisition judged no-match must NOT relocate on
    re-entry — the standpoint's sweep never finished. The masked id no
    longer triggers the peek; the next goal-labeled object does."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="fp-1"),
                               mk_hit(age_s=0.1, hid="real-2")])
    rtsm.label_results_fn = lambda n: [mk_hit(age_s=0.1, hid="fp-1"),
                                       mk_hit(age_s=0.1, hid="real-2")]
    rtsm.clearance = _clear_now()
    bridge = FakeBridge()
    s = mk_searcher(FAST, bridge, rtsm)
    acq1 = s.acquire("m", "t-ee3", budget_s=8.0)
    assert acq1.status == "acquired" and "early" in acq1.detail

    bridge.drive_calls.clear()
    acq2 = s.acquire("m", "t-ee3", budget_s=8.0,
                     exclude_ids=frozenset({"fp-1"}))
    assert acq2.status == "acquired"
    assert [h.id for h in acq2.hits] == ["real-2"]
    walks = [c for c in bridge.drive_calls if c[1] == c[2] and c[1] > 0]
    assert walks == [], "early-exit rejection must re-observe IN PLACE"


def test_below_floor_label_hit_never_livelocks_the_standpoint():
    """REVIEW 2026-08-28: the peek must apply the SAME score floor as
    the confirm — a persistent goal-labeled hit below the floor used to
    fire the peek every round while the confirm dropped it, spinning the
    car at one standpoint for the whole search cap (and miscoding the
    trial not_found). With aligned gates the sweep completes and the
    empty round RELOCATES."""
    cfg = replace(FAST, baseline=replace(FAST.baseline,
                                         min_candidate_score=0.05))
    rtsm = StubRtsm(lambda n: [])
    rtsm.label_results_fn = lambda n: [_mk_scored("weak", 0.03)]
    rtsm.clearance = _clear_now()
    bridge = FakeBridge()
    acq = mk_searcher(cfg, bridge, rtsm).acquire("m", "t-llk", budget_s=3.0)
    assert acq.status == "timeout"
    walks = [c for c in bridge.drive_calls if c[1] == c[2] and c[1] > 0]
    assert walks, "empty rounds must still relocate — never spin in place"


def test_wasted_early_exit_suppresses_peek_then_relocates():
    """REVIEW 2026-08-28: a peek that fired but confirmed to nothing
    (hit evaporated between peek and confirm) must not fire again next
    round — the following sweep runs FULL and its empty confirm
    relocates. Worst case one wasted partial round per standpoint."""
    rtsm = StubRtsm(lambda n: [])
    # The label hit exists only for the FIRST label call (the peek);
    # every later call — the confirm's label side included — sees none.
    rtsm.label_results_fn = (
        lambda n: [mk_hit(age_s=0.1, hid="ghost")] if n == 1 else [])
    rtsm.clearance = _clear_now()
    bridge = FakeBridge()
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-gh", budget_s=4.0)
    assert acq.status == "timeout"
    walks = [c for c in bridge.drive_calls if c[1] == c[2] and c[1] > 0]
    assert walks, "the suppressed full round's empty confirm relocates"


def test_peek_skipped_on_the_rounds_final_step():
    """REVIEW 2026-08-28: a peek on the final step of the final sweep
    would tag a FULLY observed standpoint as partial (sweeps undercount;
    rejection would wrongly re-observe instead of relocating). The final
    step skips the peek — the confirm runs moments later anyway."""
    cfg = replace(FAST, baseline=replace(FAST.baseline, sweeps_per_round=1))
    calls = {"peeks": 0}
    rtsm = StubRtsm(lambda n: [])
    # Hits appear from label call 3 on. Steps per sweep is 3: peeks run
    # at steps 1 and 2 (empty), step 3's peek is SKIPPED, so call 3 is
    # the confirm's label fetch — a full-round, non-early acquisition.
    rtsm.label_results_fn = (
        lambda n: [mk_hit(age_s=0.1, hid="late")] if n >= 3 else [])
    acq = mk_searcher(cfg, FakeBridge(), rtsm).acquire("m", "t-fs",
                                                       budget_s=8.0)
    assert acq.status == "acquired"
    assert acq.hits[0].id == "late"
    assert "early" not in acq.detail
    assert acq.sweeps == 1, "the completed sweep must be counted"


def test_leash_anchor_persists_across_reentries():
    """REVIEW 2026-08-28: the leash must confine the TRIAL, not each
    call — a per-call anchor let every no-match rejection re-anchor the
    2 m radius at the just-judged standpoint, compounding the mandatory
    relocate hop out of the venue ~1 m per rejection."""
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1, hid="junk-a"),
                               mk_hit(age_s=0.1, hid="mug-b")])
    rtsm.clearance = _clear_now()
    s = mk_searcher(FAST, FakeBridge(), rtsm)
    assert s.acquire("m", "t-an", budget_s=8.0).status == "acquired"
    anchor = s._start_xy
    assert anchor is not None
    acq2 = s.acquire("m", "t-an", budget_s=8.0,
                     exclude_ids=frozenset({"junk-a"}))
    assert acq2.status == "acquired"
    assert s._start_xy == anchor, "re-entry must NOT re-anchor the leash"


def test_epoch_change_at_acquisition_poll_aborts():
    """The query the searcher ACTS on is epoch-guarded too: a Lens
    restart during the round must not let a fresh-stamped, old-frame
    observation be acquired (the drive would then adopt the NEW epoch and
    its own guard could never fire)."""
    bridge = FakeBridge()
    rtsm = StubRtsm(lambda n: [mk_hit(age_s=0.1)])   # always "fresh"
    rtsm.bump_epoch_at_query = 1    # dwell polls keep epoch 7; the round
    acq = mk_searcher(FAST, bridge, rtsm).acquire("m", "t-2b", budget_s=8.0)
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


def _mk_scored(hid, score, age_s=0.1):
    return SemanticHit(id=hid, score=score, confirmed=True, stability=0.9,
                       xyz_world=[1.0, 0.3, 1.0],
                       last_seen_wall_utc=time.time() - age_s)


def test_score_floor_disabled_admits_the_flat_band_target():
    """REGRESSION (2026-08-28, operator-caught): the measured single-
    standpoint band is a flat 0.028-0.045 with the TRUE target mid-pack —
    a 0.05 floor filtered the actual tissue box. Default floor must be
    disabled: a 0.03-scoring candidate reaches the confirm call."""
    assert FAST.baseline.min_candidate_score <= 0
    rtsm = StubRtsm(lambda n: [_mk_scored("target-ish", 0.03)])
    acq = mk_searcher(FAST, FakeBridge(), rtsm).acquire("m", "t-fl",
                                                        budget_s=8.0)
    assert acq.status == "acquired"
    assert acq.hits[0].id == "target-ish"


def test_score_floor_when_enabled_trims_the_tail():
    cfg = replace(FAST, baseline=replace(FAST.baseline,
                                         min_candidate_score=0.05))
    rtsm = StubRtsm(lambda n: [_mk_scored("band", 0.06),
                               _mk_scored("tail", 0.01)])
    acq = mk_searcher(cfg, FakeBridge(), rtsm).acquire("m", "t-fl2",
                                                       budget_s=8.0)
    assert acq.status == "acquired"
    assert [h.id for h in acq.hits] == ["band"]


def test_stride_aims_walk_max_into_free_depth_keeping_buffer():
    # New rule (2026-08-28): stride = clearance - keep_clear, capped at
    # walk_max (~the 1 m hop), floored; overrun protection lives in the
    # chunked walk's live re-checks.
    assert relocation_stride_m(1.3, 0.12, 1.0, keep_clear_m=0.6) == \
        pytest.approx(0.7)                                # depth-limited
    assert relocation_stride_m(2.0, 0.12, 1.0, keep_clear_m=0.6) == 1.0  # cap
    assert relocation_stride_m(0.5, 0.12, 1.0, keep_clear_m=0.6) == 0.12  # floor
    assert relocation_stride_m(None, 0.12, 1.0, keep_clear_m=0.6) == 0.12
    assert relocation_stride_m(5.0, 0.12, 1.2) == 1.2     # no buffer -> cap


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
        (0.0, 1.9, _math.pi, 1.3),     # heading back toward start
    ] + [None] * 10
    k, stride, mode = leashed_choice(samples, (0.0, 0.0), 2.0, 0.6,
                                     0.12, 1.0)
    assert (k, mode) == (1, "open")
    assert abs(stride - 0.7) < 1e-9    # 1.3 depth minus the 0.6 buffer


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
    acq = s.acquire("m", "t-9", budget_s=3.0)
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
    rtsm.clearance_live = False                          # keep it ancient
    s, progress = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-10", budget_s=3.0)
    assert acq.status == "timeout"
    assert progress.get("walk_blocked_skips", 0) >= 1


def test_fresh_clearance_allows_walk():
    rtsm = StubRtsm(lambda n: [])
    rtsm.clearance = _clear_now()
    s, _ = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-11", budget_s=2.5)   # round 1 + relocate fit
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
        # Blocked through the round's 6 dwell samples AND the steered
        # walk's pre-check — the fallback loop's first re-check then
        # finds open space: rotate-away-from-the-wall, then walk.
        if calls["n"] <= 7:
            return dict(blocked, timestamp=time.time())
        return _clear_now()

    rtsm.get_forward_clearance = clearance_script
    s, progress = _searcher_with_progress(FAST, rtsm)
    acq = s.acquire("m", "t-12", budget_s=3.0)
    assert acq.status == "timeout"
    walks = [c for c in s._bridge.drive_calls
             if c[1] == c[2] and c[1] > 0]
    assert walks                                     # eventually walked
    assert progress.get("walk_blocked_skips", 0) == 0  # found a way, no skip
