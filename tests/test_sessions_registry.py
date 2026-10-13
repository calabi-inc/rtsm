"""Stage 2 of the session registry (rtsm/core/sessions.py) against fakes: boundary
rules, freeze content, trivial discard, routing and refinement, aliases, merge
rules, persistence round trip, keep cap, retention, ledger lines. No models, no
sockets, no threads except the ordering test in tests/test_viz_sessions.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rtsm.core.sessions import (
    Boundary, LiveApplyResult, LiveComponents, LiveVizState, SessionRegistry, SessionSnapshot,
    frame_deltas_from_history, packet_frame_id, subsample_cloud,
)
from rtsm.evaluation.event_log import SessionEvent

T0 = 1_700_000_000.0


# ───────────────────────────── fakes ─────────────────────────────

class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def obj(oid: str, xyz, fid: str, *, confirmed: bool = True) -> dict:
    return {"id": oid, "xyz_world": np.asarray(xyz, dtype=np.float32), "confirmed": confirmed, "hits": 3,
            "stability": 0.8, "label_primary": "cup", "label_scores": {"cup": 0.9}, "emb_mean": [0.1, 0.2],
            "last_update_frame_id": fid, "image_crops": [b"\xff\xd8jpeg"], "view_bins": {1, 2}}


class FakeWM:
    def __init__(self, objects=None, thresh: float = 0.5) -> None:
        self.objects = list(objects or [])
        self.cleared = 0
        self.pose_demote_thresh_m = thresh
        self.last_correction = None
        self.applied = []

    def _index(self):
        idx = {}
        for o in self.objects:
            idx.setdefault(o["last_update_frame_id"], []).append(o["id"])
        return idx

    def snapshot_for_session(self) -> SessionSnapshot:
        return SessionSnapshot(objects=[dict(o) for o in self.objects], frame_to_objects=self._index(),
                               crops={o["id"]: list(o["image_crops"]) for o in self.objects})

    def clear(self):
        self.cleared += 1
        n = len(self.objects)
        self.objects = []
        return {"objects": n}

    def apply_pose_corrections(self, frame_corrections):
        self.applied.append(dict(frame_corrections))
        moved = 0
        for o in self.objects:
            d = frame_corrections.get(o["last_update_frame_id"])
            if d is None:
                continue
            _p, dR, dt = d
            o["xyz_world"] = (dR @ o["xyz_world"] + dt).astype(np.float32)
            moved += 1
        self.last_correction = {"corrected": moved, "demoted": []}
        return moved


class FakeVectors:
    def __init__(self, ids=None) -> None:
        self.ids = list(ids or [])
        self.cleared = 0

    def export(self):
        return list(self.ids), np.full((len(self.ids), 4), 0.5, dtype=np.float32)

    def clear(self):
        self.cleared += 1
        self.ids = []
        return {"vectors": 0}


class FakeSweep:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self):
        self.cleared += 1
        return {}


def kf_record(mesh_id: str, ts_ns: int, pose=None, n: int = 10):
    return SimpleNamespace(mesh_id=mesh_id, timestamp_ns=ts_ns, positions=np.random.default_rng(1).random((n, 3)).astype(np.float32),
                           colors=np.full((n, 3), 7, dtype=np.uint8), pose=(np.eye(4, dtype=np.float32) if pose is None else pose))


class FakeViz:
    def __init__(self) -> None:
        self.records = []
        self.pose_history = {}
        self.kf_alias = {}
        self.alias_stamp = {}
        self.freeze_calls = []

    def add(self, mesh_id, ts_ns, pose=None, kf_id=None):
        r = kf_record(mesh_id, ts_ns, pose)
        self.records.append(r)
        self.pose_history[mesh_id] = r.pose.copy()
        self.alias_stamp[mesh_id] = ts_ns
        if kf_id is not None:
            self.kf_alias[str(kf_id)] = mesh_id
        return r

    def freeze_live(self, next_key):
        self.freeze_calls.append(next_key)
        st = LiveVizState(records=list(self.records), pose_history=dict(self.pose_history), kf_alias=dict(self.kf_alias),
                          alias_stamp=dict(self.alias_stamp))
        self.records, self.pose_history, self.kf_alias, self.alias_stamp = [], {}, {}, {}
        return st


def pose_at(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = (x, y, z)
    return T


def pkt(seq, ts_ns, *, epoch=0, sid="A", kf=True, pose=None, kf_id=None, rx_seq=None):
    T = pose_at(float(seq if seq is not None else (rx_seq or 0))) if pose is None else pose
    return SimpleNamespace(time=SimpleNamespace(seq=seq, t_sensor_ns=ts_ns), frame_epoch=epoch, session_id=sid,
                           is_keyframe=kf, pose=SimpleNamespace(T_wc=lambda: T), kf_id=kf_id,
                           ingest=SimpleNamespace(rx_seq=rx_seq if rx_seq is not None else seq))


class World:
    """A registry with every fake wired, inline persistence, a controllable clock."""

    def __init__(self, tmp_path, **sessions_cfg) -> None:
        self.clock = Clock()
        self.events: list = []
        self.wm = FakeWM()
        self.vectors = FakeVectors()
        self.sweep = FakeSweep()
        self.viz = FakeViz()
        cfg = {"sessions": {"dir": str(tmp_path / "sessions"), "persist_on_freeze": True, "keep_in_memory": 3,
                            "cloud_max_points": 0, "retention": {"keep_on_disk": 0, "max_age_days": 0}, **sessions_cfg}}
        self.reg = SessionRegistry(cfg, live=LiveComponents(wm=self.wm, vectors=self.vectors, sweep_cache=self.sweep,
                                                            vis_server=self.viz),
                                   event_sink=self.events.append, source="fake", now=self.clock.now, persist_async=False)

    def frames(self, seqs, *, epoch=0, sid="A", kf=True, t0_ns=1_000_000_000, step_ns=100_000_000):
        out = []
        for i, s in enumerate(seqs):
            out.append(self.reg.observe_frame(pkt(s, t0_ns + i * step_ns, epoch=epoch, sid=sid, kf=kf)))
        return out

    def populate(self, *, objects=True, keyframes=("ws_1", "ws_2", "ws_3")):
        for i, mid in enumerate(keyframes):
            self.viz.add(mid, 1_000_000_000 + i * 100_000_000, pose_at(float(i + 1)))
        if objects:
            self.wm.objects = [obj("o1", (1.0, 0.0, 0.5), "ws_1"), obj("o2", (2.0, 0.0, 0.5), "ws_2", confirmed=False)]
            self.vectors.ids = ["o1"]

    def actions(self):
        return [e.action for e in self.events if isinstance(e, SessionEvent)]


# ───────────────────────────── helpers ─────────────────────────────

def test_packet_frame_id_rule():
    assert packet_frame_id(pkt(7, 1)) == "ws_7"
    assert packet_frame_id(pkt(None, 1, rx_seq=12)) == "rx_12"
    assert packet_frame_id(SimpleNamespace(time=SimpleNamespace(seq=None), ingest=None)) is None


def test_subsample_is_deterministic_and_bounded():
    pos = np.random.default_rng(0).random((1000, 3)).astype(np.float32)
    col = np.zeros((1000, 3), dtype=np.uint8)
    a, _ = subsample_cloud("ws_9", pos, col, 100)
    b, _ = subsample_cloud("ws_9", pos, col, 100)
    assert a.shape == (100, 3) and np.array_equal(a, b)
    c, _ = subsample_cloud("ws_9", pos, col, 0)
    assert c.shape == (1000, 3) and c is not pos


def test_frame_deltas_match_viz_math():
    hist = {"ws_1": pose_at(1.0)}
    deltas = frame_deltas_from_history(hist, {"ws_1": pose_at(1.0, 0.0, 0.25), "ws_9": pose_at(0.0)})
    assert list(deltas) == ["ws_1"]
    old_pos, dR, dt = deltas["ws_1"]
    assert np.allclose(old_pos, [1.0, 0.0, 0.0]) and np.allclose(dR, np.eye(3)) and np.allclose(dt, [0.0, 0.0, 0.25])


# ───────────────────────────── boundaries ─────────────────────────────

def test_first_packet_opens_live_without_freeze(tmp_path):
    w = World(tmp_path)
    assert w.reg.observe_frame(pkt(1, 1_000_000_000)) is None
    assert w.reg.live is not None and w.reg.frozen == []
    assert [e.kind for e in w.reg.live.timeline] == ["start"]
    assert w.actions() == ["start"]
    assert "ws_1" in w.reg.live.pose_history


def test_boundary_on_epoch_change_with_new_id(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.frames([1, 2, 3])
    b = w.reg.observe_frame(pkt(4, 2_000_000_000, epoch=1, sid="B"))
    assert isinstance(b, Boundary) and b.reason == "session_id" and not b.discarded
    assert [r.key for r in w.reg.frozen] == [b.frozen_key]
    assert w.reg.live.key == b.live_key and w.reg.live.session_id == "B" and w.reg.live.epoch == 1
    assert w.reg.frozen[0].end_reason == "session_id"
    assert w.actions() == ["start", "freeze", "start"]


def test_same_id_reconnect_is_not_a_boundary(tmp_path):
    w = World(tmp_path)
    w.frames([1, 2])
    w.reg.on_session_change("A", 0)
    assert w.reg.frozen == []
    assert [e.kind for e in w.reg.live.timeline] == ["start", "reconnect"]
    w.reg.on_session_change("B", 1)                 # the boundary comes from the packets, not from here
    assert w.reg.frozen == [] and w.reg.live.session_id == "A"


@pytest.mark.parametrize("flag", [True, False])
def test_clock_rebase_without_epoch_change(tmp_path, flag):
    w = World(tmp_path, boundary_on_clock_rebase=flag)
    w.populate()
    w.reg.observe_frame(pkt(1, 1_000_000_000), clock_rebases=0)
    b = w.reg.observe_frame(pkt(2, 1_100_000_000), clock_rebases=1)
    if flag:
        assert b is not None and b.reason == "restart_inferred"
    else:
        assert b is None and w.reg.frozen == []


def test_epoch_change_and_rebase_count_once(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.reg.observe_frame(pkt(1, 1_000_000_000), clock_rebases=0)
    b = w.reg.observe_frame(pkt(2, 1_100_000_000, epoch=1, sid="B"), clock_rebases=1)
    assert b.reason == "session_id"
    assert w.reg.observe_frame(pkt(3, 1_200_000_000, epoch=1, sid="B"), clock_rebases=1) is None


def test_tracking_gap_event(tmp_path):
    w = World(tmp_path, tracking_gap_s=2.0)
    for s, ts in ((1, 0), (2, 500_000_000), (3, 3_500_000_000), (4, 3_600_000_000)):
        assert w.reg.observe_frame(pkt(s, ts)) is None
    gaps = [e for e in w.reg.live.timeline if e.kind == "tracking_gap"]
    assert len(gaps) == 1 and gaps[0].detail["gap_s"] == 3.0 and gaps[0].detail["prev_t_sensor_ns"] == 500_000_000
    assert w.reg.live.counters["tracking_gaps"] == 1


# ───────────────────────────── freeze ─────────────────────────────

def test_freeze_snapshot_content_and_copies(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.frames([1, 2, 3])
    b = w.reg.observe_frame(pkt(4, 2_000_000_000, epoch=1, sid="B"))
    f = w.reg.frozen[0]
    assert f.key == b.frozen_key
    assert sorted(o["id"] for o in f.objects) == ["o1", "o2"]
    assert f.objects[0]["label_scores"] == {"cup": 0.9} and f.objects[0]["demoted_by_correction"] is False
    assert f.frame_to_objects == {"ws_1": ["o1"], "ws_2": ["o2"]}
    assert sorted(f.keyframes) == ["ws_1", "ws_2", "ws_3"] and f.keyframes["ws_2"].n_points == 10
    assert set(f.pose_history) >= {"ws_1", "ws_2", "ws_3"}
    assert f.vectors[0] == ["o1"] and f.vectors[1].shape == (1, 4)
    assert f.counters["objects"] == 2 and f.counters["confirmed"] == 1 and f.counters["keyframes"] == 3
    assert f.timeline[-1].kind == "freeze" and f.on_disk and f.persisted
    # the live stores were cleared (in /reset order minus the clock and the pairing window)
    assert w.wm.cleared == 1 and w.vectors.cleared == 1 and w.sweep.cleared == 1
    assert w.viz.freeze_calls == [b.live_key] and w.viz.records == []
    # copies, not aliases: mutating the fakes afterwards changes nothing frozen
    w.viz.add("ws_9", 5, pose_at(9.0))
    assert "ws_9" not in f.keyframes
    f_xyz_before = f.objects[0]["xyz_world"].copy()
    w.wm.objects = [obj("o1", (99.0, 0.0, 0.0), "ws_1")]
    assert np.array_equal(f.objects[0]["xyz_world"], f_xyz_before)


def test_freeze_subsamples_clouds_deterministically(tmp_path):
    w = World(tmp_path, cloud_max_points=4)
    w.populate()
    w.frames([1, 2, 3])
    w.reg.observe_frame(pkt(4, 2_000_000_000, epoch=1, sid="B"))
    f = w.reg.frozen[0]
    assert all(p.shape == (4, 3) for p, _c in f.clouds.values())
    again, _ = subsample_cloud("ws_1", kf_record("ws_1", 0).positions, np.zeros((10, 3), np.uint8), 4)
    assert np.array_equal(f.clouds["ws_1"][0], again)


def test_trivial_session_is_discarded_not_frozen(tmp_path):
    w = World(tmp_path, min_keyframes=1)
    w.frames([1, 2], kf=False)                       # frames, no keyframes, no objects
    b = w.reg.observe_frame(pkt(3, 2_000_000_000, epoch=1, sid="B"))
    assert b.discarded and b.frozen_key is None and w.reg.frozen == []
    assert w.reg.live.timeline[0].detail["discarded_predecessor"]["session_id"] == "A"
    assert w.reg.live.discarded_predecessor["frames"] == 2
    assert w.reg.counters["discarded"] == 1
    assert w.actions() == ["start", "discard", "start"]
    assert w.wm.cleared == 1                         # the live stores are still cleared


def test_end_live_and_discard_live(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.frames([1, 2, 3])
    b = w.reg.end_live("manual")
    assert b.reason == "manual" and w.reg.frozen[0].end_reason == "manual"
    assert w.reg.live.session_id == "A" and w.reg.live.epoch == 0 and w.reg.live.key != b.frozen_key
    old = w.reg.live.key
    assert w.reg.discard_live() != old and w.reg.frozen[0].key == b.frozen_key


def test_process_exit_freezes_without_reopening(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.frames([1, 2, 3])
    w.reg.on_process_exit()
    assert w.reg.live is None and w.reg.frozen[0].end_reason == "process_exit" and w.reg.frozen[0].on_disk


# ───────────────────────────── corrections ─────────────────────────────

def _frozen_world(tmp_path, **cfg):
    """Session A frozen with ws_1..ws_3 and objects o1 (ws_1), o2 (ws_2); live session B with ws_10."""
    w = World(tmp_path, **cfg)
    w.populate()
    w.frames([1, 2, 3])
    w.reg.observe_frame(pkt(10, 2_000_000_000, epoch=1, sid="B"))
    w.viz.add("ws_10", 2_000_000_000, pose_at(10.0))
    return w


def test_route_live_before_frozen_on_id_collision(tmp_path):
    w = _frozen_world(tmp_path)
    w.reg.live.pose_history["ws_3"] = pose_at(3.0)   # the id exists in both sessions
    calls = []
    def live_apply(part):
        calls.append(dict(part))
        return LiveApplyResult(frame_deltas=frame_deltas_from_history(w.reg.live.pose_history, part), objects_moved=0)
    s = w.reg.apply_corrections({"ws_3": pose_at(3.0, 0.0, 0.2)}, live_apply=live_apply)
    assert s["live"] == 1 and s["frozen"] == {} and list(calls[0]) == ["ws_3"]
    assert w.reg.frozen[0].counters["refinements"] == 0
    assert np.allclose(w.reg.live.pose_history["ws_3"][:3, 3], [3.0, 0.0, 0.2])
    ev = w.reg.live.timeline[-1]
    assert ev.kind == "refine" and ev.detail["n_keyframes"] == 1 and ev.detail["max_shift_m"] == pytest.approx(0.2)


def test_route_to_frozen_when_live_lacks_the_id(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    s = w.reg.apply_corrections({"ws_1": pose_at(1.0, 0.0, 0.1)}, live_apply=lambda part: pytest.fail("live must not be called"))
    assert s["frozen"] == {f.key: 1} and s["live"] == 0
    o1 = next(o for o in f.objects if o["id"] == "o1")
    assert np.allclose(o1["xyz_world"], [1.0, 0.0, 0.6]) and o1["demoted_by_correction"] is False
    assert np.allclose(f.keyframes["ws_1"].pose[:3, 3], [1.0, 0.0, 0.1])
    assert f.counters["refinements"] == 1 and f.timeline[-1].kind == "refine"
    on_disk = json.loads((w.reg.dir / f.key / "keyframes.json").read_text())
    assert [k for k in on_disk["keyframes"] if k["mesh_id"] == "ws_1"][0]["pose"][11] == pytest.approx(0.1)


def test_frozen_refinement_demotes_large_shift_and_uses_nearest_fallback(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    w.reg.apply_corrections({"ws_1": pose_at(1.0, 0.0, 0.6)})
    o1 = next(o for o in f.objects if o["id"] == "o1")
    o2 = next(o for o in f.objects if o["id"] == "o2")
    assert o1["demoted_by_correction"] is True
    assert np.allclose(o2["xyz_world"], [2.0, 0.0, 1.1]) and o2["demoted_by_correction"] is True   # nearest corrected camera
    assert f.counters["objects_demoted_by_corrections"] == 2


def test_unknown_ws_id_is_unrouted_and_announced_only_batch_is_an_event(tmp_path):
    w = _frozen_world(tmp_path)
    s = w.reg.apply_corrections({"ws_77": pose_at(0.0)})
    assert s["unrouted"] == ["ws_77"] and s["announced_only"] is True
    assert w.reg.counters["corrections_unrouted"] == 1 and w.reg.counters["announced_only"] == 1
    s2 = w.reg.apply_corrections({}, kind="loop_closure")
    assert s2["announced_only"] is True
    ev = w.reg.live.timeline[-1]
    assert ev.kind == "refine" and ev.detail["announced_only"] is True and ev.detail["n_keyframes"] == 0
    assert w.reg.live.counters["corrections_applied"] == 0 and w.reg.live.counters["announced_only"] == 2


def test_alias_by_node_id_and_by_stamp(tmp_path):
    w = _frozen_world(tmp_path, alias_stamp_tol_ms=30)
    w.reg.live.kf_alias["42"] = "ws_10"
    w.reg.live.alias_stamp["ws_10"] = 2_000_000_000
    seen = {}
    def live_apply(part):
        seen.update(part)
        return LiveApplyResult()
    s = w.reg.apply_corrections({"42": pose_at(10.0, 0.0, 0.1), "ts:2012000000": pose_at(10.0, 0.0, 0.2),
                                 "ts:2080000000": pose_at(0.0), "bogus": pose_at(0.0)}, live_apply=live_apply)
    assert list(seen) == ["ws_10"]
    assert sorted(s["unaliased"]) == ["bogus", "ts:2080000000"]
    assert w.reg.counters["corrections_unaliased"] == 2
    # a frozen session's alias and its keyframe stamps resolve too
    f = w.reg.frozen[0]
    f.kf_alias["7"] = "ws_2"
    s2 = w.reg.apply_corrections({"7": pose_at(2.0, 0.0, 0.1), "ts:1112000000": pose_at(2.0, 0.0, 0.1)})
    assert s2["frozen"] == {f.key: 1}                # both keys name ws_2: one delta
    assert np.allclose(f.keyframes["ws_2"].pose[:3, 3], [2.0, 0.0, 0.1])


def test_relocalized_sets_aligned_to_and_merges_even_without_poses(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    s = w.reg.apply_corrections({}, kind="relocalized_in_prior_map", map_ref="A")
    assert s["merged"] == f.key
    assert w.reg.live.aligned_to["session_key"] == f.key and w.reg.live.aligned_to["kind"] == "relocalized_in_prior_map"
    assert f.merged_into == w.reg.live.key and w.reg.live.merged_from == [f.key]
    assert [e.kind for e in w.reg.live.timeline[-3:]] == ["refine", "relocalized", "merge"]
    meta = json.loads((w.reg.dir / f.key / "meta.json").read_text())
    assert meta["merged_into"] == w.reg.live.key


def test_implicit_merge_on_batch_spanning_live_and_one_frozen(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    s = w.reg.apply_corrections({"ws_10": pose_at(10.0, 0.0, 0.1), "ws_1": pose_at(1.0, 0.0, 0.1)},
                                live_apply=lambda part: LiveApplyResult())
    assert s["live"] == 1 and s["frozen"] == {f.key: 1} and s["merged"] == f.key
    assert f.merged_into == w.reg.live.key


def test_no_merge_when_batch_spans_two_frozen(tmp_path):
    w = _frozen_world(tmp_path)
    a = w.reg.frozen[0]
    w.populate(keyframes=("ws_20", "ws_21"))
    w.wm.objects = []
    w.frames([20, 21], epoch=1, sid="B")
    w.reg.observe_frame(pkt(30, 3_000_000_000, epoch=2, sid="C"))
    b = w.reg.frozen[0]
    assert b.key != a.key
    s = w.reg.apply_corrections({"ws_1": pose_at(1.0, 0.0, 0.1), "ws_20": pose_at(20.0, 0.0, 0.1)})
    assert sorted(s["frozen"].values()) == [1, 1] and s["merged"] is None
    assert w.reg.counters["merge_ambiguous"] == 1 and a.merged_into is None and b.merged_into is None


def test_merge_on_link_can_be_disabled(tmp_path):
    w = _frozen_world(tmp_path, merge_on_link=False)
    s = w.reg.apply_corrections({}, kind="relocalized_in_prior_map", map_ref="A")
    assert s["merged"] is None and w.reg.frozen[0].merged_into is None and w.reg.live.aligned_to is not None


def test_merge_unmerge_by_hand(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    assert w.reg.merge(f.key) is None
    assert w.reg.merge(f.key) == "already_merged"
    assert w.reg.merge("nope") == "unknown_session"
    assert f.merged_into == w.reg.live.key
    assert w.reg.unmerge(f.key) is None
    assert f.merged_into is None and w.reg.live.merged_from == []
    assert w.reg.unmerge(f.key) == "not_merged"
    assert [e.kind for e in f.timeline[-2:]] == ["merge", "unmerge"]


def test_reset_kind_defers_boundary_to_next_frame(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.frames([1, 2])
    s = w.reg.apply_corrections({"ws_1": pose_at(1.0)}, kind="reset")
    assert s["live"] == 0 and w.reg.frozen == []
    b = w.reg.observe_frame(pkt(3, 1_300_000_000))
    assert b is not None and b.reason == "map_reset" and w.reg.frozen[0].end_reason == "map_reset"


def test_unknown_kind_is_normalised(tmp_path):
    w = _frozen_world(tmp_path)
    s = w.reg.apply_corrections({}, kind="bogus")
    assert s["kind"] == "unknown"


# ───────────────────────────── persistence / retention ─────────────────────────────

def test_persist_and_reload_round_trip(tmp_path):
    w = World(tmp_path)
    w.populate()
    w.viz.kf_alias["42"] = "ws_1"
    w.frames([1, 2, 3])
    w.reg.observe_frame(pkt(4, 2_000_000_000, epoch=1, sid="B"))
    f = w.reg.frozen[0]
    folder = w.reg.dir / f.key
    assert {p.name for p in folder.iterdir()} >= {"meta.json", "objects.json", "frame_index.json", "keyframes.json",
                                                  "clouds.npz", "vectors.npz", "timeline.jsonl", "crops"}
    assert (folder / "crops" / "o1" / "0.jpg").read_bytes() == b"\xff\xd8jpeg"
    r2 = SessionRegistry({"sessions": {"dir": str(w.reg.dir)}}, source="fake", now=w.clock.now, persist_async=False)
    assert r2.load_from_disk() == 1
    g = r2.frozen[0]
    assert r2.summary(g)["key"] == f.key and g.payload_loaded is False and g.clouds is None
    assert [e.kind for e in g.timeline] == [e.kind for e in f.timeline]
    assert g.kf_alias == {"42": "ws_1"} and set(g.pose_history) >= {"ws_1", "ws_2", "ws_3"}
    d = r2.detail(f.key)
    assert sorted(o["id"] for o in d["objects"]) == ["o1", "o2"] and d["retention"]["protected_by"] == []
    assert g.payload_loaded and g.clouds is not None
    assert np.array_equal(g.clouds["ws_2"][0], f.clouds["ws_2"][0]) and g.vectors[0] == ["o1"]
    assert np.allclose(next(o for o in g.objects if o["id"] == "o1")["xyz_world"], [1.0, 0.0, 0.5])


def test_persist_failure_keeps_record_in_memory(tmp_path):
    w = World(tmp_path)
    (tmp_path / "sessions").write_text("not a directory")
    w.populate()
    w.frames([1, 2, 3])
    w.reg.observe_frame(pkt(4, 2_000_000_000, epoch=1, sid="B"))
    f = w.reg.frozen[0]
    assert f.persisted is False and f.persist_error and f.on_disk is False and len(f.objects) == 2


def test_keep_in_memory_cap(tmp_path):
    w = World(tmp_path, keep_in_memory=2)
    for i in range(4):
        w.populate(keyframes=(f"ws_{i * 10 + 1}",))
        w.frames([i * 10 + 1], epoch=i, sid=f"S{i}")
        w.reg.observe_frame(pkt(i * 10 + 9, 9_000_000_000 + i, epoch=i + 1, sid=f"S{i + 1}"))
    assert len(w.reg.frozen) == 4
    assert [r.clouds is not None for r in w.reg.frozen] == [True, True, False, False]
    old = w.reg.frozen[-1]
    w.reg._ensure_loaded(old)
    assert old.clouds is not None and old.payload_loaded


def test_gc_count_and_age(tmp_path):
    w = World(tmp_path, retention={"keep_on_disk": 3, "max_age_days": 0})
    for i in range(6):
        w.populate(keyframes=(f"ws_{i * 10 + 1}",))
        w.frames([i * 10 + 1], epoch=i, sid=f"S{i}")
        w.clock.advance(60)
        w.reg.observe_frame(pkt(i * 10 + 9, 9_000_000_000 + i, epoch=i + 1, sid=f"S{i + 1}"))
    assert len(w.reg.frozen) == 3 and w.reg.counters["gc_removed"] == 3
    assert len([p for p in w.reg.dir.iterdir()]) == 3 and "gc" in w.actions()
    assert [r.session_id for r in w.reg.frozen] == ["S5", "S4", "S3"]
    w2 = World(tmp_path / "age", retention={"keep_on_disk": 0, "max_age_days": 1})
    w2.populate()
    w2.frames([1], epoch=0, sid="OLD")
    w2.reg.observe_frame(pkt(9, 9_000_000_000, epoch=1, sid="NEW"))
    w2.clock.advance(2 * 86400)
    w2.populate(keyframes=("ws_11",))
    w2.frames([11], epoch=1, sid="NEW")
    w2.reg.observe_frame(pkt(19, 19_000_000_000, epoch=2, sid="NEWER"))
    assert [r.session_id for r in w2.reg.frozen] == ["NEW"] and w2.reg.counters["gc_removed"] == 1


def test_gc_protected_set(tmp_path):
    from dataclasses import replace
    w = World(tmp_path, retention={"keep_on_disk": 0, "max_age_days": 0, "grace_served_min": 10})
    for i in range(4):                               # S0..S3 frozen, S4 live; no retention while building
        w.populate(keyframes=(f"ws_{i * 10 + 1}",))
        w.frames([i * 10 + 1], epoch=i, sid=f"S{i}")
        w.clock.advance(60)
        w.reg.observe_frame(pkt(i * 10 + 9, 9_000_000_000 + i, epoch=i + 1, sid=f"S{i + 1}"))
    by_sid = {r.session_id: r.key for r in w.reg.frozen}
    assert set(by_sid) == {"S0", "S1", "S2", "S3"}
    assert w.reg.merge(by_sid["S0"]) is None        # in the live chain
    assert w.reg.pin(by_sid["S1"]) is None          # pinned
    w.reg.mark_served(by_sid["S2"])                  # served within the grace window
    w.reg.cfg = replace(w.reg.cfg, keep_on_disk=1)
    assert w.reg.gc() == 1
    remaining = {r.session_id for r in w.reg.frozen}
    assert remaining == {"S0", "S1", "S2"}
    assert not (w.reg.dir / by_sid["S3"]).exists()
    assert w.reg.detail(by_sid["S1"])["retention"]["protected_by"] == ["pinned"]
    assert "live_chain" in w.reg.detail(by_sid["S0"])["retention"]["protected_by"]
    w.clock.advance(11 * 60)                         # the grace window passes: S2 goes on the next run
    assert w.reg.gc() == 1 and {r.session_id for r in w.reg.frozen} == {"S0", "S1"}


def test_gc_removes_stray_tmp_and_gc_folders_at_start(tmp_path):
    d = tmp_path / "sessions"
    (d / "x.gc").mkdir(parents=True)
    (d / "y.tmp").mkdir()
    r = SessionRegistry({"sessions": {"dir": str(d)}}, source="fake", persist_async=False)
    assert r.load_from_disk() == 0
    assert not (d / "x.gc").exists() and not (d / "y.tmp").exists()


def test_delete_rules(tmp_path):
    w = _frozen_world(tmp_path)
    f = w.reg.frozen[0]
    assert w.reg.delete(w.reg.live.key) == "live_session"
    assert w.reg.delete("nope") == "unknown_session"
    w.reg.pin(f.key)
    assert w.reg.delete(f.key) == "pinned"
    w.reg.pin(f.key, False)
    w.reg.merge(f.key)
    assert w.reg.delete(f.key) == "in_live_chain"
    w.reg.unmerge(f.key)
    assert w.reg.delete(f.key) is None
    assert w.reg.frozen == [] and not (w.reg.dir / f.key).exists()
    assert w.actions()[-1] == "delete"


def test_session_event_ledger_lines(tmp_path):
    w = _frozen_world(tmp_path)
    w.reg.apply_corrections({"ws_1": pose_at(1.0, 0.0, 0.1)})
    w.reg.merge(w.reg.frozen[0].key)
    acts = w.actions()
    assert acts[:3] == ["start", "freeze", "start"] and "refine" in acts and "merge" in acts
    fz = next(e for e in w.events if e.action == "freeze")
    assert fz.kind == "session" and fz.reason == "session_id" and fz.counts["objects"] == 2


def test_health_block(tmp_path):
    w = _frozen_world(tmp_path)
    h = w.reg.health()
    assert h["live"] == w.reg.live.key and h["n_frozen"] == 1 and h["n_on_disk"] == 1 and h["boundaries"] == 1
    assert h["last_boundary"]["reason"] == "session_id"


def test_disabled_registry_is_a_no_op(tmp_path):
    w = World(tmp_path, enable=False)
    assert w.reg.observe_frame(pkt(1, 1)) is None and w.reg.live is None
    assert w.reg.observe_frame(pkt(2, 2, epoch=1, sid="B")) is None and w.reg.frozen == []
    assert w.reg.freeze() is None and w.reg.end_live() is None
    calls = []
    s = w.reg.apply_corrections({"ws_1": pose_at(1.0)}, live_apply=lambda part: calls.append(part) or LiveApplyResult())
    assert s["enabled"] is False and calls and w.wm.cleared == 0
