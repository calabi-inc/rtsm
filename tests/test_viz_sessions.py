"""Stages 4 and 7 of the session registry: the visualization server as the live
side of a session. Keyframes tagged with the live key, the source keyframe id
alias learnt from the packet, freeze_live() handing everything to the registry,
the one correction entry, the kf_pose_update coalescing adapter, and the
two-thread ordering guarantee (a batch lands entirely before or entirely after
a freeze). No GPU, no sockets: a real VisualizationServer with a fake working
memory and a real SessionRegistry with persistence off.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rtsm.core.datamodel import FramePacket, PinholeIntrinsics, PoseStamped, TimeBundle
from rtsm.core.sessions import LiveComponents, SessionRegistry, SessionSnapshot
from rtsm.visualization.server import VisualizationServer

INTR = PinholeIntrinsics(width=8, height=6, fx=5.0, fy=5.0, cx=4.0, cy=3.0)
IDENT_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def pose_at(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = (x, y, z)
    return T


def make_pkt(seq: int, ts_ns: int, x: float, *, kf_id=None, epoch: int = 0, sid: str = "A") -> FramePacket:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    depth = np.ones((6, 8), dtype=np.float32)
    pose = PoseStamped(stamp_ns=ts_ns, frame_id="world", t_wc=np.array([x, 0.0, 0.0], np.float32), q_wc_xyzw=IDENT_Q)
    return FramePacket(time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=1.7e9, t_sensor_ns=ts_ns, seq=seq), rgb=rgb, depth_m=depth,
                       pose=pose, intr=INTR, is_keyframe=True, frame_epoch=epoch, session_id=sid, kf_id=kf_id)


class FakeWM:
    """Objects linked to frames; apply_pose_corrections moves them by their frame's delta."""

    def __init__(self, objects=None) -> None:
        self.objects = list(objects or [])
        self.pose_demote_thresh_m = 0.5
        self.last_correction = None
        self.applied = []
        self.cleared = 0

    def snapshot_for_session(self) -> SessionSnapshot:
        idx = {}
        for o in self.objects:
            idx.setdefault(o["last_update_frame_id"], []).append(o["id"])
        return SessionSnapshot(objects=[dict(o) for o in self.objects], frame_to_objects=idx, crops={})

    def clear(self):
        self.cleared += 1
        self.objects = []
        return {}

    def apply_pose_corrections(self, fc):
        self.applied.append(dict(fc))
        moved = 0
        for o in self.objects:
            d = fc.get(o["last_update_frame_id"])
            if d is not None:
                _p, dR, dt = d
                o["xyz_world"] = (dR @ o["xyz_world"] + dt).astype(np.float32)
                moved += 1
        self.last_correction = {"corrected": moved, "demoted": [], "max_shift_m": 0.0}
        return moved


def obj(oid, xyz, fid):
    return {"id": oid, "xyz_world": np.asarray(xyz, dtype=np.float32), "confirmed": True, "label_scores": {},
            "last_update_frame_id": fid}


def world(tmp_path, objects=None):
    wm = FakeWM(objects)
    vs = VisualizationServer(cfg={}, working_memory=wm)
    reg = SessionRegistry({"sessions": {"dir": str(tmp_path / "s"), "persist_on_freeze": False}},
                          live=LiveComponents(wm=wm, vis_server=vs), source="fake", persist_async=False)
    vs.sessions = reg
    return wm, vs, reg


def feed(vs, reg, pkt):
    reg.observe_frame(pkt)
    vs.handle_frame_packet(pkt)


def test_frame_packet_tags_session_and_learns_alias(tmp_path):
    wm, vs, reg = world(tmp_path)
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0, kf_id="42"))
    rec = vs.registry.get("ws_1")
    assert rec is not None and rec.session_key == reg.live.key and vs._live_session_key == reg.live.key
    assert vs._kf_alias == {"42": "ws_1"} and vs._alias_stamp == {"ws_1": 1_000_000_000}
    assert "ws_1" in vs._pose_history and "ws_1" in reg.live.pose_history
    assert vs.session_stats()["aliases"] == 1


def test_kf_pose_update_coalesced_into_one_batch(tmp_path):
    wm, vs, reg = world(tmp_path, [obj("o1", (1.0, 0.0, 0.5), "ws_1")])
    for i in (1, 2, 3):
        feed(vs, reg, make_pkt(i, 1_000_000_000 * i, float(i), kf_id=str(40 + i)))
    for i in (1, 2, 3):
        vs.handle_kf_pose_update(str(40 + i), pose_at(float(i), 0.0, 0.1))
    assert vs.session_stats()["pending_updates"] == 3 and vs._pending_timer is not None
    vs.flush_pending_corrections()
    assert vs._pending_timer is None and vs.session_stats()["pending_updates"] == 0
    assert len(wm.applied) == 1 and sorted(wm.applied[0]) == ["ws_1", "ws_2", "ws_3"]
    assert reg.live.counters["refinements"] == 1 and reg.live.counters["corrections_applied"] == 3
    assert np.allclose(reg.live.pose_history["ws_2"][:3, 3], [2.0, 0.0, 0.1])
    assert np.allclose(vs._pose_history["ws_2"][:3, 3], [2.0, 0.0, 0.1])
    assert np.allclose(wm.objects[0]["xyz_world"], [1.0, 0.0, 0.6])
    ev = reg.live.timeline[-1]
    assert ev.kind == "refine" and ev.detail["n_keyframes"] == 3 and ev.detail["objects_moved"] == 1
    vs.flush_pending_corrections()                       # nothing pending: no second batch
    assert len(wm.applied) == 1


def test_kf_pose_update_timer_flush(tmp_path):
    wm, vs, reg = world(tmp_path)
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0, kf_id="41"))
    vs.handle_kf_pose_update("41", pose_at(1.0, 0.0, 0.1))
    deadline = time.monotonic() + 2.0
    while reg.live.counters["refinements"] < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert reg.live.counters["refinements"] == 1 and vs._pending_timer is None


def test_kf_pose_update_unknown_alias_counts(tmp_path):
    wm, vs, reg = world(tmp_path)
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0, kf_id="41"))
    vs.handle_kf_pose_update("99", pose_at(0.0))
    assert vs.session_stats()["corrections_unaliased"] == 1 and vs.session_stats()["pending_updates"] == 0
    assert reg.live.counters["refinements"] == 0


def test_freeze_live_detaches_and_clears(tmp_path):
    wm, vs, reg = world(tmp_path, [obj("o1", (1.0, 0.0, 0.5), "ws_1")])
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0, kf_id="41"))
    feed(vs, reg, make_pkt(2, 2_000_000_000, 2.0, kf_id="42"))
    vs.handle_kf_pose_update("42", pose_at(2.0, 0.0, 0.3))   # pending at the boundary: lands before the freeze
    old_key = reg.live.key
    b = reg.end_live()
    f = reg.frozen[0]
    assert f.key == old_key and sorted(f.keyframes) == ["ws_1", "ws_2"] and f.clouds["ws_1"][0].shape[0] > 0
    assert f.kf_alias == {"41": "ws_1", "42": "ws_2"} and f.alias_stamp["ws_2"] == 2_000_000_000
    assert np.allclose(f.keyframes["ws_2"].pose[:3, 3], [2.0, 0.0, 0.3]) and f.counters["refinements"] == 1
    assert vs.registry.count() == 0 and vs._pose_history == {} and vs._kf_alias == {} and vs._alias_stamp == {}
    assert vs._live_session_key == reg.live.key == b.live_key and wm.cleared == 1
    # a later correction for the frozen keyframe refines the frozen snapshot, not the (empty) live one
    s = vs.handle_pose_corrections_batch({"41": pose_at(1.0, 0.0, 0.2)})
    assert s["frozen"] == {f.key: 1} and np.allclose(next(o for o in f.objects if o["id"] == "o1")["xyz_world"], [1.0, 0.0, 0.7])


def test_batch_without_registry_runs_todays_path(tmp_path):
    wm = FakeWM([obj("o1", (1.0, 0.0, 0.5), "ws_1")])
    vs = VisualizationServer(cfg={}, working_memory=wm)
    vs._pose_history["ws_1"] = pose_at(1.0)
    assert vs.handle_pose_corrections_batch({}) is None and wm.applied == []
    vs.handle_pose_corrections_batch({"ws_1": pose_at(1.0, 0.0, 0.2)})
    assert len(wm.applied) == 1 and np.allclose(wm.objects[0]["xyz_world"], [1.0, 0.0, 0.7])
    assert np.allclose(vs._pose_history["ws_1"][:3, 3], [1.0, 0.0, 0.2])


def test_announced_only_batch_reaches_registry(tmp_path):
    wm, vs, reg = world(tmp_path)
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0))
    s = vs.handle_pose_corrections_batch({}, kind="relocalized_in_prior_map", map_ref="X")
    assert s["announced_only"] is True and wm.applied == []
    ev = reg.live.timeline[-2]
    assert ev.kind == "refine" and ev.detail["announced_only"] is True and ev.detail["map_ref"] == "X"
    assert reg.live.aligned_to["session_key"] == "X"


@pytest.mark.parametrize("iteration", range(25))
def test_correction_during_freeze_is_applied_once(tmp_path, iteration):
    wm, vs, reg = world(tmp_path, [obj("o1", (1.0, 0.0, 0.5), "ws_1")])
    feed(vs, reg, make_pkt(1, 1_000_000_000, 1.0))
    go = threading.Event()

    def correct():
        go.wait()
        vs.handle_pose_corrections_batch({"ws_1": pose_at(1.0, 0.0, 0.2)})

    def freeze():
        go.wait()
        reg.end_live()

    threads = [threading.Thread(target=correct), threading.Thread(target=freeze)]
    for t in threads:
        t.start()
    go.set()
    for t in threads:
        t.join(5.0)
    f = reg.frozen[0]
    o1 = next(o for o in f.objects if o["id"] == "o1")
    assert np.allclose(o1["xyz_world"], [1.0, 0.0, 0.7]), "the delta must land exactly once"
    assert np.allclose(f.pose_history["ws_1"][:3, 3], [1.0, 0.0, 0.2])
    assert f.counters["refinements"] + reg.live.counters["refinements"] == 1 or f.counters["corrections_applied"] == 1
