"""The session picker's server side: registry viewer hooks and accessors, the
clouds / keyframes / snapshots endpoints, the websocket's session_update,
resync and viewing, and the viz server's broadcast hooks. Real registry with
persistence inline, real VisualizationServer, API through TestClient.
"""
from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import ResetComponents, create_app
from rtsm.core.sessions import LiveComponents, SessionRegistry, SessionSnapshot
from rtsm.visualization.broadcaster import pack_mesh_create
from rtsm.visualization.server import VisualizationServer

JPEG_A = b"\xff\xd8A\xff\xd9"
JPEG_B = b"\xff\xd8B\xff\xd9"


class _Obj:
    def __init__(self, oid, xyz, label):
        self.id, self.xyz_world, self.label_primary = oid, np.asarray(xyz, dtype=np.float32), label
        self.confirmed, self.hits, self.stability = True, 3, 0.8
        self.label_scores = {label: 0.9}
        self.view_bins = {0: None}
        self.created_wall_utc = self.created_mono = 1.0
        self.last_seen_mono = 2.0
        self.last_seen_wall_utc = 1751000000.0
        self.last_seen_px = (10.0, 12.0)
        self.cov_world = np.eye(3, dtype=np.float32)
        self.image_crops = [JPEG_A, JPEG_B]                   # oldest first, as the WM keeps them


class _WM:
    def __init__(self):
        self.objs = [_Obj("obj_7", [0.5, 0.8, 1.5], "tissue box")]
        self.pose_demote_thresh_m = 0.3
        self.last_correction = None

    def stats(self):
        return {"objects": len(self.objs), "confirmed": sum(o.confirmed for o in self.objs)}

    def iter_objects(self):
        return list(self.objs)

    def get(self, oid):
        return next((o for o in self.objs if o.id == oid), None)

    def get_robot_pose(self):
        return None

    def clear(self):
        n = len(self.objs)
        self.objs = []
        return {"objects": n}

    def snapshot_for_session(self):
        objs = [{"id": o.id, "xyz_world": o.xyz_world.copy(), "confirmed": o.confirmed, "label_primary": o.label_primary,
                 "label_scores": dict(o.label_scores), "last_update_frame_id": "ws_1", "n_crops": len(o.image_crops)} for o in self.objs]
        return SessionSnapshot(objects=objs, frame_to_objects={"ws_1": [o["id"] for o in objs]},
                               crops={o.id: list(o.image_crops) for o in self.objs})

    def apply_pose_corrections(self, fc):
        self.last_correction = {"corrected": 0, "demoted": []}
        return 0


def pkt(seq, ts_ns, epoch=0, sid="A", x=None):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = float(seq if x is None else x)
    return SimpleNamespace(time=SimpleNamespace(seq=seq, t_sensor_ns=ts_ns), frame_epoch=epoch, session_id=sid,
                           is_keyframe=True, pose=SimpleNamespace(T_wc=lambda: T), kf_id=None,
                           ingest=SimpleNamespace(rx_seq=seq))


def unpack_clouds(blob: bytes):
    """The dashboard's parser, in Python: consecutive PCLD packets."""
    out = []
    off = 0
    while off < len(blob):
        magic, id_len, n = struct.unpack_from("<4sHI", blob, off)
        assert magic == b"PCLD"
        off += 10
        mid = blob[off:off + id_len].decode()
        off += id_len
        pos = np.frombuffer(blob[off:off + n * 12], dtype=np.float32).reshape(n, 3)
        off += n * 12
        col = np.frombuffer(blob[off:off + n * 3], dtype=np.uint8).reshape(n, 3)
        off += n * 3
        has_pose = blob[off]
        off += 1
        pose = None
        if has_pose:
            pose = np.frombuffer(blob[off:off + 64], dtype=np.float32).reshape(4, 4)
            off += 64
        out.append((mid, pos, col, pose))
    return out


@pytest.fixture
def world(tmp_path):
    """A viz server + registry with a frozen session A (two keyframe clouds, one object with two crops) and a live session."""
    wm = _WM()
    vs = VisualizationServer(cfg={}, working_memory=wm)
    reg = SessionRegistry({"sessions": {"dir": str(tmp_path / "sessions"), "persist_on_freeze": True, "cloud_max_points": 0}},
                          live=LiveComponents(wm=wm, vis_server=vs), source="fake", persist_async=False)
    vs.sessions = reg
    reg.on_change = vs.broadcast_session_update
    reg.on_refined = vs.broadcast_session_refined
    reg.observe_frame(pkt(1, 1_000_000_000, x=1.0))
    reg.observe_frame(pkt(2, 1_100_000_000, x=2.0))
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    cols = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    T1 = np.eye(4, dtype=np.float32); T1[0, 3] = 1.0
    T2 = np.eye(4, dtype=np.float32); T2[0, 3] = 2.0
    vs.registry.register("ws_1", 1_000_000_000, pts, cols, T1, session_key=reg.live.key)
    vs.registry.register("ws_2", 1_100_000_000, pts, cols, T2, session_key=reg.live.key)
    vs._pose_history["ws_1"] = T1.copy()
    vs._pose_history["ws_2"] = T2.copy()
    b = reg.end_live()
    app = create_app(working_memory=wm, registry=CollectorRegistry(), reset_components=ResetComponents(), sessions=reg,
                     vis_server=vs, vis_broadcaster=vs.broadcaster, vis_registry=vs.registry)
    return SimpleNamespace(wm=wm, vs=vs, reg=reg, frozen=b.frozen_key, api=TestClient(app))


# ───────────────────────────── registry hooks and accessors ─────────────────────────────

def test_on_change_fires_on_every_list_change(tmp_path):
    calls = []
    wm = _WM()
    reg = SessionRegistry({"sessions": {"dir": str(tmp_path / "s"), "persist_on_freeze": False}}, live=LiveComponents(wm=wm),
                          persist_async=False)
    reg.on_change = lambda: calls.append(1)
    reg.observe_frame(pkt(1, 1_000_000_000))                 # open live
    assert len(calls) == 1
    b = reg.end_live()                                        # freeze + open next
    assert len(calls) == 2
    reg.merge(b.frozen_key); reg.unmerge(b.frozen_key); reg.pin(b.frozen_key); reg.pin(b.frozen_key, False)
    assert len(calls) == 6
    reg.delete(b.frozen_key)
    assert len(calls) == 7
    reg.on_change = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    reg.observe_frame(pkt(9, 9_000_000_000, epoch=1, sid="B"))   # a failing hook never breaks the registry
    assert reg.live.session_id == "B"


def test_on_refined_payload(world):
    w = world
    seen = []
    w.reg.on_refined = lambda key, poses, objects: seen.append((key, poses, objects))
    T = np.eye(4, dtype=np.float32); T[0, 3] = 1.0; T[2, 3] = 0.25
    s = w.reg.apply_corrections({"ws_1": T})
    assert s["frozen"] == {w.frozen: 1}
    key, poses, objects = seen[0]
    assert key == w.frozen and list(poses) == ["ws_1"] and poses["ws_1"][11] == pytest.approx(0.25)
    assert objects == [{"id": "obj_7", "xyz_world": pytest.approx([0.5, 0.8, 1.75]), "demoted_by_correction": False}]


def test_accessors_keyframes_clouds_snapshots(world):
    w = world
    kfs = w.reg.keyframes(w.frozen)
    assert sorted(k["mesh_id"] for k in kfs) == ["ws_1", "ws_2"] and len(kfs[0]["pose"]) == 16 and kfs[0]["n_points"] == 2
    clouds = w.reg.clouds(w.frozen)
    assert sorted(clouds) == ["ws_1", "ws_2"] and clouds["ws_1"][0].shape == (2, 3)
    assert w.reg.get(w.frozen).last_served_wall_utc is not None
    assert w.reg.snapshots(w.frozen, "obj_7") == [JPEG_B, JPEG_A]            # most recent first, from disk (persisted)
    assert w.reg.snapshots(w.frozen, "ghost") == []
    assert w.reg.keyframes("nope") is None and w.reg.clouds("nope") is None and w.reg.snapshots("nope", "x") is None


def test_snapshots_in_memory_before_persistence(tmp_path):
    wm = _WM()
    reg = SessionRegistry({"sessions": {"dir": str(tmp_path / "s"), "persist_on_freeze": False}}, live=LiveComponents(wm=wm),
                          persist_async=False)
    reg.observe_frame(pkt(1, 1_000_000_000))
    b = reg.end_live()
    assert reg.snapshots(b.frozen_key, "obj_7") == [JPEG_B, JPEG_A]


# ───────────────────────────── REST ─────────────────────────────

def test_keyframes_and_clouds_endpoints(world):
    w = world
    r = w.api.get(f"/sessions/{w.frozen}/keyframes").json()
    assert r["key"] == w.frozen and sorted(k["mesh_id"] for k in r["keyframes"]) == ["ws_1", "ws_2"]
    resp = w.api.get(f"/sessions/{w.frozen}/clouds")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("application/octet-stream")
    assert resp.headers["x-clouds"] == "2"
    packets = unpack_clouds(resp.content)
    by_id = {mid: (pos, col, pose) for mid, pos, col, pose in packets}
    assert sorted(by_id) == ["ws_1", "ws_2"]
    pos, col, pose = by_id["ws_2"]
    assert np.allclose(pos, [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]) and col.tolist() == [[10, 20, 30], [40, 50, 60]]
    assert pose is not None and pose[0, 3] == pytest.approx(2.0)
    assert resp.content == b"".join(pack_mesh_create(mid, p, c, q) for mid, p, c, q in packets)
    assert w.api.get("/sessions/nope/clouds").status_code == 404
    assert w.api.get("/sessions/nope/keyframes").status_code == 404


def test_session_snapshots_mirror_live_shapes(world):
    w = world
    r = w.api.get(f"/sessions/{w.frozen}/objects/obj_7/snapshots").json()
    assert r["id"] == "obj_7" and r["count"] == 2
    assert r["snapshots"][0]["index"] == 0 and r["snapshots"][0]["data"].startswith("data:image/jpeg;base64,")
    assert r["snapshots"][0]["size_bytes"] == len(JPEG_B)
    one = w.api.get(f"/sessions/{w.frozen}/objects/obj_7/snapshots?index=1").json()
    assert one["index"] == 1 and one["total"] == 2 and one["snapshot"].startswith("data:image/jpeg;base64,")
    img = w.api.get(f"/sessions/{w.frozen}/objects/obj_7/snapshots/0/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and img.content == JPEG_B
    assert w.api.get(f"/sessions/{w.frozen}/objects/obj_7/snapshots/5/image").status_code == 404
    assert w.api.get(f"/sessions/{w.frozen}/objects/ghost/snapshots").json() == {"id": "ghost", "count": 0, "snapshots": []}
    assert w.api.get("/sessions/nope/objects/obj_7/snapshots").status_code == 404


# ───────────────────────────── WebSocket ─────────────────────────────

def test_ws_session_update_first_then_viewing_and_resync(world):
    w = world
    w.vs.registry.register("ws_9", 2_000_000_000, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                           np.eye(4, dtype=np.float32), session_key=w.reg.live.key)
    with w.api.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "session_update" and first["enabled"] is True and first["live"] == w.reg.live.key
        assert [s["key"] for s in first["sessions"]] == [w.reg.live.key, w.frozen]
        synced = ws.receive_bytes()                                       # the live keyframe
        assert synced[:4] == b"PCLD"
        ws.send_json({"cmd": "viewing", "session": w.frozen})
        ws.send_json({"cmd": "resync"})
        again = ws.receive_bytes()
        assert again[:4] == b"PCLD"
        objs = ws.receive_json()
        assert objs["type"] == "objects_update" and objs["objects"] == []   # the live memory is empty after the freeze
    assert w.reg.get(w.frozen).last_served_wall_utc is not None


def test_viz_hooks_and_payload(world):
    w = world
    assert w.reg.on_change == w.vs.broadcast_session_update and w.reg.on_refined == w.vs.broadcast_session_refined
    p = w.vs.session_update_payload()
    assert p["enabled"] is True and p["live"] == w.reg.live.key and len(p["sessions"]) == 2
    w.vs.broadcast_session_update()                                      # not running: no scheduling, no error
    w.vs.broadcast_session_refined(w.frozen, {}, [])
    w.vs.note_viewing("live")
    w.vs.note_viewing(None)
    vs2 = VisualizationServer(cfg={}, working_memory=_WM())
    assert vs2.session_update_payload() == {"enabled": False, "live": None, "sessions": []}


def test_ws_ping_pong(world):
    w = world
    with w.api.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "session_update"
        ws.send_json({"cmd": "ping"})
        assert ws.receive_json() == {"type": "pong"}
