"""Stage 9 of the session registry: the REST surface over a real SessionRegistry
(persistence off) with the API test fakes: /sessions, /sessions/{key},
/sessions/end, merge / unmerge / pin / unpin / delete, /reset with discard,
/healthz.sessions, and the shapes when no registry is wired.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import ResetComponents, create_app
from rtsm.core.sessions import LiveComponents, SessionRegistry, SessionSnapshot


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
        self.image_crops = []


class _WM:
    def __init__(self):
        self.objs = [_Obj("obj_7", [0.5, 0.8, 1.5], "tissue box")]
        self.pose_demote_thresh_m = 0.3
        self.last_correction = None
        self.cleared = 0

    def stats(self):
        return {"objects": len(self.objs), "confirmed": sum(o.confirmed for o in self.objs)}

    def iter_objects(self):
        return list(self.objs)

    def get(self, oid):
        return next((o for o in self.objs if o.id == oid), None)

    def get_robot_pose(self):
        return None

    def clear(self):
        self.cleared += 1
        n = len(self.objs)
        self.objs = []
        return {"objects": n}

    def snapshot_for_session(self):
        objs = [{"id": o.id, "xyz_world": o.xyz_world.copy(), "confirmed": o.confirmed, "label_primary": o.label_primary,
                 "label_scores": dict(o.label_scores), "last_update_frame_id": "ws_1", "n_crops": 0} for o in self.objs]
        return SessionSnapshot(objects=objs, frame_to_objects={"ws_1": [o["id"] for o in objs]}, crops={})

    def apply_pose_corrections(self, fc):
        return 0


def pkt(seq, ts_ns, epoch=0, sid="A"):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = float(seq)
    return SimpleNamespace(time=SimpleNamespace(seq=seq, t_sensor_ns=ts_ns), frame_epoch=epoch, session_id=sid,
                           is_keyframe=True, pose=SimpleNamespace(T_wc=lambda: T), kf_id=None,
                           ingest=SimpleNamespace(rx_seq=seq))


@pytest.fixture
def world(tmp_path):
    wm = _WM()
    reg = SessionRegistry({"sessions": {"dir": str(tmp_path / "sessions"), "persist_on_freeze": False}},
                          live=LiveComponents(wm=wm), source="fake", persist_async=False)
    reg.observe_frame(pkt(1, 1_000_000_000))
    reg.observe_frame(pkt(2, 1_100_000_000))
    app = create_app(working_memory=wm, registry=CollectorRegistry(), reset_components=ResetComponents(), sessions=reg)
    return SimpleNamespace(wm=wm, reg=reg, api=TestClient(app))


def _refill(w):
    w.wm.objs = [_Obj("obj_8", [1.0, 0.0, 1.0], "mug")]
    w.reg.observe_frame(pkt(3, 2_000_000_000, epoch=w.reg.live.epoch, sid=w.reg.live.session_id))


def test_sessions_list_and_detail(world):
    w = world
    live_key = w.reg.live.key
    r = w.api.get("/sessions").json()
    assert r["enabled"] is True and r["live"] == live_key
    assert r["sessions"][0]["key"] == live_key and r["sessions"][0]["live"] is True and r["sessions"][0]["ended_wall_utc"] is None
    end = w.api.post("/sessions/end").json()
    assert end["frozen"] == live_key and end["live"] != live_key and end["discarded"] is False
    r = w.api.get("/sessions").json()
    assert [s["key"] for s in r["sessions"]] == [end["live"], live_key]
    d = w.api.get(f"/sessions/{live_key}").json()
    assert d["key"] == live_key and d["end_reason"] == "manual" and d["counters"]["objects"] == 1
    assert [o["id"] for o in d["objects"]] == ["obj_7"] and d["objects"][0]["xyz_world"] == pytest.approx([0.5, 0.8, 1.5])
    assert [e["kind"] for e in d["timeline"]] == ["start", "freeze"] and d["retention"]["protected_by"] == []
    assert w.api.get("/sessions/nope").status_code == 404
    assert w.wm.cleared == 1                                 # the freeze cleared the live memory


def test_sessions_end_discards_an_empty_live_session(world):
    w = world
    first = w.api.post("/sessions/end").json()
    assert first["discarded"] is False
    second = w.api.post("/sessions/end").json()             # nothing happened since: trivial, discarded
    assert second["frozen"] is None and second["discarded"] is True
    assert w.api.get("/healthz").json()["sessions"]["discarded"] == 1


def test_reset_freezes_by_default_and_discards_on_request(world):
    w = world
    live_key = w.reg.live.key
    r = w.api.post("/reset").json()
    assert r["status"] == "ok" and r["cleared"]["working_memory"] == {"objects": 0}   # the freeze cleared first
    assert r["session"] == {"frozen": live_key, "live": w.reg.live.key, "discarded": False}
    assert len(w.reg.frozen) == 1
    _refill(w)
    key2 = w.reg.live.key
    r2 = w.api.post("/reset", json={"discard": True}).json()
    assert r2["session"]["discarded"] is True and r2["session"]["frozen"] is None and r2["session"]["live"] != key2
    assert len(w.reg.frozen) == 1 and r2["cleared"]["working_memory"] == {"objects": 1}


def test_merge_pin_delete_endpoints(world):
    w = world
    frozen = w.api.post("/sessions/end").json()["frozen"]
    _refill(w)
    m = w.api.post(f"/sessions/{frozen}/merge", json={"into": "live"})
    assert m.status_code == 200 and m.json()["merged_into"] == w.reg.live.key
    again = w.api.post(f"/sessions/{frozen}/merge")
    assert again.status_code == 409 and again.json()["detail"]["reason"] == "already_merged"
    assert w.api.delete(f"/sessions/{frozen}").status_code == 409                 # in the live chain
    assert w.api.post(f"/sessions/{frozen}/unmerge").json()["merged_into"] is None
    assert w.api.post(f"/sessions/{frozen}/pin").json()["pinned"] is True
    assert "pinned" in w.api.get(f"/sessions/{frozen}").json()["retention"]["protected_by"]
    d = w.api.delete(f"/sessions/{frozen}")
    assert d.status_code == 409 and d.json()["detail"]["reason"] == "pinned"
    assert w.api.delete(f"/sessions/{w.reg.live.key}").json()["detail"]["reason"] == "live_session"
    assert w.api.delete("/sessions/nope").status_code == 404
    assert w.api.post("/sessions/nope/pin").status_code == 404
    assert w.api.delete(f"/sessions/{frozen}?force=true").status_code == 204
    assert w.api.get("/sessions").json()["sessions"][0]["key"] == w.reg.live.key and len(w.reg.frozen) == 0


def test_healthz_sessions_block(world):
    w = world
    h = w.api.get("/healthz").json()
    assert h["status"] == "ok"
    s = h["sessions"]
    assert s["enabled"] is True and s["live"] == w.reg.live.key and s["n_frozen"] == 0 and s["boundaries"] == 0
    w.api.post("/sessions/end")
    s = w.api.get("/healthz").json()["sessions"]
    assert s["n_frozen"] == 1 and s["boundaries"] == 1 and s["last_boundary"]["reason"] == "manual"


def test_shapes_without_a_registry():
    api = TestClient(create_app(working_memory=_WM(), registry=CollectorRegistry(), reset_components=ResetComponents()))
    assert api.get("/sessions").json() == {"enabled": False, "live": None, "sessions": []}
    assert api.get("/sessions/x").status_code == 404
    assert api.post("/sessions/end").status_code == 404
    r = api.post("/reset").json()
    assert "session" not in r and r["status"] == "ok"
    assert "sessions" not in api.get("/healthz").json()


def test_disabled_registry_reset_keeps_todays_behaviour(tmp_path):
    wm = _WM()
    reg = SessionRegistry({"sessions": {"enable": False, "dir": str(tmp_path)}}, live=LiveComponents(wm=wm), persist_async=False)
    api = TestClient(create_app(working_memory=wm, registry=CollectorRegistry(), reset_components=ResetComponents(), sessions=reg))
    r = api.post("/reset").json()
    assert "session" not in r and r["cleared"]["working_memory"] == {"objects": 1}
    assert api.post("/sessions/end").status_code == 409
    assert api.get("/sessions").json()["enabled"] is False


def test_dashboard_page_is_not_cached(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html><script src=\"/assets/index-abc.js\"></script></html>")
    (static / "assets" / "index-abc.js").write_text("console.log(1)")
    api = TestClient(create_app(working_memory=_WM(), registry=CollectorRegistry(), static_dir=str(static)))
    page = api.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache, must-revalidate"
    asset = api.get("/assets/index-abc.js")
    assert asset.status_code == 200 and "no-cache" not in asset.headers.get("cache-control", "")
    assert "cache-control" not in {k.lower() for k in api.get("/healthz").headers} or "no-cache" not in api.get("/healthz").headers.get("cache-control", "")
