"""
The server-side API contract the Python SDK (``rtsm/client.py``) reads, pinned
against ``create_app`` with fakes (owed from the repo split, 2026-09-29).

Every assertion is a field the client parses -- nothing more. The server's
JSON is fed through the client's own parsers (``_parse_pose``, ``_parse_search``)
so a renamed or dropped field fails here before it reaches a robot:
  /healthz.status; /stats (objects, confirmed, robot_pose.{xyz, quaternion_xyzw,
  timestamp, frame_epoch}, forward_clearance.{clearance_m, valid_frac, timestamp});
  /objects + /objects/{id} (id, label_primary, xyz_world, confirmed, stability,
  hits); /objects/{id}/snapshots/0/image (image/jpeg, the latest crop);
  /search/semantic and /search/label (query, robot_pose, results[{id, score,
  confirmed, stability, xyz_world, last_seen_wall_utc}], label results also
  matched_label).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import create_app
from rtsm.client import PoseSample, RtsmClient, SemanticResult

POSE = {"xyz": [1.0, 0.3, -2.0], "quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071], "timestamp": 1751000000.25,
        "frame_epoch": 2, "sensor_ts_ns": 12345, "pose_clock": "sender", "age_s": 0.1, "stale": False}
CLEARANCE = {"clearance_m": 1.4, "valid_frac": 0.9, "timestamp": 1751000000.3}
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"


class _Obj:
    def __init__(self, oid, xyz, label, confirmed=True, hits=3, stability=0.8, crops=()):
        self.id, self.xyz_world, self.label_primary = oid, np.asarray(xyz, dtype=np.float32), label
        self.confirmed, self.hits, self.stability = confirmed, hits, stability
        self.label_scores = {label: 0.9, "thing": 0.1}
        self.view_bins = {0: None}
        self.created_wall_utc = self.created_mono = 1.0
        self.last_seen_mono = 2.0
        self.last_seen_wall_utc = 1751000000.0
        self.last_seen_px = (10.0, 12.0)
        self.cov_world = np.eye(3, dtype=np.float32)
        self.image_crops = list(crops)


class _WM:
    def __init__(self):
        self.objs = [_Obj("obj_7", [0.5, 0.8, 1.5], "tissue box", crops=[b"old", JPEG]), _Obj("obj_9", [2.0, 0.0, 1.0], "mug", confirmed=False, hits=1)]

    def stats(self):
        return {"objects": len(self.objs), "confirmed": sum(o.confirmed for o in self.objs), "avg_hits": 2.0, "upserts_total": 4, "robot_pose": dict(POSE)}

    def iter_objects(self):
        return list(self.objs)

    def get(self, oid):
        return next((o for o in self.objs if o.id == oid), None)

    def get_robot_pose(self):
        return dict(POSE)


class _Clip:
    _prompt_wrap = False

    def encode_text(self, text):
        return np.ones(4, dtype=np.float32)


class _Vectors:
    def search(self, emb, top_k=5):
        return [("obj_7", 0.42), ("ghost", 0.1)]

    def stats(self):
        return {"count": 2}


@pytest.fixture
def api():
    app = create_app(working_memory=_WM(), clip_adapter=_Clip(), vectors=_Vectors(), registry=CollectorRegistry(),
                     extra_stats_provider=lambda: {"forward_clearance": dict(CLEARANCE)})
    return TestClient(app)


def test_healthz_and_stats_shapes_parse_through_the_sdk(api):
    assert api.get("/healthz").json()["status"] in ("ok", "degraded")
    stats = api.get("/stats").json()
    assert stats["objects"] == 2 and stats["confirmed"] == 1
    pose = RtsmClient._parse_pose(stats["robot_pose"])
    assert isinstance(pose, PoseSample) and list(pose.xyz) == [1.0, 0.3, -2.0] and list(pose.quaternion_xyzw) == [0.0, 0.7071, 0.0, 0.7071]
    assert pose.timestamp == 1751000000.25 and pose.frame_epoch == 2 and pose.fetched_at_mono > 0
    assert RtsmClient._parse_pose(None) is None
    c = stats["forward_clearance"]
    assert set(c) >= {"clearance_m", "valid_frac", "timestamp"}


def test_objects_list_and_detail_carry_what_the_client_reads(api):
    body = api.get("/objects").json()
    assert {"total", "offset", "limit", "count", "objects"} <= set(body) and body["total"] == 2
    o = body["objects"][0]
    assert {"id", "label_primary", "xyz_world", "confirmed", "stability", "hits"} <= set(o)
    assert o["id"] == "obj_7" and o["label_primary"] == "tissue box" and o["xyz_world"] == pytest.approx([0.5, 0.8, 1.5])
    detail = api.get("/objects/obj_7").json()
    assert detail["label_primary"] == "tissue box" and "label_scores" in detail
    missing = api.get("/objects/nope")                                       # a missing object is a 200 with an error field (the client maps it to None)
    assert missing.status_code == 200 and missing.json() == {"error": "not_found", "id": "nope"}
    only = api.get("/objects", params={"confirmed_only": "true"}).json()
    assert [x["id"] for x in only["objects"]] == ["obj_7"]


def test_snapshot_image_is_the_latest_crop_as_jpeg(api):
    r = api.get("/objects/obj_7/snapshots/0/image")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/jpeg") and r.content == JPEG
    assert api.get("/objects/obj_9/snapshots/0/image").status_code == 404      # no crops
    assert api.get("/objects/obj_7/snapshots/5/image").status_code == 404      # out of range


def test_semantic_and_label_search_parse_through_the_sdk(api):
    sem = api.get("/search/semantic", params={"query": "tissue box", "top_k": 5}).json()
    assert sem["query"] == "tissue box" and sem["robot_pose"]["xyz"] == POSE["xyz"]
    res = RtsmClient("http://unused")._parse_search(sem, "tissue box")
    assert isinstance(res, SemanticResult) and res.robot_pose.frame_epoch == 2
    assert [h.id for h in res.results] == ["obj_7", "ghost"]
    top = res.results[0]
    assert top.score == pytest.approx(0.42) and top.confirmed is True and top.stability == pytest.approx(0.8)
    assert top.xyz_world == pytest.approx([0.5, 0.8, 1.5]) and top.last_seen_wall_utc == pytest.approx(1751000000.0)
    ghost = res.results[1]                                                    # a vector id without a memory object is never confirmed
    assert ghost.confirmed is False and ghost.xyz_world is None
    lab = api.get("/search/label", params={"query": "tissue", "top_k": 5}).json()
    res2 = RtsmClient("http://unused")._parse_search(lab, "tissue")
    assert [h.id for h in res2.results] == ["obj_7"] and lab["results"][0]["matched_label"] == "tissue box"
    assert lab["results"][0]["score"] == pytest.approx(0.9)
    assert api.get("/search/label", params={"query": "   "}).status_code == 422
