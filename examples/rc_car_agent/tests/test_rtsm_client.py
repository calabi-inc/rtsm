"""rtsm_client parse tests against a canned local HTTP server (shapes match
rtsm/api/server.py: /healthz, /stats, /search/semantic)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rtsm_client import RtsmClient

POSE = {"xyz": [1.0, 0.3, -2.0], "quaternion_xyzw": [0, 0.7071, 0, 0.7071],
        "timestamp": 1751000000.25}

CANNED = {
    "/healthz": {"status": "ok"},
    "/stats": {"objects": 21, "confirmed": 12, "avg_hits": 3.4,
               "upserts_total": 40, "robot_pose": POSE},
    "/search/semantic": {
        "query": "red mug",
        "robot_pose": POSE,
        "results": [
            {"id": "obj_7", "score": 0.1412, "confirmed": True,
             "stability": 0.71, "xyz_world": [0.5, 0.8, 1.5]},
            {"id": "obj_9", "score": 0.09, "confirmed": False,
             "stability": 0.3, "xyz_world": None},
        ],
    },
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        payload = dict(self.server.canned.get(path, {}))
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_rtsm():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.canned = {k: dict(v) for k, v in CANNED.items()}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_healthz(mock_rtsm):
    _, url = mock_rtsm
    assert RtsmClient(url).healthz() is True
    assert RtsmClient("http://127.0.0.1:1", timeout_s=0.2).healthz() is False


def test_pose_parse(mock_rtsm):
    _, url = mock_rtsm
    pose = RtsmClient(url).get_robot_pose()
    assert pose is not None
    assert pose.xyz == [1.0, 0.3, -2.0]
    assert pose.quaternion_xyzw == [0, 0.7071, 0, 0.7071]
    assert pose.timestamp == 1751000000.25
    assert pose.fetched_at_mono > 0


def test_pose_none_before_first_frame(mock_rtsm):
    srv, url = mock_rtsm
    srv.canned["/stats"] = {"objects": 0, "confirmed": 0, "robot_pose": None}
    assert RtsmClient(url).get_robot_pose() is None
    assert RtsmClient(url).object_count() == 0


def test_semantic_query_assembles_snapshot(mock_rtsm):
    _, url = mock_rtsm
    res = RtsmClient(url).semantic_query("red mug", top_k=5)
    assert res.query == "red mug"
    assert res.robot_pose is not None            # pose + results in ONE response
    assert len(res.results) == 2
    top = res.results[0]
    assert (top.id, top.confirmed) == ("obj_7", True)
    assert top.xyz_world == [0.5, 0.8, 1.5]
    assert res.results[1].xyz_world is None      # unconfirmed hit without xyz
