"""Agent-server integration tests — FastAPI TestClient over mock RTSM +
mock (recording) ESP32. Covers every Gate-E software item."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from config import load_config
from esp32_bridge import Esp32Bridge
from rtsm_client import RtsmClient
from server import BENCH_DUMMY_GOAL, create_app
from test_esp32_bridge import _RecordingHandler

POSE = {"xyz": [0.0, 0.3, 0.0], "quaternion_xyzw": [0, 0, 0, 1],
        "timestamp": 1751000000.0}
GOOD_STATS = {"objects": 12, "confirmed": 8, "robot_pose": POSE}
COLD_STATS = {"objects": 0, "confirmed": 0, "robot_pose": None}


class _RtsmHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        payload = self.server.routes.get(path)
        if payload is None:
            self.send_response(404); self.end_headers(); return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def env():
    rtsm_srv = ThreadingHTTPServer(("127.0.0.1", 0), _RtsmHandler)
    rtsm_srv.routes = {"/healthz": {"status": "ok"}, "/stats": dict(GOOD_STATS)}
    threading.Thread(target=rtsm_srv.serve_forever, daemon=True).start()

    esp_srv = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    esp_srv.recorded = []
    threading.Thread(target=esp_srv.serve_forever, daemon=True).start()

    cfg = load_config()
    rtsm = RtsmClient(f"http://127.0.0.1:{rtsm_srv.server_address[1]}", timeout_s=1.0)
    bridge = Esp32Bridge(f"http://127.0.0.1:{esp_srv.server_address[1]}",
                         http_timeout_s=0.4)
    yield cfg, rtsm, bridge, rtsm_srv, esp_srv
    rtsm_srv.shutdown()
    esp_srv.shutdown()


def _wait(pred, timeout=2.0, every=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(every)
    return False


# ── boot & readiness ─────────────────────────────────────────────────────


def test_ready_boot(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        s = c.get("/status").json()
        assert s["state"] == "READY"
        assert s["not_ready_reasons"] == []
        assert s["bench"] is False


def test_not_ready_503_with_reasons(env):
    cfg, rtsm, bridge, rtsm_srv, _ = env
    rtsm_srv.routes["/stats"] = dict(COLD_STATS)      # no pose, empty map
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        s = c.get("/status").json()
        assert s["state"] == "NOT_READY"
        assert any("robot_pose" in r for r in s["not_ready_reasons"])
        assert any("map empty" in r for r in s["not_ready_reasons"])
        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 503
        assert "not_ready" in r.json()["detail"]


def test_command_reprobes_preflight_after_recovery(env):
    cfg, rtsm, bridge, rtsm_srv, _ = env
    rtsm_srv.routes["/stats"] = dict(COLD_STATS)
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        assert c.get("/status").json()["state"] == "NOT_READY"
        rtsm_srv.routes["/stats"] = dict(GOOD_STATS)  # phone "just connected"
        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 200                   # re-probe let it through
        assert r.json()["accepted"] is True


# ── stub worker: run / preempt / cancel / soft stop ──────────────────────


def test_command_runs_stub_and_ticks(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert r["accepted"] and r["condition"] == "rtsm"
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("stub_ticks", 0) > 5)
        s = c.get("/status").json()
        assert s["state"] == "RUNNING"
        assert s["task"]["task_id"] == r["task_id"]


def test_second_command_preempts(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        a = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("stub_ticks", 0) > 2)
        b = c.post("/command", json={"goal": "go to the blue backpack"}).json()
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("task_id") == b["task_id"])
        s = c.get("/status").json()
        assert s["last_result"]["task_id"] == a["task_id"]
        assert s["last_result"]["result"] == "preempted"


def test_cancel_returns_to_idle(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")
        c.post("/cancel")
        assert _wait(lambda: c.get("/status").json()["state"] == "READY")
        assert c.get("/status").json()["last_result"]["result"] == "cancelled"


def test_soft_stop_hits_wire_and_cancels(env):
    cfg, rtsm, bridge, _, esp_srv = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")
        c.post("/stop")
        assert _wait(lambda: c.get("/status").json()["state"] == "READY")
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)


# ── e-stop semantics ─────────────────────────────────────────────────────


def test_estop_abandons_blocks_and_rearms(env):
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")

        app.state.srv.monitor.trigger("test-estop")   # the hard channel
        assert _wait(lambda: c.get("/status").json()["state"] == "ESTOPPED")
        assert c.get("/status").json()["last_result"]["result"] == "estopped"

        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 503                   # abandoned, no auto-resume
        assert "ESTOPPED" in str(r.json()["detail"])

        assert c.post("/reset_estop").json()["state"] == "READY"
        assert c.post("/command",
                      json={"goal": "go to the red mug"}).status_code == 200


# ── bench dummy drive (the Gate-E hardware test vehicle) ─────────────────


def test_bench_goal_forbidden_without_bench_flag(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": BENCH_DUMMY_GOAL})
        assert r.status_code == 403


def test_bench_goal_bypasses_preflight_and_drives(env):
    cfg, rtsm, bridge, rtsm_srv, esp_srv = env
    rtsm_srv.routes["/stats"] = dict(COLD_STATS)      # NOT_READY on purpose
    with TestClient(create_app(cfg, rtsm, bridge, bench=True)) as c:
        assert c.get("/status").json()["state"] == "NOT_READY"
        r = c.post("/command", json={"goal": BENCH_DUMMY_GOAL})
        assert r.status_code == 200                   # bench bypass BY DESIGN
        assert _wait(lambda: any(rec["path"] == "/drive"
                                 for rec in esp_srv.recorded))
        c.post("/cancel")
        assert _wait(lambda: c.get("/status").json()["state"] != "RUNNING")
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)
