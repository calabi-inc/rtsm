"""Agent-server integration tests — FastAPI TestClient over mock RTSM +
mock (recording) ESP32. Phase E items (boot/preflight/preempt/cancel/
e-stop/bench) plus the Phase-F mission path: planner -> nav -> monitor,
closed-loop convergence on the kinematic fake car, frame_epoch abort,
bounded stale abort, and trial-JSONL output."""

import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from config import load_config
from esp32_bridge import Esp32Bridge
from fake_car import FakeCar, FakeCarEsp32Handler, FakeRtsmHandler
from rtsm_client import RtsmClient
from server import BENCH_DUMMY_GOAL, create_app
from test_esp32_bridge import _RecordingHandler

POSE = {"xyz": [0.0, 0.3, 0.0], "quaternion_xyzw": [0, 0, 0, 1],
        "timestamp": 1751000000.0, "frame_epoch": 7}
GOOD_STATS = {"objects": 12, "confirmed": 8, "robot_pose": POSE}
COLD_STATS = {"objects": 0, "confirmed": 0, "robot_pose": None}
SEMANTIC = {"query": "red mug", "robot_pose": POSE, "results": [
    {"id": "mug-1", "score": 0.81, "confirmed": True, "stability": 0.9,
     "xyz_world": [0.5, 0.3, 1.5]}]}


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
def env(monkeypatch, tmp_path):
    """Static-pose environment: RTSM serves one frozen pose, so missions
    hold-then-stale_stop (~2.5 s) — long enough to interrupt, short enough
    for tests. The fake-car env below is the one that converges."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # deterministic top1

    rtsm_srv = ThreadingHTTPServer(("127.0.0.1", 0), _RtsmHandler)
    rtsm_srv.routes = {
        "/healthz": {"status": "ok"},
        "/stats": dict(GOOD_STATS),
        "/search/semantic": json.loads(json.dumps(SEMANTIC)),
        "/objects/mug-1": {"label_primary": "red mug"},
    }
    threading.Thread(target=rtsm_srv.serve_forever, daemon=True).start()

    esp_srv = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    esp_srv.recorded = []
    threading.Thread(target=esp_srv.serve_forever, daemon=True).start()

    base = load_config()
    # No physical pad in mock envs — the live-fire gate is tested on its
    # own (test_kill_switch_gate_*), everything else runs with it off.
    cfg = replace(base, trials_output_dir=str(tmp_path),
                  server=replace(base.server, require_verified_estop=False))
    rtsm = RtsmClient(f"http://127.0.0.1:{rtsm_srv.server_address[1]}", timeout_s=1.0)
    bridge = Esp32Bridge(f"http://127.0.0.1:{esp_srv.server_address[1]}",
                         http_timeout_s=0.4)
    yield cfg, rtsm, bridge, rtsm_srv, esp_srv
    rtsm_srv.shutdown()
    esp_srv.shutdown()


@pytest.fixture()
def env_car(monkeypatch, tmp_path):
    """Kinematic fake-car environment: pose advances in response to /drive,
    so the loop actually closes. Sped-up nav timings for test runtime."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    car = FakeCar(x=0.0, z=0.0, yaw=0.0)
    rtsm_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeRtsmHandler)
    rtsm_srv.car = car
    rtsm_srv.semantic_results = [
        {"id": "mug-1", "score": 0.81, "confirmed": True, "stability": 0.9,
         "xyz_world": [0.3, 0.3, 1.5]}]
    rtsm_srv.objects = {"mug-1": {"label_primary": "red mug"}}
    threading.Thread(target=rtsm_srv.serve_forever, daemon=True).start()

    esp_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeCarEsp32Handler)
    esp_srv.recorded = []
    esp_srv.car = car
    threading.Thread(target=esp_srv.serve_forever, daemon=True).start()

    base = load_config()
    cfg = replace(
        base,
        trials_output_dir=str(tmp_path),
        server=replace(base.server, require_verified_estop=False),
        nav=replace(base.nav, tick_s=0.02, poll_hz=20.0, stale_abort_s=1.0),
        baseline=replace(base.baseline, sweep_step_s=0.25, dwell_s=0.3,
                         steps_per_sweep=8, walk_s=0.3, walk_speed=0.2),
    )
    rtsm = RtsmClient(f"http://127.0.0.1:{rtsm_srv.server_address[1]}", timeout_s=1.0)
    bridge = Esp32Bridge(f"http://127.0.0.1:{esp_srv.server_address[1]}",
                         http_timeout_s=0.4)
    yield cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path
    rtsm_srv.shutdown()
    esp_srv.shutdown()


def _wait(pred, timeout=2.0, every=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(every)
    return False


def _result(client):
    return (client.get("/status").json().get("last_result") or {}).get("result")


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


# ── mission worker: run / preempt / cancel / soft stop ───────────────────


def test_command_runs_mission_dead_feed_never_drives(env):
    """The static env IS a dead feed (pose timestamp never advances past
    the plan-time sample) — the mission must run, but the car must NEVER
    be energized on a feed that died before the mission started."""
    cfg, rtsm, bridge, _, esp_srv = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert r["accepted"] and r["condition"] == "rtsm"
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 3)
        s = c.get("/status").json()
        assert s["state"] == "RUNNING"
        assert s["task"]["task_id"] == r["task_id"]
        assert s["task"]["phase"] == "driving"
        assert s["task"]["planner_path"] == "top1_no_llm"   # no API key
        assert s["task"]["target_id"] == "mug-1"
        assert _wait(lambda: _result(c) == "stale_stop",
                     timeout=cfg.nav.stale_abort_s + 3.0)
        assert not any(rec["path"] == "/drive" for rec in esp_srv.recorded)


def test_not_found_goal_no_motion(env):
    cfg, rtsm, bridge, rtsm_srv, esp_srv = env
    rtsm_srv.routes["/search/semantic"] = {"query": "unicorn",
                                           "robot_pose": POSE, "results": []}
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the unicorn"})
        assert _wait(lambda: _result(c) == "not_found")
        assert not any(rec["path"] == "/drive" for rec in esp_srv.recorded)


def test_second_command_preempts(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        a = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 2)
        b = c.post("/command", json={"goal": "go to the red mug"}).json()
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


def test_static_pose_mission_stale_stops_bounded(env):
    """A frozen pose feed must end the mission via stale_stop well before
    the 60 s trial timeout (bounded blind-driving, audit pin)."""
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        t0 = time.monotonic()
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: _result(c) == "stale_stop",
                     timeout=cfg.nav.stale_abort_s + 3.0)
        assert time.monotonic() - t0 < cfg.nav.stale_abort_s + 3.0


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


def test_reset_estop_refused_while_mission_live(env):
    """CRITICAL review fix: /reset_estop during a live mission would clear
    the stop_event/latch before nav observes it — the e-stop would be
    silently un-fired and the car would resume. Re-arm is idle-only."""
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")
        r = c.post("/reset_estop")                    # mission still live
        assert r.status_code == 409
        # After the mission finalizes (here: estop it), re-arm works.
        app.state.srv.monitor.trigger("test-estop")
        assert _wait(lambda: c.get("/status").json()["state"] == "ESTOPPED")
        assert c.post("/reset_estop").status_code == 200


def test_goal_queued_before_estop_never_autoruns(env):
    """A command queued before the e-stop must NOT auto-drive the car
    after /reset_estop — every post-e-stop motion needs a fresh command."""
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")
        c.post("/command", json={"goal": "go to the blue backpack"})  # queued
        app.state.srv.monitor.trigger("test-estop")
        assert _wait(lambda: c.get("/status").json()["state"] == "ESTOPPED")
        c.post("/reset_estop")
        time.sleep(0.3)                               # worker would have started it
        s = c.get("/status").json()
        assert s["state"] != "RUNNING"
        assert s["task"] is None


def test_stale_preempt_flag_does_not_kill_next_mission(env):
    """Edge-triggered preempt/cancel signals die at the task boundary —
    a stale flag must never insta-abort a freshly dequeued mission."""
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        app.state.srv._preempt.set()                  # stale, no mission running
        app.state.srv._cancel.set()
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 3)            # mission survives + runs
        assert c.get("/status").json()["state"] == "RUNNING"


def test_baseline_condition_accepted(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": "go to the red mug",
                                     "condition": "baseline"})
        assert r.status_code == 200
        assert r.json()["condition"] == "baseline"


def test_plan_error_does_not_kill_worker(env):
    """CRITICAL review fix: a planning-time RTSM error must be a failed
    TRIAL, not a dead worker thread wedged in RUNNING forever."""
    cfg, rtsm, bridge, rtsm_srv, _ = env
    del rtsm_srv.routes["/search/semantic"]           # semantic_query -> HTTP 404
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: _result(c) == "plan_error")
        s = c.get("/status").json()
        assert s["state"] == "READY"                  # worker alive, not wedged
        rtsm_srv.routes["/search/semantic"] = json.loads(json.dumps(SEMANTIC))
        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 200                   # and still accepting work


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


# ── Phase-F Gate: closed loop on the kinematic fake car ──────────────────


def test_closed_loop_converges_e2e(env_car):
    cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path = env_car
    target = rtsm_srv.semantic_results[0]["xyz_world"]
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert _wait(lambda: _result(c) == "arrived", timeout=15.0)
        assert car.ground_dist_to(target) <= cfg.nav.arrival_threshold_m + 0.2
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)

        # Trial JSONL: exists, parses, carries the E1 fields.
        log = tmp_path / f"{r['task_id']}.jsonl"
        records = [json.loads(l) for l in log.read_text().splitlines()]
        types = [rec["type"] for rec in records]
        assert types[0] == "trial_start" and types[-1] == "trial_end"
        assert "tick" in types
        assert records[0]["frame_epoch"] == car.epoch
        assert records[-1]["result"] == "arrived"
        assert records[-1]["tta_s"] is not None
        assert records[-1]["censored"] is False


def test_frame_epoch_abort_e2e(env_car):
    cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path = env_car
    rtsm_srv.semantic_results[0]["xyz_world"] = [0.0, 0.3, 8.0]   # far target
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 3)
        car.epoch = car.epoch + 1                     # new sender session
        assert _wait(lambda: _result(c) == "frame_reset", timeout=5.0)
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)


def test_stale_abort_bounded_e2e(env_car):
    cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path = env_car
    rtsm_srv.semantic_results[0]["xyz_world"] = [0.0, 0.3, 8.0]   # far target
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 3)
        frozen_at = time.monotonic()
        car.freeze()
        assert _wait(lambda: _result(c) == "stale_stop",
                     timeout=cfg.nav.stale_abort_s + 3.0)
        # Bounded: stopped within stale_abort_s (+ generous slack), never
        # blind-driving until the 60 s trial timeout.
        assert time.monotonic() - frozen_at < cfg.nav.stale_abort_s + 2.0
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)


def test_baseline_e2e_sweeps_acquires_arrives(env_car):
    """Phase-H gate: condition (b) end-to-end. Target starts OUTSIDE the
    camera's field of view (behind the car); the freshness gate keeps the
    memory answer invisible; the sweep rotates until the target is
    actually seen; then the same closed-loop drive arrives."""
    cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path = env_car
    rtsm_srv.visibility = {"fov_deg": 70, "range_m": 8.0}
    target = [0.0, 0.3, -1.5]                        # directly behind (yaw 0)
    rtsm_srv.semantic_results[0]["xyz_world"] = target
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/command", json={"goal": "go to the red mug",
                                     "condition": "baseline"}).json()
        assert _wait(lambda: _result(c) == "arrived", timeout=30.0)
        assert car.ground_dist_to(target) <= cfg.nav.arrival_threshold_m + 0.2

        log_path = tmp_path / f"{r['task_id']}.jsonl"
        records = [json.loads(l) for l in log_path.read_text().splitlines()]
        start = records[0]
        assert start["condition"] == "baseline"
        assert start["planner"]["planner_path"] == "baseline_fresh"
        assert start["rng_seed"] is not None
        acquired = [rec for rec in records if rec["type"] == "event"
                    and rec["name"] == "baseline_acquired"]
        assert len(acquired) == 1
        ev = acquired[0]
        assert ev["search_time_s"] > 0
        assert ev["planner_path"] == "baseline_fresh_top1"   # no API key
        assert ev["pose"] is not None and ev["xyz_world"] is not None
        assert ev["hit_age_s"] is not None
        assert records[-1]["result"] == "arrived"

        # The server-written log must satisfy the aggregation parser —
        # including the baseline PE denominator (writer/parser contract).
        from aggregate import parse_trial
        parsed = parse_trial(log_path)
        assert parsed["optimal_m"] is not None and parsed["optimal_m"] > 0.5
        assert parsed["path_len_m"] is not None
        assert parsed["search_time_s"] == ev["search_time_s"]
        # Drive ticks share the MISSION clock: they come after acquisition.
        ticks = [rec for rec in records if rec["type"] == "tick"]
        assert ticks and ticks[0]["t"] >= ev["t"]


def test_preempt_during_real_nav(env_car):
    cfg, rtsm, bridge, rtsm_srv, esp_srv, car, tmp_path = env_car
    rtsm_srv.semantic_results[0]["xyz_world"] = [0.0, 0.3, 8.0]   # far target
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        a = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("ticks", 0) > 3)
        b = c.post("/command", json={"goal": "go to the red mug"}).json()
        assert _wait(lambda: (c.get("/status").json().get("task") or {})
                     .get("task_id") == b["task_id"], timeout=5.0)
        s = c.get("/status").json()
        assert s["last_result"]["task_id"] == a["task_id"]
        assert s["last_result"]["result"] == "preempted"
        # Safe stop hit the wire between the two drives.
        assert any(rec["path"] == "/stop" for rec in esp_srv.recorded)


# ── operator console: /ui, /trial_log, /preflight, status extensions ─────


def test_ui_served_at_root_and_ui(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        for path in ("/", "/ui"):
            r = c.get(path)
            assert r.status_code == 200
            assert "text/html" in r.headers["content-type"]
            assert "rc-car console" in r.text
            # Locked safety model: the page must not carry a hard-e-stop
            # HTTP control — soft stop only.
            assert "reset_estop" in r.text            # re-arm is allowed
            assert "Stop (soft)" in r.text


def test_status_carries_clock_and_limits(env):
    """The console renders elapsed/budget from SERVER truth — wall clock
    and config limits must ride every /status payload."""
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        s = c.get("/status").json()
        assert abs(s["now"] - time.time()) < 5.0
        assert s["limits"]["timeout_rtsm_s"] == cfg.nav.timeout_rtsm_s
        assert s["limits"]["timeout_baseline_s"] == cfg.nav.timeout_baseline_s
        assert s["limits"]["arrival_threshold_m"] == cfg.nav.arrival_threshold_m


def test_trial_log_404_before_first_trial(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        assert c.get("/trial_log").status_code == 404


def test_trial_log_tails_current_then_finished_trial(env):
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: _result(c) == "stale_stop",
                     timeout=cfg.nav.stale_abort_s + 3.0)
        full = c.get("/trial_log").json()
        assert full["records"][0]["type"] == "trial_start"
        assert full["records"][-1]["type"] == "trial_end"
        assert full["records"][0]["planner"]["target_id"] == "mug-1"
        # tail=N returns the LAST N records (the live view wants the newest).
        tail = c.get("/trial_log", params={"tail": 2}).json()
        assert len(tail["records"]) == 2
        assert tail["records"][-1]["type"] == "trial_end"
        assert tail["total_lines"] == full["total_lines"]


def test_status_surfaces_planner_score_and_reason_keys(env):
    """The console shows the Haiku pick without reading the log — score
    and reason must ride the task dict (pre-seeded, updated at plan)."""
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: _result(c) is not None,
                     timeout=cfg.nav.stale_abort_s + 3.0)
        last = c.get("/status").json()["last_result"]
        assert last["target_score"] == pytest.approx(0.81)
        assert "planner_reason" in last               # None on the top1 path


def test_preflight_endpoint_reprobes(env):
    cfg, rtsm, bridge, rtsm_srv, _ = env
    rtsm_srv.routes["/stats"] = dict(COLD_STATS)
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        r = c.post("/preflight").json()
        assert r["state"] == "NOT_READY" and r["reasons"]
        rtsm_srv.routes["/stats"] = dict(GOOD_STATS)  # phone reconnected
        r = c.post("/preflight").json()
        assert r["state"] == "READY" and r["reasons"] == []


# ── adversarial-review fixes (2026-08-09 console review) ─────────────────


def test_kill_switch_gate_blocks_all_motion_until_verified(env):
    """require_verified_estop: EVERY motion goal — bench included — 503s
    until the pad is bound AND press-proven. The remote-console gate."""
    cfg, rtsm, bridge, *_ = env
    from dataclasses import replace as _r
    gated = _r(cfg, server=_r(cfg.server, require_verified_estop=True))
    app = create_app(gated, rtsm, bridge, bench=True)
    with TestClient(app) as c:
        s = c.get("/status").json()
        assert any("kill switch" in r for r in s["not_ready_reasons"])
        for goal in ("go to the red mug", BENCH_DUMMY_GOAL):
            r = c.post("/command", json={"goal": goal})
            assert r.status_code == 503
            assert "kill_switch" in r.json()["detail"]
        # Operator plugs in the pad and presses a button (live-fire).
        mon = app.state.srv.monitor
        mon.gamepad_available = True
        mon.binding_verified = True
        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 200
        # Pad sleeps mid-session: binding_verified resets — the NEXT
        # command must block again even though preflight's _ready is
        # still cached True (the gate reads the monitor fresh).
        assert _wait(lambda: _result(c) is not None,
                     timeout=cfg.nav.stale_abort_s + 3.0)
        mon.binding_verified = False
        r = c.post("/command", json={"goal": "go to the red mug"})
        assert r.status_code == 503
        assert "kill_switch" in r.json()["detail"]


def test_worker_never_claims_task_after_estop_latch(env, tmp_path):
    """X pressed in the submit->claim window (e.g. during the previous
    task's preempt safe-stop): the worker must finalize the queued task as
    estopped IMMEDIATELY — no RUNNING state hiding the ESTOPPED banner,
    no phantom trial JSONL in the paper data."""
    cfg, rtsm, bridge, *_ = env
    from server import AgentServer
    srv = AgentServer(cfg, rtsm, bridge)
    srv.run_preflight()
    srv.submit("go to the red mug", None)            # queued; worker not running
    srv.monitor.trigger("test-x")                    # latch lands before claim
    worker = threading.Thread(target=srv._worker_loop, daemon=True)
    worker.start()
    assert _wait(lambda: srv.last_result is not None)
    srv._shutdown.set()
    assert srv.last_result["result"] == "estopped"
    assert "before mission start" in srv.last_result["detail"]
    assert srv.status()["state"] == "ESTOPPED"
    assert srv.last_result["trial_log"] is None      # no logger was opened
    assert list(tmp_path.glob("*.jsonl")) == []      # no phantom trial file


def test_preflight_409_while_mission_live(env):
    """Preflight probes (2 s-budget ESP32 GETs) must never contend with a
    live drive's heartbeat — idle-only, like reset_estop."""
    cfg, rtsm, bridge, *_ = env
    with TestClient(create_app(cfg, rtsm, bridge)) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: c.get("/status").json()["state"] == "RUNNING")
        assert c.post("/preflight").status_code == 409
        assert _wait(lambda: _result(c) is not None,
                     timeout=cfg.nav.stale_abort_s + 3.0)
        assert c.post("/preflight").status_code == 200


def test_preflight_reports_derived_state_when_idle_latched(env):
    """Idle e-stop never passes through a mission finalize, so raw state
    stays READY — /preflight must report the derived ESTOPPED, matching
    /status and /command's 503."""
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        app.state.srv.monitor.trigger("test-estop")  # idle latch
        r = c.post("/preflight").json()
        assert r["state"] == "ESTOPPED"


def test_trial_log_falls_back_past_logless_current_task(env):
    """A current task whose logger isn't up (or failed on OSError, or a
    bench run) must not 404 away the previous trial's existing file."""
    cfg, rtsm, bridge, *_ = env
    app = create_app(cfg, rtsm, bridge)
    with TestClient(app) as c:
        c.post("/command", json={"goal": "go to the red mug"})
        assert _wait(lambda: _result(c) is not None,
                     timeout=cfg.nav.stale_abort_s + 3.0)
        done_id = c.get("/status").json()["last_result"]["task_id"]
        srv = app.state.srv
        with srv._lock:                              # simulate the logless window
            srv.current = {"task_id": "t-logless", "trial_log": None}
        try:
            r = c.get("/trial_log")
            assert r.status_code == 200
            assert r.json()["task_id"] == done_id    # previous trial served
        finally:
            with srv._lock:
                srv.current = None


def test_api_token_guards_posts_but_not_reads(env):
    cfg, rtsm, bridge, *_ = env
    from dataclasses import replace as _r
    secured = _r(cfg, server=_r(cfg.server, api_token="s3cret"))
    with TestClient(create_app(secured, rtsm, bridge)) as c:
        assert c.get("/status").status_code == 200   # telemetry stays open
        assert c.get("/ui").status_code == 200
        for path in ("/cancel", "/stop", "/preflight", "/reset_estop"):
            assert c.post(path).status_code == 401
        assert c.post("/command",
                      json={"goal": "go to the red mug"}).status_code == 401
        assert c.post("/cancel",
                      headers={"X-Auth-Token": "s3cret"}).status_code == 200
        r = c.post("/command", json={"goal": "go to the red mug"},
                   headers={"X-Auth-Token": "s3cret"})
        assert r.status_code == 200
