"""
esp32_bridge tests against a real local HTTP server (not mocked requests) —
the send-gating discipline is about actual wire behavior.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from esp32_bridge import Esp32Bridge


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records every request (path, body, headers) into server.recorded."""

    def _respond(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.server.recorded.append(
            {
                "path": self.path,
                "body": json.loads(raw) if raw else None,
                "connection": self.headers.get("Connection"),
                "t": time.monotonic(),
            }
        )
        self._respond({"ok": True})

    def do_GET(self):
        self.server.recorded.append(
            {"path": self.path, "body": None,
             "connection": self.headers.get("Connection"), "t": time.monotonic()}
        )
        if self.path == "/battery":
            self._respond({"mv": 7400})
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ESP32 mock online\n")

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def mock_esp32():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    srv.recorded = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _bridge(url, **kw):
    defaults = dict(drive_rate_hz=10, heartbeat_s=0.25,
                    change_epsilon=0.04, http_timeout_s=0.6)
    defaults.update(kw)
    return Esp32Bridge(url, **defaults)


def _drive_posts(srv):
    return [r for r in srv.recorded if r["path"] == "/drive"]


# ── send gating ──────────────────────────────────────────────────────────


def test_send_gating_heartbeat_not_spam(mock_esp32):
    # Unchanged command called at ~50 Hz for 1 s → only initial + heartbeats
    # (~0.25 s apart) hit the wire: expect ~4-5 sends, NOT ~50.
    srv, url = mock_esp32
    b = _bridge(url)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        b.drive(0.5, 0.5)
        time.sleep(0.02)
    n = len(_drive_posts(srv))
    assert 3 <= n <= 6, f"expected ~4-5 heartbeat-gated sends, got {n}"


def test_change_below_epsilon_not_sent(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    assert b.drive(0.50, 0.50) is True
    assert b.drive(0.52, 0.51) is False       # < epsilon: no wire traffic
    assert len(_drive_posts(srv)) == 1


def test_change_above_epsilon_sent_after_rate_gap(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    assert b.drive(0.2, 0.2) is True
    assert b.drive(0.8, 0.8) is False         # changed, but inside the 10 Hz cap
    time.sleep(0.12)
    assert b.drive(0.8, 0.8) is True          # after the cap window it goes out
    bodies = [r["body"] for r in _drive_posts(srv)]
    assert bodies == [{"left": 0.2, "right": 0.2}, {"left": 0.8, "right": 0.8}]


def test_connection_close_header(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.drive(0.3, 0.3)
    b.stop()
    assert all(r["connection"] == "close" for r in srv.recorded)


def test_clamp(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.drive(2.0, -3.0)
    assert _drive_posts(srv)[0]["body"] == {"left": 1.0, "right": -1.0}


# ── stop / e-stop latch ──────────────────────────────────────────────────


def test_stop_bypasses_gating_and_zeroes_state(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.drive(0.5, 0.5)
    assert b.stop() is True                   # immediately, no rate gap needed
    paths = [r["path"] for r in srv.recorded]
    assert paths[-1] == "/stop"


def test_estop_latches_drive(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.drive(0.5, 0.5)
    b.stop(estop=True)
    assert b.estopped is True
    time.sleep(0.3)                           # well past heartbeat + rate cap
    assert b.drive(0.9, 0.9) is False         # latched: no-op
    assert _drive_posts(srv)[-1]["body"] == {"left": 0.5, "right": 0.5}
    b.reset_estop()
    time.sleep(0.12)
    assert b.drive(0.9, 0.9) is True          # explicit reset re-enables


def test_normal_stop_does_not_latch(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.stop()                                  # arrival/preempt stop
    assert b.estopped is False
    assert b.drive(0.4, 0.4) is True


# ── e-stop decoupling (the Phase-E safety core) ──────────────────────────


class _SlowDriveHandler(_RecordingHandler):
    """Records on arrival, then /drive stalls 0.5 s before responding —
    simulates the worker's HTTP hanging mid-flight. /stop stays fast."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.server.recorded.append(
            {"path": self.path, "body": json.loads(raw) if raw else None,
             "connection": self.headers.get("Connection"), "t": time.monotonic()}
        )
        if self.path == "/drive":
            time.sleep(0.5)
        self._respond({"ok": True})


@pytest.fixture()
def slow_esp32():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowDriveHandler)
    srv.recorded = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_estop_latch_not_blocked_by_inflight_drive(slow_esp32):
    # An in-flight /drive (0.5 s server stall) must NOT delay the e-stop:
    # the latch is lock-only (no HTTP under it) and stop() posts on its own.
    srv, url = slow_esp32
    b = _bridge(url, http_timeout_s=1.5)
    worker = threading.Thread(target=lambda: b.drive(0.5, 0.5))
    worker.start()
    time.sleep(0.1)                       # drive is now in flight

    t0 = time.monotonic()
    ok = b.stop(estop=True)
    dt = time.monotonic() - t0

    assert ok is True
    assert dt < 0.35, f"e-stop took {dt:.2f}s — serialized behind in-flight drive"
    assert b.estopped is True
    worker.join()

    # Compensating stop: the drive landed AFTER the e-stop's /stop, so the
    # drive thread must have fired another /stop behind its own command.
    drives = [r for r in srv.recorded if r["path"] == "/drive"]
    stops = [r for r in srv.recorded if r["path"] == "/stop"]
    assert drives and stops
    assert any(s["t"] >= drives[0]["t"] + 0.45 for s in stops), (
        "no compensating /stop after the in-flight drive landed"
    )


def test_no_compensating_stop_when_not_latched(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    b.drive(0.5, 0.5)                      # normal drive, no e-stop
    assert [r["path"] for r in srv.recorded] == ["/drive"]


# ── failure handling ─────────────────────────────────────────────────────


def test_failures_drop_frames_never_raise():
    b = _bridge("http://127.0.0.1:1", http_timeout_s=0.15)  # dead port
    assert b.drive(0.5, 0.5) is False         # no exception into control loop
    assert b.consecutive_failures == 1
    assert b.stop() is False                  # stop also survives dead network
    assert b.battery_mv() is None
    assert b.ping() is None


def test_failure_retry_is_rate_capped():
    b = _bridge("http://127.0.0.1:1", http_timeout_s=0.1)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.65:
        b.drive(0.5, 0.5)
        time.sleep(0.01)
    # ~0.65 s at a 10 Hz attempt cap (+0.1 s timeout each) → a handful of
    # attempts, not hundreds.
    assert 2 <= b.consecutive_failures <= 8


# ── telemetry ────────────────────────────────────────────────────────────


def test_ping_and_battery(mock_esp32):
    srv, url = mock_esp32
    b = _bridge(url)
    assert "ESP32 mock online" in (b.ping() or "")
    assert b.battery_mv() == 7400
