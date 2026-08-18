"""Tests for the frame-flow watchdog (rtsm/core/watchdog.py) and its
/healthz surfacing.

The monitor is pure logic with injectable clock and sources, so every state
is tested deterministically without threads or sleeps.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import create_app
from rtsm.core.watchdog import (
    DEGRADED_STATES,
    FrameFlowMonitor,
    PipelineHeartbeat,
    Watchdog,
)


class FakeReceiver:
    def __init__(self):
        self.alive = True
        self.last_rx_mono = None
        self.last_enqueue_mono = None
        self.tracking_drops = 0

    def liveness(self):
        return {
            "alive": self.alive,
            "last_rx_mono": self.last_rx_mono,
            "last_enqueue_mono": self.last_enqueue_mono,
            "tracking_drops": self.tracking_drops,
        }


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_monitor(recv, hb, clock, qsize=0, starved=5.0, hung=10.0):
    return FrameFlowMonitor(
        heartbeat=hb,
        queue_size=lambda: qsize,
        receiver_liveness=recv.liveness,
        starved_after_s=starved,
        hung_after_s=hung,
        now_fn=clock,
    )


def test_waiting_before_any_client():
    """Boot with no client ever connected -> waiting, not degraded."""
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    hb.last_step_mono = clock.t - 0.5  # loop is turning
    result = make_monitor(recv, hb, clock).evaluate()
    assert result["state"] == "waiting"
    assert result["degraded"] is False


def test_ok_when_everything_fresh():
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 0.5
    hb.last_step_mono = clock.t - 0.2
    hb.last_frame_mono = clock.t - 0.5
    result = make_monitor(recv, hb, clock).evaluate()
    assert result["state"] == "ok"
    assert result["degraded"] is False
    assert result["last_input_age_s"] == pytest.approx(0.1, abs=0.05)


def test_starved_when_input_stops():
    """Client connected once, then the stream died (phone slept, WiFi drop)."""
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 30.0
    recv.last_enqueue_mono = clock.t - 30.0
    hb.last_step_mono = clock.t - 0.2  # loop still turning (idle)
    result = make_monitor(recv, hb, clock).evaluate()
    assert result["state"] == "starved"
    assert result["degraded"] is True
    assert result["reasons"]


def test_hung_when_frames_wait_but_loop_silent():
    """Queue non-empty but the pipeline loop stopped turning -> hung."""
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 0.1
    hb.last_step_mono = clock.t - 60.0  # loop silent for a minute
    result = make_monitor(recv, hb, clock, qsize=512).evaluate()
    assert result["state"] == "hung"
    assert result["degraded"] is True


def test_hung_detected_via_recent_enqueue_even_with_empty_queue():
    """Enqueue happened after the last step -> hung even if qsize reads 0."""
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 1.0
    hb.last_step_mono = clock.t - 60.0
    result = make_monitor(recv, hb, clock, qsize=0).evaluate()
    assert result["state"] == "hung"


def test_receiver_dead_takes_priority():
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.alive = False
    recv.last_rx_mono = clock.t - 0.1
    hb.last_step_mono = clock.t - 0.1
    result = make_monitor(recv, hb, clock).evaluate()
    assert result["state"] == "receiver_dead"
    assert result["degraded"] is True


def test_pose_degraded_when_tracking_drops_climb():
    """Messages arrive but nothing enqueues and tracking-drops climb
    (ARKit tracking limited/lost) -> pose_degraded."""
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 20.0
    hb.last_step_mono = clock.t - 0.2
    mon = make_monitor(recv, hb, clock)
    mon.evaluate()  # baseline for the drops delta
    recv.tracking_drops = 25
    result = mon.evaluate()
    assert result["state"] == "pose_degraded"
    assert result["degraded"] is True


def test_no_ingestible_input_without_tracking_drops():
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 20.0
    hb.last_step_mono = clock.t - 0.2
    mon = make_monitor(recv, hb, clock)
    mon.evaluate()
    result = mon.evaluate()  # drops unchanged
    assert result["state"] == "no_ingestible_input"
    assert result["degraded"] is True


def test_recovery_back_to_ok():
    recv, hb, clock = FakeReceiver(), PipelineHeartbeat(), Clock()
    recv.last_rx_mono = clock.t - 30.0
    hb.last_step_mono = clock.t - 0.2
    mon = make_monitor(recv, hb, clock)
    assert mon.evaluate()["state"] == "starved"
    # Stream resumes
    clock.t += 10.0
    recv.last_rx_mono = clock.t - 0.1
    recv.last_enqueue_mono = clock.t - 0.2
    hb.last_step_mono = clock.t - 0.1
    result = mon.evaluate()
    assert result["state"] == "ok"
    assert result["degraded"] is False


def test_degraded_states_registry_consistent():
    assert "ok" not in DEGRADED_STATES
    assert "waiting" not in DEGRADED_STATES


def test_heartbeat_stamps_monotonic():
    hb = PipelineHeartbeat()
    assert hb.last_step_mono is None and hb.last_frame_mono is None
    hb.beat_step()
    hb.beat_frame()
    now = time.monotonic()
    assert hb.last_step_mono is not None and now - hb.last_step_mono < 1.0
    assert hb.last_frame_mono is not None and now - hb.last_frame_mono < 1.0


def test_watchdog_thread_status_and_transition_logging():
    """Watchdog thread evaluates, caches status, and survives source errors."""
    recv, hb = FakeReceiver(), PipelineHeartbeat()
    recv.last_rx_mono = time.monotonic() - 30.0
    hb.last_step_mono = time.monotonic()
    wd = Watchdog(
        heartbeat=hb,
        queue_size=lambda: 0,
        receiver_liveness=recv.liveness,
        starved_after_s=5.0,
        hung_after_s=10.0,
        poll_interval_s=0.1,
    )
    wd.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if wd.status().get("state") == "starved":
                break
            time.sleep(0.05)
        assert wd.status()["state"] == "starved"
        assert wd.status()["degraded"] is True
    finally:
        wd.stop()
        wd.join(timeout=2.0)
    assert not wd.is_alive()


# ---------------- /healthz surfacing ----------------


class _StubWM:
    def stats(self):
        return {"objects": 0, "confirmed": 0, "upserts_total": 0}


def _client(provider=None):
    app = create_app(
        working_memory=_StubWM(),
        registry=CollectorRegistry(),
        frame_flow_provider=provider,
    )
    return TestClient(app)


def test_healthz_unchanged_without_watchdog():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_reports_frame_flow_ok():
    provider = lambda: {"state": "ok", "degraded": False, "reasons": []}
    body = _client(provider).get("/healthz").json()
    assert body["status"] == "ok"
    assert body["frame_flow"]["state"] == "ok"
    assert "reasons" not in body


def test_healthz_degraded_with_reasons():
    provider = lambda: {
        "state": "starved",
        "degraded": True,
        "reasons": ["no input from client for 30.0s"],
    }
    body = _client(provider).get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["frame_flow"]["state"] == "starved"
    assert body["reasons"] == ["no input from client for 30.0s"]


def test_healthz_survives_provider_exception():
    def bad_provider():
        raise RuntimeError("watchdog exploded")

    body = _client(bad_provider).get("/healthz").json()
    assert body["status"] == "ok"
    assert body["frame_flow"] == {"state": "unknown"}
