"""EstopMonitor tests — fake bridge, no pygame/gamepad required."""

import sys
import threading
import time
import types

from estop import EstopMonitor


class FakeBridge:
    """Records calls; snapshots whether the stop_event was already set at
    stop() time so tests can assert the safety ordering (latch BEFORE wake)."""

    def __init__(self, stop_event):
        self._event = stop_event
        self.calls = []
        self._estopped = False

    def stop(self, estop=False):
        self.calls.append(
            {"estop": estop, "event_already_set": self._event.is_set()}
        )
        if estop:
            self._estopped = True
        return True

    def reset_estop(self):
        self._estopped = False

    @property
    def estopped(self):
        return self._estopped


def _monitor(**kw):
    ev = threading.Event()
    b = FakeBridge(ev)
    m = EstopMonitor(b, ev, **kw)
    return m, b, ev


def test_trigger_latches_bridge_before_waking_worker():
    m, b, ev = _monitor()
    assert m.trigger("test") is True
    assert b.calls == [{"estop": True, "event_already_set": False}]  # order!
    assert ev.is_set()
    assert m.triggered and m.trigger_source == "test"


def test_trigger_is_idempotent():
    m, b, ev = _monitor()
    assert m.trigger("first") is True
    assert m.trigger("second") is False
    assert len(b.calls) == 1                       # exactly one hardware stop
    assert m.trigger_source == "first"


def test_reset_rearms_monitor_but_not_bridge():
    m, b, ev = _monitor()
    m.trigger("test")
    m.reset()
    assert m.triggered is False and not ev.is_set()
    assert b.estopped is True                      # bridge latch is a separate,
    b.reset_estop()                                # deliberate second step
    assert m.trigger("again") is True              # re-armed


def test_thread_runs_degraded_without_gamepad():
    # joystick_index=9 guarantees "no gamepad" even if a controller is paired.
    m, b, ev = _monitor(joystick_index=9)
    m.start()
    time.sleep(0.3)
    assert m._thread is not None and m._thread.is_alive()
    assert m.gamepad_available is False
    assert m.status()["triggered"] is False
    assert m.trigger("ctrl-c") is True             # software path still works
    m.shutdown()
    assert not m._thread.is_alive()


class FakeJoystick:
    def __init__(self):
        self.pressed = False

    def get_button(self, index):
        return 1 if self.pressed else 0


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_gamepad_paired_after_start_is_detected_without_restart(monkeypatch):
    # The poll loop only needs pygame.event.pump(); stub it so no real
    # pygame/display is required.
    fake_pygame = types.ModuleType("pygame")
    fake_pygame.event = types.SimpleNamespace(pump=lambda: None)
    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)

    m, b, ev = _monitor()
    m._retry_s = 0.05                              # fast re-detection for test
    joy = FakeJoystick()
    calls = {"n": 0}

    def fake_init_gamepad():
        calls["n"] += 1
        if calls["n"] <= 2:                        # controller not paired yet
            m.gamepad_available = False
            return None
        m.gamepad_available = True                 # mimic the real contract
        m.gamepad_name = "Fake PS4 Controller"
        return joy

    m._init_gamepad = fake_init_gamepad
    m.start()

    assert m.trigger("probe") is True              # trigger() works while degraded
    m.reset()
    b.calls.clear()

    # Detection must happen on the SAME thread run — no restart.
    assert _wait_for(lambda: m.gamepad_available)
    assert calls["n"] >= 3
    assert m.gamepad_name == "Fake PS4 Controller"
    assert m._thread.is_alive()

    joy.pressed = True                             # now press the e-stop button
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad-button0"
    assert b.calls == [{"estop": True, "event_already_set": False}]
    m.shutdown()
    assert not m._thread.is_alive()
