"""EstopMonitor tests — fake bridge, no pygame/gamepad required."""

import threading
import time

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
