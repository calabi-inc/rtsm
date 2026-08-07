"""EstopMonitor tests — fake bridge, no real gamepad required.

HARD LIMIT OF THIS FILE (read before trusting green):
A mocked pygame CANNOT catch the class of failure that motivated the
2026-08-07 rewrite — a binding that enumerates correctly (get_count /
get_name fine) while SDL never delivers its input reports, so
get_button() reads 0 forever. 39d7ec7's mocked retry test was green
while a real operator's mid-drive X press was silently lost. These
tests verify the monitor's LOGIC (event dispatch, binding lifecycle,
liveness bookkeeping, trigger ordering); only the E1 protocol's
live-fire X check — and hw_estop_check.py after any estop.py change —
verifies the hardware path.
"""

import collections
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


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ── trigger semantics (no pygame involved) ──────────────────────────────


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


def test_status_exposes_liveness_fields():
    # gamepad_available alone must never again read as proof the kill
    # switch works — status() must carry the honest liveness pair.
    m, b, ev = _monitor()
    s = m.status()
    assert s["binding_verified"] is False
    assert s["last_button_press_mono"] is None
    assert s["gamepad_available"] is False


# ── degraded mode (real pygame if installed) ────────────────────────────


def test_thread_runs_degraded_without_gamepad():
    # joystick_index=9 guarantees "no gamepad" even if a controller is
    # paired: the boot scan needs count > 9, and hotplug JOYDEVICEADDED
    # events carry device_index 0 for a single pad, which != 9.
    m, b, ev = _monitor(joystick_index=9)
    m.start()
    time.sleep(0.3)
    assert m._thread is not None and m._thread.is_alive()
    assert m.gamepad_available is False
    assert m.status()["triggered"] is False
    assert m.trigger("ctrl-c") is True             # software path still works
    m.shutdown()
    assert not m._thread.is_alive()


# ── hotplug lifecycle against a fake pygame ─────────────────────────────


class FakeJoystick:
    _next_iid = 0

    def __init__(self, name="Fake PS4 Controller"):
        self._name = name
        self._iid = FakeJoystick._next_iid
        FakeJoystick._next_iid += 1
        self.buttons = collections.defaultdict(int)

    def get_name(self):
        return self._name

    def get_instance_id(self):
        return self._iid

    def get_button(self, index):
        return self.buttons[index]


def _make_fake_pygame():
    """Fake of exactly the pygame surface estop.py uses.

    get_init() returns True: the process looks ALREADY initialized, the
    precondition of the 2026-08-07 incident. The monitor must bind purely
    from the device list + event queue — a pygame.joystick.quit()/init()
    rescan (the old, broken path) would raise AttributeError here and
    fail the test, which is intentional: the fix forbids subsystem
    restarts."""
    pg = types.ModuleType("pygame")
    pg.JOYDEVICEADDED = 100
    pg.JOYDEVICEREMOVED = 101
    pg.JOYBUTTONDOWN = 102
    pg._devices = []                       # device_index -> FakeJoystick
    pg._queue = collections.deque()        # atomic append/popleft

    def _get():
        evs = []
        while True:
            try:
                evs.append(pg._queue.popleft())
            except IndexError:
                return evs

    pg.get_init = lambda: True
    pg.init = lambda: None
    pg.event = types.SimpleNamespace(get=_get, pump=lambda: None)
    pg.joystick = types.SimpleNamespace(
        get_init=lambda: True,
        init=lambda: None,
        get_count=lambda: len(pg._devices),
        Joystick=lambda i: pg._devices[i],
    )
    return pg


def _plug(pg, joy, index=0):
    pg._devices.insert(index, joy)
    pg._queue.append(types.SimpleNamespace(
        type=pg.JOYDEVICEADDED, device_index=index))


def _unplug(pg, joy):
    pg._devices.remove(joy)
    pg._queue.append(types.SimpleNamespace(
        type=pg.JOYDEVICEREMOVED, instance_id=joy.get_instance_id()))


def _press_event(pg, joy, button=0):
    pg._queue.append(types.SimpleNamespace(
        type=pg.JOYBUTTONDOWN, instance_id=joy.get_instance_id(),
        button=button))


def test_gamepad_paired_after_start_binds_and_button_reads(monkeypatch):
    # Successor to 39d7ec7's retry test — but exercising the REAL binding
    # code against the event queue instead of stubbing _init_gamepad.
    # Still cannot prove the hardware path (see module docstring).
    pg = _make_fake_pygame()
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()

    time.sleep(0.1)                                # let the loop spin unbound
    assert m.gamepad_available is False
    assert m.trigger("probe") is True              # trigger() works degraded
    m.reset()
    b.calls.clear()

    joy = FakeJoystick()
    _plug(pg, joy)                                 # pad pairs mid-run
    assert _wait_for(lambda: m.gamepad_available)  # same thread — no restart
    assert m.gamepad_name == "Fake PS4 Controller"
    assert m.binding_verified is False             # not proof until a press
    assert m.last_button_press_mono is None

    _press_event(pg, joy, button=0)                # operator presses X
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad-button0"
    assert b.calls == [{"estop": True, "event_already_set": False}]
    assert m.binding_verified is True
    assert m.last_button_press_mono is not None
    m.shutdown()
    assert not m._thread.is_alive()


def test_boot_time_pad_binds_by_index_and_polled_button_triggers(monkeypatch):
    # Fresh-boot path: pad present before start(), no events needed; the
    # 50 Hz get_button() poll alone must latch and count as liveness.
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert _wait_for(lambda: m.gamepad_available)
    assert m.binding_verified is False

    joy.buttons[0] = 1                             # held X, seen by the poll
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad-button0"
    assert m.binding_verified is True
    m.shutdown()


def test_non_estop_button_verifies_binding_without_triggering(monkeypatch):
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert _wait_for(lambda: m.gamepad_available)

    _press_event(pg, joy, button=3)                # e.g. Triangle
    assert _wait_for(lambda: m.binding_verified)   # liveness proven...
    assert m.triggered is False                    # ...without an e-stop
    assert b.calls == []
    m.shutdown()


def test_disconnect_flips_available_and_rebind_resets_verified(monkeypatch):
    # The incident shape end-to-end: verified pad sleeps → available goes
    # False at once (no silent zeros); it reconnects → bound again but
    # binding_verified is False until a NEW press on the NEW binding.
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert _wait_for(lambda: m.gamepad_available)

    _press_event(pg, joy, button=0)                # live-fire: prove binding 1
    assert _wait_for(lambda: m.binding_verified)
    first_press = m.last_button_press_mono
    m.reset()                                      # operator re-arm
    b.calls.clear()

    _unplug(pg, joy)                               # pad sleeps mid-session
    assert _wait_for(lambda: not m.gamepad_available)
    assert m.gamepad_name is None
    assert m.binding_verified is False

    _plug(pg, joy)                                 # pad wakes / re-pairs
    assert _wait_for(lambda: m.gamepad_available)
    assert m.binding_verified is False             # old press proves NOTHING
    assert m.last_button_press_mono == first_press  # history is kept, honest

    _press_event(pg, joy, button=0)                # live-fire on binding 2
    assert _wait_for(lambda: m.triggered)
    assert m.binding_verified is True
    assert m.last_button_press_mono > first_press
    assert b.calls == [{"estop": True, "event_already_set": False}]
    m.shutdown()


def test_other_joystick_events_are_ignored(monkeypatch):
    # A second pad's removal or presses must not touch our binding.
    pg = _make_fake_pygame()
    ours, other = FakeJoystick(), FakeJoystick()
    pg._devices.append(ours)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert _wait_for(lambda: m.gamepad_available)

    _press_event(pg, other, button=0)              # not our instance_id
    pg._queue.append(types.SimpleNamespace(
        type=pg.JOYDEVICEREMOVED, instance_id=other.get_instance_id()))
    time.sleep(0.2)
    assert m.gamepad_available is True             # still bound
    assert m.binding_verified is False             # stranger's press ≠ ours
    assert m.triggered is False
    m.shutdown()
