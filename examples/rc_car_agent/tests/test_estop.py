"""EstopMonitor tests — fake bridge + fake pygame, no real gamepad.

HARD LIMIT OF THIS FILE (read before trusting green):
Mocked pygame CANNOT catch the class of failure that motivated the
2026-08-07 rewrites — bindings that enumerate correctly while SDL never
delivers input reports. Mocked tests were green TWICE while a real
operator's mid-drive X press was silently lost (deaf HIDAPI rebind, then
the thread-affinity gap: DirectInput delivers input only to the SDL-
initializing thread). These tests verify the monitor's LOGIC (trigger
ordering, all-pads binding, liveness bookkeeping, hotplug lifecycle,
main-thread/poll-thread split); only the E1 protocol's live-fire X check
verifies the hardware path — before the first trial, and after any pad
sleep/reconnect.

Design under test (hardware-verified live 2026-08-07):
  * start()/pump_once()/init_gamepads() = MAIN-thread half (SDL owner);
    the daemon thread ONLY reads button state.
  * ALL pads bound, kill buttons {0, 1} (DS4 X on both SDL channels) —
    any pad's kill button latches; any button press proves liveness.
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


# ── fake pygame matching exactly the surface estop.py uses ──────────────


class FakeJoystick:
    def __init__(self, name="Fake PS4 Controller", numbuttons=14):
        self._name = name
        self._n = numbuttons
        self.buttons = collections.defaultdict(int)

    def init(self):
        pass

    def get_name(self):
        return self._name

    def get_numbuttons(self):
        return self._n

    def get_button(self, index):
        return self.buttons[index]


def _make_fake_pygame():
    pg = types.ModuleType("pygame")
    pg.JOYDEVICEADDED = 100
    pg.JOYDEVICEREMOVED = 101
    pg._devices = []
    pg._queue = collections.deque()

    def _get(type_filter=None):
        keep, out = collections.deque(), []
        while True:
            try:
                e = pg._queue.popleft()
            except IndexError:
                break
            if type_filter is None or e.type in type_filter:
                out.append(e)
            else:
                keep.append(e)
        pg._queue = keep
        return out

    pg.get_init = lambda: True
    pg.init = lambda: None
    pg.event = types.SimpleNamespace(get=_get, pump=lambda: None,
                                     clear=lambda: pg._queue.clear())
    pg.joystick = types.SimpleNamespace(
        quit=lambda: None,
        init=lambda: None,
        get_count=lambda: len(pg._devices),
        Joystick=lambda i: pg._devices[i],
    )
    return pg


def _device_event(pg, kind):
    pg._queue.append(types.SimpleNamespace(type=kind))


# ── binding + kill semantics ────────────────────────────────────────────


def test_boot_bind_and_kill_button_0(monkeypatch):
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert m.gamepad_available is True             # bound on caller thread
    assert m.binding_verified is False             # enumeration is not proof

    joy.buttons[0] = 1                             # X on the HIDAPI mapping
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad0-button0"
    assert m.binding_verified is True
    assert m.last_button_press_mono is not None
    assert b.calls == [{"estop": True, "event_already_set": False}]
    m.shutdown()
    assert not m._thread.is_alive()


def test_kill_button_1_also_latches(monkeypatch):
    # DS4 X is button 1 on the DirectInput channel — the mapping that a
    # single-button watch silently missed on hardware (2026-08-07).
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    joy.buttons[1] = 1
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad0-button1"
    m.shutdown()


def test_second_pads_kill_press_latches(monkeypatch):
    # ALL pads are armed. On hardware the winning press arrived via the
    # USB enumeration while index 0 held the deaf Bluetooth ghost — a
    # single-index binding is a lottery (2026-08-07). A stranger's press
    # costing a re-arm is acceptable; a missed press is not.
    pg = _make_fake_pygame()
    ghost, wired = FakeJoystick("BT ghost"), FakeJoystick("USB pad")
    pg._devices.extend([ghost, wired])
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert m.gamepad_available is True
    assert "USB pad" in m.gamepad_name

    wired.buttons[1] = 1                           # press on pad #2 only
    assert _wait_for(lambda: m.triggered)
    assert m.trigger_source == "gamepad1-button1"
    m.shutdown()


def test_non_kill_button_verifies_binding_without_triggering(monkeypatch):
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    joy.buttons[5] = 1                             # e.g. a shoulder button
    assert _wait_for(lambda: m.binding_verified)   # liveness proven...
    assert m.triggered is False                    # ...without an e-stop
    assert b.calls == []
    m.shutdown()


# ── degraded mode + main-thread retry ───────────────────────────────────


def test_degraded_without_gamepad_trigger_still_works(monkeypatch):
    pg = _make_fake_pygame()                       # zero devices
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert m.gamepad_available is False
    assert m._thread.is_alive()
    assert m.trigger("ctrl-c") is True             # software path still works
    m.shutdown()
    assert not m._thread.is_alive()


def test_pad_paired_after_start_binds_via_pump(monkeypatch):
    # The mid-run pairing path: pump_once() (MAIN thread) retries and
    # binds — no server restart. The daemon thread then hears the press.
    pg = _make_fake_pygame()
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    assert m.gamepad_available is False

    joy = FakeJoystick()
    pg._devices.append(joy)                        # pad pairs mid-run
    m._last_reinit_mono = 0                        # bypass retry throttle
    m._last_housekeep_mono = 0                     # bypass housekeep throttle
    m.pump_once()                                  # main-thread retry
    assert m.gamepad_available is True

    joy.buttons[0] = 1
    assert _wait_for(lambda: m.triggered)
    m.shutdown()


# ── hotplug lifecycle (the incident shape, end-to-end) ──────────────────


def test_sleep_wake_resets_verified_and_new_press_reverifies(monkeypatch):
    # Verified pad sleeps → available flips False at once (no silent
    # zeros); it reconnects → bound again but binding_verified is False
    # until a NEW press on the NEW binding. Press HISTORY is kept —
    # an old press proves nothing, hiding it would lie the other way.
    pg = _make_fake_pygame()
    joy = FakeJoystick()
    pg._devices.append(joy)
    monkeypatch.setitem(sys.modules, "pygame", pg)

    m, b, ev = _monitor()
    m.start()
    joy.buttons[0] = 1                             # live-fire: prove binding 1
    assert _wait_for(lambda: m.triggered)
    first_press = m.last_button_press_mono
    joy.buttons[0] = 0
    m.reset()
    b.calls.clear()

    pg._devices.remove(joy)                        # pad sleeps mid-session
    _device_event(pg, pg.JOYDEVICEREMOVED)
    m._last_housekeep_mono = 0                     # bypass housekeep throttle
    m.pump_once()                                  # main thread notices
    assert m.gamepad_available is False
    assert m.binding_verified is False

    pg._devices.append(joy)                        # pad wakes / re-pairs
    _device_event(pg, pg.JOYDEVICEADDED)
    m._last_housekeep_mono = 0
    m.pump_once()
    assert m.gamepad_available is True
    assert m.binding_verified is False             # old press proves NOTHING
    assert m.last_button_press_mono == first_press  # history kept, honest

    joy.buttons[0] = 1                             # live-fire on binding 2
    assert _wait_for(lambda: m.triggered)
    assert m.binding_verified is True
    assert m.last_button_press_mono > first_press
    assert b.calls == [{"estop": True, "event_already_set": False}]
    m.shutdown()
