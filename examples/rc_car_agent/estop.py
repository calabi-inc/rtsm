"""
Independent e-stop monitor — the hard interrupt channel (channel b).

Runs on its OWN daemon thread, decoupled from the worker/nav loop, so a
GIL-blocked or crashed control loop can never delay the stop. On trigger:

    1. bridge.stop(estop=True)  — latches the bridge FIRST (microseconds,
       no HTTP under the latch lock), then posts /stop with retry.
    2. sets the shared stop_event — the worker aborts at its next check.

Triggers:
  * gamepad button (PS4 X = button 0, per the hardware-verified teleop
    mapping) — JOYBUTTONDOWN events plus a ~50 Hz get_button() poll via
    pygame (headless-safe).
  * trigger() called directly — used by the server's Ctrl-C/SIGINT path
    and by tests.

Hotplug (2026-08-07 incident, mission t20260806-172610-001):
  The old retry loop re-detected a late-connecting pad by cycling
  pygame.joystick.quit()/init() and re-opening by index. On Windows
  (SDL 2.28) a joystick opened after such a mid-run subsystem restart can
  ENUMERATE correctly (get_count()/get_name() fine) while its input
  reports are never associated with the restarted subsystem instance —
  get_button() reads 0 forever. gamepad_available said True and the
  operator's real mid-drive X press was silently lost. Fix: initialize
  the joystick subsystem exactly ONCE per process and NEVER restart it;
  SDL2's supported hotplug path pushes JOYDEVICEADDED / JOYDEVICEREMOVED
  events (including ADDED for pads already present at init), and a
  Joystick opened from those events shares the driver association that
  actually delivers input.

  Because that failure mode is invisible to every software check
  (detection, name, and a mocked unit test all looked fine), status()
  additionally reports binding_verified — True only after a real button
  press has been observed on the CURRENT binding, reset on every rebind —
  and last_button_press_mono. gamepad_available means "a device is
  bound", never "the kill switch works"; the E1 protocol's live-fire X
  check before motion remains mandatory.

Degraded mode: if no gamepad is connected at start, the thread stays up
(trigger() still works, Ctrl-C still works), exposes
gamepad_available=False so /status can warn the operator, and a
controller paired after server boot binds via JOYDEVICEADDED without a
restart. A pad that sleeps or unpairs mid-run flips
gamepad_available=False on its JOYDEVICEREMOVED event instead of reading
silent zeros forever.

A chat/HTTP "stop" is NOT this channel — POST /stop is a convenience;
THIS is the safety guarantee (together with the ESP32 300 ms watchdog,
which needs no desktop at all).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from esp32_bridge import Esp32Bridge


class EstopMonitor:
    def __init__(
        self,
        bridge: Esp32Bridge,
        stop_event: threading.Event,
        button_index: int = 0,        # PS4 X on Windows (teleop mapping)
        joystick_index: int = 0,
        poll_hz: float = 50.0,
    ):
        self._bridge = bridge
        self._stop_event = stop_event
        self._button = int(button_index)
        self._joy_index = int(joystick_index)
        self._period_s = 1.0 / float(poll_hz)

        self._thread: Optional[threading.Thread] = None
        self._retry_s = 1.0           # backoff after pygame init/poll failures
        self._shutdown = threading.Event()
        self._trigger_lock = threading.Lock()
        self._joy = None              # bound Joystick — poll thread only

        self.triggered = False
        self.trigger_source: Optional[str] = None
        self.triggered_at_mono: Optional[float] = None
        self.gamepad_available = False
        self.gamepad_name: Optional[str] = None
        # Honest liveness: True only once a real press has been observed on
        # the CURRENT binding (reset on every rebind). A binding that has
        # not been live-fired must never be trusted — see module docstring.
        self.binding_verified = False
        self.last_button_press_mono: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> "EstopMonitor":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="estop-monitor", daemon=True
        )
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop the polling thread (does NOT trigger an e-stop)."""
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── the trigger (idempotent; callable from any thread / signal path) ─

    def trigger(self, source: str) -> bool:
        """Fire the e-stop. Returns True if this call was the first."""
        with self._trigger_lock:
            if self.triggered:
                return False
            self.triggered = True
            self.trigger_source = source
            self.triggered_at_mono = time.monotonic()
        # Order matters: latch + /stop FIRST (hardware safety), then wake
        # the software loop.
        self._bridge.stop(estop=True)
        self._stop_event.set()
        return True

    def reset(self) -> None:
        """Operator re-arm after an e-stop (does not clear the bridge latch —
        call bridge.reset_estop() explicitly; two deliberate steps)."""
        with self._trigger_lock:
            self.triggered = False
            self.trigger_source = None
            self.triggered_at_mono = None
        self._stop_event.clear()

    def status(self) -> dict:
        return {
            "gamepad_available": self.gamepad_available,
            "gamepad_name": self.gamepad_name,
            "binding_verified": self.binding_verified,
            "last_button_press_mono": self.last_button_press_mono,
            "triggered": self.triggered,
            "trigger_source": self.trigger_source,
        }

    # ── polling thread ───────────────────────────────────────────────────

    def _run(self) -> None:
        pygame = None
        while pygame is None and not self._shutdown.is_set():
            pygame = self._init_pygame()
            if pygame is None:
                # Degraded: pygame unusable. trigger() remains the (only)
                # software path (Ctrl-C wiring lives in the server).
                self._shutdown.wait(self._retry_s)
        if pygame is None:
            return

        # Bind a pad already present at boot by configured index — the
        # hardware-proven fresh-boot path. Late or re-connecting pads bind
        # via JOYDEVICEADDED below; the subsystem is never restarted.
        self._bind_by_index(pygame)

        while not self._shutdown.is_set():
            try:
                for event in pygame.event.get():
                    self._handle_event(pygame, event)
                if self._joy is not None and self._joy.get_button(self._button):
                    self._note_button_press()
                    self.trigger(f"gamepad-button{self._button}")
                    # keep polling; trigger() is idempotent
            except Exception:  # noqa: BLE001 — monitor must never die silently
                self._drop_binding()
                self._shutdown.wait(self._retry_s)
                continue
            self._shutdown.wait(self._period_s)

    def _init_pygame(self):
        """Init pygame + the joystick subsystem exactly once per process.

        NEVER call pygame.joystick.quit() after this: a mid-run subsystem
        restart is what produced enumerate-but-deaf bindings (module
        docstring). Returns the module, or None on failure."""
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            return pygame
        except Exception:  # noqa: BLE001
            self.gamepad_available = False
            return None

    def _bind_by_index(self, pygame) -> None:
        try:
            pygame.event.pump()       # let SDL run one device-detection pass
            if pygame.joystick.get_count() > self._joy_index:
                self._bind(pygame.joystick.Joystick(self._joy_index))
        except Exception:  # noqa: BLE001
            self._drop_binding()

    def _handle_event(self, pygame, event) -> None:
        if event.type == pygame.JOYDEVICEADDED:
            # SDL also delivers this for pads already present at subsystem
            # init, so boot-time and hotplug binding share one path. The
            # device_index match keeps joystick_index semantics; on a
            # single-pad rig a reconnecting pad always re-adds at index 0.
            if self._joy is None and event.device_index == self._joy_index:
                self._bind(pygame.joystick.Joystick(event.device_index))
        elif event.type == pygame.JOYDEVICEREMOVED:
            # Sleeping/unpaired pad → honest unavailable, immediately.
            if self._joy is not None and \
                    event.instance_id == self._joy.get_instance_id():
                self._drop_binding()
        elif event.type == pygame.JOYBUTTONDOWN:
            if self._joy is not None and \
                    event.instance_id == self._joy.get_instance_id():
                self._note_button_press()   # ANY button proves liveness
                if event.button == self._button:
                    self.trigger(f"gamepad-button{self._button}")

    def _bind(self, joy) -> None:
        self._joy = joy
        self.gamepad_available = True
        self.gamepad_name = joy.get_name()
        # Every (re)bind must re-prove itself with a real press — exactly
        # the state the 2026-08-07 incident showed can be deaf.
        self.binding_verified = False

    def _drop_binding(self) -> None:
        self._joy = None
        self.gamepad_available = False
        self.gamepad_name = None
        self.binding_verified = False

    def _note_button_press(self) -> None:
        # last_button_press_mono is deliberately NOT cleared on rebind: it is
        # global history; binding_verified carries the per-binding truth.
        self.last_button_press_mono = time.monotonic()
        self.binding_verified = True
