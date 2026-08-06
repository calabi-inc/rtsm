"""
Independent e-stop monitor — the hard interrupt channel (channel b).

Runs on its OWN daemon thread, decoupled from the worker/nav loop, so a
GIL-blocked or crashed control loop can never delay the stop. On trigger:

    1. bridge.stop(estop=True)  — latches the bridge FIRST (microseconds,
       no HTTP under the latch lock), then posts /stop with retry.
    2. sets the shared stop_event — the worker aborts at its next check.

Triggers:
  * gamepad button (PS4 X = button 0, per the hardware-verified teleop
    mapping) — polled at ~50 Hz via pygame.joystick (headless-safe).
  * trigger() called directly — used by the server's Ctrl-C/SIGINT path
    and by tests.

Degraded mode: if no gamepad is connected at start, the thread stays up
(trigger() still works, Ctrl-C still works), exposes
gamepad_available=False so /status can warn the operator, and keeps
retrying detection so a controller paired after server boot is picked
up without a restart.

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
        self._retry_s = 1.0           # gamepad re-detection cadence
        self._shutdown = threading.Event()
        self._trigger_lock = threading.Lock()

        self.triggered = False
        self.trigger_source: Optional[str] = None
        self.triggered_at_mono: Optional[float] = None
        self.gamepad_available = False
        self.gamepad_name: Optional[str] = None

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
            "triggered": self.triggered,
            "trigger_source": self.trigger_source,
        }

    # ── polling thread ───────────────────────────────────────────────────

    def _run(self) -> None:
        joy = self._init_gamepad()
        while joy is None and not self._shutdown.is_set():
            # Degraded: no gamepad. trigger() remains the (only) software
            # path (Ctrl-C wiring lives in the server); keep retrying so a
            # controller paired after boot is detected without a restart.
            self._shutdown.wait(self._retry_s)
            joy = self._init_gamepad()
        if joy is None:
            return

        import pygame
        while not self._shutdown.is_set():
            try:
                pygame.event.pump()
                if joy.get_button(self._button):
                    self.trigger(f"gamepad-button{self._button}")
                    # keep polling; trigger() is idempotent
            except Exception:  # noqa: BLE001 — monitor must never die silently
                self.gamepad_available = False
                self.gamepad_name = None
                joy = None
                while not self._shutdown.is_set() and joy is None:
                    self._shutdown.wait(self._retry_s)
                    joy = self._init_gamepad()
                continue
            time.sleep(self._period_s)

    def _init_gamepad(self):
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            # SDL only enumerates devices when the joystick subsystem
            # initializes — a plain init() after the first is a no-op, so
            # quit first to force a rescan (we hold no live Joystick here).
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= self._joy_index:
                self.gamepad_available = False
                return None
            joy = pygame.joystick.Joystick(self._joy_index)
            joy.init()
            self.gamepad_available = True
            self.gamepad_name = joy.get_name()
            return joy
        except Exception:  # noqa: BLE001
            self.gamepad_available = False
            return None
