"""
Independent e-stop monitor — the hard interrupt channel (channel b).

Runs on its OWN daemon thread, decoupled from the worker/nav loop, so a
GIL-blocked or crashed control loop can never delay the stop. On trigger:

    1. bridge.stop(estop=True)  — latches the bridge FIRST (microseconds,
       no HTTP under the latch lock), then posts /stop with retry.
    2. sets the shared stop_event — the worker aborts at its next check.

Triggers:
  * ANY kill button on ANY bound gamepad — polled at ~50 Hz.
  * trigger() called directly — used by the server's Ctrl-C/SIGINT path
    and by tests.

A chat/HTTP "stop" is NOT this channel — POST /stop is a convenience;
THIS is the safety guarantee (together with the ESP32 300 ms watchdog,
which needs no desktop at all).

════════════════════════════════════════════════════════════════════════
HARD-WON ARCHITECTURE (a full hardware session of forensics, 2026-08-07;
a real operator's mid-drive kill press was silently lost to the old
design — read before "simplifying"):

1. THREAD AFFINITY. Windows DirectInput delivers input only to the
   thread that initialized SDL; a background thread reads permanent
   zeros while the main thread sees every press. (SDL's HIDAPI channel
   is thread-agnostic — which is why the old all-in-daemon-thread
   structure ever appeared to work, and why it failed unpredictably as
   the pad's Bluetooth session hopped channels.) Therefore: start()
   binds pads on the CALLER's thread, the host must call pump_once()
   periodically from that SAME thread (the agent server runs an asyncio
   pump task; calibrate.py pumps in its drive wrapper), and the daemon
   poll thread ONLY READS button state.

2. ALL PADS, PLURAL KILL BUTTONS. A wired pad and its lingering
   Bluetooth ghost enumerate as separate devices in shuffling order, and
   the DS4's X button is index 0 on SDL-HIDAPI but index 1 on
   DirectInput. Binding one index and watching one button is a double
   lottery where the losing ticket is a deaf kill switch. ANY bound
   device hearing ANY kill button latches. (A stranger's pad triggering
   an e-stop is a re-arm's worth of nuisance; a missed press is a crash.)

3. CHANNEL PINNING. The SDL_JOYSTICK_* env below pins the DirectInput
   path — the channel Windows' own joy.cpl validates — and allows
   background event delivery for a windowless server process.

4. HONEST LIVENESS. gamepad_available means ENUMERATED, nothing more —
   a binding can enumerate perfectly and still be deaf.
   binding_verified flips True only when a real button press is observed
   on the CURRENT binding (any button counts as proof of life; only kill
   buttons trigger) and resets on every rebind. The E1 protocol's
   live-fire X check before the first trial remains mandatory; mocked
   tests were green TWICE while the hardware path was broken.

Degraded mode: if no gamepad is bound, the thread stays up (trigger()
still works, Ctrl-C still works), exposes gamepad_available=False, and
pump_once() keeps retrying detection on the main thread.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

# MUST be set before pygame/SDL initializes — see ARCHITECTURE note 3.
os.environ.setdefault("SDL_JOYSTICK_HIDAPI", "0")
os.environ.setdefault("SDL_JOYSTICK_RAWINPUT", "0")
os.environ.setdefault("SDL_JOYSTICK_THREAD", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

from esp32_bridge import Esp32Bridge


class EstopMonitor:
    def __init__(
        self,
        bridge: Esp32Bridge,
        stop_event: threading.Event,
        button_indices: tuple = (0, 1),   # DS4 X on both SDL channels —
                                          # see ARCHITECTURE note 2
        joystick_index: int = 0,          # kept for API compat; ALL pads
                                          # are bound regardless
        poll_hz: float = 50.0,
    ):
        self._bridge = bridge
        self._stop_event = stop_event
        self._buttons = tuple(int(b) for b in button_indices)
        self._joy_index = int(joystick_index)
        self._period_s = 1.0 / float(poll_hz)

        self._thread: Optional[threading.Thread] = None
        self._retry_s = 1.0           # gamepad re-detection cadence
        self._shutdown = threading.Event()
        self._trigger_lock = threading.Lock()
        self._joys: list = []         # bound on the MAIN thread only
        self._last_reinit_mono = 0.0
        self._last_housekeep_mono = 0.0

        self.triggered = False
        self.trigger_source: Optional[str] = None
        self.triggered_at_mono: Optional[float] = None
        self.gamepad_available = False
        self.gamepad_name: Optional[str] = None
        # Honest liveness (ARCHITECTURE note 4): True only after a real
        # press on the CURRENT binding; reset on every rebind. The press
        # history (last_button_press_mono) is kept across rebinds — an
        # old press proves nothing about a new binding, but hiding the
        # history would be lying in the other direction.
        self.binding_verified = False
        self.last_button_press_mono: Optional[float] = None

    # ── lifecycle (main-thread half) ─────────────────────────────────────

    def start(self) -> "EstopMonitor":
        """Bind pads on the CALLER's thread (must be the thread that will
        keep calling pump_once()), then start the read-only poll thread."""
        if self._thread is not None:
            return self
        self.init_gamepads()
        self._thread = threading.Thread(
            target=self._run, name="estop-monitor", daemon=True
        )
        self._thread.start()
        return self

    def pump_once(self) -> None:
        """MAIN-THREAD ONLY, call at ~20 Hz: pumps SDL so input flows
        (note 1). The expensive parts — hotplug-event drain (a sleeping
        pad must flip gamepad_available, not read silent zeros), queue
        clearing, and unbound-retry — are internally throttled to ~2 Hz
        so a host event loop is never saturated (50 Hz of full SDL work
        starved the GIL enough to slow the whole server, found in tests
        2026-08-07)."""
        try:
            import pygame
            if self._joys:
                pygame.event.pump()               # cheap; input delivery
            now = time.monotonic()
            if (now - self._last_housekeep_mono) < 0.5:
                return
            self._last_housekeep_mono = now
            added = getattr(pygame, "JOYDEVICEADDED", None)
            removed = getattr(pygame, "JOYDEVICEREMOVED", None)
            events = []
            if added is not None and removed is not None:
                try:
                    events = pygame.event.get([added, removed])
                except Exception:  # noqa: BLE001 — queue quirks never fatal
                    events = []
            if events:
                # Any device topology change → full main-thread rebind.
                self.init_gamepads(settle_s=0.3)
                return
            if self._joys:
                # Keep the queue from silting up with unread button/axis
                # events (state is POLLED, not event-driven): a full SDL
                # queue drops new events — including the hotplug ones.
                try:
                    pygame.event.clear()
                except Exception:  # noqa: BLE001
                    pass
            elif (now - self._last_reinit_mono) >= self._retry_s:
                self.init_gamepads(settle_s=0.3)
        except Exception:  # noqa: BLE001 — the pump must never kill the host
            pass

    def init_gamepads(self, settle_s: float = 1.0) -> bool:
        """MAIN-THREAD ONLY: (re)bind EVERY connected joystick (note 2).
        Resets binding_verified — a new binding is unproven by definition."""
        self._last_reinit_mono = time.monotonic()
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            # SDL only (re)enumerates when the joystick subsystem inits —
            # quit first to force a rescan. Safe here and ONLY here: this
            # runs on the SDL-owning thread.
            pygame.joystick.quit()
            pygame.joystick.init()
            # Enumeration after a rescan is not instantaneous on Windows;
            # an instant check misses pads and a 1 s retry then wipes and
            # re-asks, racing itself forever. Give it a settle window.
            settle_deadline = time.monotonic() + settle_s
            while time.monotonic() < settle_deadline:
                pygame.event.pump()
                if pygame.joystick.get_count() > 0:
                    break
                time.sleep(0.05)
            n = pygame.joystick.get_count()
            if n == 0:
                self._joys = []
                self.gamepad_available = False
                self.gamepad_name = None
                self.binding_verified = False
                return False
            joys = []
            for i in range(n):
                j = pygame.joystick.Joystick(i)
                j.init()
                joys.append(j)
            self._joys = joys
            self.gamepad_available = True
            self.gamepad_name = ", ".join(j.get_name() for j in joys)
            self.binding_verified = False
            return True
        except Exception:  # noqa: BLE001
            self._joys = []
            self.gamepad_available = False
            self.gamepad_name = None
            self.binding_verified = False
            return False

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

    # ── polling thread (read-only half) ──────────────────────────────────

    def _run(self) -> None:
        """READ-ONLY poll loop (daemon thread). Never touches SDL init or
        the event pump — main-thread work (note 1). Scans EVERY button of
        EVERY bound pad: any press proves liveness; kill buttons latch."""
        while not self._shutdown.is_set():
            joys = self._joys
            if not joys:
                self._shutdown.wait(self._retry_s)
                continue
            try:
                for joy_i, j in enumerate(joys):
                    n = j.get_numbuttons()
                    for b in range(n):
                        if not j.get_button(b):
                            continue
                        self.binding_verified = True
                        self.last_button_press_mono = time.monotonic()
                        if b in self._buttons:
                            self.trigger(f"gamepad{joy_i}-button{b}")
                            # keep polling; trigger() is idempotent
            except Exception:  # noqa: BLE001 — monitor must never die silently
                self._joys = []
                self.gamepad_available = False
                self.gamepad_name = None
                self.binding_verified = False
            time.sleep(self._period_s)
