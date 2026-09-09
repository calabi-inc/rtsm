"""
Send-gated HTTP client for the ESP32 motor controller.

Ports the hardware-proven discipline from teleop_gamepad.py verbatim:

  * `Connection: close` on every request — the Arduino WebServer has no
    HTTP keep-alive; pooled sockets go stale and fail mysteriously.
    (Deliberately NO requests.Session.)
  * Send /drive only when the command CHANGED (> change_epsilon) or a
    HEARTBEAT is due (< 0.3 s — feeds the firmware's 300 ms watchdog).
  * Attempts are rate-capped at drive_rate_hz (~10 Hz) — faster exhausts
    the ESP32's connection pool with TIME_WAIT sockets and can starve /stop.
  * Failures DROP THE FRAME (never raise into the control loop); the
    firmware watchdog is the safety net if failures persist.

Fail-safe semantics: if the caller stops calling drive() for any reason
(crash, GIL stall, WiFi loss), the firmware watchdog stops the motors
within 300 ms. The bridge deliberately has NO background heartbeat thread —
a dead control loop MUST mean a stopped car.

E-stop latch: stop(estop=True) latches the bridge; every subsequent
drive() becomes a no-op until reset_estop() is called explicitly. This
prevents a queued/racing drive() from re-energizing the motors after an
emergency stop. A normal stop() (arrival, preempt) does not latch.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import requests

_HEADERS = {"Connection": "close"}


class Esp32Bridge:
    def __init__(
        self,
        url: str,
        drive_rate_hz: float = 10.0,
        heartbeat_s: float = 0.25,
        change_epsilon: float = 0.04,
        http_timeout_s: float = 0.6,
    ):
        self.url = url.rstrip("/")
        self._min_interval_s = 1.0 / float(drive_rate_hz)
        self._heartbeat_s = float(heartbeat_s)
        self._eps = float(change_epsilon)
        self._timeout_s = float(http_timeout_s)

        self._lock = threading.Lock()
        self._estopped = False
        self._last_sent: Optional[tuple] = None   # (left, right) last SUCCESSFUL send
        self._last_success_mono = 0.0             # heartbeat reference
        self._last_attempt_mono = 0.0             # rate-cap reference (incl. failures)
        self.consecutive_failures = 0

    # ── drive ────────────────────────────────────────────────────────────

    def drive(self, left: float, right: float) -> bool:
        """Send-gated /drive. Returns True iff an HTTP send succeeded now.

        Call this every control tick; the bridge decides whether the wire
        needs a packet. Clamps to [-1, 1]. No-op while e-stop is latched.

        SAFETY: the HTTP POST happens OUTSIDE the lock — the lock guards
        only gating state, so stop(estop=True) can latch in microseconds
        even while a drive request is in flight. If the latch was set
        during our flight, we fire a COMPENSATING /stop behind our own
        landed command, bounding stray motion to network jitter instead
        of the 300 ms watchdog window.
        """
        left = max(-1.0, min(1.0, float(left)))
        right = max(-1.0, min(1.0, float(right)))

        with self._lock:
            if self._estopped:
                return False
            now = time.monotonic()

            changed = (
                self._last_sent is None
                or abs(left - self._last_sent[0]) > self._eps
                or abs(right - self._last_sent[1]) > self._eps
            )
            heartbeat_due = (now - self._last_success_mono) >= self._heartbeat_s
            if not (changed or heartbeat_due):
                return False
            # Rate-cap ALL attempts (including failure retries): >~10 Hz
            # exhausts the ESP32 socket pool. Reserve the slot under lock.
            if (now - self._last_attempt_mono) < self._min_interval_s:
                return False
            self._last_attempt_mono = now

        # ---- HTTP outside the lock (never blocks the e-stop latch) ----
        ok = True
        try:
            requests.post(
                f"{self.url}/drive",
                json={"left": left, "right": right},
                timeout=self._timeout_s,
                headers=_HEADERS,
            )
        except requests.RequestException:
            ok = False

        with self._lock:
            if ok:
                self._last_sent = (left, right)
                self._last_success_mono = time.monotonic()
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            latched_during_flight = self._estopped

        if ok and latched_during_flight:
            # Our command may have landed AFTER the e-stop's /stop —
            # compensate immediately from this thread.
            self._post_stop_once()
        return ok

    def _post_stop_once(self) -> bool:
        try:
            requests.post(
                f"{self.url}/stop", timeout=self._timeout_s, headers=_HEADERS
            )
            return True
        except requests.RequestException:
            return False

    # ── stop ─────────────────────────────────────────────────────────────

    def stop(self, estop: bool = False) -> bool:
        """POST /stop, bypassing all gating. Retries once on failure.

        estop=True latches the bridge (drive() no-ops until reset_estop());
        the latch is set BEFORE the HTTP call so a concurrent drive() cannot
        slip through even if the network is slow. If HTTP fails entirely,
        the firmware watchdog stops the car within 300 ms anyway.
        """
        if estop:
            with self._lock:            # microseconds — no HTTP under this lock
                self._estopped = True
        ok = False
        for _ in range(2):
            if self._post_stop_once():
                ok = True
                break
        with self._lock:
            self._last_sent = (0.0, 0.0)
        return ok

    def reset_estop(self) -> None:
        """Explicitly clear the e-stop latch (operator/new-trial action)."""
        with self._lock:
            self._estopped = False

    @property
    def estopped(self) -> bool:
        with self._lock:
            return self._estopped

    # ── preflight / telemetry ───────────────────────────────────────────

    def ping(self) -> Optional[str]:
        """GET / — returns the status text (battery mV, RSSI) or None."""
        try:
            r = requests.get(self.url + "/", timeout=2.0, headers=_HEADERS)
            return r.text
        except requests.RequestException:
            return None

    def battery_mv(self) -> Optional[int]:
        """GET /battery -> mV, or None if unreachable/unparseable."""
        try:
            r = requests.get(
                f"{self.url}/battery", timeout=2.0, headers=_HEADERS
            )
            return int(r.json().get("mv"))
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return None
