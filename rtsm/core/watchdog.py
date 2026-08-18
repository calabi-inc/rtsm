"""
Frame-flow watchdog: distinguishes "pipeline alive but starved", "pipeline
hung", and "receiver dead" — states that are indistinguishable from the
outside today because /healthz has no frame-flow awareness.

Purely observational: it never restarts anything. It classifies, exposes a
status dict for /healthz, and logs state transitions.

Components:
- PipelineHeartbeat: two monotonic stamps the pipeline updates as it runs.
- FrameFlowMonitor: pure classification logic (injectable clock, unit-testable).
- Watchdog: daemon thread polling the monitor and logging transitions.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# States that mean "something is wrong". "waiting" (no client has ever
# connected) and "ok" are healthy.
DEGRADED_STATES = frozenset(
    {"receiver_dead", "hung", "starved", "pose_degraded", "no_ingestible_input"}
)


class PipelineHeartbeat:
    """Monotonic stamps updated by the pipeline loop.

    beat_step() proves the loop is turning (stamped every run_one_step call,
    including idle ones); beat_frame() proves frames are actually being
    processed. Bare float assignment is atomic under the GIL, so readers on
    other threads need no lock.
    """

    def __init__(self) -> None:
        self.last_step_mono: Optional[float] = None
        self.last_frame_mono: Optional[float] = None

    def beat_step(self) -> None:
        self.last_step_mono = time.monotonic()

    def beat_frame(self) -> None:
        self.last_frame_mono = time.monotonic()


class FrameFlowMonitor:
    """Classifies frame flow into one state per evaluation.

    Sources are callables so the monitor stays decoupled and testable:
    - heartbeat: PipelineHeartbeat (read-only)
    - queue_size: () -> int, current ingest queue depth
    - receiver_liveness: () -> {"alive": bool,
                                "last_rx_mono": float|None,
                                "last_enqueue_mono": float|None,
                                "tracking_drops": int}

    States, highest priority first:
    - receiver_dead: receiver thread is gone; nothing will ever arrive.
    - hung: frames are waiting (queue non-empty, or enqueued more recently
      than the last pipeline step) but the loop stopped turning.
    - starved: nothing has arrived from the client for starved_after_s
      (stream died, phone slept, WiFi drop).
    - pose_degraded: messages ARE arriving but nothing is ingestible and
      tracking drops are climbing (ARKit tracking limited/lost).
    - no_ingestible_input: messages arriving, nothing enqueued, no tracking
      drops — e.g. text-only traffic or every frame throttled/failing parse.
    - waiting: no client has ever connected since boot (normal at startup).
    - ok: everything else.
    """

    def __init__(
        self,
        *,
        heartbeat: PipelineHeartbeat,
        queue_size: Callable[[], int],
        receiver_liveness: Callable[[], Dict[str, Any]],
        starved_after_s: float = 5.0,
        hung_after_s: float = 10.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._hb = heartbeat
        self._queue_size = queue_size
        self._receiver_liveness = receiver_liveness
        self._starved_after_s = float(starved_after_s)
        self._hung_after_s = float(hung_after_s)
        self._now = now_fn
        self._prev_tracking_drops: Optional[int] = None

    @staticmethod
    def _age(now: float, stamp: Optional[float]) -> Optional[float]:
        return None if stamp is None else max(0.0, now - stamp)

    def evaluate(self) -> Dict[str, Any]:
        now = self._now()
        recv = self._receiver_liveness() or {}
        alive = bool(recv.get("alive", False))
        rx_age = self._age(now, recv.get("last_rx_mono"))
        enq_age = self._age(now, recv.get("last_enqueue_mono"))
        step_age = self._age(now, self._hb.last_step_mono)
        frame_age = self._age(now, self._hb.last_frame_mono)
        drops = int(recv.get("tracking_drops", 0) or 0)
        drops_delta = 0 if self._prev_tracking_drops is None else max(
            0, drops - self._prev_tracking_drops
        )
        self._prev_tracking_drops = drops
        try:
            qsize = int(self._queue_size())
        except Exception:
            qsize = -1

        state = "ok"
        reasons = []
        if not alive:
            state = "receiver_dead"
            reasons.append("receiver thread is not alive")
        elif step_age is not None and step_age > self._hung_after_s and (
            qsize > 0 or (enq_age is not None and enq_age < step_age)
        ):
            # Frames are waiting but the pipeline loop stopped turning.
            state = "hung"
            reasons.append(
                f"pipeline loop silent for {step_age:.1f}s with frames waiting"
            )
        elif rx_age is None:
            state = "waiting"
        elif rx_age > self._starved_after_s:
            state = "starved"
            reasons.append(f"no input from client for {rx_age:.1f}s")
        elif enq_age is not None and enq_age > self._starved_after_s:
            if drops_delta > 0:
                state = "pose_degraded"
                reasons.append(
                    f"input arriving but tracking-drops climbing "
                    f"(+{drops_delta}); nothing enqueued for {enq_age:.1f}s"
                )
            else:
                state = "no_ingestible_input"
                reasons.append(
                    f"input arriving but nothing enqueued for {enq_age:.1f}s"
                )
        elif enq_age is None and rx_age is not None:
            # Client is talking but not one frame has ever been accepted.
            if drops_delta > 0:
                state = "pose_degraded"
                reasons.append(
                    f"client connected but no frame ever enqueued "
                    f"(tracking-drops +{drops_delta})"
                )

        def _r(v: Optional[float]) -> Optional[float]:
            return None if v is None else round(v, 2)

        return {
            "state": state,
            "degraded": state in DEGRADED_STATES,
            "reasons": reasons,
            "receiver_alive": alive,
            "last_input_age_s": _r(rx_age),
            "last_ingest_age_s": _r(enq_age),
            "last_step_age_s": _r(step_age),
            "last_frame_age_s": _r(frame_age),
            "ingest_queue_depth": qsize,
            "tracking_drops": drops,
        }


class Watchdog(threading.Thread):
    """Daemon thread: polls the monitor, caches the result for /healthz,
    and logs state transitions (WARNING on degradation, INFO on recovery)."""

    def __init__(
        self,
        *,
        heartbeat: PipelineHeartbeat,
        queue_size: Callable[[], int],
        receiver_liveness: Callable[[], Dict[str, Any]],
        starved_after_s: float = 5.0,
        hung_after_s: float = 10.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        super().__init__(name="frame-flow-watchdog", daemon=True)
        self._monitor = FrameFlowMonitor(
            heartbeat=heartbeat,
            queue_size=queue_size,
            receiver_liveness=receiver_liveness,
            starved_after_s=starved_after_s,
            hung_after_s=hung_after_s,
        )
        self._poll_interval_s = max(0.1, float(poll_interval_s))
        self._stop_event = threading.Event()
        self._last: Dict[str, Any] = {
            "state": "waiting",
            "degraded": False,
            "reasons": [],
        }
        self._last_logged_state = "waiting"

    def status(self) -> Dict[str, Any]:
        return dict(self._last)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            try:
                result = self._monitor.evaluate()
            except Exception as e:  # never let the watchdog die silently
                logger.warning(f"[watchdog] evaluation error: {e}")
                continue
            self._last = result
            state = result["state"]
            if state != self._last_logged_state:
                msg = (
                    f"[watchdog] frame flow: {self._last_logged_state} -> {state}"
                    + (f" ({'; '.join(result['reasons'])})" if result["reasons"] else "")
                )
                if result["degraded"]:
                    logger.warning(msg)
                else:
                    logger.info(msg)
                self._last_logged_state = state
