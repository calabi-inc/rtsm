"""
Analytics ticker — the ONE owner of the Tier-1 -> Tier-2 rollup (P1 task 5).

Before this module the rollup (``roll_up_second`` on both analytics buffers)
ran only inside the visualization server's push loop, and only while a browser
client was attached: a headless process — ``python -m rtsm --replay``, the eval
harness, ``rtsm demo --no-viz``, a robot without a dashboard — never produced
a single per-second bucket, ``aggregate()['input_hz']`` read 0.0 and
``effective_ratio`` read ~1000 (processing / 0.001).

``AnalyticsTicker`` is a 1 Hz daemon thread that runs whenever
``analytics.enable`` is true, independent of the visualization server, of its
client count and of replay/live. Each tick, in order, each step isolated so a
raising collaborator cannot stop the thread:

  1. ``latency.snapshot_wm(total, confirmed, proto)`` from ``wm.stats()``;
  2. ``roll_up_second(now_mono=now, stale_after_s=max(2 s, 2 x interval))`` on
     both buffers with ONE clock reading;
  3. publish ``TickRecord(tick, latency, seg, mono)`` into a bounded deque for
     consumers (``latest()``, ``since(tick)``) — the visualization server reads
     those and never rolls up itself.

``stats()`` is the ticker's own health, served as ``/stats/analytics.rollup``:
``ticks``, ``late_ticks`` (a tick-to-tick gap > 2 x interval — a 1 Hz thread
that cannot get scheduled is the GIL-wedge symptom the wedge gate looks for),
``stale_rollups`` / ``ring_truncated`` (the buffers' own flags, summed),
``last_tick_age_s``, ``stalled`` (read-time: no tick for > 2 x interval — the
counters above only move when a tick eventually happens), ``alive``. None of
it is reset by ``POST /reset``.

``build_analytics(cfg, wm=...)`` is how the runners get the buffers AND the
ticker in one call (``AnalyticsBundle``); ``bundle.start()`` belongs
immediately before ``pipe.run_forever()`` — after every model load, the
receiver and the API server — so the counters measure the run, not startup.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

from rtsm.analytics.latency_analytics import PipelineLatencyBuffer, STALE_INTERVAL_S
from rtsm.analytics.seg_analytics import SegAnalyticsBuffer

logger = logging.getLogger(__name__)

TICK_INTERVAL_S = 1.0          # the buckets are per-second by name; not a yaml key
TICK_HISTORY = 64              # published ticks kept for consumers (since())


@dataclass(frozen=True)
class TickRecord:
    """One published tick: the bucket each buffer rolled (None for an absent
    buffer or a rollup that raised) and the ticker's monotonic stamp."""
    tick: int
    latency: Any = None
    seg: Any = None
    mono: float = 0.0
    late: bool = False          # tick-to-tick gap exceeded 2 x interval


class AnalyticsTicker(threading.Thread):
    """Daemon thread owning the Tier-2 rollup. See the module docstring."""

    def __init__(
        self,
        latency: Optional[PipelineLatencyBuffer] = None,
        seg: Optional[SegAnalyticsBuffer] = None,
        *,
        wm: Any = None,
        interval_s: float = TICK_INTERVAL_S,
        now_fn: Callable[[], float] = time.monotonic,
        history: int = TICK_HISTORY,
    ) -> None:
        super().__init__(name="analytics-ticker", daemon=True)
        if latency is None and seg is None:
            raise ValueError("AnalyticsTicker needs at least one analytics buffer")
        if not (float(interval_s) > 0):
            raise ValueError(f"interval_s must be > 0 s; got {interval_s!r}")
        self._latency = latency
        self._seg = seg
        self._wm = wm
        self._interval = float(interval_s)
        self._now = now_fn
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._recent: Deque[TickRecord] = deque(maxlen=max(1, int(history)))
        self._ticks = 0
        self._late_ticks = 0
        self._started_mono: Optional[float] = None
        self._last_tick_mono: Optional[float] = None

    # ── identity ────────────────────────────────────────────────────────

    @property
    def interval_s(self) -> float:
        return self._interval

    @property
    def stale_after_s(self) -> float:
        """Threshold passed to the buffers: a rollup interval above it is
        flagged ``stale_interval``. Equals the ``late_ticks`` rule
        (2 x interval) at any interval >= 1 s."""
        return max(STALE_INTERVAL_S, 2.0 * self._interval)

    # ── lifecycle ───────────────────────────────────────────────────────

    def arm(self, now_mono: Optional[float] = None) -> None:
        """Re-anchor both buffers' Tier-2 clocks to now and take the reference
        stamp for the first tick's lateness rule. ``start()`` calls this; tests
        that drive ``tick()`` with an injected clock call it directly. The
        buffers were built before the model loads; without the re-anchor the
        first bucket would span the whole startup and be flagged stale."""
        now = self._now() if now_mono is None else float(now_mono)
        for buf in (self._latency, self._seg):
            if buf is not None:
                try:
                    buf.reset_rollup_clock(now_mono=now)
                except Exception:
                    logger.debug("[analytics] reset_rollup_clock failed", exc_info=True)
        self._started_mono = now

    @property
    def armed(self) -> bool:
        return self._started_mono is not None

    def start(self) -> None:  # type: ignore[override]
        """``arm()`` then start the daemon thread. A ticker that was already
        stopped refuses to start (loudly) instead of running a dead loop."""
        if self._stop_evt.is_set():
            raise RuntimeError("AnalyticsTicker: stop() was called before start(); a stopped ticker cannot be restarted")
        self.arm()
        super().start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to exit and join it (no-op when never started)."""
        self._stop_evt.set()
        if self._started_mono is not None and self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)

    @staticmethod
    def next_deadline(deadline: float, now: float, interval: float, late: bool) -> float:
        """Cadence rule. On time: the schedule advances by one interval (no
        drift). Late (gap > 2 x interval, already counted): restart from now
        rather than catching up with a burst. In between — a wake-up delay of
        one to two intervals — the schedule would already be in the past and
        the next tick would fire immediately, producing a compressed bucket
        whose rates divide by a few milliseconds without any flag; the clamp
        keeps every interval at least half an interval long."""
        if late:
            return now + interval
        return max(deadline + interval, now + 0.5 * interval)

    def run(self) -> None:
        deadline = self._now() + self._interval
        while True:
            wait_s = max(0.0, deadline - self._now())
            if self._stop_evt.wait(wait_s):
                return
            now = self._now()
            rec = self.tick(now_mono=now)
            deadline = self.next_deadline(deadline, now, self._interval, rec.late)

    # ── one tick (also the test seam) ───────────────────────────────────

    def tick(self, now_mono: Optional[float] = None) -> TickRecord:
        """Perform one tick now. Public so tests can drive the ticker with an
        injected clock instead of sleeping; ``run()`` calls exactly this."""
        now = self._now() if now_mono is None else float(now_mono)
        prev = self._last_tick_mono if self._last_tick_mono is not None else self._started_mono
        late = prev is not None and (now - prev) > 2.0 * self._interval

        # (1) WM snapshot -> latency buffer (object-health fields of the bucket)
        if self._latency is not None and self._wm is not None:
            try:
                st = self._wm.stats() or {}
                total = int(st.get("objects", 0) or 0)
                confirmed = int(st.get("confirmed", 0) or 0)
                self._latency.snapshot_wm(total=total, confirmed=confirmed, proto=max(0, total - confirmed))
            except Exception:
                logger.debug("[analytics] wm.stats() failed; bucket keeps the previous snapshot", exc_info=True)

        # (2) rollups, one clock reading for both buffers
        lat_b = seg_b = None
        if self._latency is not None:
            try:
                lat_b = self._latency.roll_up_second(now_mono=now, stale_after_s=self.stale_after_s)
            except Exception:
                logger.warning("[analytics] latency roll_up_second failed", exc_info=True)
        if self._seg is not None:
            try:
                seg_b = self._seg.roll_up_second(now_mono=now, stale_after_s=self.stale_after_s)
            except Exception:
                logger.warning("[analytics] segmentation roll_up_second failed", exc_info=True)

        # (3) publish
        with self._lock:
            self._ticks += 1
            if late:
                self._late_ticks += 1
            self._last_tick_mono = now
            rec = TickRecord(tick=self._ticks, latency=lat_b, seg=seg_b, mono=now, late=late)
            self._recent.append(rec)
        return rec

    # ── consumers ───────────────────────────────────────────────────────

    def latest(self) -> Optional[TickRecord]:
        with self._lock:
            return self._recent[-1] if self._recent else None

    def since(self, tick: Optional[int]) -> List[TickRecord]:
        """Published ticks after ``tick`` (all retained ticks when None), oldest
        first; bounded by the history depth, so a consumer that fell further
        behind must resync from ``hourly_history()``."""
        with self._lock:
            if tick is None:
                return list(self._recent)
            return [r for r in self._recent if r.tick > tick]

    def stats(self, now_mono: Optional[float] = None) -> Dict[str, Any]:
        now = self._now() if now_mono is None else float(now_mono)
        stale = truncated = 0
        for buf in (self._latency, self._seg):
            if buf is None:
                continue
            try:
                rs = buf.rollup_stats(now_mono=now)
                stale += int(rs.get("stale_rollups", 0) or 0)
                truncated += int(rs.get("ring_truncated", 0) or 0)
            except Exception:
                pass
        with self._lock:
            last = self._last_tick_mono
            ref = last if last is not None else self._started_mono
            # A ticker that stopped ticking (wedged in wm.stats(), starved
            # thread) never increments late_ticks — that counter only moves
            # when a tick eventually happens. `stalled` is the read-time view:
            # more than 2 x interval since the last tick (or since arm()).
            stalled = ref is not None and (now - ref) > 2.0 * self._interval
            return {
                "interval_s": self._interval,
                "ticks": self._ticks,
                "late_ticks": self._late_ticks,
                "stale_rollups": stale,
                "ring_truncated": truncated,
                "last_tick_age_s": None if last is None else round(max(0.0, now - last), 2),
                "stalled": bool(stalled),
                "alive": self.is_alive(),
            }


# ── runner-facing bundle ────────────────────────────────────────────────


@dataclass
class AnalyticsBundle:
    """What ``build_analytics`` hands the runners: the two Tier-1 buffers and
    the rollup owner (all None when ``analytics.enable`` is false)."""
    seg: Optional[SegAnalyticsBuffer] = None
    latency: Optional[PipelineLatencyBuffer] = None
    ticker: Optional[AnalyticsTicker] = None
    retention_s: float = 3600.0
    buffer_frames: int = 300

    @property
    def enabled(self) -> bool:
        return self.ticker is not None

    def start(self) -> None:
        """Start the rollup owner. Call immediately before ``pipe.run_forever()``.
        Idempotent while running; a ticker that already ran and stopped is not
        restarted (threads cannot be), and that is logged instead of silent."""
        if self.ticker is None:
            return
        if self.ticker.is_alive():
            return
        if self.ticker.armed:
            logger.warning("[analytics] ticker was already started and stopped; the Tier-2 rollup stays off")
            return
        self.ticker.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self.ticker is not None:
            self.ticker.stop(timeout=timeout)


def build_analytics(cfg: Dict[str, Any], *, wm: Any = None, interval_s: float = TICK_INTERVAL_S) -> AnalyticsBundle:
    """Build the analytics buffers and their rollup owner from the ``analytics:``
    block (``enable`` default true, ``retention_s`` 3600, ``buffer_frames``
    300). The ticker is constructed, not started (``bundle.start()``)."""
    a = (cfg or {}).get("analytics") or {}
    if isinstance(a, dict) and not bool(a.get("enable", True)):
        return AnalyticsBundle()
    if not isinstance(a, dict):
        a = {}
    retention_s = float(a.get("retention_s", 3600))
    buffer_frames = int(a.get("buffer_frames", 300))
    seg = SegAnalyticsBuffer(max_frames=buffer_frames, retention_s=retention_s)
    lat = PipelineLatencyBuffer(max_frames=buffer_frames, retention_s=retention_s)
    ticker = AnalyticsTicker(lat, seg, wm=wm, interval_s=interval_s)
    return AnalyticsBundle(seg=seg, latency=lat, ticker=ticker, retention_s=retention_s, buffer_frames=buffer_frames)
