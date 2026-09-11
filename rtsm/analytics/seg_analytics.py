"""
Segmentation analytics buffer — tracks dual confirmation breakdown per frame.

Tier 1: Per-frame SegFrameStats in a ring buffer (deque, maxlen=300).
Tier 2: Per-second SegSecondBucket with wall-clock retention (default 1 hour).

Works for all backends (fastsam/yoloe/dual). Single-backend modes leave
irrelevant fields at 0 — never None, never crashes.

The Tier-1 -> Tier-2 rollup is owned by ``rtsm.analytics.ticker.AnalyticsTicker``
(one per process, runs whenever ``analytics.enable`` is true, viz or not);
see rtsm/analytics/latency_analytics.py for the ownership and the count-based
selection contract, which this buffer shares.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rtsm.analytics.latency_analytics import STALE_INTERVAL_S


@dataclass
class SegFrameStats:
    """One entry per processed frame — appended by the pipeline."""
    timestamp: float        # time.monotonic()
    frame_seq: int = 0
    backend: str = "dual"   # "dual" | "fastsam" | "yoloe"
    # Post-merge counts
    n_dual: int = 0
    n_fastsam_only: int = 0
    n_yoloe_only: int = 0
    n_total: int = 0
    # Pre-merge raw model output counts (dual only)
    n_fastsam_raw: int = 0
    n_yoloe_raw: int = 0
    # Post-staging survival
    staged_dual: int = 0
    staged_fastsam_only: int = 0
    staged_yoloe_only: int = 0
    # Post-selection (top-K)
    selected_dual: int = 0
    selected_fastsam_only: int = 0
    selected_yoloe_only: int = 0


@dataclass
class SegSecondBucket:
    """One-second aggregate for Tier 2 time-series."""
    wall_ts: float          # time.time() — x-axis for charts
    backend: str = "dual"
    dual_rate: float = 0.0
    fastsam_only_rate: float = 0.0
    yoloe_only_rate: float = 0.0
    mean_total: float = 0.0
    mean_fastsam_raw: float = 0.0
    mean_yoloe_raw: float = 0.0
    staged_survival_rate: float = 0.0
    frames_in_bucket: int = 0
    # Rollup interval bookkeeping (P1 task 5)
    elapsed_s: float = 0.0
    stale_interval: bool = False


def _safe_mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


class SegAnalyticsBuffer:
    """Thread-safe segmentation analytics with two-tier storage."""

    def __init__(self, max_frames: int = 300, retention_s: float = 3600.0):
        # Tier 1 — per-frame ring buffer
        self._buffer: deque[SegFrameStats] = deque(maxlen=max_frames)
        # Tier 2 — per-second buckets (time-evicted, no maxlen)
        self._second_buckets: deque[SegSecondBucket] = deque()
        self._retention_s = retention_s
        # Rollup cursors (time for rates, count for frame selection)
        self._last_rollup_ts: float = time.monotonic()
        self._total_appended: int = 0
        self._last_rollup_total: int = 0
        # Rollup-mechanism diagnostics (survive clear(); see rollup_stats())
        self._rollups: int = 0
        self._stale_rollups: int = 0
        self._ring_truncated: int = 0
        self._lock = threading.Lock()

    def append(self, entry: SegFrameStats) -> None:
        """Tier 1 append — called from pipeline thread."""
        if entry is None:
            return
        with self._lock:
            self._buffer.append(entry)
            self._total_appended += 1

    def roll_up_second(self, now_mono: Optional[float] = None,
                       stale_after_s: float = STALE_INTERVAL_S) -> SegSecondBucket:
        """Aggregate the frames appended since the previous rollup into one
        Tier-2 bucket. Called by the analytics ticker (the one owner per
        process); never by the pipeline. Count-based selection and the
        ``elapsed_s`` / ``stale_interval`` semantics are as in
        PipelineLatencyBuffer.roll_up_second (nothing is skipped on a long
        interval; the means cover the ring's tail if it truncated)."""
        with self._lock:
            now = time.monotonic() if now_mono is None else float(now_mono)
            elapsed = max(0.001, now - self._last_rollup_ts)
            stale = elapsed > float(stale_after_s)

            n = max(0, self._total_appended - self._last_rollup_total)
            truncated = max(0, n - len(self._buffer))
            recent = list(self._buffer)[-n:] if n > 0 else []
            m = len(recent)   # frames available for the means (== n unless the ring truncated)

            total_masks = sum(f.n_total for f in recent) if recent else 0
            total_staged = (
                sum(f.staged_dual + f.staged_fastsam_only + f.staged_yoloe_only for f in recent)
                if recent else 0
            )

            bucket = SegSecondBucket(
                wall_ts=time.time(),
                backend=recent[0].backend if recent else "unknown",
                dual_rate=round(sum(f.n_dual for f in recent) / max(1, total_masks), 3),
                fastsam_only_rate=round(sum(f.n_fastsam_only for f in recent) / max(1, total_masks), 3),
                yoloe_only_rate=round(sum(f.n_yoloe_only for f in recent) / max(1, total_masks), 3),
                mean_total=round(total_masks / max(1, m), 1),
                mean_fastsam_raw=round(sum(f.n_fastsam_raw for f in recent) / max(1, m), 1),
                mean_yoloe_raw=round(sum(f.n_yoloe_raw for f in recent) / max(1, m), 1),
                staged_survival_rate=round(total_staged / max(1, total_masks), 3),
                frames_in_bucket=n,
                elapsed_s=round(elapsed, 3),
                stale_interval=stale,
            )

            self._second_buckets.append(bucket)
            self._evict_old()
            self._last_rollup_ts = now
            self._last_rollup_total = self._total_appended
            self._rollups += 1
            if stale:
                self._stale_rollups += 1
            self._ring_truncated += truncated
            return bucket

    def reset_rollup_clock(self, now_mono: Optional[float] = None) -> None:
        """Start the Tier-2 clock now (ticker start()); the count cursor is left
        alone so frames appended before the owner started still land in the
        first bucket — see PipelineLatencyBuffer.reset_rollup_clock."""
        with self._lock:
            self._last_rollup_ts = time.monotonic() if now_mono is None else float(now_mono)

    def rollup_stats(self, now_mono: Optional[float] = None) -> Dict[str, Any]:
        """Health of the rollup mechanism itself; survives clear()."""
        with self._lock:
            now = time.monotonic() if now_mono is None else float(now_mono)
            return {
                "rollups": self._rollups,
                "stale_rollups": self._stale_rollups,
                "ring_truncated": self._ring_truncated,
                "last_rollup_age_s": round(max(0.0, now - self._last_rollup_ts), 2),
            }

    def aggregate(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Compute rolling stats over Tier 1 for real-time text display."""
        with self._lock:
            entries = list(self._buffer)
            if last_n is not None:
                entries = entries[-last_n:]

        if not entries:
            return {
                "frame_count": 0,
                "backend": "unknown",
                "dual_rate": 0.0,
                "fastsam_only_rate": 0.0,
                "yoloe_only_rate": 0.0,
                "mean_total": 0.0,
                "mean_fastsam_raw": 0.0,
                "mean_yoloe_raw": 0.0,
                "staged_survival_rate": 0.0,
                "selected_rate_by_source": {"dual": 0.0, "fastsam_only": 0.0, "yoloe_only": 0.0},
            }

        total_masks = sum(f.n_total for f in entries)
        total_selected = sum(f.selected_dual + f.selected_fastsam_only + f.selected_yoloe_only for f in entries)
        n = len(entries)

        return {
            "frame_count": n,
            "backend": entries[-1].backend,
            "dual_rate": round(sum(f.n_dual for f in entries) / max(1, total_masks), 3),
            "fastsam_only_rate": round(sum(f.n_fastsam_only for f in entries) / max(1, total_masks), 3),
            "yoloe_only_rate": round(sum(f.n_yoloe_only for f in entries) / max(1, total_masks), 3),
            "mean_total": round(total_masks / max(1, n), 1),
            "mean_fastsam_raw": round(sum(f.n_fastsam_raw for f in entries) / max(1, n), 1),
            "mean_yoloe_raw": round(sum(f.n_yoloe_raw for f in entries) / max(1, n), 1),
            "staged_survival_rate": round(
                sum(f.staged_dual + f.staged_fastsam_only + f.staged_yoloe_only for f in entries)
                / max(1, total_masks), 3
            ),
            "selected_rate_by_source": {
                "dual": round(sum(f.selected_dual for f in entries) / max(1, total_selected), 3),
                "fastsam_only": round(sum(f.selected_fastsam_only for f in entries) / max(1, total_selected), 3),
                "yoloe_only": round(sum(f.selected_yoloe_only for f in entries) / max(1, total_selected), 3),
            },
        }

    def hourly_history(self) -> List[Dict[str, Any]]:
        """Return Tier 2 buckets as list of dicts for chart rendering (snapshot
        under the lock, serialisation outside it — see PipelineLatencyBuffer)."""
        with self._lock:
            buckets = list(self._second_buckets)
        return [dataclasses.asdict(b) for b in buckets]

    def clear(self) -> None:
        """Reset the session's data — called on /reset (rollup_stats survive)."""
        with self._lock:
            self._buffer.clear()
            self._second_buckets.clear()
            self._total_appended = 0
            self._last_rollup_total = 0
            # _last_rollup_ts deliberately untouched (see PipelineLatencyBuffer.clear)

    def _evict_old(self) -> None:
        """Remove Tier 2 buckets older than retention window. Must hold lock."""
        cutoff = time.time() - self._retention_s
        while self._second_buckets and self._second_buckets[0].wall_ts < cutoff:
            self._second_buckets.popleft()
