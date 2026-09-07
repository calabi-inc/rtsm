"""
Pipeline latency & throughput analytics buffer.

Tracks per-frame stage timings, input/processing rates, queue pressure,
and 4 drop points (tracking, throttle, queue full, gate rejection).

Tier 1: Per-frame FrameTimingStats in a ring buffer (deque, maxlen=300).
Tier 2: Per-second LatencySecondBucket with wall-clock retention (default 1 hour).
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FrameTimingStats:
    """One entry per processed frame — appended by the pipeline."""
    timestamp: float          # time.monotonic()
    frame_seq: int = 0
    is_keyframe: bool = False
    t_segmentation: float = 0.0
    t_heuristics: float = 0.0
    t_scoring: float = 0.0
    t_clip: float = 0.0
    t_association: float = 0.0
    t_total: float = 0.0
    queue_depth: int = 0
    n_masks_in: int = 0
    n_masks_staged: int = 0
    n_candidates: int = 0
    # Association outcome (per frame)
    assoc_matched: int = 0
    assoc_created: int = 0


@dataclass
class LatencySecondBucket:
    """One-second aggregate for Tier 2 time-series."""
    wall_ts: float              # time.time() — x-axis for charts
    input_hz: float = 0.0
    processing_hz: float = 0.0
    effective_ratio: float = 0.0
    # Drop counters (per second)
    queue_drops: int = 0
    gate_rejections: int = 0
    frame_rejections: int = 0   # frame-quality gate (gates.*)
    throttle_skips: int = 0
    tracking_drops: int = 0
    # Queue pressure
    queue_depth_mean: float = 0.0
    queue_depth_max: int = 0
    # Stage timing (ms)
    t_total_ms: float = 0.0
    t_seg_ms: float = 0.0
    t_heur_ms: float = 0.0
    t_scoring_ms: float = 0.0
    t_clip_ms: float = 0.0
    t_assoc_ms: float = 0.0
    frames_in_bucket: int = 0
    # Association / WM health (per second)
    assoc_matched: int = 0          # objects matched this second
    assoc_created: int = 0          # new protos spawned this second
    wm_total: int = 0               # total WM objects at snapshot time
    wm_confirmed: int = 0           # confirmed objects at snapshot time
    wm_proto: int = 0               # proto objects at snapshot time


def _mean(values: list) -> float:
    return sum(values) / max(1, len(values))


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * pct)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _timing_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean/p50/p95/max from a list of durations."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "mean": round(_mean(s), 4),
        "p50": round(_percentile(s, 0.50), 4),
        "p95": round(_percentile(s, 0.95), 4),
        "max": round(s[-1], 4),
    }


class PipelineLatencyBuffer:
    """Thread-safe latency/throughput analytics with two-tier storage."""

    def __init__(self, max_frames: int = 300, retention_s: float = 3600.0, warmup_skip: int = 5):
        # Tier 1 — per-frame ring buffer
        self._buffer: deque[FrameTimingStats] = deque(maxlen=max_frames)
        # Tier 2 — per-second buckets (time-evicted)
        self._second_buckets: deque[LatencySecondBucket] = deque()
        self._retention_s = retention_s
        self._warmup_skip = warmup_skip  # skip first N frames for percentile stats
        self._total_appended: int = 0    # lifetime frame counter

        # Input rate tracking (monotonically increasing)
        self._received_count: int = 0

        # Drop counters (all monotonically increasing, diffed at rollup)
        self._gate_rejections: int = 0
        self._frame_rejections: int = 0
        self._queue_drops: int = 0
        self._throttle_skips: int = 0
        self._tracking_drops: int = 0

        # Queue depth sampling (capped to prevent unbounded growth if no clients)
        self._queue_depth_samples: deque[int] = deque(maxlen=120)  # ~4s at 30 Hz

        # Rollup cursors
        self._last_rollup_ts: float = time.monotonic()
        self._last_rollup_received: int = 0
        self._last_rollup_rejections: int = 0
        self._last_rollup_frame_rejections: int = 0
        self._last_rollup_queue_drops: int = 0
        self._last_rollup_throttle_skips: int = 0
        self._last_rollup_tracking_drops: int = 0

        # WM snapshot (updated by vis server before rollup)
        self._wm_snapshot: dict = {"total": 0, "confirmed": 0, "proto": 0}

        self._lock = threading.Lock()

    # ---- Tier 1 append (pipeline thread) ----

    def append(self, entry: FrameTimingStats) -> None:
        """Append per-frame timing stats. Called from the pipeline thread."""
        if entry is None:
            return
        with self._lock:
            self._buffer.append(entry)
            self._total_appended += 1

    # ---- Receiver-side counters (receiver thread, ~30 Hz) ----

    def record_frame_received(self) -> None:
        """Called on every arriving frame, before any filtering."""
        with self._lock:
            self._received_count += 1

    def record_gate_rejection(self) -> None:
        """Called when the ingest gate rejects a frame."""
        with self._lock:
            self._gate_rejections += 1

    def record_frame_rejection(self) -> None:
        """Called when the frame-quality gate skips a frame."""
        with self._lock:
            self._frame_rejections += 1

    def record_queue_drop(self) -> None:
        """Called when IngestQueue.put() returns False (queue full)."""
        with self._lock:
            self._queue_drops += 1

    def record_throttle_skip(self) -> None:
        """Called when non-KF throttle skips a frame (by design)."""
        with self._lock:
            self._throttle_skips += 1

    def record_tracking_drop(self) -> None:
        """Called when tracking_state != normal drops a frame."""
        with self._lock:
            self._tracking_drops += 1

    def snapshot_wm(self, total: int, confirmed: int, proto: int) -> None:
        """Inject WM state snapshot for next rollup. Called by vis server."""
        with self._lock:
            self._wm_snapshot = {"total": total, "confirmed": confirmed, "proto": proto}

    def sample_queue_depth(self, depth: int) -> None:
        """Record queue depth at enqueue time for mean/max stats."""
        with self._lock:
            self._queue_depth_samples.append(depth)

    # ---- Tier 2 rollup (vis server thread, ~1 Hz) ----

    def roll_up_second(self) -> LatencySecondBucket:
        """Aggregate recent Tier 1 data into a Tier 2 bucket.

        Called by the vis server's push loop. Never called by the pipeline.
        """
        with self._lock:
            now_mono = time.monotonic()
            elapsed = now_mono - self._last_rollup_ts

            # Stale cursor guard: if no rollup for >2s (e.g., no clients),
            # skip accumulated data — rates would be meaningless.
            if elapsed > 2.0:
                self._last_rollup_ts = now_mono
                self._last_rollup_received = self._received_count
                self._last_rollup_rejections = self._gate_rejections
                self._last_rollup_frame_rejections = self._frame_rejections
                self._last_rollup_queue_drops = self._queue_drops
                self._last_rollup_throttle_skips = self._throttle_skips
                self._last_rollup_tracking_drops = self._tracking_drops
                self._queue_depth_samples.clear()
                bucket = LatencySecondBucket(wall_ts=time.time(), frames_in_bucket=0)
                self._second_buckets.append(bucket)
                self._evict_old()
                return bucket

            elapsed = max(0.001, elapsed)

            # Frames processed since last rollup
            recent = [f for f in self._buffer if f.timestamp > self._last_rollup_ts]
            n = len(recent)

            # Input rate since last rollup
            received_since = self._received_count - self._last_rollup_received
            input_hz = received_since / elapsed

            # Drop deltas since last rollup
            gate_rej = self._gate_rejections - self._last_rollup_rejections
            frame_rej = self._frame_rejections - self._last_rollup_frame_rejections
            q_drops = self._queue_drops - self._last_rollup_queue_drops
            throttle = self._throttle_skips - self._last_rollup_throttle_skips
            tracking = self._tracking_drops - self._last_rollup_tracking_drops

            # Queue depth stats
            samples = list(self._queue_depth_samples)
            q_mean = round(_mean(samples), 1) if samples else 0.0
            q_max = max(samples) if samples else 0

            processing_hz = n / elapsed

            bucket = LatencySecondBucket(
                wall_ts=time.time(),
                input_hz=round(input_hz, 1),
                processing_hz=round(processing_hz, 2),
                effective_ratio=round(processing_hz / max(0.001, input_hz), 3),
                queue_drops=q_drops,
                gate_rejections=gate_rej,
                frame_rejections=frame_rej,
                throttle_skips=throttle,
                tracking_drops=tracking,
                queue_depth_mean=q_mean,
                queue_depth_max=q_max,
                t_total_ms=round(_mean([f.t_total for f in recent]) * 1000, 1) if recent else 0.0,
                t_seg_ms=round(_mean([f.t_segmentation for f in recent]) * 1000, 1) if recent else 0.0,
                t_heur_ms=round(_mean([f.t_heuristics for f in recent]) * 1000, 1) if recent else 0.0,
                t_scoring_ms=round(_mean([f.t_scoring for f in recent]) * 1000, 1) if recent else 0.0,
                t_clip_ms=round(_mean([f.t_clip for f in recent]) * 1000, 1) if recent else 0.0,
                t_assoc_ms=round(_mean([f.t_association for f in recent]) * 1000, 1) if recent else 0.0,
                frames_in_bucket=n,
                assoc_matched=sum(f.assoc_matched for f in recent),
                assoc_created=sum(f.assoc_created for f in recent),
                wm_total=self._wm_snapshot.get('total', 0),
                wm_confirmed=self._wm_snapshot.get('confirmed', 0),
                wm_proto=self._wm_snapshot.get('proto', 0),
            )

            self._second_buckets.append(bucket)
            self._evict_old()

            # Advance rollup cursors + reset per-interval state
            self._last_rollup_ts = now_mono
            self._last_rollup_received = self._received_count
            self._last_rollup_rejections = self._gate_rejections
            self._last_rollup_frame_rejections = self._frame_rejections
            self._last_rollup_queue_drops = self._queue_drops
            self._last_rollup_throttle_skips = self._throttle_skips
            self._last_rollup_tracking_drops = self._tracking_drops
            self._queue_depth_samples.clear()

            return bucket

    # ---- Read methods (vis server + API) ----

    def aggregate(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Compute rolling stats over Tier 1 for real-time text display.

        Returns percentile stats (p50/p95/max) for stage timings, plus
        throughput rates and mask counts.

        First `warmup_skip` frames (default 5) are excluded from percentile
        calculations to avoid CUDA warmup spikes skewing p95/max. They are
        still counted for Hz and throughput metrics.
        """
        with self._lock:
            entries = list(self._buffer)
            total_appended = self._total_appended
            if last_n is not None:
                entries = entries[-last_n:]
            # Grab recent Tier 2 buckets for input_hz estimation
            recent_t2 = [b for b in self._second_buckets if b.frames_in_bucket > 0][-10:]

        # For percentile stats, skip warmup frames (only matters early in session)
        # Warmup frames are the first N globally appended, not per-window
        warmup_count = min(self._warmup_skip, total_appended)
        entries_for_timing = entries
        if total_appended <= self._warmup_skip + len(entries):
            # Some warmup frames might still be in the buffer — skip them
            skip_in_buffer = max(0, self._warmup_skip - (total_appended - len(entries)))
            entries_for_timing = entries[skip_in_buffer:]

        if not entries:
            empty_timing = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            return {
                "frame_count": 0,
                "window_duration_s": 0.0,
                "input_hz": 0.0,
                "processing_hz": 0.0,
                "effective_ratio": 0.0,
                "gate_acceptance_rate": 0.0,
                "t_segmentation": empty_timing,
                "t_heuristics": empty_timing,
                "t_scoring": empty_timing,
                "t_clip": empty_timing,
                "t_association": empty_timing,
                "t_total": empty_timing,
                "mean_queue_depth": 0.0,
                "mean_masks_in": 0.0,
                "mean_candidates": 0.0,
                "mask_survival_rate": 0.0,
            }

        n = len(entries)
        duration = max(0.001, entries[-1].timestamp - entries[0].timestamp) if n > 1 else 1.0
        processing_hz = n / duration

        # Use most recent Tier 2 buckets for input_hz (accurate per-second rate).
        # Tier 1 doesn't track received timestamps, so computing from total count is inaccurate.
        input_hz = round(_mean([b.input_hz for b in recent_t2]), 1) if recent_t2 else 0.0

        total_masks = sum(e.n_masks_in for e in entries)
        total_cands = sum(e.n_candidates for e in entries)

        return {
            "frame_count": n,
            "window_duration_s": round(duration, 1),
            "input_hz": round(input_hz, 1),
            "processing_hz": round(processing_hz, 2),
            "effective_ratio": round(processing_hz / max(0.001, input_hz), 3),
            "gate_acceptance_rate": 1.0,  # Tier 1 only has accepted frames
            "warmup_skipped": len(entries) - len(entries_for_timing),
            "t_segmentation": _timing_stats([e.t_segmentation for e in entries_for_timing]),
            "t_heuristics": _timing_stats([e.t_heuristics for e in entries_for_timing]),
            "t_scoring": _timing_stats([e.t_scoring for e in entries_for_timing]),
            "t_clip": _timing_stats([e.t_clip for e in entries_for_timing]),
            "t_association": _timing_stats([e.t_association for e in entries_for_timing]),
            "t_total": _timing_stats([e.t_total for e in entries_for_timing]),
            "mean_queue_depth": round(_mean([e.queue_depth for e in entries]), 1),
            "mean_masks_in": round(_mean([e.n_masks_in for e in entries]), 1),
            "mean_candidates": round(_mean([e.n_candidates for e in entries]), 1),
            "mask_survival_rate": round(total_cands / max(1, total_masks), 3),
        }

    def hourly_history(self) -> List[Dict[str, Any]]:
        """Return Tier 2 buckets as list of dicts for chart rendering."""
        with self._lock:
            return [dataclasses.asdict(b) for b in self._second_buckets]

    def clear(self) -> None:
        """Reset all state — called on /reset."""
        with self._lock:
            self._buffer.clear()
            self._second_buckets.clear()
            self._received_count = 0
            self._gate_rejections = 0
            self._frame_rejections = 0
            self._queue_drops = 0
            self._throttle_skips = 0
            self._tracking_drops = 0
            self._queue_depth_samples.clear()
            self._last_rollup_ts = time.monotonic()
            self._last_rollup_received = 0
            self._last_rollup_rejections = 0
            self._last_rollup_frame_rejections = 0
            self._last_rollup_queue_drops = 0
            self._last_rollup_throttle_skips = 0
            self._last_rollup_tracking_drops = 0
            self._total_appended = 0
            self._wm_snapshot = {"total": 0, "confirmed": 0, "proto": 0}

    def _evict_old(self) -> None:
        """Remove Tier 2 buckets older than retention window. Must hold lock."""
        cutoff = time.time() - self._retention_s
        while self._second_buckets and self._second_buckets[0].wall_ts < cutoff:
            self._second_buckets.popleft()
