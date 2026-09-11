"""
Pipeline latency & throughput analytics buffer.

Tracks per-frame stage timings, input/processing rates, queue pressure,
and 4 drop points (tracking, throttle, queue full, gate rejection).

Tier 1: Per-frame FrameTimingStats in a ring buffer (deque, maxlen=300).
Tier 2: Per-second LatencySecondBucket with wall-clock retention (default 1 hour).

The Tier-1 -> Tier-2 rollup (``roll_up_second``) has ONE owner per process:
``rtsm.analytics.ticker.AnalyticsTicker`` (built by ``build_analytics``), a
1 Hz daemon thread that runs whenever ``analytics.enable`` is true — with or
without a visualization client (P1 task 5; before, the rollup lived in the
visualization push loop and a headless run never produced a bucket). The
visualization server only consumes the buckets the ticker publishes. Frame
selection for a bucket is COUNT-based (every appended frame lands in exactly
one bucket, whatever clock stamped it) and rates are computed over the
interval actually elapsed.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# A rollup interval longer than this is flagged ``stale_interval`` (a late
# tick); the owner passes max(STALE_INTERVAL_S, 2 x its interval). Counts stay
# exact either way — the flag only marks the bucket as covering a long gap.
STALE_INTERVAL_S = 2.0


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
    frames_received: int = 0    # frames that arrived this interval (input_hz = frames_received / elapsed_s)
    processing_hz: float = 0.0
    effective_ratio: float = 0.0
    # Drop counters (per second)
    queue_drops: int = 0        # refused by the ingest queue / oldest keyframe dropped (frames lost to capacity)
    superseded: int = 0         # waiting non-keyframe replaced by a newer one (ingest.policy=latest; not a loss)
    age_drops: int = 0          # non-keyframe discarded at dequeue for age (ingest.policy=latest; a stall)
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
    # Rollup interval bookkeeping (P1 task 5)
    elapsed_s: float = 0.0          # monotonic seconds this bucket covers (~1.0 at the ticker's cadence)
    stale_interval: bool = False    # elapsed_s exceeded the stale threshold (a late tick); counts are still exact


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
        self._age_drops: int = 0
        self._superseded: int = 0
        self._throttle_skips: int = 0
        self._tracking_drops: int = 0

        # Queue depth samples for one rollup interval (bounded in case the ticker is stopped)
        self._queue_depth_samples: deque[int] = deque(maxlen=120)  # ~4s at 30 Hz

        # Rollup cursors
        self._last_rollup_ts: float = time.monotonic()
        self._last_rollup_received: int = 0
        self._last_rollup_rejections: int = 0
        self._last_rollup_frame_rejections: int = 0
        self._last_rollup_queue_drops: int = 0
        self._last_rollup_age_drops: int = 0
        self._last_rollup_superseded: int = 0
        self._last_rollup_throttle_skips: int = 0
        self._last_rollup_tracking_drops: int = 0
        self._last_rollup_total: int = 0        # count cursor: frames already rolled into a bucket
        # Rollup-mechanism diagnostics (NOT reset by clear(): they describe the
        # rollup owner's health, not the session's data; see rollup_stats()).
        self._rollups: int = 0
        self._stale_rollups: int = 0
        self._ring_truncated: int = 0

        # WM snapshot (updated by the analytics ticker before each rollup)
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
        """A frame lost to ingest capacity: a receiver-side refusal
        (queue_full / kf_lane_full, before the RGB decode) or, under
        ingest.policy=latest, the oldest waiting keyframe dropped on overflow.
        Comparable across policies as "frames the ingest stage did not hand to
        the pipeline". Superseded non-keyframes are NOT counted here (see
        record_superseded): replacing a waiting frame with a newer one is the
        designed steady state under congestion, not a loss."""
        with self._lock:
            self._queue_drops += 1

    def record_superseded(self) -> None:
        """A waiting non-keyframe was replaced by a newer one in the latest
        slot (ingest.policy=latest). Informational: how often input outpaced
        the pipeline; the dashboard shows it as a neutral count."""
        with self._lock:
            self._superseded += 1

    def record_age_drop(self) -> None:
        """A non-keyframe discarded at dequeue for exceeding max_frame_age_s
        (ingest.policy=latest only). Counted apart from queue drops: it means
        the pipeline stalled, not that input outpaced it."""
        with self._lock:
            self._age_drops += 1

    def record_throttle_skip(self) -> None:
        """Called when non-KF throttle skips a frame (by design)."""
        with self._lock:
            self._throttle_skips += 1

    def record_tracking_drop(self) -> None:
        """Called when tracking_state != normal drops a frame."""
        with self._lock:
            self._tracking_drops += 1

    def snapshot_wm(self, total: int, confirmed: int, proto: int) -> None:
        """Inject WM state snapshot for the next rollup. Called by the analytics ticker."""
        with self._lock:
            self._wm_snapshot = {"total": total, "confirmed": confirmed, "proto": proto}

    def sample_queue_depth(self, depth: int) -> None:
        """Record queue depth at enqueue time for mean/max stats."""
        with self._lock:
            self._queue_depth_samples.append(depth)

    # ---- Tier 2 rollup (analytics ticker thread, 1 Hz) ----

    def roll_up_second(self, now_mono: Optional[float] = None,
                       stale_after_s: float = STALE_INTERVAL_S) -> LatencySecondBucket:
        """Aggregate everything appended / recorded since the previous rollup
        into one Tier-2 bucket. Called by the analytics ticker (the one owner
        per process); never by the pipeline, never by the visualization server.

        ``now_mono`` lets the owner stamp both buffers with one clock reading;
        ``stale_after_s`` is the threshold above which the interval is flagged
        ``stale_interval`` (the ticker passes max(2 s, 2 x its interval)).

        Counts are exact over ANY interval: ``frames_in_bucket`` is the number
        of frames appended since the previous rollup (a count delta, so each
        frame is in exactly one bucket regardless of the clock that stamped
        it — the pipeline stamps with perf_counter, the cursor is monotonic),
        and the drop fields are cumulative-counter deltas. Rates divide by the
        interval actually elapsed. A long interval (a late tick) therefore
        yields one wide, flagged bucket with nothing skipped: the old guard
        that discarded the interval's data is gone, only its flag remains.
        When more frames were appended than the ring holds, the timing means
        cover the ring's tail and ``ring_truncated`` (rollup_stats) grows.
        """
        with self._lock:
            now = time.monotonic() if now_mono is None else float(now_mono)
            elapsed = max(0.001, now - self._last_rollup_ts)
            stale = elapsed > float(stale_after_s)

            # Frames appended since the last rollup (count-based selection)
            n = max(0, self._total_appended - self._last_rollup_total)
            truncated = max(0, n - len(self._buffer))
            recent = list(self._buffer)[-n:] if n > 0 else []

            # Input rate since last rollup
            received_since = self._received_count - self._last_rollup_received
            input_hz = received_since / elapsed

            # Drop deltas since last rollup
            gate_rej = self._gate_rejections - self._last_rollup_rejections
            frame_rej = self._frame_rejections - self._last_rollup_frame_rejections
            q_drops = self._queue_drops - self._last_rollup_queue_drops
            age_drops = self._age_drops - self._last_rollup_age_drops
            superseded = self._superseded - self._last_rollup_superseded
            throttle = self._throttle_skips - self._last_rollup_throttle_skips
            tracking = self._tracking_drops - self._last_rollup_tracking_drops

            # Queue depth stats (samples taken by the receivers at enqueue)
            samples = list(self._queue_depth_samples)
            q_mean = round(_mean(samples), 1) if samples else 0.0
            q_max = max(samples) if samples else 0

            processing_hz = n / elapsed

            bucket = LatencySecondBucket(
                wall_ts=time.time(),
                input_hz=round(input_hz, 1),
                frames_received=received_since,
                processing_hz=round(processing_hz, 2),
                # Undefined without input this interval (the lossless drain after a
                # replay ends, a live pause): 0.0, not processing / 0.001.
                effective_ratio=round(processing_hz / input_hz, 3) if received_since > 0 else 0.0,
                queue_drops=q_drops,
                superseded=superseded,
                age_drops=age_drops,
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
                elapsed_s=round(elapsed, 3),
                stale_interval=stale,
            )

            self._second_buckets.append(bucket)
            self._evict_old()
            self._advance_cursors_locked(now)
            self._rollups += 1
            if stale:
                self._stale_rollups += 1
            self._ring_truncated += truncated
            return bucket

    def _advance_cursors_locked(self, now: float) -> None:
        """Move every rollup cursor to the present. Must hold lock."""
        self._last_rollup_ts = now
        self._last_rollup_total = self._total_appended
        self._last_rollup_received = self._received_count
        self._last_rollup_rejections = self._gate_rejections
        self._last_rollup_frame_rejections = self._frame_rejections
        self._last_rollup_queue_drops = self._queue_drops
        self._last_rollup_age_drops = self._age_drops
        self._last_rollup_superseded = self._superseded
        self._last_rollup_throttle_skips = self._throttle_skips
        self._last_rollup_tracking_drops = self._tracking_drops
        self._queue_depth_samples.clear()

    def reset_rollup_clock(self, now_mono: Optional[float] = None) -> None:
        """Start the Tier-2 clock now (called by the ticker's start()): ONLY the
        time cursor moves, so the first bucket the owner rolls covers one
        interval rather than the time since construction (model loads happen in
        between) and is not flagged stale. Every COUNT cursor stays where it is:
        frames appended, frames received and drops recorded before the owner
        started (the receiver runs before the ticker) land in the first bucket,
        so for every counter sum(bucket field) == the lifetime counter over the
        whole process. The first gate run proved the need: with the drop
        cursors re-anchored, six throttle skips recorded between the replayer's
        start and the ticker's start were in no bucket (148 vs 154)."""
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
            # The last 10 Tier-2 buckets that carried frames, for input_hz
            # estimation (bounded reverse scan: the deque holds up to 3600).
            recent_t2: List[LatencySecondBucket] = []
            for b in reversed(self._second_buckets):
                if b.frames_in_bucket > 0:
                    recent_t2.append(b)
                    if len(recent_t2) == 10:
                        break
            recent_t2.reverse()
            # Lifetime (process-monotonic) counters, independent of the Tier-2
            # rollup: every drop point is visible even before a bucket exists.
            counters = {
                "received": self._received_count,
                "processed": total_appended,
                "gate_rejections": self._gate_rejections,
                "frame_rejections": self._frame_rejections,
                "queue_drops": self._queue_drops,
                "superseded": self._superseded,
                "age_drops": self._age_drops,
                "throttle_skips": self._throttle_skips,
                "tracking_drops": self._tracking_drops,
            }

        # Lifetime acceptance of DEQUEUED frames by the two pipeline gates
        # (ingest gate + frame-quality gate), i.e. admitted / (admitted +
        # gate-rejected + frame-gate-rejected). Lifetime and cumulative, unlike
        # the windowed rates around it; frames dropped for a failed pose
        # conversion never reach the gates and are counted on the pipeline
        # (/stats.pose_conversion_failures), not here. Was hardcoded to 1.0
        # ("Tier 1 only has accepted frames"), which misled the datasheet: it
        # read as "nothing was ever gated".
        dequeued = total_appended + counters["gate_rejections"] + counters["frame_rejections"]
        gate_acceptance_rate = round(total_appended / dequeued, 4) if dequeued > 0 else 0.0

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
                "gate_acceptance_rate": gate_acceptance_rate,
                "counters": counters,
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
            "gate_acceptance_rate": gate_acceptance_rate,
            "counters": counters,
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
        """Return Tier 2 buckets as list of dicts for chart rendering. The
        buckets are immutable once appended, so only the snapshot is taken
        under the lock; serialising up to 3600 of them (~13 ms) happens
        outside it, off the receiver's and the pipeline's hot path."""
        with self._lock:
            buckets = list(self._second_buckets)
        return [dataclasses.asdict(b) for b in buckets]

    def clear(self) -> None:
        """Reset the session's data — called on /reset. The rollup-mechanism
        counters (rollup_stats) survive: they describe the owner, not the data."""
        with self._lock:
            self._buffer.clear()
            self._second_buckets.clear()
            self._last_rollup_total = 0
            self._received_count = 0
            self._gate_rejections = 0
            self._frame_rejections = 0
            self._queue_drops = 0
            self._age_drops = 0
            self._superseded = 0
            self._throttle_skips = 0
            self._tracking_drops = 0
            self._queue_depth_samples.clear()
            self._last_rollup_received = 0
            self._last_rollup_rejections = 0
            self._last_rollup_frame_rejections = 0
            self._last_rollup_queue_drops = 0
            self._last_rollup_age_drops = 0
            self._last_rollup_superseded = 0
            self._last_rollup_throttle_skips = 0
            self._last_rollup_tracking_drops = 0
            self._total_appended = 0
            self._wm_snapshot = {"total": 0, "confirmed": 0, "proto": 0}
            # _last_rollup_ts is deliberately NOT touched: the count cursors
            # were reset with the counters, so the next bucket's deltas are
            # already post-reset, and keeping the time cursor tick-to-tick
            # keeps elapsed_s the real interval (a reset 0.1 s before a tick
            # would otherwise produce a 10x rate spike) on the ticker's clock.

    def _evict_old(self) -> None:
        """Remove Tier 2 buckets older than retention window. Must hold lock."""
        cutoff = time.time() - self._retention_s
        while self._second_buckets and self._second_buckets[0].wall_ts < cutoff:
            self._second_buckets.popleft()
