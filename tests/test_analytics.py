"""
Tests for rtsm.analytics — SegAnalyticsBuffer and PipelineLatencyBuffer.

Covers:
- Tier 1 append/aggregate
- Tier 2 rollup + time-based eviction
- Stale-interval flag (a long gap is flagged, never skipped; rollup owner = the analytics ticker)
- Drop counters and queue depth sampling
- Thread safety (concurrent append + rollup)
- Empty buffer edge cases
- Single-backend mapping
- Clear/reset
"""
import time
import threading

import pytest

from rtsm.analytics.seg_analytics import SegAnalyticsBuffer, SegFrameStats, SegSecondBucket
from rtsm.analytics.latency_analytics import PipelineLatencyBuffer, FrameTimingStats, LatencySecondBucket


# ============================================================================
# SegAnalyticsBuffer
# ============================================================================

class TestSegFrameStats:
    def test_defaults_all_zero(self):
        s = SegFrameStats(timestamp=1.0)
        assert s.n_dual == 0
        assert s.n_fastsam_only == 0
        assert s.n_yoloe_only == 0
        assert s.n_total == 0
        assert s.backend == "dual"

    def test_single_backend_fastsam(self):
        s = SegFrameStats(timestamp=1.0, backend="fastsam", n_fastsam_only=10, n_total=10)
        assert s.n_dual == 0
        assert s.n_yoloe_only == 0
        assert s.n_fastsam_only == 10


class TestSegAnalyticsBuffer:
    def test_append_and_aggregate_empty(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        agg = buf.aggregate()
        assert agg["frame_count"] == 0
        assert agg["dual_rate"] == 0.0
        assert agg["fastsam_only_rate"] == 0.0

    def test_append_and_aggregate_dual(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        for i in range(10):
            buf.append(SegFrameStats(
                timestamp=time.monotonic(),
                backend="dual",
                n_dual=3, n_fastsam_only=2, n_yoloe_only=1, n_total=6,
                n_fastsam_raw=8, n_yoloe_raw=4,
            ))
        agg = buf.aggregate()
        assert agg["frame_count"] == 10
        assert agg["backend"] == "dual"
        # 30 dual out of 60 total = 0.5
        assert agg["dual_rate"] == 0.5
        assert agg["fastsam_only_rate"] == pytest.approx(2 / 6, abs=0.01)
        assert agg["yoloe_only_rate"] == pytest.approx(1 / 6, abs=0.01)
        assert agg["mean_total"] == 6.0
        assert agg["mean_fastsam_raw"] == 8.0

    def test_append_single_backend_yoloe(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        for _ in range(5):
            buf.append(SegFrameStats(
                timestamp=time.monotonic(),
                backend="yoloe",
                n_yoloe_only=7, n_total=7,
            ))
        agg = buf.aggregate()
        assert agg["dual_rate"] == 0.0
        assert agg["fastsam_only_rate"] == 0.0
        assert agg["yoloe_only_rate"] == 1.0

    def test_append_none_is_noop(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        buf.append(None)
        assert buf.aggregate()["frame_count"] == 0

    def test_deque_maxlen_eviction(self):
        buf = SegAnalyticsBuffer(max_frames=5)
        for i in range(10):
            buf.append(SegFrameStats(timestamp=time.monotonic(), n_total=1, n_dual=1))
        agg = buf.aggregate()
        assert agg["frame_count"] == 5  # oldest 5 evicted

    def test_rollup_produces_bucket(self):
        buf = SegAnalyticsBuffer(max_frames=100, retention_s=3600)
        # Widen the interval so the rates are finite (selection is count-based; the
        # cursor no longer decides visibility)
        time.sleep(0.02)
        buf._last_rollup_ts = time.monotonic() - 0.5
        buf.append(SegFrameStats(
            timestamp=time.monotonic(),
            backend="dual",
            n_dual=3, n_fastsam_only=2, n_yoloe_only=1, n_total=6,
        ))
        time.sleep(0.01)
        bucket = buf.roll_up_second()
        assert isinstance(bucket, SegSecondBucket)
        assert bucket.frames_in_bucket == 1
        assert bucket.dual_rate == 0.5
        assert bucket.wall_ts > 0

    def test_rollup_empty_buffer(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        time.sleep(0.01)
        bucket = buf.roll_up_second()
        assert bucket.frames_in_bucket == 0
        assert bucket.dual_rate == 0.0

    def test_stale_interval_is_flagged_not_skipped(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        buf.append(SegFrameStats(timestamp=time.monotonic(), n_dual=5, n_total=5))
        # A long gap since the previous rollup (a late tick)
        buf._last_rollup_ts = time.monotonic() - 10.0
        bucket = buf.roll_up_second()
        # P1 task 5: the interval is flagged, the frame is still counted
        assert bucket.frames_in_bucket == 1 and bucket.dual_rate == 1.0
        assert bucket.stale_interval is True and bucket.elapsed_s >= 10.0
        assert buf.rollup_stats()["stale_rollups"] == 1
        time.sleep(0.01)
        assert buf.roll_up_second().stale_interval is False

    def test_hourly_history(self):
        buf = SegAnalyticsBuffer(max_frames=100, retention_s=3600)
        buf.append(SegFrameStats(timestamp=time.monotonic(), n_dual=1, n_total=1))
        time.sleep(0.01)
        buf.roll_up_second()
        history = buf.hourly_history()
        assert len(history) == 1
        assert "wall_ts" in history[0]
        assert "dual_rate" in history[0]

    def test_time_based_eviction(self):
        buf = SegAnalyticsBuffer(max_frames=100, retention_s=0.05)  # 50ms retention
        buf.append(SegFrameStats(timestamp=time.monotonic(), n_dual=1, n_total=1))
        time.sleep(0.01)
        buf.roll_up_second()
        assert len(buf.hourly_history()) == 1
        # Wait for retention to expire
        time.sleep(0.1)
        buf.append(SegFrameStats(timestamp=time.monotonic(), n_dual=1, n_total=1))
        time.sleep(0.01)
        buf.roll_up_second()
        # Old bucket evicted, only new one remains
        assert len(buf.hourly_history()) == 1

    def test_clear(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        buf.append(SegFrameStats(timestamp=time.monotonic(), n_dual=1, n_total=1))
        time.sleep(0.01)
        buf.roll_up_second()
        buf.clear()
        assert buf.aggregate()["frame_count"] == 0
        assert len(buf.hourly_history()) == 0

    def test_selected_rate_by_source(self):
        buf = SegAnalyticsBuffer(max_frames=100)
        buf.append(SegFrameStats(
            timestamp=time.monotonic(),
            n_dual=3, n_fastsam_only=2, n_yoloe_only=1, n_total=6,
            selected_dual=2, selected_fastsam_only=1, selected_yoloe_only=0,
        ))
        agg = buf.aggregate()
        rates = agg["selected_rate_by_source"]
        assert rates["dual"] == pytest.approx(2 / 3, abs=0.01)
        assert rates["fastsam_only"] == pytest.approx(1 / 3, abs=0.01)
        assert rates["yoloe_only"] == 0.0


# ============================================================================
# PipelineLatencyBuffer
# ============================================================================

class TestFrameTimingStats:
    def test_defaults_all_zero(self):
        f = FrameTimingStats(timestamp=1.0)
        assert f.t_segmentation == 0.0
        assert f.t_total == 0.0
        assert f.queue_depth == 0
        assert f.n_masks_in == 0


class TestPipelineLatencyBuffer:
    def test_append_and_aggregate_empty(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        agg = buf.aggregate()
        assert agg["frame_count"] == 0
        assert agg["processing_hz"] == 0.0
        assert agg["t_total"]["mean"] == 0.0

    def test_append_and_aggregate(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        for _ in range(10):
            buf.append(FrameTimingStats(
                timestamp=time.monotonic(),
                t_segmentation=0.1, t_heuristics=0.02, t_scoring=0.005,
                t_clip=0.05, t_association=0.03, t_total=0.205,
                queue_depth=5, n_masks_in=12, n_candidates=5,
            ))
        agg = buf.aggregate()
        assert agg["frame_count"] == 10
        assert agg["t_segmentation"]["mean"] == pytest.approx(0.1, abs=0.001)
        assert agg["t_total"]["mean"] == pytest.approx(0.205, abs=0.001)
        assert agg["mean_queue_depth"] == 5.0
        assert agg["mean_masks_in"] == 12.0
        assert agg["mean_candidates"] == 5.0
        assert agg["mask_survival_rate"] == pytest.approx(5 / 12, abs=0.01)

    def test_percentiles(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        # Insert values with known distribution
        for i in range(100):
            buf.append(FrameTimingStats(
                timestamp=time.monotonic(),
                t_total=float(i) / 1000.0,  # 0ms to 99ms
            ))
        agg = buf.aggregate()
        assert agg["t_total"]["p50"] == pytest.approx(0.050, abs=0.002)
        assert agg["t_total"]["p95"] == pytest.approx(0.095, abs=0.002)
        assert agg["t_total"]["max"] == pytest.approx(0.099, abs=0.001)

    def test_append_none_is_noop(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.append(None)
        assert buf.aggregate()["frame_count"] == 0

    def test_deque_maxlen_eviction(self):
        buf = PipelineLatencyBuffer(max_frames=5)
        for _ in range(10):
            buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        assert buf.aggregate()["frame_count"] == 5

    # ---- Drop counters ----

    def test_record_frame_received(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        for _ in range(30):
            buf.record_frame_received()
        assert buf._received_count == 30

    def test_record_gate_rejection(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.record_gate_rejection()
        buf.record_gate_rejection()
        assert buf._gate_rejections == 2

    def test_record_queue_drop(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.record_queue_drop()
        assert buf._queue_drops == 1

    def test_record_throttle_skip(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.record_throttle_skip()
        buf.record_throttle_skip()
        buf.record_throttle_skip()
        assert buf._throttle_skips == 3

    def test_record_tracking_drop(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.record_tracking_drop()
        assert buf._tracking_drops == 1

    def test_sample_queue_depth(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        for d in [3, 5, 10, 2]:
            buf.sample_queue_depth(d)
        assert list(buf._queue_depth_samples) == [3, 5, 10, 2]

    def test_queue_depth_samples_capped(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        for i in range(200):
            buf.sample_queue_depth(i)
        # deque(maxlen=120) caps at 120
        assert len(buf._queue_depth_samples) == 120

    # ---- Rollup ----

    def test_rollup_produces_bucket(self):
        buf = PipelineLatencyBuffer(max_frames=100, retention_s=3600)
        # Widen the interval so the rates are finite (selection is count-based; the
        # cursor no longer decides visibility)
        time.sleep(0.02)
        buf._last_rollup_ts = time.monotonic() - 0.5
        for _ in range(5):
            buf.record_frame_received()
        buf.append(FrameTimingStats(
            timestamp=time.monotonic(),
            t_segmentation=0.1, t_total=0.2,
        ))
        buf.record_gate_rejection()
        buf.record_queue_drop()
        buf.sample_queue_depth(10)
        buf.sample_queue_depth(20)
        time.sleep(0.01)

        bucket = buf.roll_up_second()
        assert isinstance(bucket, LatencySecondBucket)
        assert bucket.frames_in_bucket == 1
        assert bucket.input_hz > 0
        assert bucket.processing_hz > 0
        assert bucket.gate_rejections == 1
        assert bucket.queue_drops == 1
        assert bucket.queue_depth_mean == 15.0
        assert bucket.queue_depth_max == 20
        assert bucket.t_seg_ms == pytest.approx(100.0, abs=1.0)
        assert bucket.t_total_ms == pytest.approx(200.0, abs=1.0)

    def test_rollup_clears_queue_depth_samples(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.sample_queue_depth(5)
        buf.sample_queue_depth(10)
        time.sleep(0.01)
        buf.roll_up_second()
        assert len(buf._queue_depth_samples) == 0

    def test_rollup_advances_cursors(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        for _ in range(3):
            buf.record_frame_received()
        buf.record_gate_rejection()
        buf.record_queue_drop()
        time.sleep(0.01)
        buf.roll_up_second()

        # Second rollup should see zero deltas
        time.sleep(0.01)
        bucket2 = buf.roll_up_second()
        assert bucket2.gate_rejections == 0
        assert bucket2.queue_drops == 0
        assert bucket2.frames_in_bucket == 0

    def test_stale_interval_is_flagged_not_skipped(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        buf.record_frame_received()
        buf.record_queue_drop()
        # A long gap since the previous rollup (a late tick)
        buf._last_rollup_ts = time.monotonic() - 10.0
        bucket = buf.roll_up_second()
        # P1 task 5: counts are exact over any interval, rates use the real elapsed time
        assert bucket.frames_in_bucket == 1
        assert bucket.queue_drops == 1
        assert bucket.input_hz == pytest.approx(0.1, abs=0.01)
        assert bucket.stale_interval is True and bucket.elapsed_s >= 10.0
        assert buf.rollup_stats()["stale_rollups"] == 1

    def test_hourly_history(self):
        buf = PipelineLatencyBuffer(max_frames=100, retention_s=3600)
        buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        time.sleep(0.01)
        buf.roll_up_second()
        history = buf.hourly_history()
        assert len(history) == 1
        assert "wall_ts" in history[0]
        assert "input_hz" in history[0]
        assert "queue_drops" in history[0]
        assert "t_total_ms" in history[0]

    def test_time_based_eviction(self):
        buf = PipelineLatencyBuffer(max_frames=100, retention_s=0.05)
        buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        time.sleep(0.01)
        buf.roll_up_second()
        assert len(buf.hourly_history()) == 1
        time.sleep(0.1)
        buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        time.sleep(0.01)
        buf.roll_up_second()
        assert len(buf.hourly_history()) == 1  # old bucket evicted

    def test_clear_resets_everything(self):
        buf = PipelineLatencyBuffer(max_frames=100)
        buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.1))
        buf.record_frame_received()
        buf.record_gate_rejection()
        buf.record_queue_drop()
        buf.record_throttle_skip()
        buf.record_tracking_drop()
        buf.sample_queue_depth(10)
        time.sleep(0.01)
        buf.roll_up_second()
        buf.clear()
        assert buf.aggregate()["frame_count"] == 0
        assert len(buf.hourly_history()) == 0
        assert buf._received_count == 0
        assert buf._gate_rejections == 0
        assert buf._queue_drops == 0
        assert buf._throttle_skips == 0
        assert buf._tracking_drops == 0
        assert len(buf._queue_depth_samples) == 0

    # ---- Thread safety ----

    def test_concurrent_append_and_rollup(self):
        buf = PipelineLatencyBuffer(max_frames=300, retention_s=3600)
        errors = []

        def append_loop():
            try:
                for _ in range(200):
                    buf.append(FrameTimingStats(timestamp=time.monotonic(), t_total=0.01))
                    buf.record_frame_received()
                    buf.sample_queue_depth(5)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def rollup_loop():
            try:
                for _ in range(20):
                    buf.roll_up_second()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append_loop)
        t2 = threading.Thread(target=rollup_loop)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"
        assert buf.aggregate()["frame_count"] > 0
        assert len(buf.hourly_history()) > 0


class TestSegAnalyticsBufferThreadSafety:
    def test_concurrent_append_and_rollup(self):
        buf = SegAnalyticsBuffer(max_frames=300, retention_s=3600)
        errors = []

        def append_loop():
            try:
                for _ in range(200):
                    buf.append(SegFrameStats(
                        timestamp=time.monotonic(),
                        n_dual=2, n_fastsam_only=1, n_total=3,
                    ))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def rollup_loop():
            try:
                for _ in range(20):
                    buf.roll_up_second()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append_loop)
        t2 = threading.Thread(target=rollup_loop)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"
