"""Frame-quality gate: pure numpy, no GPU or model downloads."""

import logging

import numpy as np
import pytest

from rtsm.core.frame_gate import (
    REASON_DARK,
    REASON_DEPTH,
    REASON_FLAT,
    FrameQualityGate,
)

H, W = 96, 128
_rng = np.random.default_rng(0)


def scene_rgb():
    return _rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def scene_depth():
    d = _rng.uniform(0.5, 4.0, size=(H, W)).astype(np.float32)
    d[:8] = np.nan  # a band of invalid depth is normal
    return d


def gate(**overrides):
    return FrameQualityGate({"gates": overrides})


def test_normal_frame_is_accepted():
    d = gate().check(scene_rgb(), scene_depth())
    assert d.accept and d.reason == ""
    assert d.brightness > 100 and d.contrast > 30
    assert 0.9 < d.depth_valid < 1.0


def test_black_frame_is_dark():
    d = gate().check(np.zeros((H, W, 3), np.uint8), scene_depth())
    assert not d.accept and d.reason == REASON_DARK


def test_uniform_frame_is_flat():
    d = gate().check(np.full((H, W, 3), 120, np.uint8), scene_depth())
    assert not d.accept and d.reason == REASON_FLAT
    assert d.brightness == pytest.approx(120.0) and d.contrast == 0.0


@pytest.mark.parametrize("bad", [np.full((H, W), np.nan, np.float32), np.zeros((H, W), np.float32)])
def test_depth_failure_is_rejected(bad):
    d = gate().check(scene_rgb(), bad)
    assert not d.accept and d.reason == REASON_DEPTH and d.depth_valid == 0.0


def test_missing_depth_does_not_reject():
    d = gate().check(scene_rgb(), None)
    assert d.accept and d.depth_valid == 1.0


def test_disabled_gate_still_measures_but_accepts():
    d = gate(enable=False).check(np.zeros((H, W, 3), np.uint8), None)
    assert d.accept and d.brightness == 0.0


def test_thresholds_come_from_config():
    assert gate(min_brightness=200).check(np.full((H, W, 3), 150, np.uint8), None).reason == REASON_DARK
    quarter = np.where(np.arange(W) < W // 4, 1.0, np.nan).astype(np.float32)[None].repeat(H, 0)
    d = gate(min_depth_valid=0.5).check(scene_rgb(), quarter)
    assert d.reason == REASON_DEPTH and d.depth_valid == pytest.approx(0.25)


def test_stride_odd_shapes_and_array_like_input():
    rgb, depth = scene_rgb()[:95, :127], scene_depth()[:95, :127]
    assert gate(sample_stride=7).check(rgb, depth).accept
    assert gate(sample_stride=1).check(list(rgb), depth).accept
    assert gate().check(rgb[..., 0], depth).accept  # greyscale input


def test_counters_and_rate_limited_log(caplog):
    g = FrameQualityGate({}, log_interval_s=10.0)
    black = np.zeros((H, W, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger="rtsm.frame_gate"):
        assert g.maybe_log(g.check(black, None), now_mono=0.0)
        assert not g.maybe_log(g.check(black, None), now_mono=5.0)
        assert g.check(scene_rgb(), None).accept
        assert g.maybe_log(g.check(black, None), now_mono=10.0)
    assert g.checked == 4
    assert g.rejections == {REASON_DARK: 3, REASON_FLAT: 0, REASON_DEPTH: 0}
    assert "skipped 2 frame(s)" in caplog.records[-1].getMessage()


def test_latency_analytics_rolls_up_frame_rejections():
    from rtsm.analytics.latency_analytics import PipelineLatencyBuffer

    la = PipelineLatencyBuffer()
    la.roll_up_second()  # prime the rollup cursor
    la.record_frame_rejection()
    la.record_frame_rejection()
    la.record_gate_rejection()
    bucket = la.roll_up_second()
    assert bucket.frame_rejections == 2 and bucket.gate_rejections == 1
    assert la.roll_up_second().frame_rejections == 0  # cursor advanced
    la.clear()
    la.roll_up_second()
    la.record_frame_rejection()
    assert la.roll_up_second().frame_rejections == 1
