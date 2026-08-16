"""forward_clearance_from_depth — the wall-guard signal for blind agent
motion (2026-08-16). Lives in the io layer (receive-time, frame-packet
level — before the ingest queue/gate and any GPU work). Pure-numpy
contract tests: robust nearest-surface estimate from the central band,
fail-closed on missing/invalid depth."""

import numpy as np

from rtsm.io.websocket import forward_clearance_from_depth


def _depth(fill=3.0, h=192, w=256):
    return np.full((h, w), fill, dtype=np.float32)


def test_open_scene_reports_far_clearance():
    c, frac = forward_clearance_from_depth(_depth(3.0))
    assert abs(c - 3.0) < 1e-5
    assert frac > 0.9


def test_near_wall_reports_near():
    d = _depth(4.0)
    d[:, 85:171] = 0.4                     # wall filling the central columns
    c, _ = forward_clearance_from_depth(d)
    assert c < 0.5


def test_floor_below_band_is_ignored():
    # Close floor in the lower half must not read as an obstacle: the
    # band is rows 30-55%, chosen above the floor line.
    d = _depth(3.0)
    d[int(d.shape[0] * 0.6):, :] = 0.5     # near floor, below the band
    c, _ = forward_clearance_from_depth(d)
    assert c > 2.9


def test_none_and_empty_fail_closed():
    assert forward_clearance_from_depth(None) == (0.0, 0.0)
    assert forward_clearance_from_depth(np.zeros((0, 0), dtype=np.float32)) == (0.0, 0.0)


def test_mostly_invalid_band_fails_closed():
    # LiDAR too close / no return -> NaN. A blind sensor must read as
    # blocked, not as open space.
    d = _depth(np.nan)
    c, frac = forward_clearance_from_depth(d)
    assert c == 0.0
    assert frac < 0.2


def test_tenth_percentile_is_robust_to_speckle():
    # A few far-away speckle pixels must not raise the estimate above the
    # dominant near surface.
    d = _depth(0.6)
    d[::13, ::17] = 5.0                    # sparse far outliers
    c, _ = forward_clearance_from_depth(d)
    assert c < 0.7
