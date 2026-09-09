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


# ─────────────── wiring: io.clearance.enable carve (2026-09) ───────────────
#
# Flag off (packaged default): the receiver gets no sink and WorkingMemory
# publishes no forward_clearance key -- the pre-feature /stats contract.
# Flag on: the sink fires once per DECODED frame (post tracking filter and
# non-KF throttle) with the header's wall timestamp, and WM publishes the
# sample (None until the first depth frame).

import json
import struct
import time

import cv2
import pytest

from rtsm.cfg import load_config
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver
from rtsm.stores.working_memory import WorkingMemory


def _T_wc_col_major():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.0, 2.0, 3.0]
    return T.flatten(order="F").tolist()


def _full_frame(unix_ts: float = 1234.5, depth_mm: int = 2500) -> bytes:
    """Header + 2x2 JPEG + 20x20 uint16 depth (depth_mm everywhere)."""
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", img)
    depth = np.full((20, 20), depth_mm, dtype=np.uint16)
    header = {
        "T_wc": _T_wc_col_major(),
        "pose_format": "matrix4x4_col_major",
        "tracking_state": "normal",
        "unix_timestamp": unix_ts,
        "timestamp_ns": 42,
        "frame_id": 7,
        "rgb_format": "jpeg", "rgb_width": 2, "rgb_height": 2,
        "depth_format": "uint16_mm", "depth_width": 20, "depth_height": 20,
        "depth_scale": 0.001,
        "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0,
    }
    hj = json.dumps(header).encode("utf-8")
    rgb_b = jpeg.tobytes()
    dep_b = depth.tobytes()
    return (struct.pack("<I", len(hj)) + hj
            + struct.pack("<I", len(rgb_b)) + rgb_b
            + struct.pack("<I", len(dep_b)) + dep_b)


def _receiver(clearance_sink) -> WebSocketReceiver:
    return WebSocketReceiver(
        ingest_queue=IngestQueue(maxsize=4),
        require_tracking_normal=True,
        keyframe_every_n=30,
        nonkf_min_interval_s=0.5,
        clearance_sink=clearance_sink,
    )


def test_packaged_default_is_off_and_flag_is_a_known_override():
    cfg = load_config("rtsm.yaml")
    assert cfg["io"]["clearance"]["enable"] is False
    try:
        on = load_config("rtsm.yaml", set_values=["io.clearance.enable=true"])
    except TypeError:  # pre-PR#25 loader (demo2 tip) has no --set support
        pytest.skip("load_config without set_values (pre-PR#25 loader)")
    assert on["io"]["clearance"]["enable"] is True


def test_sink_fires_once_per_decoded_frame_with_header_timestamp():
    calls = []
    rx = _receiver(lambda c_m, frac, ts: calls.append((c_m, frac, ts)))
    pkt = rx._parse_binary_message(_full_frame(unix_ts=1234.5))  # frame 1 -> KF -> decoded
    assert pkt is not None
    assert len(calls) == 1
    c_m, frac, ts = calls[0]
    assert abs(c_m - 2.5) < 1e-3
    assert frac == 1.0
    assert ts == 1234.5


def test_sink_not_called_for_throttled_frame():
    """Unlike the pose sink, clearance needs decoded depth, so a frame the
    non-KF throttle drops (no decode at all) produces no sample."""
    calls = []
    rx = _receiver(lambda *a: calls.append(a))
    rx._frame_count = 1
    rx._last_nonkf_enq_mono = time.monotonic()
    assert rx._parse_binary_message(_full_frame()) is None
    assert calls == []


def test_no_sink_parses_normally():
    rx = _receiver(None)
    pkt = rx._parse_binary_message(_full_frame())
    assert pkt is not None
    assert pkt.depth_m is not None and pkt.depth_m.shape == (20, 20)


def test_sink_error_does_not_break_parse():
    def boom(*_):
        raise RuntimeError("sink failure")
    rx = _receiver(boom)
    assert rx._parse_binary_message(_full_frame()) is not None


def test_wm_publishes_key_only_when_enabled():
    off = WorkingMemory(cfg={})
    assert off.clearance_enabled is False
    assert "forward_clearance" not in off.stats()

    on = WorkingMemory(cfg={"io": {"clearance": {"enable": True}}})
    assert on.clearance_enabled is True
    assert on.stats()["forward_clearance"] is None  # enabled, no depth frame yet
    on.set_forward_clearance(0.8, 0.95, 1234.5)
    sample = {"clearance_m": 0.8, "valid_frac": 0.95, "timestamp": 1234.5}
    assert on.stats()["forward_clearance"] == sample
    assert on.get_forward_clearance() == sample
    on.clear()
    assert on.stats()["forward_clearance"] is None  # reset drops the stale sample
