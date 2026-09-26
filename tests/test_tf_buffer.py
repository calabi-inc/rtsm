"""P3 task 1 -- TfBuffer pinned against analytic answers (the CLAUDE.md pose-math rule)."""
from __future__ import annotations

import numpy as np
import pytest

from rtsm.io.tf_buffer import (
    TfBuffer, TfLookupError, make_T, norm_frame, odometry_buffer, quat_slerp, quat_to_rotmat, split_T,
)

S = 1_000_000_000


def q_about_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def q_about_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg) / 2
    return np.array([np.sin(a), 0.0, 0.0, np.cos(a)])


IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def test_norm_frame_strips_ros1_slashes():
    assert norm_frame("/world") == "world" and norm_frame("world") == "world" and norm_frame(None) == ""


def test_quaternion_helpers():
    R = quat_to_rotmat(q_about_z(90))
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)
    mid = quat_slerp(IDENT, q_about_z(90), 0.5)
    assert np.allclose(mid, q_about_z(45), atol=1e-12)
    # q and -q are the same rotation: slerp takes the short arc
    mid2 = quat_slerp(IDENT, -q_about_z(90), 0.5)
    assert np.allclose(np.abs(mid2), np.abs(q_about_z(45)), atol=1e-12)
    t, q = split_T(make_T([1, 2, 3], q_about_z(30)))
    assert t.dtype == np.float32 and np.allclose(t, [1, 2, 3]) and np.allclose(np.abs(q), np.abs(q_about_z(30)), atol=1e-6)


def test_single_static_hop():
    buf = TfBuffer()
    buf.add("world", "cam", 0, [1, 2, 3], q_about_z(90), static=True)
    t, q = buf.lookup("world", "cam", 12345)
    assert np.allclose(t, [1, 2, 3]) and np.allclose(np.abs(q), np.abs(q_about_z(90)), atol=1e-6)
    assert buf.roots() == ["world"] and buf.hops() == [("world", "cam", True)]


def test_moving_hop_interpolates_translation_and_rotation():
    buf = TfBuffer()
    buf.add("/world", "/kinect", 0, [0, 0, 0], IDENT)
    buf.add("/world", "/kinect", 1 * S, [1, 0, 0], q_about_z(90))
    t, q = buf.lookup("world", "kinect", S // 2)
    assert np.allclose(t, [0.5, 0, 0], atol=1e-6)
    assert np.allclose(np.abs(q), np.abs(q_about_z(45)), atol=1e-6)
    t, q = buf.lookup("world", "kinect", S // 4)
    assert np.allclose(t, [0.25, 0, 0], atol=1e-6) and np.allclose(np.abs(q), np.abs(q_about_z(22.5)), atol=1e-6)
    # exact sample stamps return the samples
    t, q = buf.lookup("world", "kinect", S)
    assert np.allclose(t, [1, 0, 0]) and np.allclose(np.abs(q), np.abs(q_about_z(90)), atol=1e-6)


def test_two_hops_compose_in_parent_to_child_order():
    """T_world_cam = T_world_base @ T_base_cam. A wrong order would put the
    static camera offset in the world frame instead of the rotated base frame."""
    buf = TfBuffer()
    buf.add("world", "base", 0, [0, 0, 0], IDENT)
    buf.add("world", "base", 2 * S, [2, 0, 0], q_about_z(90))
    buf.add("base", "cam", 0, [1, 0, 0], q_about_x(90), static=True)
    T = buf.lookup_T("world", "cam", 2 * S)
    expected = make_T([2, 0, 0], q_about_z(90)) @ make_T([1, 0, 0], q_about_x(90))
    assert np.allclose(T, expected, atol=1e-9)
    # the camera's x offset is rotated by the base yaw: it lands at world y
    assert np.allclose(T[:3, 3], [2, 1, 0], atol=1e-9)
    # midpoint: base half way and at 45 deg
    T = buf.lookup_T("world", "cam", S)
    exp_mid = make_T([1, 0, 0], q_about_z(45)) @ make_T([1, 0, 0], q_about_x(90))
    assert np.allclose(T, exp_mid, atol=1e-9)
    assert buf.chain("world", "cam") == [("world", "base"), ("base", "cam")]


def test_tum_style_four_hop_chain_with_leading_slashes():
    buf = TfBuffer()
    buf.add("/world", "/kinect", 0, [0, 0, 1], IDENT)
    buf.add("/world", "/kinect", S, [1, 0, 1], IDENT)
    buf.add("/kinect", "/openni_camera", 0, [0, 0, 0], IDENT, static=True)
    buf.add("/openni_camera", "/openni_rgb_frame", 0, [0, -0.045, 0], IDENT, static=True)
    buf.add("/openni_rgb_frame", "/openni_rgb_optical_frame", 0, [0, 0, 0], np.array([-0.5, 0.5, -0.5, 0.5]), static=True)
    assert buf.chain("world", "openni_rgb_optical_frame") == [
        ("world", "kinect"), ("kinect", "openni_camera"), ("openni_camera", "openni_rgb_frame"), ("openni_rgb_frame", "openni_rgb_optical_frame")]
    t, q = buf.lookup("world", "/openni_rgb_optical_frame", S // 2)
    assert np.allclose(t, [0.5, -0.045, 1], atol=1e-6)
    assert np.allclose(np.abs(q), [0.5, 0.5, 0.5, 0.5], atol=1e-6)


def test_no_chain_and_extrapolation_limits():
    buf = TfBuffer(extrapolation_s=0.05)
    buf.add("world", "base", S, [1, 0, 0], IDENT)
    buf.add("world", "base", 2 * S, [2, 0, 0], IDENT)
    with pytest.raises(TfLookupError) as e:
        buf.lookup("world", "cam", S)
    assert e.value.reason == "no_chain"
    with pytest.raises(TfLookupError) as e:
        buf.lookup("world", "base", 3 * S)                       # 1 s past the last sample
    assert e.value.reason == "extrapolation"
    t, _ = buf.lookup("world", "base", 2 * S + 40_000_000)      # 40 ms past: clamped to the last sample
    assert np.allclose(t, [2, 0, 0])
    t, _ = buf.lookup("world", "base", S - 40_000_000)          # 40 ms before the first: clamped
    assert np.allclose(t, [1, 0, 0])
    with pytest.raises(TfLookupError) as e:
        buf.lookup("world", "base", S - 60_000_000)
    assert e.value.reason == "extrapolation"


def test_unsorted_samples_are_sorted_lazily():
    buf = TfBuffer()
    buf.add("world", "base", 2 * S, [2, 0, 0], IDENT)
    buf.add("world", "base", 0, [0, 0, 0], IDENT)
    buf.add("world", "base", S, [1, 0, 0], IDENT)
    t, _ = buf.lookup("world", "base", S + S // 2)
    assert np.allclose(t, [1.5, 0, 0], atol=1e-6)
    assert buf.span(("world", "base")) == (0, 2 * S)


def test_odometry_buffer_with_static_extrinsic():
    buf = odometry_buffer([(0, [0, 0, 0], IDENT), (S, [0, 2, 0], q_about_z(-90))],
                          world="odom", body="base_link", extrinsic=([0.5, 0, 0], IDENT), camera="cam")
    T = buf.lookup_T("odom", "cam", S)
    expected = make_T([0, 2, 0], q_about_z(-90)) @ make_T([0.5, 0, 0], IDENT)
    assert np.allclose(T, expected, atol=1e-9)
    assert np.allclose(T[:3, 3], [0, 1.5, 0], atol=1e-9)        # the 0.5 m forward offset now points along -y


def test_bad_frames_rejected():
    buf = TfBuffer()
    with pytest.raises(ValueError):
        buf.add("world", "world", 0, [0, 0, 0], IDENT)
    with pytest.raises(ValueError):
        buf.add("", "cam", 0, [0, 0, 0], IDENT)


def test_single_hop_exact_sample_round_trips_bit_for_bit():
    """The receiver's own pose (float32 t, q) stored on a one-hop chain comes back
    unchanged; a 4x4 round trip would perturb the quaternion at 1e-7."""
    t = np.array([-0.8506309, 0.6687498, 0.11896142], dtype=np.float32)
    q = np.array([0.09346367, 0.2436344, -0.6842305, 0.68098134], dtype=np.float32)
    buf = TfBuffer()
    buf.add("map", "cam", 5 * S, t.astype(np.float64), q.astype(np.float64))
    buf.add("map", "cam", 6 * S, [0, 0, 0], IDENT)
    t2, q2 = buf.lookup("map", "cam", 5 * S)
    assert np.array_equal(t2, t) and np.array_equal(q2, q) and t2.dtype == q2.dtype == np.float32
    tm, qm = buf.lookup("map", "cam", 5 * S + S // 2)                        # between samples: interpolated, not exact
    assert not np.array_equal(qm, q)


def test_republished_constant_hop_has_no_extrapolation_limit():
    """TUM re-publishes the calibration chain on /tf as timed samples starting
    76 ms AFTER the first image: a hop whose samples never change is constant
    and clamps at any stamp; a hop that moves keeps the limit."""
    buf = TfBuffer(extrapolation_s=0.05)
    for k in range(3):
        buf.add("rgb_frame", "rgb_optical", S + k * 100_000_000, [0, 0, 0], np.array([-0.5, 0.5, -0.5, 0.5]))
    t, q = buf.lookup("rgb_frame", "rgb_optical", S - 500_000_000)             # half a second before its first sample
    assert np.allclose(np.abs(q), 0.5)
    buf.add("world", "kinect", S, [0, 0, 0], IDENT)
    buf.add("world", "kinect", 2 * S, [1, 0, 0], IDENT)
    with pytest.raises(TfLookupError):
        buf.lookup("world", "kinect", S - 500_000_000)                            # a moving hop keeps the limit
    buf.add("kinect", "rgb_frame", 0, [0, -0.045, 0], IDENT, static=True)
    t, q = buf.lookup("world", "rgb_optical", S + 40_000_000)                    # 40 ms before the constant hop's first sample, inside the moving hop
    assert np.allclose(t, [0.04, -0.045, 0], atol=1e-6)
