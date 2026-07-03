"""
Unit tests for geometry.py — the X–Y-vs-X–Z bug zone.

Every expected value below is derived from the pinned conventions, NOT from
running the implementation:

  * world: ARKit, Y UP → ground plane is X–Z
  * camera quat (xyzw): OpenCV convention → forward = R(q) @ [0,0,1]
  * yaw = atan2(x, z): 0 = facing world +Z, positive = CCW from above = LEFT
    (matches ESP32 firmware +angle = CCW); facing +Z, the robot's left = +X
"""

import math

import numpy as np
import pytest

from geometry import (
    DegenerateHeadingError,
    camera_to_car,
    desired_yaw,
    forward_from_quat_xyzw,
    ground_dist,
    ground_vec_xz,
    wrap_angle,
    yaw_from_quat_xyzw,
)

DEG = math.pi / 180.0


# ───────────────────────── quaternion helpers (test-local, independent) ──


def quat_axis_angle(axis, angle_rad):
    """xyzw quaternion for a right-handed rotation about `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def quat_mul(q1, q2):
    """Hamilton product (xyzw): rotation q1∘q2 applies q2 first, then q1."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


# ───────────────────────────────────────────── ground-plane projection ──


def test_ground_vec_xz_ignores_y():
    # A mug on a table (y=0.8) vs the floor spot beneath it: same target.
    robot = [1.0, 0.1, 2.0]
    target_high = [1.0, 0.9, 2.0]
    assert ground_vec_xz(target_high, robot) == (0.0, 0.0)
    assert ground_dist(target_high, robot) == 0.0


def test_ground_vec_and_dist_basic():
    dx, dz = ground_vec_xz([3.0, 0.0, 4.0], [0.0, 5.0, 0.0])
    assert (dx, dz) == (3.0, 4.0)
    assert ground_dist([3.0, 0.0, 4.0], [0.0, 5.0, 0.0]) == pytest.approx(5.0)


# ─────────────────────────────────────────────────────── bearing (yaw) ──


def test_desired_yaw_quadrants():
    # atan2(dx, dz) — pinned argument order.
    assert desired_yaw(0.0, 1.0) == pytest.approx(0.0)              # ahead (+Z)
    assert desired_yaw(1.0, 0.0) == pytest.approx(+90 * DEG)        # +X = LEFT
    assert desired_yaw(-1.0, 0.0) == pytest.approx(-90 * DEG)       # −X = right
    assert abs(desired_yaw(0.0, -1.0)) == pytest.approx(180 * DEG)  # behind
    assert desired_yaw(1.0, 1.0) == pytest.approx(+45 * DEG)        # front-left


def test_wrap_angle():
    assert wrap_angle(1.5 * math.pi) == pytest.approx(-0.5 * math.pi)
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(math.pi) == pytest.approx(math.pi)      # π stays π …
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)     # … −π maps to π
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-0.25 * math.pi) == pytest.approx(-0.25 * math.pi)


# ─────────────────────────────────────────────────── yaw from quaternion ──


def test_yaw_identity_quat_faces_plus_z():
    # R = I → forward = +Z (OpenCV convention) → yaw 0.
    assert yaw_from_quat_xyzw([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_yaw_plus_90_about_world_up_is_left():
    # +90° about +Y (right-handed) takes +Z → +X: R_y(θ)·ẑ = (sin θ, 0, cos θ).
    q = quat_axis_angle([0, 1, 0], +90 * DEG)
    fwd = forward_from_quat_xyzw(q)
    assert fwd == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert yaw_from_quat_xyzw(q) == pytest.approx(+90 * DEG)


def test_yaw_minus_90_and_180():
    assert yaw_from_quat_xyzw(quat_axis_angle([0, 1, 0], -90 * DEG)) == pytest.approx(
        -90 * DEG
    )
    assert abs(yaw_from_quat_xyzw(quat_axis_angle([0, 1, 0], 180 * DEG))) == (
        pytest.approx(180 * DEG)
    )


def test_yaw_unnormalized_quat_is_tolerated():
    q = 2.0 * quat_axis_angle([0, 1, 0], +30 * DEG)
    assert yaw_from_quat_xyzw(q) == pytest.approx(+30 * DEG)


def test_yaw_survives_camera_pitch():
    # Phone tilted 30° DOWN while heading +60°: in OpenCV camera convention
    # (Y down, Z forward) looking DOWN is a POSITIVE rotation about local X
    # (R_x(+θ)·ẑ has negative world-Y). Yaw first (world up), then pitch
    # about the camera's LOCAL X — local composition = right-multiply.
    q = quat_mul(
        quat_axis_angle([0, 1, 0], +60 * DEG),
        quat_axis_angle([1, 0, 0], +30 * DEG),
    )
    assert yaw_from_quat_xyzw(q) == pytest.approx(+60 * DEG, abs=1e-6)


def test_yaw_degenerate_when_facing_straight_down():
    # Pitch +90° (OpenCV: positive local-X = down): forward = R_x(+90°)·ẑ =
    # (0, −1, 0) = straight down → vertical, no ground heading.
    q = quat_axis_angle([1, 0, 0], +90 * DEG)
    with pytest.raises(DegenerateHeadingError):
        yaw_from_quat_xyzw(q)


def test_yaw_degenerate_when_facing_straight_up():
    # Pitch −90°: forward = R_x(−90°)·ẑ = (0, +1, 0) = straight up.
    q = quat_axis_angle([1, 0, 0], -90 * DEG)
    with pytest.raises(DegenerateHeadingError):
        yaw_from_quat_xyzw(q)


def test_err_sign_left_positive():
    # Robot faces +Z (yaw 0); target at +X (robot's LEFT) → err = +90° → CCW,
    # matching firmware +angle = CCW.
    robot_yaw = yaw_from_quat_xyzw([0.0, 0.0, 0.0, 1.0])
    dx, dz = ground_vec_xz([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    err = wrap_angle(desired_yaw(dx, dz) - robot_yaw)
    assert err == pytest.approx(+90 * DEG)


# ─────────────────────────────────────────────────────── camera → car ──


def test_camera_to_car_yaw_offset_applied_and_wrapped():
    _, _, car_yaw = camera_to_car([0, 0, 0], cam_yaw=+90 * DEG, yaw_offset=+90 * DEG)
    assert car_yaw == pytest.approx(+180 * DEG)
    _, _, car_yaw = camera_to_car([0, 0, 0], cam_yaw=+170 * DEG, yaw_offset=+20 * DEG)
    assert car_yaw == pytest.approx(-170 * DEG)  # wraps past π


def test_camera_to_car_lever_arm_forward_at_yaw_zero():
    # Facing +Z, drive center 10 cm ahead of the camera → +0.10 on Z.
    car_x, car_z, _ = camera_to_car(
        [1.0, 0.3, 2.0], cam_yaw=0.0, yaw_offset=0.0, lever_arm_rf=(0.0, 0.10)
    )
    assert car_x == pytest.approx(1.0)
    assert car_z == pytest.approx(2.10)


def test_camera_to_car_lever_arm_rotates_with_heading():
    # Same forward lever arm, car heading +90° (facing +X) → offset lands on +X.
    car_x, car_z, _ = camera_to_car(
        [0.0, 0.0, 0.0], cam_yaw=+90 * DEG, yaw_offset=0.0, lever_arm_rf=(0.0, 0.10)
    )
    assert car_x == pytest.approx(0.10)
    assert car_z == pytest.approx(0.0, abs=1e-12)


def test_camera_to_car_right_lever_arm():
    # Facing +Z (yaw 0), right = −X: a drive center 5 cm to the camera's
    # right sits at x = −0.05.
    car_x, car_z, _ = camera_to_car(
        [0.0, 0.0, 0.0], cam_yaw=0.0, yaw_offset=0.0, lever_arm_rf=(0.05, 0.0)
    )
    assert car_x == pytest.approx(-0.05)
    assert car_z == pytest.approx(0.0, abs=1e-12)


def test_camera_to_car_zero_lever_is_identity_position():
    car_x, car_z, car_yaw = camera_to_car([4.0, 1.0, -2.0], cam_yaw=0.7, yaw_offset=0.0)
    assert (car_x, car_z) == (4.0, -2.0)
    assert car_yaw == pytest.approx(0.7)
