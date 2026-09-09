"""
Tier-1 geometry for the RC-car agent — pure math, zero I/O, zero tunables.

════════════════════════════════════════════════════════════════════════
COORDINATE CONVENTIONS (load-bearing — every sign in this file hangs on
these four facts; change nothing here without re-running test_geometry.py
and the on-car calibration acceptance gate)
════════════════════════════════════════════════════════════════════════

1. WORLD frame = ARKit world: **Y is UP** (gravity-aligned).
   The ground plane is X–Z. Heading/distance math uses indices [0] (x)
   and [2] (z) — NEVER [1] (that's the vertical axis; using it was the
   X–Y bug in the original demo2 sketch).

2. CAMERA orientation, as served by RTSM (`robot_pose.quaternion_xyzw`),
   is in **OpenCV camera convention**: X right, Y down, **Z forward**.
   RTSM flips ARKit→OpenCV once at ingestion (rtsm/io/websocket.py,
   `_ARKIT_TO_OPENCV`, right-multiplied), so the camera's forward
   direction in world coordinates is  R(q) @ [0, 0, 1].
   ⚠ CONTRACT DEPENDENCY: this holds only when the RTSM config sets
   `visualization.apply_camera_flip: true` (shipped configs do; the code
   default is False if the key is deleted) AND the pose comes from the
   ARKit WebSocket receiver path (not the ZMQ/RTABMap path). If the flip
   is off, R(q)@[0,0,1] is the camera's BACKWARD direction and every
   heading is 180° wrong — preflight must verify the flip is active.

3. YAW convention: yaw = atan2(v_x, v_z) of a direction v on the ground
   plane. Zero = facing world +Z; range (−π, π]; **positive = CCW viewed
   from above = a LEFT turn** — matching the ESP32 firmware convention
   (+angle = CCW). At yaw 0 the robot's left hand points to +X.

4. Quaternions are **xyzw** order throughout (RTSM wire format).

Derived identity used by nav: err = wrap_angle(desired_yaw − car_yaw);
err > 0 → target is to the LEFT → turn CCW.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "wrap_angle",
    "ground_vec_xz",
    "ground_dist",
    "desired_yaw",
    "forward_from_quat_xyzw",
    "yaw_from_quat_xyzw",
    "camera_to_car",
    "DegenerateHeadingError",
]

# Below this XZ-projection magnitude the camera is pointing almost straight
# up/down and a ground heading is meaningless (phone knocked over, mount
# failure). Callers must treat this as a fault, not a heading.
_MIN_GROUND_PROJECTION = 0.05  # ≈ within ~3° of vertical


class DegenerateHeadingError(ValueError):
    """Camera forward is (near-)vertical — no ground heading exists."""


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to the half-open interval (−π, π]."""
    wrapped = math.fmod(a + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def ground_vec_xz(
    target_xyz: Sequence[float], robot_xyz: Sequence[float]
) -> Tuple[float, float]:
    """Ground-plane vector robot→target: (dx, dz) = indices [0] and [2].

    Index [1] (Y, vertical) is deliberately ignored — a mug on a table and
    the floor spot beneath it are the same navigation target.
    """
    dx = float(target_xyz[0]) - float(robot_xyz[0])
    dz = float(target_xyz[2]) - float(robot_xyz[2])
    return dx, dz


def ground_dist(target_xyz: Sequence[float], robot_xyz: Sequence[float]) -> float:
    """Euclidean distance on the ground plane (X–Z only)."""
    dx, dz = ground_vec_xz(target_xyz, robot_xyz)
    return math.hypot(dx, dz)


def desired_yaw(dx: float, dz: float) -> float:
    """World-frame bearing of a ground vector.

    **Argument order is atan2(dx, dz)** — pinned. atan2(dz, dx) is the
    90°-rotated wrong answer; test_desired_yaw_quadrants guards this.
    """
    return math.atan2(dx, dz)


def _quat_to_rotmat(q_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (normalized internally)."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be length-4 xyzw, got shape {q.shape}")
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = (q / n).tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def forward_from_quat_xyzw(q_xyzw: Sequence[float]) -> np.ndarray:
    """Camera forward direction in world coordinates.

    OpenCV camera convention (see module docstring fact #2): forward is the
    camera's +Z axis, i.e. the third column of R(q).
    """
    return _quat_to_rotmat(q_xyzw)[:, 2]


def yaw_from_quat_xyzw(q_xyzw: Sequence[float]) -> float:
    """Ground-plane heading of the camera from `robot_pose.quaternion_xyzw`.

    Projects the camera forward onto X–Z; pitch (phone tilted up/down) does
    not bias the heading. Raises DegenerateHeadingError when the forward is
    near-vertical and no heading exists.
    """
    fwd = forward_from_quat_xyzw(q_xyzw)
    fx, fz = float(fwd[0]), float(fwd[2])
    if math.hypot(fx, fz) < _MIN_GROUND_PROJECTION:
        raise DegenerateHeadingError(
            f"camera forward {fwd.tolist()} is near-vertical; no ground heading"
        )
    return math.atan2(fx, fz)


def camera_to_car(
    cam_xyz: Sequence[float],
    cam_yaw: float,
    yaw_offset: float,
    lever_arm_rf: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float, float]:
    """Camera pose → car body pose on the ground plane.

    The iPhone is not mounted at the car's drive center nor necessarily
    facing the car's forward; calibrate.py measures both constants.

    Args:
        cam_xyz: camera position in world (ARKit world, Y up).
        cam_yaw: camera ground heading from yaw_from_quat_xyzw().
        yaw_offset: calibrated camera→car heading offset (rad); car_yaw =
            wrap(cam_yaw + yaw_offset).
        lever_arm_rf: (right, forward) offset in METERS from the CAMERA to
            the CAR DRIVE CENTER, expressed in the CAR body frame.
            Example: camera mounted 10 cm behind the drive center →
            lever_arm_rf = (0.0, +0.10).

    Returns:
        (car_x, car_z, car_yaw).

    Body→world axes at heading ψ (see module docstring fact #3):
        forward_unit = ( sin ψ, cos ψ )      # in (x, z)
        right_unit   = (−cos ψ, sin ψ )      # facing +Z (ψ=0): right = −X
    """
    r, f = float(lever_arm_rf[0]), float(lever_arm_rf[1])
    car_yaw = wrap_angle(float(cam_yaw) + float(yaw_offset))
    s, c = math.sin(car_yaw), math.cos(car_yaw)
    car_x = float(cam_xyz[0]) + f * s - r * c
    car_z = float(cam_xyz[2]) + f * c + r * s
    return car_x, car_z, car_yaw
