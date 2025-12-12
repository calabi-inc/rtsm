"""
Coordinate transform utilities for RTSM.

Provides conversions between rotation representations (Euler angles, quaternions, rotation matrices).
"""

from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> NDArray[np.float32]:
    """
    Convert Euler angles (radians) to quaternion [x, y, z, w].

    Uses ZYX convention (yaw-pitch-roll), which is standard for robotics:
    - First rotate around Z by yaw
    - Then rotate around Y by pitch
    - Then rotate around X by roll

    This matches RTABMap's convention for T_wc = [x, y, z, roll, pitch, yaw].

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        Quaternion as numpy array [x, y, z, w] (Hamilton convention)
    """
    # Half angles
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # ZYX convention quaternion multiplication
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w], dtype=np.float32)


def quat_xyzw_to_euler(q: NDArray[np.float32]) -> tuple[float, float, float]:
    """
    Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw) in radians.

    Uses ZYX convention (inverse of euler_to_quat_xyzw).

    Args:
        q: Quaternion as numpy array [x, y, z, w]

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    x, y, z, w = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)  # Gimbal lock
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float32]:
    """
    Convert Euler angles (radians) to 3x3 rotation matrix.

    Uses ZYX convention (same as euler_to_quat_xyzw).

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        3x3 rotation matrix as numpy array
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # ZYX: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ]
    ], dtype=np.float32)
