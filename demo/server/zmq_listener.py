"""
ZMQ Listener for RTSM Demo

Subscribes to RTABMap topics and processes incoming keyframe data.

Topics:
- rtabmap.kf_packet: [topic, json, rgb_jpeg, depth_png]
  Full keyframe data for point cloud building

- rtabmap.kf_pose: [topic, json]
  Notify downstream of new keyframe pose

- rtabmap.kf_pose_update: [topic, json]
  Corrected pose after loop closure / graph optimization
"""

import asyncio
import json
import math
from typing import Callable, Awaitable, Optional, Dict, Any
import numpy as np
import zmq
import zmq.asyncio


def euler_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert position + euler angles to 4x4 transformation matrix.

    Args:
        x, y, z: Translation
        roll, pitch, yaw: Rotation angles in radians

    Returns:
        4x4 transformation matrix (row-major)
    """
    # Rotation matrices
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Combined rotation matrix (ZYX order: yaw, pitch, roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=np.float32)

    # Build 4x4 matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    return T


def quaternion_to_matrix(t: list, q: list) -> np.ndarray:
    """
    Convert translation + quaternion to 4x4 transformation matrix.

    Args:
        t: [x, y, z] translation
        q: [qx, qy, qz, qw] quaternion

    Returns:
        4x4 transformation matrix (row-major)
    """
    x, y, z = t
    qx, qy, qz, qw = q

    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float32)

    # Build 4x4 matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    return T


def parse_pose(pose_data) -> Optional[np.ndarray]:
    """
    Parse pose from various formats.

    Supports:
    - [x, y, z, roll, pitch, yaw] (6 floats)
    - {"t": [x,y,z], "q": [qx,qy,qz,qw]} (quaternion dict)

    Returns:
        4x4 transformation matrix or None if invalid
    """
    if isinstance(pose_data, list) and len(pose_data) == 6:
        # Euler format: [x, y, z, roll, pitch, yaw]
        return euler_to_matrix(*pose_data)
    elif isinstance(pose_data, dict) and 't' in pose_data and 'q' in pose_data:
        # Quaternion format
        return quaternion_to_matrix(pose_data['t'], pose_data['q'])
    else:
        return None


class ZMQListener:
    """
    Async ZMQ subscriber for RTABMap keyframe data.

    Callbacks are invoked for each message type:
    - on_kf_packet: New keyframe with RGB-D data
    - on_kf_pose: Initial pose for a keyframe
    - on_kf_pose_update: Pose correction from BA/loop closure
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:6001",
        on_kf_packet: Optional[Callable[..., Awaitable[None]]] = None,
        on_kf_pose: Optional[Callable[..., Awaitable[None]]] = None,
        on_kf_pose_update: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self.endpoint = endpoint
        self.on_kf_packet = on_kf_packet
        self.on_kf_pose = on_kf_pose
        self.on_kf_pose_update = on_kf_pose_update

        self._ctx: Optional[zmq.asyncio.Context] = None
        self._sub: Optional[zmq.asyncio.Socket] = None
        self._running = False

    async def start(self) -> None:
        """Start the ZMQ listener."""
        if self._running:
            return

        self._ctx = zmq.asyncio.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(self.endpoint)

        # Subscribe to RTABMap topics
        topics = [
            b'rtabmap.kf_packet',
            b'rtabmap.kf_pose',
            b'rtabmap.kf_pose_update',
        ]
        for topic in topics:
            self._sub.subscribe(topic)

        self._running = True
        print(f"[zmq-listener] Connected to {self.endpoint}")
        print(f"[zmq-listener] Subscribed to: {[t.decode() for t in topics]}")

    async def stop(self) -> None:
        """Stop the ZMQ listener."""
        self._running = False
        if self._sub:
            self._sub.close()
            self._sub = None
        if self._ctx:
            self._ctx.term()
            self._ctx = None

    async def run(self) -> None:
        """
        Main loop - receive and dispatch ZMQ messages.

        Should be run as an asyncio task.
        """
        if not self._running or not self._sub:
            raise RuntimeError("Listener not started. Call start() first.")

        while self._running:
            try:
                parts = await self._sub.recv_multipart()
                await self._handle_message(parts)
            except zmq.ZMQError as e:
                if self._running:
                    print(f"[zmq-listener] ZMQ error: {e}")
                    await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[zmq-listener] Error processing message: {e}")
                import traceback
                traceback.print_exc()

    async def _handle_message(self, parts: list) -> None:
        """Dispatch message to appropriate handler."""
        if not parts:
            return

        topic = parts[0].decode('utf-8')

        try:
            if topic == 'rtabmap.kf_packet':
                if len(parts) >= 4:
                    await self._handle_kf_packet(parts)
                else:
                    # kf_packet without images - skip silently
                    pass
            elif topic == 'rtabmap.kf_pose' and len(parts) >= 2:
                await self._handle_kf_pose(parts)
            elif topic == 'rtabmap.kf_pose_update' and len(parts) >= 2:
                await self._handle_kf_pose_update(parts)
            else:
                print(f"[zmq-listener] Unhandled topic: {topic} ({len(parts)} parts)")
        except Exception as e:
            print(f"[zmq-listener] Error handling {topic}: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_kf_packet(self, parts: list) -> None:
        """
        Handle rtabmap.kf_packet message.

        Format: [topic, json, rgb_jpeg, depth_png]

        JSON structure:
        {
            "kf_id": 123,
            "ts_ns": 1234567890000000,
            "map_id": 0,
            "T_w_c": {"t": [x,y,z], "q": [qx,qy,qz,qw]},
            "intrinsics": {
                "fx": 615.0, "fy": 615.0,
                "cx": 320.0, "cy": 240.0,
                "width": 640, "height": 480
            },
            "depth_units_m": 0.001
        }
        """
        if not self.on_kf_packet:
            return

        # Parse JSON metadata
        try:
            metadata = json.loads(parts[1].decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"[zmq-listener] Invalid JSON in kf_packet: {e}")
            return

        kf_id = str(metadata.get('kf_id', 0))
        map_id = str(metadata.get('map_id', 0))
        timestamp_ns = metadata.get('ts_ns', 0)

        # Parse intrinsics
        intrinsics = metadata.get('intrinsics', {})
        fx = intrinsics.get('fx', 615.0)
        fy = intrinsics.get('fy', 615.0)
        cx = intrinsics.get('cx', 320.0)
        cy = intrinsics.get('cy', 240.0)
        width = intrinsics.get('width', 640)
        height = intrinsics.get('height', 480)

        # Build K matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Get depth scale
        depth_scale = metadata.get('depth_units_m', 0.001)

        # Parse pose (included in kf_packet)
        pose = None
        if 'T_w_c' in metadata:
            pose = parse_pose(metadata['T_w_c'])

        # Binary parts
        jpeg_bytes = bytes(parts[2])
        depth_png_bytes = bytes(parts[3])

        await self.on_kf_packet(
            map_id=map_id,
            kf_id=kf_id,
            timestamp_ns=timestamp_ns,
            K=K,
            jpeg_bytes=jpeg_bytes,
            depth_png_bytes=depth_png_bytes,
            depth_scale=depth_scale,
            width=width,
            height=height,
            pose=pose
        )

    async def _handle_kf_pose(self, parts: list) -> None:
        """
        Handle rtabmap.kf_pose message.

        Format: [topic, json]

        JSON structure:
        {
            "kf_id": 123,
            "T_wc": [x, y, z, roll, pitch, yaw]
        }
        """
        if not self.on_kf_pose:
            return

        try:
            metadata = json.loads(parts[1].decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"[zmq-listener] Invalid JSON in kf_pose: {e}")
            return

        kf_id = str(metadata.get('kf_id', 0))

        pose = parse_pose(metadata.get('T_wc'))
        if pose is None:
            print(f"[zmq-listener] Invalid pose in kf_pose")
            return

        await self.on_kf_pose(
            kf_id=kf_id,
            pose=pose
        )

    async def _handle_kf_pose_update(self, parts: list) -> None:
        """
        Handle rtabmap.kf_pose_update message.

        Format: [topic, json]

        JSON structure:
        {
            "kf_id": 123,
            "T_wc": [x, y, z, roll, pitch, yaw]
        }
        """
        if not self.on_kf_pose_update:
            return

        try:
            metadata = json.loads(parts[1].decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"[zmq-listener] Invalid JSON in kf_pose_update: {e}")
            return

        kf_id = str(metadata.get('kf_id', 0))

        pose = parse_pose(metadata.get('T_wc'))
        if pose is None:
            print(f"[zmq-listener] Invalid pose in kf_pose_update")
            return

        await self.on_kf_pose_update(
            kf_id=kf_id,
            pose=pose
        )
