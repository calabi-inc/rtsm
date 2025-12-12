"""
WebSocket Broadcaster for RTSM Visualization

Handles:
- Client connection management
- Binary mesh_create messages (point clouds)
- JSON mesh_update_pose and mesh_delete messages
- New client sync (send all existing keyframes)
- WM objects_update messages (real-time object overlay)
"""

import struct
import asyncio
from typing import Set, Optional, List, Dict, Any
from fastapi import WebSocket
import numpy as np

from rtsm.visualization.registry import KeyframeRegistry, KeyframeRecord


# Binary message format for mesh_create:
# [magic:4][mesh_id_len:2][mesh_id:N][num_points:4][positions:N*12][colors:N*3][has_pose:1][pose:64?]
MAGIC_MESH_CREATE = b'PCLD'  # 0x50434C44


class WSBroadcaster:
    """
    Manages WebSocket connections and broadcasts mesh commands.

    Backend is source of truth - assigns mesh_id and controls lifecycle.
    Frontend only renders based on commands received.
    """

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        """Register a new WebSocket client."""
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket client."""
        async with self._lock:
            self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    def _pack_mesh_create(
        self,
        mesh_id: str,
        positions: np.ndarray,
        colors: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> bytes:
        """
        Pack a mesh_create message as binary.

        Format:
        - Magic: 4 bytes ('PCLD')
        - mesh_id_len: 2 bytes (uint16 LE)
        - mesh_id: N bytes (UTF-8)
        - num_points: 4 bytes (uint32 LE)
        - positions: num_points * 12 bytes (float32 LE, xyz interleaved)
        - colors: num_points * 3 bytes (uint8 RGB)
        - has_pose: 1 byte (0 or 1)
        - pose: 64 bytes if has_pose (float32 LE, 4x4 row-major)
        """
        mesh_id_bytes = mesh_id.encode('utf-8')
        num_points = positions.shape[0]

        # Ensure correct dtypes
        pos_data = positions.astype(np.float32).tobytes()
        col_data = colors.astype(np.uint8).tobytes()

        has_pose = 1 if pose is not None else 0
        pose_data = pose.astype(np.float32).tobytes() if pose is not None else b''

        # Pack header
        header = struct.pack(
            '<4sHI',
            MAGIC_MESH_CREATE,
            len(mesh_id_bytes),
            num_points
        )

        return header + mesh_id_bytes + pos_data + col_data + bytes([has_pose]) + pose_data

    async def _broadcast_bytes(self, data: bytes) -> None:
        """Send binary data to all connected clients."""
        async with self._lock:
            dead_clients = []
            for client in self._clients:
                try:
                    await client.send_bytes(data)
                except Exception:
                    dead_clients.append(client)

            # Remove dead clients
            for client in dead_clients:
                self._clients.discard(client)

    async def _broadcast_json(self, data: dict) -> None:
        """Send JSON data to all connected clients."""
        async with self._lock:
            dead_clients = []
            for client in self._clients:
                try:
                    await client.send_json(data)
                except Exception:
                    dead_clients.append(client)

            for client in dead_clients:
                self._clients.discard(client)

    async def _send_bytes_to(self, ws: WebSocket, data: bytes) -> bool:
        """Send binary data to a specific client. Returns True on success."""
        try:
            await ws.send_bytes(data)
            return True
        except Exception:
            return False

    async def _send_json_to(self, ws: WebSocket, data: dict) -> bool:
        """Send JSON data to a specific client. Returns True on success."""
        try:
            await ws.send_json(data)
            return True
        except Exception:
            return False

    async def send_mesh_create(
        self,
        mesh_id: str,
        positions: np.ndarray,
        colors: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> None:
        """
        Broadcast mesh_create to all clients.

        Args:
            mesh_id: Unique mesh identifier (from SLAM kf_id)
            positions: (N, 3) float32 XYZ in camera frame
            colors: (N, 3) uint8 RGB
            pose: Optional (4, 4) float32 Twc transform
        """
        data = self._pack_mesh_create(mesh_id, positions, colors, pose)
        await self._broadcast_bytes(data)

    async def send_mesh_update_pose(self, mesh_id: str, pose: np.ndarray) -> None:
        """
        Broadcast pose update for an existing mesh.

        Args:
            mesh_id: Mesh to update
            pose: (4, 4) float32 Twc transform, row-major
        """
        # Send as JSON for simplicity (pose updates are small)
        pose_list = pose.astype(np.float32).flatten().tolist()
        await self._broadcast_json({
            'type': 'mesh_update_pose',
            'mesh_id': mesh_id,
            'pose': pose_list  # 16 floats, row-major
        })

    async def send_mesh_delete(self, mesh_id: str) -> None:
        """Broadcast mesh deletion command."""
        await self._broadcast_json({
            'type': 'mesh_delete',
            'mesh_id': mesh_id
        })

    async def send_objects_update(self, objects: List[Dict[str, Any]]) -> None:
        """
        Broadcast WM objects update to all clients.

        Args:
            objects: List of object dicts with id, xyz_world, label, confirmed, stability
        """
        await self._broadcast_json({
            'type': 'objects_update',
            'objects': objects
        })

    async def sync_new_client(self, ws: WebSocket, registry: KeyframeRegistry) -> int:
        """
        Send all existing keyframes to a newly connected client.

        Returns the number of keyframes synced.
        """
        keyframes = registry.get_all()
        synced = 0

        for kf in keyframes:
            data = self._pack_mesh_create(
                kf.mesh_id,
                kf.positions,
                kf.colors,
                kf.pose
            )
            if await self._send_bytes_to(ws, data):
                synced += 1
            else:
                # Client disconnected during sync
                break

        return synced
