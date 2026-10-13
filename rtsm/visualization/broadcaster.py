"""
WebSocket Broadcaster for RTSM Visualization

Handles:
- Client connection management
- Binary mesh_create messages (point clouds)
- Binary camera_frame messages (raw JPEG for PiP overlay)
- JSON mesh_update_pose and mesh_delete messages
- New client sync (send all existing keyframes)
- WM objects_update messages (real-time object overlay)
"""

import struct
import asyncio
import logging
from typing import Set, Optional, List, Dict, Any
from fastapi import WebSocket
import numpy as np

from rtsm.visualization.registry import KeyframeRegistry, KeyframeRecord

logger = logging.getLogger(__name__)

# Binary message format for mesh_create:
# [magic:4][mesh_id_len:2][mesh_id:N][num_points:4][positions:N*12][colors:N*3][has_pose:1][pose:64?]
MAGIC_MESH_CREATE = b'PCLD'  # 0x50434C44

# Binary message format for camera_frame:
# [magic:4 = 'CAMF'][jpeg_len:4 uint32 LE][jpeg_bytes:N]
MAGIC_CAMERA_FRAME = b'CAMF'  # 0x464D4143

# Timeout for sending to a single client (prevents slow client blocking)
_SEND_TIMEOUT_S = 5.0


class WSBroadcaster:
    """
    Manages WebSocket connections and broadcasts mesh commands.

    Backend is source of truth - assigns mesh_id and controls lifecycle.
    Frontend only renders based on commands received.

    Uses asyncio.gather for concurrent sends so a slow client
    doesn't block delivery to other clients.
    """

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket) -> None:
        """Register a new WebSocket client."""
        async with self._lock:
            self._clients.add(ws)
            # Capture the event loop that owns the WebSocket connections.
            # All broadcasts must run on this loop (not the viz server's loop).
            if self._client_loop is None:
                self._client_loop = asyncio.get_running_loop()

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket client."""
        async with self._lock:
            self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    # ── Thread-safe scheduling ──

    def schedule(self, coro) -> None:
        """Schedule an async broadcast from any thread.

        Uses the event loop that owns the WebSocket clients (captured in
        connect()).  Safe to call from the viz-server thread, the replay
        thread, or anywhere else.
        """
        loop = self._client_loop
        if loop is None:
            return  # no clients yet
        asyncio.run_coroutine_threadsafe(coro, loop)

    # ── Packing helpers ──

    def _pack_mesh_create(
        self,
        mesh_id: str,
        positions: np.ndarray,
        colors: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> bytes:
        return pack_mesh_create(mesh_id, positions, colors, pose)

    async def send_session_update(self, payload: Dict[str, Any]) -> None:
        """The session list changed (rtsm/core/sessions.py): {live, sessions, enabled}."""
        await self._broadcast_json({"type": "session_update", **payload})

    async def send_session_refined(self, payload: Dict[str, Any]) -> None:
        """A frozen session's keyframes / objects moved: {session, poses, objects}."""
        await self._broadcast_json({"type": "session_refined", **payload})

    @staticmethod
    def _pack_mesh_create_impl(
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

    @staticmethod
    def pack_camera_frame(jpeg_bytes: bytes) -> bytes:
        """Pack a camera_frame message as binary.

        Format: [magic:4 'CAMF'][jpeg_len:4 uint32 LE][jpeg_bytes:N]
        """
        return MAGIC_CAMERA_FRAME + struct.pack('<I', len(jpeg_bytes)) + jpeg_bytes

    @staticmethod
    def unpack_camera_frame(data: bytes) -> Optional[bytes]:
        """Unpack a camera_frame message. Returns JPEG bytes or None if invalid."""
        if len(data) < 8:
            return None
        magic = data[:4]
        if magic != MAGIC_CAMERA_FRAME:
            return None
        jpeg_len = struct.unpack_from('<I', data, 4)[0]
        if len(data) < 8 + jpeg_len:
            return None
        return data[8 : 8 + jpeg_len]

    # ── Concurrent broadcast helpers ──

    async def _try_send_bytes(self, ws: WebSocket, data: bytes) -> bool:
        """Send binary data to a single client with timeout."""
        try:
            await asyncio.wait_for(ws.send_bytes(data), timeout=_SEND_TIMEOUT_S)
            return True
        except Exception:
            return False

    async def _try_send_json(self, ws: WebSocket, data: dict) -> bool:
        """Send JSON data to a single client with timeout."""
        try:
            await asyncio.wait_for(ws.send_json(data), timeout=_SEND_TIMEOUT_S)
            return True
        except Exception:
            return False

    async def _broadcast_bytes(self, data: bytes) -> None:
        """Send binary data to all connected clients concurrently."""
        async with self._lock:
            if not self._clients:
                return
            clients = list(self._clients)

        # Send concurrently (outside lock to avoid holding it during I/O)
        results = await asyncio.gather(
            *(self._try_send_bytes(c, data) for c in clients),
            return_exceptions=True,
        )

        # Remove dead clients
        dead = [c for c, ok in zip(clients, results) if ok is not True]
        if dead:
            async with self._lock:
                for c in dead:
                    self._clients.discard(c)

    async def _broadcast_json(self, data: dict) -> None:
        """Send JSON data to all connected clients concurrently."""
        async with self._lock:
            if not self._clients:
                return
            clients = list(self._clients)

        results = await asyncio.gather(
            *(self._try_send_json(c, data) for c in clients),
            return_exceptions=True,
        )

        dead = [c for c, ok in zip(clients, results) if ok is not True]
        if dead:
            async with self._lock:
                for c in dead:
                    self._clients.discard(c)

    # ── Public send methods ──

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

    async def send_camera_frame(self, jpeg_bytes: bytes) -> None:
        """Broadcast a raw JPEG camera frame to all clients.

        Uses binary protocol (CAMF magic) for zero base64 overhead.
        Frontend uses Blob URL for native JPEG decode.
        """
        data = self.pack_camera_frame(jpeg_bytes)
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
            if await self._try_send_bytes(ws, data):
                synced += 1
            else:
                # Client disconnected during sync
                break

        return synced


def pack_mesh_create(mesh_id: str, positions: np.ndarray, colors: np.ndarray,
                     pose: Optional[np.ndarray] = None) -> bytes:
    """One mesh_create packet (the dashboard's binary format), usable outside the
    broadcaster: the REST clouds endpoint of a frozen session concatenates these."""
    return WSBroadcaster._pack_mesh_create_impl(mesh_id, positions, colors, pose)
