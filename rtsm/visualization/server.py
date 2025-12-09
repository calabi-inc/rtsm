"""
RTSM Visualization Server

Embedded visualization server that runs within RTSM core process.
Provides real-time point cloud streaming and WM object overlay via WebSocket.

Features:
- Receives kf_packet callbacks from ZeroMQSubscriber
- Processes RGB-D into point clouds
- Broadcasts meshes to WebSocket clients
- Periodically pushes WM objects for real-time overlay
"""

import asyncio
import json
import math
import threading
import logging
from typing import Optional, Callable, Any, Dict, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rtsm.visualization.processor import PointCloudProcessor, ProcessorConfig
from rtsm.visualization.registry import KeyframeRegistry
from rtsm.visualization.broadcaster import WSBroadcaster

logger = logging.getLogger(__name__)


def euler_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert position + euler angles to 4x4 transformation matrix.

    Args:
        x, y, z: Translation
        roll, pitch, yaw: Rotation angles in radians

    Returns:
        4x4 transformation matrix (row-major)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Combined rotation matrix (ZYX order: yaw, pitch, roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=np.float32)

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

    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float32)

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
        return euler_to_matrix(*pose_data)
    elif isinstance(pose_data, dict) and 't' in pose_data and 'q' in pose_data:
        return quaternion_to_matrix(pose_data['t'], pose_data['q'])
    else:
        return None


class VisualizationServer:
    """
    Embedded visualization server for RTSM.

    Runs WebSocket server in background thread, receives kf_packet callbacks
    from ZeroMQSubscriber, and can push WM object updates to clients.
    """

    def __init__(
        self,
        cfg: dict,
        working_memory: Any,  # WorkingMemory instance
        host: str = "0.0.0.0",
        port: int = 8081,
    ):
        """
        Initialize visualization server.

        Args:
            cfg: RTSM configuration dict
            working_memory: WorkingMemory instance for direct object access
            host: WebSocket server host
            port: WebSocket server port
        """
        self.cfg = cfg
        self.wm = working_memory
        self.host = host
        self.port = port

        # Visualization config
        vis_cfg = cfg.get("visualization", {})
        self._push_interval_ms = vis_cfg.get("objects", {}).get("push_interval_ms", 200)
        self._include_proto = vis_cfg.get("objects", {}).get("include_proto", True)

        # Components
        self.processor = PointCloudProcessor(ProcessorConfig())
        self.registry = KeyframeRegistry()
        self.broadcaster = WSBroadcaster()

        # Server state
        self._app: Optional[FastAPI] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._objects_task: Optional[asyncio.Task] = None

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with WebSocket endpoint."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self._running = True
            # Start periodic WM objects push
            self._objects_task = asyncio.create_task(self._push_objects_loop())
            logger.info(f"[visualization] Server started on port {self.port}")
            yield
            # Shutdown
            self._running = False
            if self._objects_task:
                self._objects_task.cancel()
                try:
                    await self._objects_task
                except asyncio.CancelledError:
                    pass

        app = FastAPI(
            title="RTSM Visualization Server",
            description="Real-time point cloud and object overlay",
            lifespan=lifespan
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self.broadcaster.connect(websocket)

            # Sync existing keyframes to new client
            synced = await self.broadcaster.sync_new_client(websocket, self.registry)
            logger.debug(f"[visualization] New client synced with {synced} keyframes")

            try:
                while True:
                    try:
                        data = await websocket.receive_json()
                        await self._handle_client_command(websocket, data)
                    except Exception:
                        break
            except WebSocketDisconnect:
                pass
            finally:
                await self.broadcaster.disconnect(websocket)
                logger.debug(f"[visualization] Client disconnected ({self.broadcaster.client_count} remaining)")

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/stats")
        async def stats():
            return {
                **self.registry.stats(),
                "clients": self.broadcaster.client_count
            }

        return app

    async def _handle_client_command(self, websocket: WebSocket, data: dict) -> None:
        """Handle commands from WebSocket clients."""
        cmd = data.get("cmd")

        if cmd == "clear":
            count = self.registry.clear()
            await self.broadcaster._broadcast_json({"type": "clear"})
            logger.info(f"[visualization] Cleared {count} keyframes")

        elif cmd == "stats":
            stats = self.registry.stats()
            stats["clients"] = self.broadcaster.client_count
            await websocket.send_json({"type": "stats", **stats})

    async def _push_objects_loop(self) -> None:
        """Periodically push WM objects to connected clients."""
        interval_s = self._push_interval_ms / 1000.0

        while self._running:
            try:
                if self.broadcaster.client_count > 0:
                    objects = self._get_wm_objects()
                    if objects:
                        await self.broadcaster.send_objects_update(objects)
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[visualization] Error pushing objects: {e}")
                await asyncio.sleep(interval_s)

    def _get_wm_objects(self) -> List[Dict[str, Any]]:
        """Get serialized WM objects for frontend."""
        try:
            # Get all objects from WorkingMemory
            result = []

            for obj in self.wm.iter_objects():
                # Skip proto objects if configured
                if not self._include_proto and not obj.confirmed:
                    continue

                result.append({
                    "id": obj.id,
                    "xyz_world": obj.xyz_world.tolist() if hasattr(obj.xyz_world, 'tolist') else list(obj.xyz_world),
                    "label_hint": obj.label_primary,  # Secondary/unreliable - use object_id as primary
                    "label_scores": dict(obj.label_scores) if obj.label_scores else {},
                    "confirmed": obj.confirmed,
                    "stability": float(obj.stability) if hasattr(obj, 'stability') else 0.0,
                })

            return result
        except Exception as e:
            logger.error(f"[visualization] Error getting WM objects: {e}")
            return []

    def handle_kf_packet(
        self,
        map_id: str,
        kf_id: str,
        timestamp_ns: int,
        K: np.ndarray,
        jpeg_bytes: bytes,
        depth_png_bytes: bytes,
        depth_scale: float,
        width: int,
        height: int,
        pose: Optional[np.ndarray]
    ) -> None:
        """
        Handle incoming keyframe packet from ZMQ (rtabmap.kf_packet).

        This is called from the ZMQ subscriber thread.
        Processes RGB-D → point cloud and broadcasts to clients.
        """
        try:
            # Debug: log pose translation for comparison with object positions
            if pose is not None:
                t_vis = pose[:3, 3]
                logger.debug(f"[visualization] KF {kf_id} pose translation: [{t_vis[0]:.3f}, {t_vis[1]:.3f}, {t_vis[2]:.3f}]")
            # Process RGB-D into point cloud
            positions, colors = self.processor.process(
                jpeg_bytes=jpeg_bytes,
                depth_bytes=depth_png_bytes,
                K=K,
                width=width,
                height=height,
                depth_scale=depth_scale,
                depth_encoding='png'
            )

            if positions.shape[0] == 0:
                logger.debug(f"[visualization] Keyframe {kf_id} produced 0 points, skipping")
                return

            mesh_id = kf_id

            # Check if we already have a pose for this keyframe
            existing = self.registry.get(mesh_id)
            if pose is None and existing is not None:
                pose = existing.pose

            # Register in registry
            self.registry.register(
                mesh_id=mesh_id,
                timestamp_ns=timestamp_ns,
                positions=positions,
                colors=colors,
                pose=pose,
                map_id=map_id
            )

            # Broadcast to clients (thread-safe via asyncio)
            if self._loop and self._running:
                asyncio.run_coroutine_threadsafe(
                    self.broadcaster.send_mesh_create(mesh_id, positions, colors, pose),
                    self._loop
                )

            stats = self.registry.stats()
            logger.debug(f"[visualization] KF {mesh_id}: {positions.shape[0]:,} pts | Total: {stats['keyframes']} KFs")

        except Exception as e:
            logger.error(f"[visualization] Error processing kf_packet {kf_id}: {e}")

    def handle_kf_pose(self, kf_id: str, pose: np.ndarray) -> None:
        """
        Handle initial pose for a keyframe (rtabmap.kf_pose).

        Called from ZMQ subscriber thread.
        """
        mesh_id = kf_id

        if self.registry.exists(mesh_id):
            self.registry.update_pose(mesh_id, pose)
            if self._loop and self._running:
                asyncio.run_coroutine_threadsafe(
                    self.broadcaster.send_mesh_update_pose(mesh_id, pose),
                    self._loop
                )
            logger.debug(f"[visualization] Pose set for existing KF {mesh_id}")
        else:
            # Cache pose for when keyframe arrives
            self.registry.register(
                mesh_id=mesh_id,
                timestamp_ns=0,
                positions=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                pose=pose,
                map_id="0"
            )
            logger.debug(f"[visualization] Pose cached for pending KF {mesh_id}")

    def handle_kf_pose_update(self, kf_id: str, pose: np.ndarray) -> None:
        """
        Handle pose correction from bundle adjustment or loop closure.

        Called from ZMQ subscriber thread.
        """
        mesh_id = kf_id

        if self.registry.update_pose(mesh_id, pose):
            if self._loop and self._running:
                asyncio.run_coroutine_threadsafe(
                    self.broadcaster.send_mesh_update_pose(mesh_id, pose),
                    self._loop
                )
            logger.debug(f"[visualization] Pose updated for KF {mesh_id} (BA/LC)")

    def _run_server(self) -> None:
        """Run uvicorn server in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._app = self._create_app()

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False
        )
        server = uvicorn.Server(config)

        self._loop.run_until_complete(server.serve())

    def start(self) -> None:
        """Start the visualization server in a background thread."""
        if self._server_thread is not None:
            logger.warning("[visualization] Server already started")
            return

        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="visualization-server"
        )
        self._server_thread.start()
        logger.info(f"[visualization] Server thread started (port {self.port})")

    def stop(self) -> None:
        """Stop the visualization server."""
        self._running = False
        # Note: Graceful shutdown of uvicorn in a thread is complex.
        # Since the thread is daemon, it will terminate with the process.
        logger.info("[visualization] Server stopped")
