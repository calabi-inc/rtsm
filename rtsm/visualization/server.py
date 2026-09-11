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
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rtsm.visualization.processor import PointCloudProcessor, ProcessorConfig
from rtsm.visualization.registry import KeyframeRegistry
from rtsm.visualization.broadcaster import WSBroadcaster

# TSDF fusion (optional � requires open3d)
try:
    from rtsm.visualization.tsdf_integrator import TSDFIntegrator, TSDFConfig
    _HAS_TSDF = True
except ImportError:
    _HAS_TSDF = False

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
        seg_analytics: Any = None,
        latency_analytics: Any = None,
        ingest_queue: Any = None,
        analytics_ticker: Any = None,
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
        self._ingest_queue = ingest_queue   # for the config echo (maxsize / policy)

        # Analytics buffers (optional) + the Tier-2 rollup owner. This server
        # only CONSUMES the buckets the ticker publishes (P1 task 5); buffers
        # without a ticker would mean either no rollup or a second owner.
        self._seg_analytics = seg_analytics
        self._latency_analytics = latency_analytics
        self._analytics_ticker = analytics_ticker
        if (seg_analytics is not None or latency_analytics is not None) and analytics_ticker is None:
            raise ValueError(
                "VisualizationServer: analytics buffers were passed without analytics_ticker; "
                "the Tier-2 rollup is owned by rtsm.analytics.AnalyticsTicker — build the buffers with "
                "rtsm.analytics.build_analytics(cfg, wm=...) and pass analytics_ticker=bundle.ticker"
            )

        # Visualization config
        vis_cfg = cfg.get("visualization", {})
        self._push_interval_ms = vis_cfg.get("objects", {}).get("push_interval_ms", 200)
        self._include_proto = vis_cfg.get("objects", {}).get("include_proto", True)

        # Analytics push config
        analytics_vis_cfg = vis_cfg.get("analytics", {})
        self._analytics_push_s = float(analytics_vis_cfg.get("push_interval_ms", 1000)) / 1000.0
        self._analytics_full_sync_interval = int(analytics_vis_cfg.get("full_sync_interval_s", 30))

        # Depth config for visualization
        depth_cfg = vis_cfg.get("depth", {})
        proc_depth_min = float(depth_cfg.get("min_m", 0.1))
        proc_depth_max = float(depth_cfg.get("max_m", 3.5))

        # Components
        self.processor = PointCloudProcessor(ProcessorConfig(
            depth_min_m=proc_depth_min,
            depth_max_m=proc_depth_max,
        ))
        self.registry = KeyframeRegistry()
        self.broadcaster = WSBroadcaster()

        # TSDF volumetric fusion (optional)
        self.tsdf = None
        tsdf_cfg = vis_cfg.get("tsdf", {})
        if tsdf_cfg.get("enable", False) and _HAS_TSDF:
            try:
                self.tsdf = TSDFIntegrator(
                    TSDFConfig(
                        voxel_size=float(tsdf_cfg.get("voxel_size", 0.01)),
                        sdf_trunc=float(tsdf_cfg.get("sdf_trunc", 0.04)),
                        max_depth_m=float(tsdf_cfg.get("max_depth_m", proc_depth_max)),
                        min_depth_m=proc_depth_min,
                        extract_every_n=int(tsdf_cfg.get("extract_every_n", 30)),
                        extract_interval_s=float(tsdf_cfg.get("extract_interval_s", 2.0)),
                        frame_buffer_size=int(tsdf_cfg.get("frame_buffer_size", 200)),
                        correction_reset_threshold_m=float(
                            tsdf_cfg.get("correction_reset_threshold_m", 0.05)
                        ),
                    ),
                )
                logger.info("[visualization] TSDF fusion enabled")
            except RuntimeError as e:
                logger.warning(f"[visualization] TSDF unavailable: {e}")
        elif tsdf_cfg.get("enable", False) and not _HAS_TSDF:
            logger.warning(
                "[visualization] TSDF enabled in config but open3d not installed"
            )

        # Max resolution for TSDF/visualization integration
        # (LiDAR depth is 256x192; upscaling to 1920x1440 RGB wastes compute)
        self._integration_max_width = int(tsdf_cfg.get("integration_max_width", 640))

        # Pose history for retroactive WM correction (frame_id -> T_wc 4x4)
        self._pose_history: Dict[str, np.ndarray] = {}
        self._pose_history_max = int(
            tsdf_cfg.get("frame_buffer_size", 200) * 2
        )  # keep slightly more history than TSDF buffer
        self._correct_wm = bool(
            cfg.get("slam", {}).get("correct_working_memory", True)
        )

        # Server state
        self._app: Optional[FastAPI] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._objects_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with WebSocket endpoint."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self._running = True
            # Start periodic WM objects push
            self._objects_task = asyncio.create_task(self._push_objects_loop())
            if self._seg_analytics or self._latency_analytics:
                self._analytics_task = asyncio.create_task(self._push_analytics_loop())
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
            if self._analytics_task:
                self._analytics_task.cancel()
                try:
                    await self._analytics_task
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

            # Sync existing keyframes/TSDF mesh to new client
            synced = await self.broadcaster.sync_new_client(websocket, self.registry)
            # Also send latest TSDF mesh if available
            if self.tsdf is not None:
                latest = self.tsdf.get_latest_mesh()
                if latest is not None:
                    positions, colors = latest
                    identity = np.eye(4, dtype=np.float32)
                    data = self.broadcaster._pack_mesh_create(
                        "tsdf_fused", positions, colors, identity
                    )
                    await self.broadcaster._try_send_bytes(websocket, data)
                    synced += 1
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
            result = {
                **self.registry.stats(),
                "clients": self.broadcaster.client_count,
            }
            if self.tsdf is not None:
                result.update(self.tsdf.stats())
            return result

        return app

    async def _handle_client_command(self, websocket: WebSocket, data: dict) -> None:
        """Handle commands from WebSocket clients."""
        cmd = data.get("cmd")

        if cmd == "clear":
            count = self.registry.clear()
            if self.tsdf is not None:
                self.tsdf.reset()
            await self.broadcaster._broadcast_json({"type": "clear"})
            logger.info(f"[visualization] Cleared {count} keyframes (TSDF reset: {self.tsdf is not None})")

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

    # ---- Runtime analytics push ----

    def _extract_analytics_config(self) -> dict:
        """Extract runtime config relevant to analytics dashboard."""
        seg = self.cfg.get("segmentation", {})
        io_cfg = self.cfg.get("io", {})
        ws = io_cfg.get("websocket", {})
        return {
            "segmentation": {
                "backend": seg.get("backend", "fastsam"),
                "retina_masks": seg.get("retina_masks", True),
                "fastsam": {k: seg.get("fastsam", {}).get(k) for k in ("model_path", "imgsz", "conf", "iou", "max_det")},
                "yoloe": {k: seg.get("yoloe", {}).get(k) for k in ("model_path", "imgsz", "conf", "iou", "max_det")},
                "dual": seg.get("dual", {}),
            },
            "receiver": {
                "type": io_cfg.get("receiver", "websocket"),
                "keyframe_every_n": ws.get("keyframe_every_n", 30),
                "nonkf_min_interval_s": ws.get("nonkf_min_interval_s", 0.5),
                "confidence_threshold": ws.get("confidence_threshold", 2),
                # From the live queue object: 512 (legacy), lossless_depth, or
                # keyframe_lane_depth + 1 (latest). 512 only if none was wired.
                "queue_maxsize": int(getattr(self._ingest_queue, "maxsize", 512) or 512),
                "ingest_policy": str(getattr(self._ingest_queue, "policy", "legacy")),
            },
            "pipeline": {
                "topk_preclip": self.cfg.get("staging", {}).get("topk_preclip", 15),
                "device": seg.get(seg.get("backend", "fastsam"), {}).get("device", "cuda"),
            },
            "sweep_policy": {
                "ttl_s": self.cfg.get("sweep_policy", {}).get("ttl_s", 2.0),
                "min_baseline_m": self.cfg.get("sweep_policy", {}).get("min_baseline_m", 0.08),
            },
        }

    async def _push_analytics_loop(self) -> None:
        """Push analytics to clients every ``visualization.analytics.push_interval_ms``.

        CONSUMER ONLY (P1 task 5): the Tier-1 -> Tier-2 rollup and the WM
        snapshot are done by the analytics ticker at 1 Hz whether or not a
        client is attached; this loop reads the ticks it has not pushed yet.

        Cursor protocol. A full sync (``mode: full``, the whole history since
        process start plus the config) is sent whenever the client count
        increased since the last push — so every attaching browser gets the
        history within one push interval — every ``full_sync_interval_s``
        pushes, and whenever the ticker's history no longer covers the cursor
        (a gap: the loop fell more than the ticker's retained ticks behind).
        Otherwise one ``append`` per new bucket (the frontend takes one bucket
        per message, so a push interval slower than the tick loses nothing),
        or one aggregate-only append (``bucket: null``) when no tick landed in
        this interval. The cursor is committed only after the messages went
        out, and a full sync reads the cursor BEFORE the history, so a bucket
        landing in between is sent twice at worst (the chart tolerates a
        duplicate x), never skipped.
        """
        ticker = self._analytics_ticker
        last_tick: Optional[int] = None
        ticks_since_full = 0
        prev_clients = 0

        while self._running:
            try:
                await asyncio.sleep(self._analytics_push_s)

                n_clients = self.broadcaster.client_count
                if n_clients == 0:
                    prev_clients = 0
                    continue

                ticks_since_full += 1
                is_full = (n_clients > prev_clients) or last_tick is None \
                    or ticks_since_full >= self._analytics_full_sync_interval
                prev_clients = n_clients

                new_ticks: list = []
                if not is_full and ticker is not None:
                    new_ticks = ticker.since(last_tick)
                    if new_ticks and new_ticks[0].tick != last_tick + 1:
                        is_full = True            # gap: the ticker's history no longer covers the cursor
                        new_ticks = []

                if is_full:
                    rec = ticker.latest() if ticker is not None else None
                    next_cursor = rec.tick if rec is not None else last_tick     # cursor first, history second
                    msgs = self._analytics_messages(True, [])
                else:
                    next_cursor = new_ticks[-1].tick if new_ticks else last_tick
                    msgs = self._analytics_messages(False, new_ticks)

                for msg in msgs:
                    await self.broadcaster._broadcast_json(msg)
                last_tick = next_cursor                # committed only after the sends
                if is_full:
                    ticks_since_full = 0

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("[visualization] analytics push error", exc_info=True)
                await asyncio.sleep(self._analytics_push_s)

    def _analytics_messages(self, is_full: bool, new_ticks: list) -> List[dict]:
        """Build the ``runtime_analytics`` messages for one push: one ``full``
        message (aggregates + whole history + config), or one ``append`` per
        tick record (``bucket`` = that tick's bucket, ``null`` for a buffer the
        record has no bucket for), or a single aggregate-only ``append`` when
        ``new_ticks`` is empty. Never rolls up, never snapshots the WM."""
        import dataclasses

        backend = self.cfg.get("segmentation", {}).get("backend", "fastsam")

        def _lat() -> dict:
            return {"aggregate": self._latency_analytics.aggregate()}

        def _seg() -> dict:
            return {"backend": backend, "aggregate": self._seg_analytics.aggregate()}

        if is_full:
            msg: dict = {"type": "runtime_analytics", "mode": "full",
                         "config": self._extract_analytics_config()}
            if self._latency_analytics:
                msg["latency"] = {**_lat(), "hourly": self._latency_analytics.hourly_history()}
            if self._seg_analytics:
                msg["segmentation"] = {**_seg(), "hourly": self._seg_analytics.hourly_history()}
            return [msg]

        out: List[dict] = []
        for rec in (list(new_ticks) or [None]):
            lat_b = getattr(rec, "latency", None) if rec is not None else None
            seg_b = getattr(rec, "seg", None) if rec is not None else None
            msg = {"type": "runtime_analytics", "mode": "append"}
            if self._latency_analytics:
                msg["latency"] = {**_lat(), "bucket": dataclasses.asdict(lat_b) if lat_b is not None else None}
            if self._seg_analytics:
                msg["segmentation"] = {**_seg(), "bucket": dataclasses.asdict(seg_b) if seg_b is not None else None}
            out.append(msg)
        return out

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

            # Broadcast to clients (thread-safe via broadcaster.schedule)
            if self._running:
                self.broadcaster.schedule(
                    self.broadcaster.send_mesh_create(mesh_id, positions, colors, pose)
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
            if self._running:
                self.broadcaster.schedule(
                    self.broadcaster.send_mesh_update_pose(mesh_id, pose)
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

        Called from ZMQ subscriber thread or WebSocket receiver.
        """
        mesh_id = kf_id

        if self.registry.update_pose(mesh_id, pose):
            if self._running:
                self.broadcaster.schedule(
                    self.broadcaster.send_mesh_update_pose(mesh_id, pose)
                )
            logger.debug(f"[visualization] Pose updated for KF {mesh_id} (BA/LC)")

    def handle_pose_corrections_batch(self, corrections: dict) -> None:
        """
        Handle a batch of pose corrections from loop closure.

        When TSDF is enabled, applies corrections to the TSDF volume.
        Otherwise delegates per-correction to handle_kf_pose_update.
        Also retroactively corrects WorkingMemory object positions.
        """
        if not corrections:
            return

        # ---- Correct WorkingMemory objects ----
        if self._correct_wm and self.wm is not None:
            self._apply_wm_pose_corrections(corrections)

        # ---- Correct TSDF / per-frame meshes ----
        if self.tsdf is not None:
            was_reset = self.tsdf.handle_pose_corrections(corrections)
            if was_reset:
                threading.Thread(
                    target=self._extract_and_broadcast_tsdf,
                    daemon=True,
                    name="tsdf-reintegrate-extract",
                ).start()
        else:
            for kf_id, corr_pose in corrections.items():
                self.handle_kf_pose_update(kf_id, corr_pose)

        # Update pose history with corrected poses
        for kf_id, new_pose in corrections.items():
            if isinstance(new_pose, np.ndarray):
                self._pose_history[kf_id] = new_pose.copy()

    def broadcast_camera_frame(self, pkt) -> None:
        """Broadcast camera JPEG to visualization clients.

        Lightweight — called for every decoded frame (not just keyframes)
        so the PiP overlay runs at full frame rate.
        """
        if not self._running:
            return
        jpeg_bytes = getattr(pkt, 'rgb_jpeg', None)
        if jpeg_bytes is not None:
            self.broadcaster.schedule(
                self.broadcaster.send_camera_frame(jpeg_bytes)
            )
        else:
            rgb = pkt.rgb
            if rgb is None:
                return
            _, jpeg_buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.broadcaster.schedule(
                self.broadcaster.send_camera_frame(jpeg_buf.tobytes())
            )

    def handle_frame_packet(self, pkt) -> None:
        """
        Handle a decoded FramePacket from the WebSocket receiver.

        Generates a point cloud from the already-decoded RGB + depth + pose
        and broadcasts it to visualization clients. Called from the WebSocket
        receiver thread for keyframes only.

        When TSDF is enabled, integrates frames into the TSDF volume and
        periodically extracts a fused point cloud.
        """
        try:
            if pkt.depth_m is None or pkt.pose is None or pkt.intr is None:
                return

            mesh_id = f"ws_{pkt.time.seq}"

            # Build 3x3 intrinsics matrix
            K = np.array([
                [pkt.intr.fx, 0, pkt.intr.cx],
                [0, pkt.intr.fy, pkt.intr.cy],
                [0, 0, 1],
            ], dtype=np.float32)

            rgb = pkt.rgb
            depth_m = pkt.depth_m

            # Apply additional confidence filtering for visualization quality
            # (pipeline already filters at ingestion, but vis may benefit from stricter threshold)
            confidence = getattr(pkt, 'confidence', None)
            if confidence is not None and depth_m is not None:
                dep_h, dep_w = depth_m.shape[:2]
                conf_h, conf_w = confidence.shape[:2]
                if conf_h != dep_h or conf_w != dep_w:
                    confidence = cv2.resize(
                        confidence, (dep_w, dep_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                # For visualization/TSDF, only use high-confidence depth (level 2)
                depth_m = depth_m.copy()
                depth_m[confidence < 2] = np.nan

            # Downsample to target resolution for efficiency
            # (LiDAR depth is 256x192; upscaling to 1920x1440 RGB wastes compute)
            rgb_h, rgb_w = rgb.shape[:2]
            max_w = self._integration_max_width
            if rgb_w > max_w:
                scale = max_w / rgb_w
                target_h = int(rgb_h * scale)
                rgb = cv2.resize(rgb, (max_w, target_h), interpolation=cv2.INTER_AREA)
                K = K.copy()
                K[0, 0] *= scale
                K[1, 1] *= scale
                K[0, 2] *= scale
                K[1, 2] *= scale
                rgb_h, rgb_w = target_h, max_w

            # Resize depth to match (downsampled) RGB if mismatched
            dep_h, dep_w = depth_m.shape[:2]
            if dep_h != rgb_h or dep_w != rgb_w:
                depth_m = cv2.resize(depth_m, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

            # Build 4x4 T_wc pose
            pose = pkt.pose.T_wc()

            # Store original pose for retroactive WM correction
            self._pose_history[mesh_id] = pose.copy()
            if len(self._pose_history) > self._pose_history_max:
                # Evict oldest entries (dict preserves insertion order in Python 3.7+)
                excess = len(self._pose_history) - self._pose_history_max
                for key in list(self._pose_history.keys())[:excess]:
                    del self._pose_history[key]

            # Note: ARKit→OpenCV camera flip is applied at ingestion
            # (WebSocketReceiver) so pose is already in OpenCV convention.

            # ---- TSDF path ----
            if self.tsdf is not None:
                # Convert BGR -> RGB (pkt.rgb is BGR from OpenCV decode)
                rgb_for_tsdf = rgb[:, :, ::-1].copy()

                should_extract = self.tsdf.integrate(
                    frame_id=mesh_id,
                    rgb=rgb_for_tsdf,
                    depth_m=depth_m,
                    K=K,
                    pose_T_wc=pose,
                    width=rgb_w,
                    height=rgb_h,
                    timestamp_ns=int(pkt.time.t_sensor_ns or 0),
                )

                if should_extract and self._running:
                    threading.Thread(
                        target=self._extract_and_broadcast_tsdf,
                        daemon=True,
                        name="tsdf-extract",
                    ).start()
                return  # Skip per-frame mesh path when TSDF is active

            # ---- Legacy per-frame path ----
            # Filter and back-project
            depth_filtered = self.processor.filter_depth(depth_m)
            positions, colors = self.processor.backproject(depth_filtered, rgb, K)

            if positions.shape[0] == 0:
                logger.debug(f"[visualization] Frame {mesh_id} produced 0 points, skipping")
                return

            # Debug: log pose translation
            t_vis = pose[:3, 3]
            logger.debug(f"[visualization] WS frame {mesh_id} pose: [{t_vis[0]:.3f}, {t_vis[1]:.3f}, {t_vis[2]:.3f}]")

            # Register and broadcast
            self.registry.register(
                mesh_id=mesh_id,
                timestamp_ns=int(pkt.time.t_sensor_ns or 0),
                positions=positions,
                colors=colors,
                pose=pose,
                map_id="websocket",
            )

            if self._running:
                self.broadcaster.schedule(
                    self.broadcaster.send_mesh_create(mesh_id, positions, colors, pose)
                )

            stats = self.registry.stats()
            logger.debug(f"[visualization] WS {mesh_id}: {positions.shape[0]:,} pts | Total: {stats['keyframes']} KFs")

        except Exception as e:
            logger.error(f"[visualization] Error processing frame packet: {e}")

    def _apply_wm_pose_corrections(self, corrections: dict) -> None:
        """
        Compute per-frame rigid correction deltas and apply them to WM objects.

        For each corrected frame where we have the original pose in history,
        computes delta_R, delta_t such that:  p_new = delta_R @ p_old + delta_t

        Passes a frame_id-keyed dict to wm.apply_pose_corrections so that
        objects are corrected using the exact delta from the frame that last
        updated them, with spatial proximity fallback for unlinked objects.
        """
        # Build frame_id -> (old_cam_pos, delta_R, delta_t)
        frame_corrections: Dict[str, tuple] = {}

        for kf_id, new_pose in corrections.items():
            old_pose = self._pose_history.get(kf_id)
            if old_pose is None:
                continue
            if not isinstance(new_pose, np.ndarray):
                continue

            old_pose = old_pose.astype(np.float64)
            new_pose_f64 = new_pose.astype(np.float64)

            # delta_T = T_wc_new @ inv(T_wc_old)
            # p_world_new = delta_T @ p_world_old  (homogeneous)
            try:
                delta_T = new_pose_f64 @ np.linalg.inv(old_pose)
            except np.linalg.LinAlgError:
                continue

            delta_R = delta_T[:3, :3].astype(np.float32)
            delta_t = delta_T[:3, 3].astype(np.float32)
            old_cam_pos = old_pose[:3, 3].astype(np.float32)

            frame_corrections[kf_id] = (old_cam_pos, delta_R, delta_t)

        if frame_corrections:
            try:
                n = self.wm.apply_pose_corrections(frame_corrections)
                logger.info(
                    f"[visualization] WM pose correction: {n} objects updated "
                    f"from {len(frame_corrections)} frame deltas"
                )
            except Exception as e:
                logger.error(f"[visualization] WM pose correction failed: {e}")

    def _extract_and_broadcast_tsdf(self) -> None:
        """Extract fused point cloud from TSDF and broadcast to clients."""
        try:
            positions, colors = self.tsdf.extract_point_cloud()
            if positions.shape[0] == 0:
                return

            # TSDF points are already in world frame; use identity pose
            identity = np.eye(4, dtype=np.float32)

            if self._running:
                self.broadcaster.schedule(
                    self.broadcaster.send_mesh_create(
                        "tsdf_fused", positions, colors, identity
                    )
                )

            logger.debug(
                f"[visualization] TSDF extracted {positions.shape[0]:,} pts "
                f"(v{self.tsdf.mesh_version})"
            )
        except Exception as e:
            logger.error(f"[visualization] TSDF extraction error: {e}")

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

    # ── Integrated mode (tasks on API server's loop) ──

    async def start_tasks(self) -> None:
        """Start periodic push tasks on the current event loop.

        Called from the API server's lifespan (integrated mode).
        The caller's loop becomes the broadcast loop.
        """
        if self._running:
            logger.warning("[visualization] Tasks already running")
            return
        self._running = True
        self.broadcaster._client_loop = asyncio.get_running_loop()
        self._objects_task = asyncio.create_task(self._push_objects_loop())
        if self._seg_analytics or self._latency_analytics:
            self._analytics_task = asyncio.create_task(self._push_analytics_loop())
        logger.info("[visualization] Periodic tasks started (integrated mode)")

    async def stop_tasks(self) -> None:
        """Cancel periodic tasks. Called from the API server's lifespan shutdown."""
        self._running = False
        for task in [self._objects_task, self._analytics_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._objects_task = None
        self._analytics_task = None
        logger.info("[visualization] Periodic tasks stopped")

    # ── Standalone mode (own uvicorn on dedicated port) ──

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
