"""
ZeroMQ Subscriber for RTSM - RTABMap Bridge Integration.

Subscribes to:
- D435i camera (port 5555): camera.rgbd topic (bundled RGB+depth+intrinsics)
- RTABMap bridge (port 6000): rtabmap.tracking_pose, rtabmap.kf_pose topics
- (Optional) RTABMap bridge: rtabmap.kf_packet, rtabmap.kf_pose_update for visualization

Forms canonical FramePacket objects and enqueues them to the ingest queue.
"""

from __future__ import annotations
import time
import sys
import json
import math
from typing import Optional, List, Callable, Any

import zmq
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

from rtsm.stores.frame_window import FrameWindow
from rtsm.core.datamodel import FramePacket, TimeBundle, PoseStamped, PinholeIntrinsics
from rtsm.io.ingest_queue import IngestQueue
from rtsm.utils.transforms import euler_to_quat_xyzw


class ZeroMQSubscriber:
    """
    Subscribes to D435i camera and RTABMap bridge via dual ZMQ sockets.
    Forms canonical FramePacket objects and enqueues them to the ingest queue.
    """

    def __init__(
        self,
        camera_endpoint: str = "tcp://127.0.0.1:5555",
        rtabmap_endpoint: str = "tcp://127.0.0.1:6000",
        ingest_queue: Optional[IngestQueue] = None,
        *,
        depth_m_per_unit: float = 0.001,
        pose_m_per_unit: float = 1.0,
        # Optional visualization callbacks
        on_kf_packet: Optional[Callable[..., Any]] = None,
        on_kf_pose_update: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Initialize dual-socket ZMQ subscriber.

        Args:
            camera_endpoint: ZMQ endpoint for D435i camera (camera.rgbd topic)
            rtabmap_endpoint: ZMQ endpoint for RTABMap bridge (pose topics)
            ingest_queue: Queue to push FramePackets to
            depth_m_per_unit: Scale factor for depth values (default 0.001 for mm to m)
            pose_m_per_unit: Scale factor for pose translation (default 1.0, already in meters)
            on_kf_packet: Optional callback for rtabmap.kf_packet (visualization)
            on_kf_pose_update: Optional callback for rtabmap.kf_pose_update (visualization)
        """
        self.camera_endpoint = camera_endpoint
        self.rtabmap_endpoint = rtabmap_endpoint
        self.ingest_q = ingest_queue

        # Unit normalization
        self._depth_scale = float(depth_m_per_unit)
        self._pose_scale = float(pose_m_per_unit)

        # Visualization callbacks
        self._on_kf_packet = on_kf_packet
        self._on_kf_pose_update = on_kf_pose_update

        # ZMQ context and sockets
        self.ctx = zmq.Context()

        # Camera socket (D435i)
        self.camera_sock = self.ctx.socket(zmq.SUB)
        self.camera_sock.connect(self.camera_endpoint)
        self.camera_sock.setsockopt(zmq.SUBSCRIBE, b"camera.rgbd")

        # RTABMap socket
        self.rtabmap_sock = self.ctx.socket(zmq.SUB)
        self.rtabmap_sock.connect(self.rtabmap_endpoint)
        self.rtabmap_sock.setsockopt(zmq.SUBSCRIBE, b"rtabmap.tracking_pose")
        self.rtabmap_sock.setsockopt(zmq.SUBSCRIBE, b"rtabmap.kf_pose")

        # Conditionally subscribe to visualization topics
        if on_kf_packet:
            self.rtabmap_sock.setsockopt(zmq.SUBSCRIBE, b"rtabmap.kf_packet")
            logger.info("[zeromq] Subscribed to rtabmap.kf_packet for visualization")
        if on_kf_pose_update:
            self.rtabmap_sock.setsockopt(zmq.SUBSCRIBE, b"rtabmap.kf_pose_update")
            logger.info("[zeromq] Subscribed to rtabmap.kf_pose_update for visualization")

        # Poller for both sockets
        self.poller = zmq.Poller()
        self.poller.register(self.camera_sock, zmq.POLLIN)
        self.poller.register(self.rtabmap_sock, zmq.POLLIN)

        # Frame window for buffering camera data
        self.fw = FrameWindow()

        # Track last enqueued timestamp to avoid duplicates
        self._last_enq_ts_ns: Optional[int] = None

        # Track latest pose for frame assembly
        self._last_pose_ts_ns: Optional[int] = None
        self._last_pose_t_wc: Optional[np.ndarray] = None
        self._last_pose_q_xyzw: Optional[np.ndarray] = None

        # Throttle non-keyframe enqueuing (pipeline can't keep up with 30Hz)
        self._last_nonkf_enq_mono: float = 0.0
        self._nonkf_min_interval_s: float = 0.5  # Max ~2 non-KF per second

    def close(self):
        """Clean up ZMQ resources."""
        try:
            self.camera_sock.close(0)
        except Exception:
            pass
        try:
            self.rtabmap_sock.close(0)
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass

    def _handle_camera_rgbd(self, parts: List[bytes]) -> None:
        """
        Handle camera.rgbd message from D435i.

        Format: [b"camera.rgbd", json_metadata, jpeg_bytes, png_bytes]

        json_metadata: {
            "ts_ns": int,
            "intrinsics": {"fx", "fy", "cx", "cy", "width", "height"},
            "depth_units_m": float,
            "encoding": {"rgb": "jpeg", "depth": "png_u16"}
        }
        """
        if len(parts) != 4:
            logger.warning(f"[zeromq] camera.rgbd: expected 4 parts, got {len(parts)}")
            return

        try:
            # Parse JSON metadata
            meta = json.loads(parts[1].decode("utf-8"))
            ts_ns = int(meta["ts_ns"])
            intr_data = meta["intrinsics"]
            depth_units = float(meta.get("depth_units_m", self._depth_scale))

            # Build intrinsics object
            intr = PinholeIntrinsics(
                width=int(intr_data["width"]),
                height=int(intr_data["height"]),
                fx=float(intr_data["fx"]),
                fy=float(intr_data["fy"]),
                cx=float(intr_data["cx"]),
                cy=float(intr_data["cy"]),
            )

            # Decode JPEG RGB
            jpg_buf = np.frombuffer(parts[2], dtype=np.uint8)
            rgb = cv2.imdecode(jpg_buf, cv2.IMREAD_COLOR)
            if rgb is None:
                logger.warning("[zeromq] camera.rgbd: failed to decode JPEG")
                return

            # Decode PNG depth (16-bit uint16)
            png_buf = np.frombuffer(parts[3], dtype=np.uint8)
            depth_u16 = cv2.imdecode(png_buf, cv2.IMREAD_UNCHANGED)
            if depth_u16 is None:
                logger.warning("[zeromq] camera.rgbd: failed to decode PNG depth")
                return

            # Convert depth to meters
            depth_m = depth_u16.astype(np.float32) * depth_units

            # Add to frame window with intrinsics
            self.fw.add_rgbd(ts_ns, rgb, depth_m, intr)
            logger.debug(f"[zmq] camera.rgbd: buffered frame ts={ts_ns}")

        except Exception as e:
            logger.error(f"[zeromq] camera.rgbd: parse error: {e}")

    def _parse_rtabmap_pose(self, json_data: dict) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Parse RTABMap pose from JSON.

        RTABMap format: T_wc = [x, y, z, roll, pitch, yaw] (Euler angles in radians)

        Returns:
            Tuple of (ts_ns, t_wc, q_xyzw)
        """
        # Get timestamp - RTABMap uses stamp_ms for tracking_pose, kf_id for kf_pose
        if "stamp_ms" in json_data:
            ts_ns = int(json_data["stamp_ms"] * 1_000_000)  # ms to ns
        elif "ts_ns" in json_data:
            ts_ns = int(json_data["ts_ns"])
        else:
            # Use current time as fallback
            ts_ns = int(time.time_ns())

        # Parse pose [x, y, z, roll, pitch, yaw]
        T_wc = json_data["T_wc"]
        x, y, z = float(T_wc[0]), float(T_wc[1]), float(T_wc[2])
        roll, pitch, yaw = float(T_wc[3]), float(T_wc[4]), float(T_wc[5])

        # Apply pose scale
        t_wc = np.array([x, y, z], dtype=np.float32) * self._pose_scale

        # Debug: log RAW pose from RTABMap (periodically to avoid spam)
        if not hasattr(self, '_pose_log_count'):
            self._pose_log_count = 0
        self._pose_log_count += 1
        if self._pose_log_count % 30 == 1:  # Log every 30th pose (~1 per second at 30Hz)
            logger.debug(f"[zmq] RAW T_wc from rtabmap: {T_wc}")
            logger.debug(f"[zmq] parsed: xyz=[{x:.4f},{y:.4f},{z:.4f}] rpy=[{roll:.3f},{pitch:.3f},{yaw:.3f}]")

        # Convert Euler to quaternion
        q_xyzw = euler_to_quat_xyzw(roll, pitch, yaw)

        return ts_ns, t_wc, q_xyzw

    def _handle_tracking_pose(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.tracking_pose message.

        Format: [b"rtabmap.tracking_pose", json_bytes]

        This is the continuous 30Hz pose stream. Triggers non-keyframe processing.
        """
        if len(parts) != 2:
            logger.warning(f"[zeromq] tracking_pose: expected 2 parts, got {len(parts)}")
            return

        try:
            json_data = json.loads(parts[1].decode("utf-8"))
            ts_ns, t_wc, q_xyzw = self._parse_rtabmap_pose(json_data)

            # Store latest pose
            self._last_pose_ts_ns = ts_ns
            self._last_pose_t_wc = t_wc
            self._last_pose_q_xyzw = q_xyzw

            # Try to assemble non-keyframe
            self._try_enqueue_frame(ts_ns, t_wc, q_xyzw, is_keyframe=False)
            logger.debug(f"[zmq] tracking_pose: ts={ts_ns}")

        except Exception as e:
            logger.error(f"[zeromq] tracking_pose: parse error: {e}")

    def _handle_kf_pose(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.kf_pose message.

        Format: [b"rtabmap.kf_pose", json_bytes]

        This signals a keyframe event. Triggers keyframe processing.
        """
        if len(parts) != 2:
            logger.warning(f"[zeromq] kf_pose: expected 2 parts, got {len(parts)}")
            return

        try:
            json_data = json.loads(parts[1].decode("utf-8"))
            ts_ns, t_wc, q_xyzw = self._parse_rtabmap_pose(json_data)

            # kf_id for logging/debugging
            kf_id = json_data.get("kf_id", -1)

            # Enqueue as keyframe
            self._try_enqueue_frame(ts_ns, t_wc, q_xyzw, is_keyframe=True)
            logger.debug(f"[zmq] kf_pose: kf_id={kf_id}")

        except Exception as e:
            logger.error(f"[zeromq] kf_pose: parse error: {e}")

    def _handle_kf_packet(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.kf_packet message for visualization.

        Format: [topic, json, rgb_jpeg, depth_png]

        JSON structure:
        {
            "kf_id": 123,
            "ts_ns": 1234567890000000,
            "map_id": 0,
            "T_w_c": {"t": [x,y,z], "q": [qx,qy,qz,qw]} or [x,y,z,roll,pitch,yaw],
            "intrinsics": {"fx", "fy", "cx", "cy", "width", "height"},
            "depth_units_m": 0.001
        }
        """
        if not self._on_kf_packet:
            return

        if len(parts) < 4:
            # kf_packet without images - skip silently
            return

        try:
            metadata = json.loads(parts[1].decode('utf-8'))

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

            depth_scale = metadata.get('depth_units_m', 0.001)

            # Parse pose
            pose = None
            if 'T_w_c' in metadata:
                pose = self._parse_vis_pose(metadata['T_w_c'])

            # Binary parts
            jpeg_bytes = bytes(parts[2])
            depth_png_bytes = bytes(parts[3])

            # Invoke callback
            self._on_kf_packet(
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

        except Exception as e:
            logger.error(f"[zeromq] kf_packet: parse error: {e}")

    def _handle_kf_pose_update(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.kf_pose_update message for visualization.

        Format: [topic, json]

        JSON structure:
        {
            "kf_id": 123,
            "T_wc": [x, y, z, roll, pitch, yaw]
        }
        """
        if not self._on_kf_pose_update:
            return

        if len(parts) < 2:
            return

        try:
            metadata = json.loads(parts[1].decode('utf-8'))
            kf_id = str(metadata.get('kf_id', 0))

            pose = self._parse_vis_pose(metadata.get('T_wc'))
            if pose is None:
                logger.warning("[zeromq] kf_pose_update: invalid pose")
                return

            self._on_kf_pose_update(kf_id=kf_id, pose=pose)

        except Exception as e:
            logger.error(f"[zeromq] kf_pose_update: parse error: {e}")

    def _parse_vis_pose(self, pose_data) -> Optional[np.ndarray]:
        """
        Parse pose for visualization (returns 4x4 matrix).

        Supports:
        - [x, y, z, roll, pitch, yaw] (6 floats)
        - {"t": [x,y,z], "q": [qx,qy,qz,qw]} (quaternion dict)
        """
        if isinstance(pose_data, list) and len(pose_data) == 6:
            return self._euler_to_matrix(*pose_data)
        elif isinstance(pose_data, dict) and 't' in pose_data and 'q' in pose_data:
            return self._quat_to_matrix(pose_data['t'], pose_data['q'])
        return None

    def _euler_to_matrix(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert position + euler angles to 4x4 matrix."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

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

    def _quat_to_matrix(self, t: list, q: list) -> np.ndarray:
        """Convert translation + quaternion to 4x4 matrix."""
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

    def _try_enqueue_frame(
        self,
        ts_ns: int,
        t_wc: np.ndarray,
        q_xyzw: np.ndarray,
        is_keyframe: bool,
    ) -> None:
        """
        Try to assemble and enqueue a FramePacket.

        Looks up RGB/depth/intrinsics from FrameWindow by timestamp.
        """
        if self.ingest_q is None:
            return

        # Skip duplicates (except keyframes always get enqueued)
        if not is_keyframe and self._last_enq_ts_ns == ts_ns:
            return

        # Throttle non-keyframes to avoid overwhelming the pipeline
        now_mono = time.monotonic()
        if not is_keyframe:
            elapsed = now_mono - self._last_nonkf_enq_mono
            if elapsed < self._nonkf_min_interval_s:
                return  # Skip, too soon since last non-KF

        # Assemble frame data from window
        rgb, depth, intr = self.fw.assemble_pair(ts_ns)
        if rgb is None:
            # No matching camera frame yet
            return

        # Build pose
        pose = PoseStamped(
            stamp_ns=ts_ns,
            frame_id="world",
            t_wc=t_wc,
            q_wc_xyzw=q_xyzw,
        )

        # Build time bundle
        tb = TimeBundle(
            t_mono_s=time.monotonic(),
            t_wall_utc_s=time.time(),
            t_sensor_ns=ts_ns,
            seq=None,
        )

        # Build frame packet
        fp = FramePacket(
            time=tb,
            rgb=rgb,
            depth_m=depth,
            pose=pose,
            intr=intr,
            is_keyframe=is_keyframe,
        )

        # Enqueue
        ok = self.ingest_q.put(fp, block=False)
        if ok:
            self._last_enq_ts_ns = ts_ns
            if not is_keyframe:
                self._last_nonkf_enq_mono = now_mono
            frame_type = "KF" if is_keyframe else "frame"
            logger.debug(f"[zmq] enqueued {frame_type} -> queue={self.ingest_q.qsize()}")
        else:
            frame_type = "keyframe" if is_keyframe else "non-KF"
            logger.warning(f"[zeromq] ingest queue full; dropping {frame_type}")

    def run_forever(self) -> None:
        """Main loop: poll both sockets and dispatch messages."""
        vis_topics = []
        if self._on_kf_packet:
            vis_topics.append("rtabmap.kf_packet")
        if self._on_kf_pose_update:
            vis_topics.append("rtabmap.kf_pose_update")

        vis_info = f", {', '.join(vis_topics)}" if vis_topics else ""

        logger.info(
            f"[zeromq] Starting dual-socket subscriber:\n"
            f"  Camera: {self.camera_endpoint} (camera.rgbd)\n"
            f"  RTABMap: {self.rtabmap_endpoint} (rtabmap.tracking_pose, rtabmap.kf_pose{vis_info})"
        )

        try:
            while True:
                # Poll with 100ms timeout
                socks = dict(self.poller.poll(100))

                # Handle camera messages
                if self.camera_sock in socks:
                    parts = self.camera_sock.recv_multipart()
                    topic = parts[0].decode(errors="ignore")
                    if topic == "camera.rgbd":
                        self._handle_camera_rgbd(parts)

                # Handle RTABMap messages
                if self.rtabmap_sock in socks:
                    parts = self.rtabmap_sock.recv_multipart()
                    topic = parts[0].decode(errors="ignore")
                    if topic == "rtabmap.tracking_pose":
                        self._handle_tracking_pose(parts)
                    elif topic == "rtabmap.kf_pose":
                        self._handle_kf_pose(parts)
                    elif topic == "rtabmap.kf_packet":
                        self._handle_kf_packet(parts)
                    elif topic == "rtabmap.kf_pose_update":
                        self._handle_kf_pose_update(parts)

        except KeyboardInterrupt:
            logger.info("[zeromq] Shutting down...")
        finally:
            self.close()


# Smoke test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG)

    p = argparse.ArgumentParser(description="ZeroMQ dual-socket subscriber for RTSM")
    p.add_argument("--camera", default="tcp://127.0.0.1:5555", help="Camera endpoint")
    p.add_argument("--rtabmap", default="tcp://127.0.0.1:6000", help="RTABMap endpoint")
    args = p.parse_args()

    sub = ZeroMQSubscriber(
        camera_endpoint=args.camera,
        rtabmap_endpoint=args.rtabmap,
    )
    sub.run_forever()
