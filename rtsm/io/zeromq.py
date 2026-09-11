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
from collections import OrderedDict
from typing import Optional, List, Callable, Any

import zmq
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

from rtsm.stores.frame_window import FrameWindow
from rtsm.core.datamodel import FramePacket, TimeBundle, PoseStamped, PinholeIntrinsics, IngestMeta
from rtsm.io.ingest_lanes import KF_SOURCE
from rtsm.io.ingest_queue import IngestQueue
from rtsm.utils.transforms import euler_to_quat_xyzw
from rtsm.evaluation.event_log import (
    RX_DROPPED, RX_DUPLICATE_TS, RX_ENQUEUED, RX_MALFORMED, RX_NO_CAMERA_FRAME, RX_PARSE_ERROR, RX_QUEUE_FULL,
    RX_THROTTLE, ReceiverEvent,
)


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
        latency_analytics: Optional[Any] = None,
        event_sink: Optional[Callable[[Any], None]] = None,
        throttle_clock: str = "wall",
        # ingest.nonkf_min_interval_s: min interval between admitted non-keyframe
        # (tracking) poses, on the throttle clock. Hardcoded at 0.5 until P1
        # task 6; the same default, now the runner passes the configured value.
        nonkf_min_interval_s: float = 0.5,
        # Receive-time robot-pose passthrough (WorkingMemory.update_robot_pose):
        # called for EVERY parsed tracking_pose at input rate (NOT for
        # kf_pose: a keyframe carries the node's pose, 50-300 ms behind the
        # tracking stream), as pose_sink(t_wc, q_xyzw, time.time(),
        # frame_epoch, sensor_ts_ns=stamp, pose_clock="server"). The epoch is
        # receiver-minted: 0, +1 whenever the tracking stamp goes backwards by
        # more than POSE_EPOCH_REBASE_S (a bag loop, a bridge restart).
        pose_sink: Optional[Callable[..., Any]] = None,
        # FrameWindow now holds ENCODED frames (JPEG / PNG bytes, ~0.35 MB per
        # 640x480 frame). Its old defaults (30 s, 2000 items) held ~1.9 GB of
        # decoded frames. The window must cover the LATENCY OF A kf_pose
        # STAMP behind the newest camera frame (RTAB-Map stamps a node when
        # it processes it: typically 50-300 ms, above 1 s during loop
        # closure / graph optimisation), not just the pairing slop -- an
        # unpaired keyframe is not retried. The TTL is meant to be the
        # binding bound, so max_items >= ttl * fps with margin (90 = 2 s *
        # 30 Hz * 1.5, <= ~32 MB); at 45+ fps the count binds first. The
        # runner passes `ingest.pair_window_s` and the count derived from it
        # and `ingest.pair_window_fps` (LaneConfig.pair_window_frames, P1 task 6).
        frame_window_ttl_s: float = 2.0,
        frame_window_max_items: int = 90,
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
            nonkf_min_interval_s: ingest.nonkf_min_interval_s (non-keyframe throttle)
            frame_window_ttl_s / frame_window_max_items: the pairing window
                (ingest.pair_window_s and the count LaneConfig derives from it)
        """
        self.camera_endpoint = camera_endpoint
        self.rtabmap_endpoint = rtabmap_endpoint
        self.ingest_q = ingest_queue

        # Unit normalization
        self._depth_scale = float(depth_m_per_unit)
        self._pose_scale = float(pose_m_per_unit)

        # Analytics
        self._latency_analytics = latency_analytics

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

        # Frame window for buffering camera data (encoded bytes; decoded only
        # after a pose is admitted — see _try_enqueue_frame / _decode_rgbd)
        self.fw = FrameWindow(ttl_sec=float(frame_window_ttl_s), max_items=int(frame_window_max_items))
        # Decode memo keyed by the MATCHED CAMERA stamp (not the pose stamp:
        # two poses within slop of one frame decode it once; a pose whose
        # nearest frame changes between calls does not get stale pixels).
        self._decode_cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._decode_cache_max = 4
        # Receiver-local running count of enqueue attempts (the only per-frame
        # id ZeroMQ frames have; TimeBundle.seq stays None so frame ids do not
        # become ws_*).
        self._rx_seq: int = 0

        # Track last enqueued timestamp to avoid duplicates
        self._last_enq_ts_ns: Optional[int] = None

        # Track latest pose for frame assembly
        self._last_pose_ts_ns: Optional[int] = None
        self._last_pose_t_wc: Optional[np.ndarray] = None
        self._last_pose_q_xyzw: Optional[np.ndarray] = None

        # Throttle non-keyframe enqueuing (pipeline can't keep up with 30Hz).
        # Stamps advance on the ADMIT decision, not on enqueue. "wall" compares
        # process time, "sensor" compares the pose timestamps (ingest.clock).
        self._throttle_clock = "sensor" if str(throttle_clock).lower() == "sensor" else "wall"
        self._last_nonkf_enq_mono: float = 0.0
        self._last_nonkf_admit_sensor_ns: Optional[int] = None
        # Frame-flow trace sink (ReceiverEvent per decision); None = off.
        self._event_sink = event_sink
        self._pose_sink = pose_sink
        self._kf_stamps_inherited: int = 0   # kf_pose messages that arrived without a stamp (counted in _handle_kf_pose)
        self._last_enq_cam_ts: Optional[int] = None   # camera stamp the last admitted frame was paired with
        # Receiver-minted session epoch (ZeroMQ has no hello/session_id): a
        # tracking stamp that jumps back by more than POSE_EPOCH_REBASE_S is a
        # restarted source -> new epoch on the pose mailbox, on every
        # FramePacket (SensorClock re-bases on it) and in liveness().
        self._frame_epoch: int = 0
        self._nonkf_min_interval_s: float = float(nonkf_min_interval_s)  # ingest.nonkf_min_interval_s

        # Frame-flow liveness stamps (read by the watchdog). The subscriber
        # thread is created externally; run.py assigns it to self._thread.
        self.last_rx_mono: Optional[float] = None
        self.last_enqueue_mono: Optional[float] = None
        self._thread: Optional[Any] = None

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
        if self._latency_analytics:
            self._latency_analytics.record_frame_received()

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

            # Buffer the ENCODED frame (admit-before-decode): JPEG bytes and
            # (PNG bytes, depth units). Decoding happens in _try_enqueue_frame
            # once a pose has paired with this frame AND the ingest queue has
            # room — a congested pipeline no longer costs a decode per 30 Hz
            # camera message that is never admitted.
            jpg_bytes = bytes(parts[2])
            png_bytes = bytes(parts[3])
            if not jpg_bytes or not png_bytes:
                logger.warning("[zeromq] camera.rgbd: empty RGB or depth payload")
                return
            self.fw.add_rgbd(ts_ns, jpg_bytes, (png_bytes, depth_units), intr)
            logger.debug(f"[zmq] camera.rgbd: buffered encoded frame ts={ts_ns}")

        except Exception as e:
            logger.error(f"[zeromq] camera.rgbd: parse error: {e}")

    def _parse_rtabmap_pose(self, json_data: dict) -> tuple[int, np.ndarray, np.ndarray]:
        """Parse an RTABMap pose (see _parse_rtabmap_pose_ex); 3-tuple form."""
        ts_ns, t_wc, q_xyzw, _stamped = self._parse_rtabmap_pose_ex(json_data)
        return ts_ns, t_wc, q_xyzw

    def _parse_rtabmap_pose_ex(self, json_data: dict) -> tuple[int, np.ndarray, np.ndarray, bool]:
        """
        Parse RTABMap pose from JSON.

        RTABMap format: T_wc = [x, y, z, roll, pitch, yaw] (Euler angles in radians)

        Returns:
            Tuple of (ts_ns, t_wc, q_xyzw, stamped). ``stamped`` is False when
            the message carried no stamp and ts_ns was INHERITED.
        """
        # Timestamp: tracking_pose carries stamp_ms; kf_pose carries kf_id only
        # (bridge-side gap). A stamp-less message inherits the last tracking
        # stamp (the keyframe IS the latest tracked pose), else the newest
        # camera frame in the window, and only as a last resort this
        # process's clock -- which used to be the FIRST resort and paired the
        # keyframe with whatever frame was newest while breaking the sensor
        # ordering the pose mailbox relies on.
        stamped = True
        if "stamp_ms" in json_data:
            ts_ns = int(json_data["stamp_ms"] * 1_000_000)  # ms to ns
        elif "ts_ns" in json_data:
            ts_ns = int(json_data["ts_ns"])
        else:
            stamped = False
            wm = int(getattr(self.fw, "watermark", 0) or 0)
            if self._last_pose_ts_ns is not None:
                ts_ns = int(self._last_pose_ts_ns)
                # That tracking pose may already have enqueued THIS image (it
                # passed the throttle): if a NEWER camera frame exists, pair
                # the keyframe with it so one image is not processed twice.
                # (Pairing is by nearest camera stamp, so compare on the
                # camera stamp the tracking pose actually used.) If the
                # enqueued image is still the newest, the keyframe re-processes
                # it -- one duplicate GPU pass per stamp-less keyframe that
                # follows a throttle-admitted tracking pose; the bridge-side
                # stamp_ms on kf_pose removes it.
                match = getattr(self.fw, "match_stamp", None)
                cam = match(ts_ns) if match is not None else None
                if cam is not None and cam == self._last_enq_cam_ts and wm > cam:
                    ts_ns = wm
            elif wm:
                ts_ns = wm
            else:
                # Nothing on the sensor clock is known yet; this value cannot
                # pair with any frame (the window is empty) and never reaches
                # the pose mailbox (kf_pose does not write the pose).
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

        return ts_ns, t_wc, q_xyzw, stamped

    def _handle_tracking_pose(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.tracking_pose message.

        Format: [b"rtabmap.tracking_pose", json_bytes]

        This is the continuous 30Hz pose stream. Triggers non-keyframe processing.
        """
        if len(parts) != 2:
            logger.warning(f"[zeromq] tracking_pose: expected 2 parts, got {len(parts)}")
            self._trace_rx(RX_DROPPED, RX_MALFORMED, None, False)
            return

        try:
            json_data = json.loads(parts[1].decode("utf-8"))
            ts_ns, t_wc, q_xyzw, _stamped = self._parse_rtabmap_pose_ex(json_data)

            # Restarted source? (bag loop, bridge restart): stamp went back by
            # more than POSE_EPOCH_REBASE_S -> new receiver-minted epoch. A
            # non-positive stamp is "no stamp" (as for SensorClock and the
            # mailbox) and neither bumps the epoch nor becomes the reference.
            last = self._last_pose_ts_ns
            if ts_ns > 0:
                if last is not None and ts_ns < int(last) - int(self.POSE_EPOCH_REBASE_S * 1e9):
                    self._frame_epoch += 1
                    logger.warning(
                        "[zeromq] tracking stamp jumped back %.1f s (%d -> %d): new frame_epoch %d",
                        (int(last) - ts_ns) / 1e9, int(last), ts_ns, self._frame_epoch,
                    )
                self._last_pose_ts_ns = ts_ns

            # Store latest pose
            self._last_pose_t_wc = t_wc
            self._last_pose_q_xyzw = q_xyzw

            # Receive-time robot pose (input rate, independent of admission)
            self._emit_pose(t_wc, q_xyzw, ts_ns)

            # Try to assemble non-keyframe
            self._try_enqueue_frame(ts_ns, t_wc, q_xyzw, is_keyframe=False)
            logger.debug(f"[zmq] tracking_pose: ts={ts_ns}")

        except Exception as e:
            logger.error(f"[zeromq] tracking_pose: parse error: {e}")
            self._trace_rx(RX_DROPPED, RX_MALFORMED, None, False)

    def _handle_kf_pose(self, parts: List[bytes]) -> None:
        """
        Handle rtabmap.kf_pose message.

        Format: [b"rtabmap.kf_pose", json_bytes]

        This signals a keyframe event. Triggers keyframe processing.
        """
        if len(parts) != 2:
            logger.warning(f"[zeromq] kf_pose: expected 2 parts, got {len(parts)}")
            self._trace_rx(RX_DROPPED, RX_MALFORMED, None, True)
            return

        try:
            json_data = json.loads(parts[1].decode("utf-8"))
            ts_ns, t_wc, q_xyzw, stamped = self._parse_rtabmap_pose_ex(json_data)

            # kf_id for logging/debugging
            kf_id = json_data.get("kf_id", -1)
            if not stamped:
                self._kf_stamps_inherited += 1

            # No pose-mailbox write for keyframes: the node pose lags the
            # tracking stream (50-300 ms, > 1 s in loop closure) and with an
            # inherited stamp it would REPLACE the fresher tracking pose at
            # the same key -- a periodic backwards blip an agent could read
            # as a discontinuity. tracking_pose already writes at input rate.

            # Enqueue as keyframe
            self._try_enqueue_frame(ts_ns, t_wc, q_xyzw, is_keyframe=True)
            logger.debug(f"[zmq] kf_pose: kf_id={kf_id}{'' if stamped else ' (stamp inherited)'}")

        except Exception as e:
            logger.error(f"[zeromq] kf_pose: parse error: {e}")
            self._trace_rx(RX_DROPPED, RX_MALFORMED, None, True)

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

            # Parse intrinsics (warn if falling back to defaults)
            intrinsics = metadata.get('intrinsics', {})
            if not intrinsics or 'fx' not in intrinsics:
                logger.warning("[zeromq] kf_packet missing intrinsics, using D435i defaults")
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
        self._rx_seq += 1
        rx_seq = self._rx_seq

        # Skip duplicates (except keyframes always get enqueued)
        if not is_keyframe and self._last_enq_ts_ns == ts_ns:
            self._trace_rx(RX_DROPPED, RX_DUPLICATE_TS, ts_ns, is_keyframe, rx_seq=rx_seq)
            return

        # Throttle non-keyframes to avoid overwhelming the pipeline. Check
        # first (cheap), assemble, then STAMP: a pose whose camera frame has
        # not arrived yet is not an admission and must not burn the window —
        # the next pose ~33 ms later retries. A full queue after the stamp
        # still thins (the stamp does not depend on put() succeeding).
        if not is_keyframe and not self._nonkf_due(ts_ns):
            self._trace_rx(RX_DROPPED, RX_THROTTLE, ts_ns, is_keyframe, rx_seq=rx_seq)
            return  # Skip, too soon since last non-KF

        # Assemble frame data from window (encoded bytes, not decoded arrays)
        rgb_raw, depth_raw, intr = self.fw.assemble_pair(ts_ns)
        if rgb_raw is None:
            # No matching camera frame yet
            self._trace_rx(RX_DROPPED, RX_NO_CAMERA_FRAME, ts_ns, is_keyframe, rx_seq=rx_seq)
            return

        # Paired: STAMP the non-KF now, before the queue check, so a refused
        # frame still burns the throttle window -- the same attempt-based
        # semantics as the websocket path (step 6 before 6b) and as main.
        # Otherwise every 30 Hz pose would probe a full queue, log a warning
        # and count a queue drop: ~15x the websocket receiver's queue_drops
        # for the same congestion.
        if not is_keyframe:
            self._stamp_nonkf(ts_ns)

        # Queue admission BEFORE the decode (admit-before-decode): ask the
        # queue whether it would refuse this frame (legacy: full; lanes: a
        # SOURCE keyframe meeting a full keyframe lane under overflow=reject),
        # so a frame about to be dropped costs no JPEG + PNG decode.
        kf_origin = KF_SOURCE if is_keyframe else None
        refusal = self.ingest_q.refusal(is_keyframe, kf_origin)
        if refusal is not None:
            if self._latency_analytics:
                self._latency_analytics.sample_queue_depth(self.ingest_q.qsize())
                self._latency_analytics.record_queue_drop()
            frame_type = "keyframe" if is_keyframe else "non-KF"
            logger.warning(f"[zeromq] ingest queue refused {frame_type} before decode ({refusal})")
            self._trace_rx(RX_DROPPED, refusal, ts_ns, is_keyframe, rx_seq=rx_seq)
            return

        # Memo key = the camera stamp the pair came from (duck-typed windows
        # without match_stamp fall back to the pose stamp).
        match = getattr(self.fw, "match_stamp", None)
        cam_ts = match(ts_ns) if match is not None else None
        decoded = self._decode_rgbd(cam_ts if cam_ts is not None else ts_ns, rgb_raw, depth_raw)
        if decoded is None:
            self._trace_rx(RX_DROPPED, RX_PARSE_ERROR, ts_ns, is_keyframe, rx_seq=rx_seq)
            return
        rgb, depth = decoded

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
            frame_epoch=self._frame_epoch,
            ingest=IngestMeta(keyframe_origin=kf_origin, rx_seq=rx_seq),
        )

        # Enqueue
        if self._latency_analytics:
            self._latency_analytics.sample_queue_depth(self.ingest_q.qsize())
        ok = self.ingest_q.put(fp, block=False)
        if ok:
            self.last_enqueue_mono = time.monotonic()
            self._last_enq_ts_ns = ts_ns
            self._last_enq_cam_ts = cam_ts
            frame_type = "KF" if is_keyframe else "frame"
            logger.debug(f"[zmq] enqueued {frame_type} -> queue={self.ingest_q.qsize()}")
            self._trace_rx(RX_ENQUEUED, "", ts_ns, is_keyframe, rx_seq=rx_seq,
                           lane=getattr(fp.ingest, "lane", None))
        else:
            if self._latency_analytics:
                self._latency_analytics.record_queue_drop()
            reason = getattr(fp.ingest, "drop_reason", None) or RX_QUEUE_FULL
            frame_type = "keyframe" if is_keyframe else "non-KF"
            logger.warning(f"[zeromq] ingest queue refused {frame_type} ({reason}); dropping")
            self._trace_rx(RX_DROPPED, reason, ts_ns, is_keyframe, rx_seq=rx_seq)

    # A tracking stamp that goes back by more than this within the stream is a
    # restarted source (bag loop, bridge restart) -> new receiver-minted epoch.
    # Same constant as SensorClock.rebase_after_s so the clock and the mailbox
    # agree on what a restart is.
    POSE_EPOCH_REBASE_S = 5.0

    def _emit_pose(self, t_wc: np.ndarray, q_xyzw: np.ndarray, ts_ns: int) -> None:
        """Receive-time pose write (tracking poses only). `timestamp` is this
        process's wall clock (RTAB-Map's stamp clock is not known to be unix),
        tagged pose_clock="server"; the mailbox key is (receiver-minted epoch,
        stamp). Never raises into the receive path."""
        if self._pose_sink is None:
            return
        try:
            self._pose_sink(t_wc, q_xyzw, time.time(), self._frame_epoch,
                            sensor_ts_ns=int(ts_ns), pose_clock="server")
        except Exception as e:
            logger.error(f"[zeromq] pose_sink callback error: {e}")

    def _decode_rgbd(self, ts_ns: int, rgb_raw: Any, depth_raw: Any):
        """Decode a paired camera frame after admission; memoised per CAMERA
        stamp (the stamp assemble_pair matched, see FrameWindow.match_stamp).

        Accepts already-decoded arrays too (tests, or a window filled by
        another producer). Returns (rgb_bgr, depth_m) or None on a decode
        failure.
        """
        hit = self._decode_cache.get(ts_ns)
        if hit is not None:
            self._decode_cache.move_to_end(ts_ns)
            return hit
        try:
            if isinstance(rgb_raw, np.ndarray):
                rgb = rgb_raw
            else:
                rgb = cv2.imdecode(np.frombuffer(rgb_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if rgb is None:
                    logger.warning("[zeromq] camera.rgbd: failed to decode JPEG")
                    return None
            if isinstance(depth_raw, np.ndarray) or depth_raw is None:
                depth_m = depth_raw
            else:
                png_bytes, depth_units = depth_raw
                depth_u16 = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if depth_u16 is None:
                    logger.warning("[zeromq] camera.rgbd: failed to decode PNG depth")
                    return None
                depth_m = depth_u16.astype(np.float32) * float(depth_units)
        except Exception as e:  # noqa: BLE001 — a bad frame must not kill the subscriber
            logger.warning(f"[zeromq] camera.rgbd: decode error: {e}")
            return None
        self._decode_cache[ts_ns] = (rgb, depth_m)
        while len(self._decode_cache) > self._decode_cache_max:
            self._decode_cache.popitem(last=False)
        return rgb, depth_m

    def _sensor_throttle_active(self, ts_ns: Optional[int]) -> bool:
        return self._throttle_clock == "sensor" and ts_ns is not None and int(ts_ns) > 0

    def _nonkf_due(self, ts_ns: Optional[int]) -> bool:
        """Min-interval throttle check for a non-keyframe (no side effects).
        sensor mode compares pose timestamps (negative delta = restarted
        clock -> due); wall mode compares process time."""
        interval = self._nonkf_min_interval_s
        if self._sensor_throttle_active(ts_ns):
            last = self._last_nonkf_admit_sensor_ns
            return not (last is not None and 0 <= (int(ts_ns) - last) < int(interval * 1e9))
        return (time.monotonic() - self._last_nonkf_enq_mono) >= interval

    def _stamp_nonkf(self, ts_ns: Optional[int]) -> None:
        """Record a non-keyframe ATTEMPT (called once the frame is paired,
        before the queue check and the decode): a refused frame still burns
        the window, exactly as on the websocket path."""
        if self._sensor_throttle_active(ts_ns):
            self._last_nonkf_admit_sensor_ns = int(ts_ns)
        else:
            self._last_nonkf_enq_mono = time.monotonic()

    def _trace_rx(self, decision: str, reason: str, ts_ns: Optional[int], is_keyframe: bool, *,
                  rx_seq: Optional[int] = None, lane: Optional[str] = None) -> None:
        """Frame-flow trace: one ReceiverEvent per receiver decision (no-op when
        no sink is set). ZeroMQ has no source seq; (t_sensor_ns, is_keyframe)
        is the join key, plus rx_seq (receiver-local attempt count) since P1
        task 3. Never raises into the receive path."""
        sink = self._event_sink
        if sink is None:
            return
        try:
            q = self.ingest_q
            sink(ReceiverEvent(
                timestamp=time.monotonic(),
                source="zeromq",
                decision=decision,
                reason=reason,
                frame_seq=None,
                t_sensor_ns=(int(ts_ns) if ts_ns is not None else None),
                is_keyframe=bool(is_keyframe),
                frame_count=None,
                queue_depth=(int(q.qsize()) if q is not None else None),
                lane=lane,
                rx_seq=rx_seq,
            ))
        except Exception:
            logger.debug("[zeromq] frame-flow trace failed", exc_info=True)

    def liveness(self) -> dict:
        """Frame-flow liveness snapshot for the watchdog."""
        t = self._thread
        return {
            "alive": bool(t is not None and t.is_alive()),
            "last_rx_mono": self.last_rx_mono,
            "last_enqueue_mono": self.last_enqueue_mono,
            "tracking_drops": 0,  # no tracking-state concept on the ZMQ path
            "frame_epoch": self._frame_epoch,
            "kf_stamps_inherited": self._kf_stamps_inherited,   # bridge-side gap: kf_pose without stamp_ms
        }

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
                    self.last_rx_mono = time.monotonic()
                    parts = self.camera_sock.recv_multipart()
                    topic = parts[0].decode(errors="ignore")
                    if topic == "camera.rgbd":
                        self._handle_camera_rgbd(parts)

                # Handle RTABMap messages
                if self.rtabmap_sock in socks:
                    self.last_rx_mono = time.monotonic()
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
