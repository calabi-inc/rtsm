"""
WebSocket Receiver for RTSM — Calabi Lens ARKit Client Integration.

Accepts binary frames over ws://host:port/stream from the Calabi Lens iOS app.
Each frame contains bundled RGB + depth + pose + intrinsics.

Protocol:
1. Client connects to /stream
2. Client sends JSON text hello: {"type": "hello", "protocol_version": 1, ...}
3. Server replies with JSON text ack: {"type": "hello_ack", "status": "ok", ...}
4. Client streams binary frames (length-prefixed: [json][rgb][depth])
"""

from __future__ import annotations
import asyncio
import json
import struct
import time
import threading
import logging
from typing import Any, Optional, Tuple

import numpy as np
import cv2

from rtsm.core.datamodel import (
    FramePacket, TimeBundle, PoseStamped, PinholeIntrinsics,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from rtsm.io.ingest_queue import IngestQueue
from rtsm.utils.transforms import rotmat_to_quat_xyzw

logger = logging.getLogger(__name__)

# Current handshake protocol version
PROTOCOL_VERSION = 1

# Supported format values
_RGB_FORMATS = {"jpeg", "png", "bgra", "nv12", "h264"}
_DEPTH_FORMATS = {"uint16_mm", "float32_m", "png_uint16"}
_POSE_FORMATS = {"matrix4x4_col_major", "quat_translation"}

# ARKit camera (Y-up, Z-toward-viewer) → OpenCV camera (Y-down, Z-forward)
# Right-multiplying T_wc by this converts camera columns from ARKit to OpenCV
# convention. diag(1, -1, -1, 1) is its own inverse.
_ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


# ─────────────────── Data Conversion Functions ───────────────────


def decode_rgb(raw: bytes, fmt: str, width: int, height: int) -> np.ndarray:
    """
    Decode RGB bytes into (H, W, 3) uint8 BGR array (OpenCV convention).

    Args:
        raw: Raw byte payload.
        fmt: ``"jpeg"`` | ``"png"`` | ``"bgra"`` | ``"nv12"``.
        width, height: Expected image dimensions.

    Returns:
        (H, W, 3) uint8 BGR array.

    Raises:
        ValueError: On unsupported format or decode failure.
    """
    if fmt in ("jpeg", "png"):
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for {fmt} ({len(raw)} bytes)")
        return img
    elif fmt == "bgra":
        expected = height * width * 4
        if len(raw) != expected:
            raise ValueError(
                f"bgra buffer size mismatch: got {len(raw)}, "
                f"expected {expected} ({width}x{height}x4)"
            )
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif fmt == "nv12":
        # NV12: Y plane (H*W) + interleaved UV plane (H/2 * W)
        # Total bytes = H * W * 3 / 2
        expected = height * width * 3 // 2
        if len(raw) != expected:
            raise ValueError(
                f"nv12 buffer size mismatch: got {len(raw)}, "
                f"expected {expected} ({width}x{height} * 1.5)"
            )
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    else:
        raise ValueError(f"Unsupported rgb_format: {fmt!r}")


def decode_depth(
    raw: bytes,
    fmt: Optional[str],
    width: int,
    height: int,
    depth_scale: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Decode depth bytes into (H, W) float32 meters with NaN for invalid pixels.

    Wire convention: 0 = invalid.  After this function: NaN = invalid.

    Args:
        raw: Raw byte payload (may be empty if no depth).
        fmt: ``"uint16_mm"`` | ``"float32_m"`` | ``"png_uint16"`` | ``None``.
        width, height: Expected depth dimensions.
        depth_scale: Scale factor (0.001 for mm→m).  Used for uint16 formats.

    Returns:
        (H, W) float32 array in meters (NaN = invalid), or ``None``.
    """
    if fmt is None or len(raw) == 0:
        return None

    if depth_scale is None:
        depth_scale = 0.001  # default: millimeters

    if fmt == "uint16_mm":
        depth_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
        depth_m = depth_u16.astype(np.float32) * depth_scale
        depth_m[depth_u16 == 0] = np.nan
        return depth_m

    elif fmt == "float32_m":
        depth_m = np.frombuffer(raw, dtype=np.float32).reshape(height, width).copy()
        depth_m[depth_m == 0.0] = np.nan
        return depth_m

    elif fmt == "png_uint16":
        buf = np.frombuffer(raw, dtype=np.uint8)
        depth_u16 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            logger.warning("[websocket] failed to decode PNG depth")
            return None
        depth_m = depth_u16.astype(np.float32) * depth_scale
        depth_m[depth_u16 == 0] = np.nan
        return depth_m

    else:
        logger.warning(f"[websocket] unsupported depth_format: {fmt!r}")
        return None


def parse_arkit_pose(
    T_wc_data: list,
    pose_format: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ARKit pose into canonical (t_wc, q_xyzw) format.

    Args:
        T_wc_data: 16 floats (col-major 4×4) or 7 floats [qx, qy, qz, qw, tx, ty, tz].
        pose_format: ``"matrix4x4_col_major"`` | ``"quat_translation"``.

    Returns:
        ``(t_wc, q_xyzw)`` — both ``np.float32`` arrays, shapes (3,) and (4,).

    Raises:
        ValueError: On unsupported format or wrong element count.
    """
    if pose_format == "quat_translation":
        if len(T_wc_data) != 7:
            raise ValueError(
                f"quat_translation expects 7 elements, got {len(T_wc_data)}"
            )
        qx, qy, qz, qw, tx, ty, tz = (float(v) for v in T_wc_data)
        t_wc = np.array([tx, ty, tz], dtype=np.float32)
        q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
        return t_wc, q_xyzw

    elif pose_format == "matrix4x4_col_major":
        if len(T_wc_data) != 16:
            raise ValueError(
                f"matrix4x4_col_major expects 16 elements, got {len(T_wc_data)}"
            )
        # Column-major → numpy array with Fortran order
        mat = np.array(T_wc_data, dtype=np.float64).reshape(4, 4, order="F")
        t_wc = mat[:3, 3].astype(np.float32)
        R = mat[:3, :3].astype(np.float32)
        q_xyzw = rotmat_to_quat_xyzw(R)
        return t_wc, q_xyzw

    else:
        raise ValueError(f"Unsupported pose_format: {pose_format!r}")


# ─────────────────── H.264 Stateful Decoder ───────────────────


class H264Decoder:
    """Stateful H.264 decoder using PyAV.

    Maintains codec context across frames for P-frame reference tracking.
    Create one instance per WebSocket connection or replay session.
    """

    def __init__(self):
        import av  # lazy import — only needed when H.264 frames arrive

        self._codec = av.CodecContext.create("h264", "r")

    def decode(self, h264_bytes: bytes) -> Optional[np.ndarray]:
        """Decode H.264 NAL units to BGR numpy array.

        Args:
            h264_bytes: Raw H.264 NAL unit bytes (one or more NAL units).

        Returns:
            (H, W, 3) uint8 BGR array (OpenCV convention), or ``None``
            if the packet produced no decodable frame (e.g. SPS/PPS only).
        """
        import av

        packet = av.Packet(h264_bytes)
        frames = self._codec.decode(packet)
        if not frames:
            return None
        return frames[0].to_ndarray(format="bgr24")


# ─────────────────── WebSocket Receiver Class ───────────────────


class WebSocketReceiver:
    """
    WebSocket receiver for Calabi Lens ARKit iOS client.

    Accepts binary frames over ``ws://host:port/stream``, decodes them,
    and enqueues canonical ``FramePacket`` objects to the ingest queue.
    """

    def __init__(
        self,
        ingest_queue: IngestQueue,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        require_tracking_normal: bool = True,
        keyframe_every_n: int = 30,
        nonkf_min_interval_s: float = 0.5,
        confidence_threshold: int = 1,
        apply_camera_flip: bool = False,
        on_keyframe: Optional[callable] = None,
        on_camera_frame: Optional[callable] = None,
        on_pose_corrections: Optional[callable] = None,
        on_pose_corrections_batch: Optional[callable] = None,
        on_raw_message: Optional[callable] = None,
        on_handshake_done: Optional[callable] = None,
        latency_analytics: Optional[Any] = None,
    ) -> None:
        self.ingest_q = ingest_queue
        self._host = host
        self._port = port
        self._require_tracking_normal = require_tracking_normal
        self._keyframe_every_n = max(1, keyframe_every_n)
        self._nonkf_min_interval_s = nonkf_min_interval_s
        self._confidence_threshold = confidence_threshold
        self._apply_camera_flip = apply_camera_flip
        self._on_keyframe = on_keyframe
        self._on_camera_frame = on_camera_frame
        self._on_pose_corrections = on_pose_corrections
        self._on_pose_corrections_batch = on_pose_corrections_batch
        self._on_raw_message = on_raw_message
        self._on_handshake_done = on_handshake_done
        self._latency_analytics = latency_analytics

        # Per-session state (reset on each new client connection)
        self._frame_count: int = 0
        self._last_nonkf_enq_mono: float = 0.0
        self._last_enq_ts_ns: Optional[int] = None
        self._active_session_id: Optional[str] = None
        self._h264_decoder: Optional[H264Decoder] = None

        # Threading
        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_event = asyncio.Event()

    # ── FastAPI app creation ──

    def _create_app(self):
        app = FastAPI(title="RTSM WebSocket Receiver")

        @app.get("/health")
        def health():
            return JSONResponse({"status": "ok", "receiver": "websocket"})

        @app.websocket("/stream")
        async def stream_endpoint(ws: WebSocket):
            await self._handle_stream(ws)

        return app

    # ── Handshake + receive loop ──

    async def _handle_stream(self, ws) -> None:
        await ws.accept()
        client_addr = ws.client.host if ws.client else "unknown"
        logger.info(f"[websocket] Client connected from {client_addr}")

        # ── Handshake: wait for hello ──
        try:
            hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"[websocket] Handshake timeout from {client_addr}")
            await ws.close(code=4002, reason="handshake timeout")
            return
        except Exception:
            logger.warning(f"[websocket] Connection lost during handshake from {client_addr}")
            return

        try:
            hello = json.loads(hello_raw)
        except json.JSONDecodeError:
            await ws.close(code=4001, reason="invalid hello JSON")
            return

        if hello.get("type") != "hello":
            await ws.close(code=4001, reason="expected hello message")
            return

        proto_ver = hello.get("protocol_version", 0)
        if proto_ver != PROTOCOL_VERSION:
            ack = {
                "type": "hello_ack",
                "status": "error",
                "reason": f"unsupported protocol_version: {proto_ver}",
            }
            await ws.send_json(ack)
            await ws.close(code=4001, reason=ack["reason"])
            return

        session_id = hello.get("session_id", "unknown")
        device_name = hello.get("device_name", "unknown")
        self._active_session_id = session_id

        ack = {
            "type": "hello_ack",
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "server": "rtsm",
            "session_id": session_id,
        }
        await ws.send_json(ack)
        logger.info(
            f"[websocket] Handshake OK: session={session_id}, device={device_name}"
        )

        if self._on_handshake_done is not None:
            try:
                self._on_handshake_done(hello, ack)
            except Exception as e:
                logger.error(f"[websocket] on_handshake_done callback error: {e}")

        # Reset per-session state
        self._frame_count = 0
        self._last_nonkf_enq_mono = 0.0
        self._last_enq_ts_ns = None
        self._h264_decoder = None  # fresh decoder per connection
        frames_received = 0
        frames_enqueued = 0
        t_start = time.monotonic()

        # ── Receive loop (binary frames + text messages) ──
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        # Binary frame
                        frames_received += 1
                        if frames_received <= 3 or frames_received % 100 == 0:
                            logger.info(f"[websocket] binary frame #{frames_received}, {len(msg['bytes'])} bytes")
                        if self._on_raw_message is not None:
                            try:
                                self._on_raw_message("binary", msg["bytes"])
                            except Exception as e:
                                logger.error(f"[websocket] on_raw_message callback error: {e}")
                        try:
                            pkt = self._parse_binary_message(msg["bytes"])
                            if pkt is not None:
                                # Broadcast camera frame at full rate (every decoded frame)
                                if self._on_camera_frame is not None:
                                    try:
                                        self._on_camera_frame(pkt)
                                    except Exception as e:
                                        logger.error(f"[websocket] on_camera_frame callback error: {e}")
                                if self._latency_analytics:
                                    self._latency_analytics.sample_queue_depth(self.ingest_q.qsize())
                                ok = self.ingest_q.put(pkt, block=False)
                                if ok:
                                    frames_enqueued += 1
                                    self._last_enq_ts_ns = pkt.time.t_sensor_ns
                                    if not pkt.is_keyframe:
                                        self._last_nonkf_enq_mono = time.monotonic()
                                    if pkt.is_keyframe and self._on_keyframe is not None:
                                        try:
                                            self._on_keyframe(pkt)
                                        except Exception as e:
                                            logger.error(f"[websocket] on_keyframe callback error: {e}")
                                    frame_type = "KF" if pkt.is_keyframe else "frame"
                                    logger.debug(
                                        f"[websocket] enqueued {frame_type} "
                                        f"-> queue={self.ingest_q.qsize()}"
                                    )
                                else:
                                    if self._latency_analytics:
                                        self._latency_analytics.record_queue_drop()
                                    logger.warning(
                                        "[websocket] ingest queue full; dropping frame"
                                    )
                        except Exception as e:
                            logger.error(f"[websocket] frame parse error: {e}")
                    elif "text" in msg and msg["text"]:
                        # Text message (pose_corrections, etc.)
                        if self._on_raw_message is not None:
                            try:
                                self._on_raw_message("text", msg["text"])
                            except Exception as e:
                                logger.error(f"[websocket] on_raw_message callback error: {e}")
                        try:
                            self._handle_text_message(msg["text"])
                        except Exception as e:
                            logger.error(f"[websocket] text message error: {e}")
                elif msg["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[websocket] connection error: {e}")
        finally:
            elapsed = time.monotonic() - t_start
            logger.info(
                f"[websocket] Session {session_id} ended: "
                f"{frames_received} received, {frames_enqueued} enqueued, "
                f"{elapsed:.1f}s duration"
            )
            self._active_session_id = None

    # ── Text message handling ──

    def _handle_text_message(self, text: str) -> None:
        """Handle a JSON text message (e.g. pose_corrections from RTAB-Map)."""
        msg = json.loads(text)
        msg_type = msg.get("type")

        if msg_type == "pose_corrections":
            corrections = msg.get("corrections", {})
            if not corrections:
                return

            # Build full corrections dict (apply camera flip if active)
            batch = {}
            for kf_id, pose_data in corrections.items():
                pose_4x4 = self._corrections_pose_to_4x4(pose_data)
                if pose_4x4 is not None:
                    if self._apply_camera_flip:
                        pose_4x4 = pose_4x4 @ _ARKIT_TO_OPENCV
                    batch[kf_id] = pose_4x4

            if not batch:
                return

            # Prefer batch callback (TSDF), fall back to per-correction
            if self._on_pose_corrections_batch is not None:
                self._on_pose_corrections_batch(batch)
            elif self._on_pose_corrections is not None:
                for kf_id, pose in batch.items():
                    self._on_pose_corrections(kf_id, pose)
            else:
                logger.debug("[websocket] pose_corrections received but no callback registered")
                return

            logger.info(f"[websocket] Applied {len(batch)} pose corrections (loop closure)")
        else:
            logger.debug(f"[websocket] Ignoring text message type: {msg_type}")

    @staticmethod
    def _corrections_pose_to_4x4(pose_data: list) -> Optional[np.ndarray]:
        """Convert pose correction data to a 4x4 row-major matrix."""
        if len(pose_data) == 16:
            # Column-major 4x4 (same as ARKit binary frames)
            return np.array(pose_data, dtype=np.float32).reshape(4, 4, order="F")
        elif len(pose_data) == 7:
            # [qx, qy, qz, qw, tx, ty, tz]
            from scipy.spatial.transform import Rotation
            qx, qy, qz, qw, tx, ty, tz = pose_data
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R
            mat[:3, 3] = [tx, ty, tz]
            return mat
        else:
            logger.warning(f"[websocket] pose_corrections: unexpected {len(pose_data)} elements, skipping")
            return None

    # ── Binary message parsing ──

    def _parse_binary_message(self, data: bytes) -> Optional[FramePacket]:
        """Parse a single binary WebSocket message into a FramePacket."""
        if self._latency_analytics:
            self._latency_analytics.record_frame_received()

        offset = 0
        n = len(data)

        # 1. JSON header
        if n < 4:
            logger.warning("[websocket] message too short for json_len")
            return None
        (json_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + json_len:
            logger.warning("[websocket] message truncated at JSON payload")
            return None
        header = json.loads(data[offset : offset + json_len].decode("utf-8"))
        offset += json_len

        # 2. RGB payload
        if n < offset + 4:
            logger.warning("[websocket] message truncated at rgb_len")
            return None
        (rgb_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + rgb_len:
            logger.warning("[websocket] message truncated at RGB payload")
            return None
        rgb_bytes = data[offset : offset + rgb_len]
        offset += rgb_len

        # 3. Depth payload
        if n < offset + 4:
            logger.warning("[websocket] message truncated at depth_len")
            return None
        (depth_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + depth_len:
            logger.warning("[websocket] message truncated at depth payload")
            return None
        depth_bytes = data[offset : offset + depth_len]
        offset += depth_len

        # 4. Confidence payload (optional � backward compatible)
        confidence_m = None
        confidence_fmt = header.get("confidence_format")
        if confidence_fmt is not None and (n - offset) >= 4:
            (conf_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            if conf_len > 0 and (n - offset) >= conf_len:
                conf_bytes = data[offset : offset + conf_len]
                offset += conf_len
                conf_w = int(header.get("confidence_width", 0) or 0)
                conf_h = int(header.get("confidence_height", 0) or 0)
                if conf_w > 0 and conf_h > 0 and len(conf_bytes) == conf_w * conf_h:
                    confidence_m = np.frombuffer(
                        conf_bytes, dtype=np.uint8
                    ).reshape(conf_h, conf_w)

        # 5. Tracking state filter
        tracking_state = header.get("tracking_state", "not_available")
        if self._require_tracking_normal and tracking_state != "normal":
            if self._latency_analytics:
                self._latency_analytics.record_tracking_drop()
            logger.info(
                f"[websocket] dropping frame: tracking_state={tracking_state}"
            )
            return None

        self._frame_count += 1

        # 5. Keyframe decision
        is_keyframe = (
            self._frame_count == 1
            or self._frame_count % self._keyframe_every_n == 0
        )

        # 6. Non-KF throttle
        if not is_keyframe:
            now_mono = time.monotonic()
            if (now_mono - self._last_nonkf_enq_mono) < self._nonkf_min_interval_s:
                if self._latency_analytics:
                    self._latency_analytics.record_throttle_skip()
                return None

        # 7. Decode RGB (capture raw JPEG for zero-copy viz forwarding)
        rgb_fmt = header.get("rgb_format", "jpeg")
        rgb_w = int(header.get("rgb_width", 0))
        rgb_h = int(header.get("rgb_height", 0))
        raw_jpeg = bytes(rgb_bytes) if rgb_fmt == "jpeg" else None

        if rgb_fmt == "h264":
            # H.264 requires stateful decoder (P-frame reference tracking)
            if self._h264_decoder is None:
                self._h264_decoder = H264Decoder()
            rgb = self._h264_decoder.decode(rgb_bytes)
            if rgb is None:
                logger.warning("[websocket] H.264 decode produced no frame")
                return None
        else:
            rgb = decode_rgb(rgb_bytes, fmt=rgb_fmt, width=rgb_w, height=rgb_h)

        # 8. Decode depth (native resolution, NaN for invalid)
        depth_fmt = header.get("depth_format")
        depth_w = int(header.get("depth_width", 0) or 0)
        depth_h = int(header.get("depth_height", 0) or 0)
        depth_scale = float(header.get("depth_scale", 0.001) or 0.001)
        depth_m = decode_depth(
            depth_bytes, fmt=depth_fmt, width=depth_w, height=depth_h,
            depth_scale=depth_scale,
        )

        # 9. Parse pose
        t_wc, q_xyzw = parse_arkit_pose(
            T_wc_data=header["T_wc"],
            pose_format=header.get("pose_format", "matrix4x4_col_major"),
        )

        # 9b. Camera convention flip: ARKit (Y-up, Z-back) → OpenCV (Y-down, Z-forward)
        # Applied once at ingestion so ALL downstream consumers (pipeline, TSDF,
        # visualization, sweep cache) see poses in OpenCV camera convention.
        if self._apply_camera_flip:
            T_wc_mat = PoseStamped(
                stamp_ns=0, frame_id="", t_wc=t_wc, q_wc_xyzw=q_xyzw
            ).T_wc() @ _ARKIT_TO_OPENCV
            t_wc = T_wc_mat[:3, 3].astype(np.float32)
            q_xyzw = rotmat_to_quat_xyzw(T_wc_mat[:3, :3].astype(np.float32))

        # 10. Build intrinsics (scale if intrinsics resolution differs from RGB)
        intr_w = int(header.get("intrinsics_width", rgb_w))
        intr_h = int(header.get("intrinsics_height", rgb_h))
        fx = float(header["fx"])
        fy = float(header["fy"])
        cx = float(header["cx"])
        cy = float(header["cy"])

        if intr_w > 0 and intr_h > 0 and rgb_w > 0 and rgb_h > 0:
            if intr_w != rgb_w or intr_h != rgb_h:
                scale_x = rgb_w / intr_w
                scale_y = rgb_h / intr_h
                fx *= scale_x
                fy *= scale_y
                cx *= scale_x
                cy *= scale_y

        intr = PinholeIntrinsics(
            width=rgb_w, height=rgb_h,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )

        # 11. Build TimeBundle
        timestamp_ns = int(header.get("timestamp_ns", 0))
        unix_ts = float(header.get("unix_timestamp", time.time()))

        tb = TimeBundle(
            t_mono_s=time.monotonic(),
            t_wall_utc_s=unix_ts,
            t_sensor_ns=timestamp_ns,
            seq=int(header.get("frame_id", 0)),
        )

        # 12. Build PoseStamped
        pose = PoseStamped(
            stamp_ns=timestamp_ns,
            frame_id="arkit",
            t_wc=t_wc,
            q_wc_xyzw=q_xyzw,
        )

        # 13. Apply confidence filtering (if confidence map available)
        if confidence_m is not None and depth_m is not None and self._confidence_threshold > 0:
            # Resize confidence to depth resolution if needed
            dep_h, dep_w = depth_m.shape[:2]
            conf_h, conf_w = confidence_m.shape[:2]
            if conf_h != dep_h or conf_w != dep_w:
                confidence_m = cv2.resize(
                    confidence_m, (dep_w, dep_h),
                    interpolation=cv2.INTER_NEAREST
                )
            # Zero out low-confidence depth pixels
            low_conf = confidence_m < self._confidence_threshold
            depth_m[low_conf] = np.nan

        # 14. Build FramePacket
        return FramePacket(
            time=tb,
            rgb=rgb,
            depth_m=depth_m,
            pose=pose,
            intr=intr,
            is_keyframe=is_keyframe,
            confidence=confidence_m,
            rgb_jpeg=raw_jpeg,
        )

    # ── Server lifecycle ──

    def _run_server(self) -> None:
        """Run uvicorn in a background thread with its own event loop."""
        import uvicorn

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            ws_max_size=16 * 1024 * 1024,  # 16 MB max message
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # no signals in child thread
        loop.run_until_complete(server.serve())

    def start(self) -> None:
        """Start the WebSocket receiver in a daemon thread."""
        if self._server_thread and self._server_thread.is_alive():
            logger.warning("[websocket] Server already running")
            return
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="websocket-receiver",
        )
        self._server_thread.start()

    def stop(self) -> None:
        """Signal shutdown (best-effort; daemon thread exits with process)."""
        logger.info("[websocket] Stop requested")
