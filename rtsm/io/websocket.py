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
from rtsm.evaluation.event_log import (
    RX_DROPPED, RX_ENQUEUED, RX_MALFORMED, RX_PARSE_ERROR, RX_QUEUE_FULL, RX_THROTTLE, RX_TRACKING,
    ReceiverEvent,
)
from rtsm.core.datamodel import IngestMeta
from rtsm.io.ingest_lanes import KF_MINTED

logger = logging.getLogger(__name__)

# Current handshake protocol version
PROTOCOL_VERSION = 1

# Supported format values
_RGB_FORMATS = {"jpeg", "png", "bgra", "nv12"}
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


def forward_clearance_from_depth(depth_m: Optional[np.ndarray],
                                 min_valid_frac: float = 0.2) -> Tuple[float, float]:
    """Meters of open space ahead of the camera, from one decoded depth
    frame. RECEIVE-TIME wall-guard sensing (2026-08-16): lives here in the
    io layer — frame-packet level, before the ingest queue/gate and any
    GPU work — so it updates at stream rate regardless of pipeline load,
    and keeps updating for stationary frames the keyframe gate drops.

    Returns (clearance_m, valid_frac). Central band of the image (rows
    30-55%, cols 33-66% — above the floor line for a roughly level
    camera, so the floor doesn't read as an obstacle); clearance is the
    10th percentile of valid depths (robust nearest-surface estimate).
    Fail-closed: a mostly-invalid band (LiDAR too close / no return)
    returns 0.0 — blind agents must not walk on a blind sensor."""
    if depth_m is None or depth_m.size == 0:
        return 0.0, 0.0
    h, w = depth_m.shape[:2]
    band = depth_m[int(h * 0.30):int(h * 0.55), int(w * 0.33):int(w * 0.66)]
    if band.size == 0:
        return 0.0, 0.0
    valid = band[np.isfinite(band) & (band > 0.05)]
    frac = float(valid.size) / float(band.size)
    if frac < min_valid_frac:
        return 0.0, frac
    return float(np.percentile(valid, 10)), frac


def depth_valid_fraction(depth_m: Optional[np.ndarray]) -> Optional[float]:
    """Fraction of finite depth pixels (the P2 pose-ledger field for dropped
    frames). None when there is no depth."""
    if depth_m is None or getattr(depth_m, "size", 0) == 0:
        return None
    return float(np.isfinite(depth_m).mean())


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
        pose_sink: Optional[callable] = None,
        clearance_sink: Optional[callable] = None,
        latency_analytics: Optional[Any] = None,
        event_sink: Optional[callable] = None,
        event_source: str = "websocket",
        admission_queue: Optional[IngestQueue] = None,
        throttle_clock: str = "wall",
    ) -> None:
        self.ingest_q = ingest_queue
        # Non-keyframe throttle clock (ingest.clock): "wall" compares process
        # time between admitted non-KFs (live default); "sensor" compares the
        # header timestamps, so the admitted set is independent of replay speed.
        self._throttle_clock = "sensor" if str(throttle_clock).lower() == "sensor" else "wall"
        # Frame-flow trace: called with a ReceiverEvent for every decision
        # (enqueued / dropped + reason). None (the default) skips it entirely.
        # admission_queue: the queue this receiver's frames are destined for.
        # Consulted BEFORE the RGB decode (admit-before-decode) and reported as
        # queue_depth on trace lines. The replayer's decoder-only instance owns
        # a dummy ingest queue and passes the real one here.
        self._event_sink = event_sink
        self._event_source = str(event_source)
        self._admission_queue = admission_queue if admission_queue is not None else ingest_queue
        self._hdr_seq: Any = None     # header ids of the message being parsed (for parse_error lines)
        self._hdr_ts: Any = None
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
        # Called as pose_sink(t_wc, q_wc_xyzw, unix_ts, frame_epoch) for
        # EVERY frame with normal tracking — including frames the
        # keyframe/interval throttle skips — so consumers (e.g.
        # WorkingMemory.update_robot_pose) see pose at the full input rate,
        # not the pipeline processing rate.
        self._pose_sink = pose_sink
        # Called as clearance_sink(clearance_m, valid_frac, wall_ts) for
        # every frame that passes the tracking filter and the non-KF
        # throttle (every DECODED frame, whether or not the ingest queue
        # then accepts it), right after depth decode — receive-time
        # like the pose sink, so the wall-guard signal updates at stream
        # rate and never stalls behind GPU processing (agent-level safety
        # consumers read it before blind motion). Added 2026-08-16.
        # Wired by rtsm/run.py only when io.clearance.enable is true; None
        # (the default) skips the depth statistic entirely.
        self._clearance_sink = clearance_sink
        self._latency_analytics = latency_analytics

        # Per-session state (reset on each new client connection)
        self._frame_count: int = 0
        # Throttle stamps: advanced on the ADMIT decision (step 6), never on
        # enqueue, so a full ingest queue cannot stop the throttle from
        # thinning. Wall stamp (process-monotonic) and sensor stamp (header ns).
        self._last_nonkf_enq_mono: float = 0.0
        self._last_nonkf_admit_sensor_ns: Optional[int] = None
        self._last_enq_ts_ns: Optional[int] = None
        # Receiver-local running count of parsed binary frames (never reset:
        # it is a join key across sessions in one trace file), and the value
        # for the frame being parsed right now.
        self._rx_seq: int = 0
        self._cur_rx_seq: Optional[int] = None
        self._active_session_id: Optional[str] = None

        # Frame-flow liveness stamps (server-lifetime, read by the watchdog).
        # Bare float/int assignment is atomic under the GIL — no lock needed.
        self.last_rx_mono: Optional[float] = None
        self.last_enqueue_mono: Optional[float] = None
        self.tracking_drops: int = 0

        # Frame epoch — SERVER-lifetime, never reset per connection. Bumps
        # when a hello carries a new session_id: the sender re-created its
        # streaming session, so its ARKit world origin may have moved and
        # poses across the boundary must not be assumed to share a world
        # frame. Same-id reconnects keep the epoch. Delivered to the pose
        # sink with every pose so consumers can detect the boundary.
        self._frame_epoch: int = 0
        self._epoch_session_id: Optional[str] = None

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

    # ── Frame epoch ──

    def _note_session(self, session_id: str) -> int:
        """Advance the frame epoch iff ``session_id`` differs from the last
        session seen. Calabi Lens mints a fresh UUID per connect and resets
        ARKit tracking per app lifecycle, so a new id marks a potential
        world-origin change; a same-id reconnect keeps the epoch."""
        if session_id != self._epoch_session_id:
            self._epoch_session_id = session_id
            self._frame_epoch += 1
        return self._frame_epoch

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
        self._note_session(session_id)

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
        self._last_nonkf_admit_sensor_ns = None
        self._last_enq_ts_ns = None
        frames_received = 0
        frames_enqueued = 0
        t_start = time.monotonic()

        # ── Receive loop (binary frames + text messages) ──
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.receive":
                    self.last_rx_mono = time.monotonic()
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
                                if self._latency_analytics:
                                    self._latency_analytics.sample_queue_depth(self.ingest_q.qsize())
                                ok = self.ingest_q.put(pkt, block=False)
                                if ok:
                                    frames_enqueued += 1
                                    self._trace_rx(RX_ENQUEUED, "", pkt=pkt)
                                    # Viz camera feed (JPEG encode for non-JPEG sources)
                                    # only for ADMITTED frames: no encode work for a
                                    # frame the queue just refused.
                                    if self._on_camera_frame is not None:
                                        try:
                                            self._on_camera_frame(pkt)
                                        except Exception as e:
                                            logger.error(f"[websocket] on_camera_frame callback error: {e}")
                                    self.last_enqueue_mono = time.monotonic()
                                    self._last_enq_ts_ns = pkt.time.t_sensor_ns
                                    # (throttle stamp advanced at the admit decision in
                                    #  _parse_binary_message step 6, not here)
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
                                    reason = getattr(getattr(pkt, "ingest", None), "drop_reason", None) or RX_QUEUE_FULL
                                    logger.warning(f"[websocket] ingest queue refused frame ({reason}); dropping")
                                    self._trace_rx(RX_DROPPED, reason, pkt=pkt)
                        except Exception as e:
                            # (the parse_error trace line is emitted inside
                            # _parse_binary_message, with the header ids)
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

    # ── Non-keyframe throttle ──

    def _admit_nonkf(self, sensor_ts_ns: Any) -> bool:
        """Decide whether a non-keyframe passes the min-interval throttle, and
        stamp the decision if it does.

        sensor mode: compare the header timestamp with the last ADMITTED
        non-KF's; a negative delta (new session / restarted clock) admits and
        re-stamps. Falls back to wall when the header carries no timestamp.
        wall mode: compare process-monotonic time, as before.
        """
        interval = self._nonkf_min_interval_s
        if self._throttle_clock == "sensor" and sensor_ts_ns:
            try:
                ts = int(sensor_ts_ns)
            except (TypeError, ValueError):
                ts = 0
            if ts > 0:
                last = self._last_nonkf_admit_sensor_ns
                if last is not None and 0 <= (ts - last) < int(interval * 1e9):
                    return False
                self._last_nonkf_admit_sensor_ns = ts
                return True
        now_mono = time.monotonic()
        if (now_mono - self._last_nonkf_enq_mono) < interval:
            return False
        self._last_nonkf_enq_mono = now_mono
        return True

    # ── Frame-flow trace ──

    def _trace_rx(self, decision: str, reason: str = "", *, pkt: Optional[FramePacket] = None,
                  seq: Any = None, ts: Any = None, is_kf: Optional[bool] = None,
                  frame_count: Optional[int] = None, queue_depth: Optional[int] = None,
                  depth_valid_frac: Optional[float] = None, lane: Optional[str] = None,
                  rx_seq: Optional[int] = None) -> None:
        """Emit one ReceiverEvent to the event sink (no-op when unset).

        Returns None so drop sites can `return self._trace_rx(...)`. When a
        FramePacket is given, its seq / t_sensor_ns / is_keyframe are used.
        Never raises into the receive path.
        """
        sink = self._event_sink
        if sink is None:
            return None
        try:
            if pkt is not None:
                seq, ts, is_kf = pkt.time.seq, pkt.time.t_sensor_ns, bool(pkt.is_keyframe)
                if frame_count is None:
                    frame_count = self._frame_count
            if queue_depth is None:
                q = self._admission_queue
                queue_depth = int(q.qsize()) if q is not None else None
            if pkt is not None:
                # One statistic on every line: the PRE-confidence-filter
                # fraction the parser put on pkt.ingest at step 5d. NEVER
                # recomputed from pkt.depth_m here -- step 13 has NaN-masked it
                # by now and the post-filter value would silently reappear.
                meta = getattr(pkt, "ingest", None)
                if depth_valid_frac is None:
                    depth_valid_frac = getattr(meta, "depth_valid_frac", None)
                if lane is None:
                    lane = getattr(meta, "lane", None)
                if rx_seq is None:
                    rx_seq = getattr(meta, "rx_seq", None)
            elif rx_seq is None:
                rx_seq = self._cur_rx_seq
            sink(ReceiverEvent(
                timestamp=time.monotonic(),
                source=self._event_source,
                decision=decision,
                reason=reason,
                frame_seq=(int(seq) if seq is not None else None),
                t_sensor_ns=(int(ts) if ts is not None else None),
                is_keyframe=is_kf,
                frame_count=frame_count,
                queue_depth=queue_depth,
                depth_valid_frac=depth_valid_frac,
                lane=lane,
                rx_seq=rx_seq,
            ))
        except Exception:
            logger.debug("[websocket] frame-flow trace failed", exc_info=True)
        return None

    # ── Binary message parsing ──

    def _parse_binary_message(self, data: bytes) -> Optional[FramePacket]:
        """Parse a single binary WebSocket message into a FramePacket.

        Thin wrapper around `_parse_binary_message_impl`: when parsing raises
        after the header was read (malformed T_wc, decode failure, missing
        field), one 'dropped / parse_error' trace line is emitted with the
        header ids, then the exception is re-raised so every caller keeps its
        existing behaviour (the stream loop logs and continues; the replayer
        propagates as before).
        """
        self._hdr_seq = None
        self._hdr_ts = None
        self._rx_seq += 1
        self._cur_rx_seq = self._rx_seq
        try:
            return self._parse_binary_message_impl(data)
        except Exception:
            self._trace_rx(RX_DROPPED, RX_PARSE_ERROR, seq=self._hdr_seq, ts=self._hdr_ts,
                           frame_count=self._frame_count)
            raise

    def _parse_binary_message_impl(self, data: bytes) -> Optional[FramePacket]:
        if self._latency_analytics:
            self._latency_analytics.record_frame_received()

        offset = 0
        n = len(data)
        hdr_seq = None   # header frame_id / timestamp_ns for the frame-flow trace
        hdr_ts = None

        # 1. JSON header
        if n < 4:
            logger.warning("[websocket] message too short for json_len")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED)
        (json_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + json_len:
            logger.warning("[websocket] message truncated at JSON payload")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED)
        header = json.loads(data[offset : offset + json_len].decode("utf-8"))
        offset += json_len
        if isinstance(header, dict):
            hdr_seq = header.get("frame_id")
            hdr_ts = header.get("timestamp_ns")
        self._hdr_seq, self._hdr_ts = hdr_seq, hdr_ts

        # 2. RGB payload
        if n < offset + 4:
            logger.warning("[websocket] message truncated at rgb_len")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED, seq=hdr_seq, ts=hdr_ts)
        (rgb_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + rgb_len:
            logger.warning("[websocket] message truncated at RGB payload")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED, seq=hdr_seq, ts=hdr_ts)
        rgb_bytes = data[offset : offset + rgb_len]
        offset += rgb_len

        # 3. Depth payload
        if n < offset + 4:
            logger.warning("[websocket] message truncated at depth_len")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED, seq=hdr_seq, ts=hdr_ts)
        (depth_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if n < offset + depth_len:
            logger.warning("[websocket] message truncated at depth payload")
            return self._trace_rx(RX_DROPPED, RX_MALFORMED, seq=hdr_seq, ts=hdr_ts)
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
            self.tracking_drops += 1
            if self._latency_analytics:
                self._latency_analytics.record_tracking_drop()
            logger.info(
                f"[websocket] dropping frame: tracking_state={tracking_state}"
            )
            return self._trace_rx(RX_DROPPED, RX_TRACKING, seq=hdr_seq, ts=hdr_ts)

        # 5b. Parse pose + wall timestamp (hoisted above the keyframe/interval
        # throttle so the pose sink fires at the full input rate).
        # Note: a malformed T_wc now raises here — before frame_count
        # increments — instead of after image decode; the caller catches and
        # logs it per frame.
        t_wc, q_xyzw = parse_arkit_pose(
            T_wc_data=header["T_wc"],
            pose_format=header.get("pose_format", "matrix4x4_col_major"),
        )

        # Camera convention flip: ARKit (Y-up, Z-back) → OpenCV (Y-down, Z-forward)
        # Applied once at ingestion so ALL downstream consumers (pipeline, TSDF,
        # visualization, sweep cache, pose sink) see poses in OpenCV camera
        # convention.
        if self._apply_camera_flip:
            T_wc_mat = PoseStamped(
                stamp_ns=0, frame_id="", t_wc=t_wc, q_wc_xyzw=q_xyzw
            ).T_wc() @ _ARKIT_TO_OPENCV
            t_wc = T_wc_mat[:3, 3].astype(np.float32)
            q_xyzw = rotmat_to_quat_xyzw(T_wc_mat[:3, :3].astype(np.float32))

        # Treat a missing/zero unix_timestamp as absent and substitute server
        # wall time; the same value flows into TimeBundle.t_wall_utc_s, so the
        # pose sink and the pipeline's later update_robot_pose call always
        # share one clock (the guard compares timestamps across the two).
        unix_ts = float(header.get("unix_timestamp") or time.time())

        # 5c. Pose sink: latest-pose passthrough for every tracking-normal
        # frame, even ones the throttle below skips. Same (post-flip) pose the
        # FramePacket carries, so consumers see one consistent convention.
        if self._pose_sink is not None:
            try:
                self._pose_sink(t_wc, q_xyzw, unix_ts, self._frame_epoch)
            except Exception as e:
                logger.error(f"[websocket] pose_sink callback error: {e}")

        # 5d. Decode depth (+ the confidence map already sliced in step 4) for
        # EVERY tracking-normal frame, before the throttle and the queue
        # admission: ~150 KB, cheap, and it lets the trace / P2 pose ledger
        # carry depth statistics for the frames that are dropped below. The
        # 8.5 MB RGB decode waits until the frame is admitted (step 7).
        # Consequence (like the T_wc hoist in 5b): a malformed depth payload
        # now raises HERE -- before frame_count and the throttle stamp, and
        # for frames the throttle would have dropped unseen -- and surfaces
        # as a parse_error line; main raised after the frame_count increment.
        depth_fmt = header.get("depth_format")
        depth_w = int(header.get("depth_width", 0) or 0)
        depth_h = int(header.get("depth_height", 0) or 0)
        depth_scale = float(header.get("depth_scale", 0.001) or 0.001)
        depth_m = decode_depth(
            depth_bytes, fmt=depth_fmt, width=depth_w, height=depth_h,
            depth_scale=depth_scale,
        )
        dvf = depth_valid_fraction(depth_m) if self._event_sink is not None else None

        self._frame_count += 1

        # 5. Keyframe decision
        is_keyframe = (
            self._frame_count == 1
            or self._frame_count % self._keyframe_every_n == 0
        )

        # 6. Non-KF throttle (ingest clock). The stamp advances on the admit
        # decision, here, before decode and enqueue — a full ingest queue must
        # not stop the throttle from thinning (the E1 wedge). Sensor mode
        # compares header timestamps so the admitted set does not depend on
        # replay speed. Wall mode therefore stamps ~3 ms EARLIER than before
        # this change (decode latency); a non-KF landing inside that window of
        # the 0.5 s boundary can flip from throttled to admitted. On Windows /
        # Python 3.12 the 15.6 ms monotonic tick absorbs it (G1-A was exact);
        # on a ns-resolution clock wall replays were never bit-reproducible,
        # which is why the sensor anchor is the reference.
        if not is_keyframe:
            if not self._admit_nonkf(hdr_ts):
                if self._latency_analytics:
                    self._latency_analytics.record_throttle_skip()
                return self._trace_rx(RX_DROPPED, RX_THROTTLE, seq=hdr_seq, ts=hdr_ts,
                                      is_kf=False, frame_count=self._frame_count,
                                      depth_valid_frac=dvf)

        # 6b. Queue admission BEFORE the RGB decode (admit-before-decode). With
        # the legacy tail-drop queue this is the same drop that put() would
        # have reported after the decode; deciding here means a congested
        # pipeline no longer costs the receiver an 8.5 MB decode per frame it
        # is about to throw away (the E1 wedge). The stream loop keeps its
        # put(): a slot can still fill between this check and the put.
        aq = self._admission_queue
        kf_origin = KF_MINTED if is_keyframe else None
        refusal = aq.refusal(is_keyframe, kf_origin) if aq is not None else None
        if refusal is not None:
            if self._latency_analytics:
                # Sample the (saturated) depth too: the stream loop samples
                # only for frames that get this far, so without this the
                # per-second queue_depth_max could never show maxsize.
                self._latency_analytics.sample_queue_depth(aq.qsize())
                self._latency_analytics.record_queue_drop()
            return self._trace_rx(RX_DROPPED, refusal, seq=hdr_seq, ts=hdr_ts,
                                  is_kf=is_keyframe, frame_count=self._frame_count,
                                  depth_valid_frac=dvf)

        # 7. Decode RGB (capture raw JPEG for zero-copy viz forwarding)
        rgb_fmt = header.get("rgb_format", "jpeg")
        rgb_w = int(header.get("rgb_width", 0))
        rgb_h = int(header.get("rgb_height", 0))
        raw_jpeg = bytes(rgb_bytes) if rgb_fmt == "jpeg" else None
        rgb = decode_rgb(rgb_bytes, fmt=rgb_fmt, width=rgb_w, height=rgb_h)

        # 8. Depth was decoded in step 5d (before throttle / admission).

        # 8b. Forward-clearance wall guard (opt-in: io.clearance.enable;
        # sink is None otherwise): computed here at RECEIVE time
        # (frame-packet level, before any heavy processing) so the signal
        # tracks the stream rate, not the GPU's mood. Reads depth_m BEFORE
        # the confidence filter in step 13 masks low-confidence pixels --
        # keep that order if steps are reshuffled (P1 admit-before-decode).
        if self._clearance_sink is not None:
            try:
                c_m, c_frac = forward_clearance_from_depth(depth_m)
                self._clearance_sink(c_m, c_frac, unix_ts)
            except Exception as e:  # noqa: BLE001 — guard telemetry never breaks ingest
                logger.error(f"[websocket] clearance_sink error: {e}")

        # 9. Pose already parsed (+ convention flip) in step 5b above; the
        # same t_wc / q_xyzw feed both the pose sink and the FramePacket.

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

        # 11. Build TimeBundle (unix_ts computed in step 5b)
        timestamp_ns = int(header.get("timestamp_ns", 0))

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
            frame_epoch=self._frame_epoch,
            ingest=IngestMeta(keyframe_origin=kf_origin, depth_valid_frac=dvf, rx_seq=self._cur_rx_seq),
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

    def liveness(self) -> dict:
        """Frame-flow liveness snapshot for the watchdog."""
        t = self._server_thread
        return {
            "alive": bool(t is not None and t.is_alive()),
            "last_rx_mono": self.last_rx_mono,
            "last_enqueue_mono": self.last_enqueue_mono,
            "tracking_drops": self.tracking_drops,
        }

    def stop(self) -> None:
        """Signal shutdown (best-effort; daemon thread exits with process)."""
        logger.info("[websocket] Stop requested")
