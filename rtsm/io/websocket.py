"""
WebSocket Receiver for RTSM — Calabi Lens ARKit Client Integration.

Accepts binary frames over ws://host:port/stream from the Calabi Lens iOS app.
Each frame contains bundled RGB + depth + pose + intrinsics.

Protocol:
1. Client connects to /stream
2. Client sends JSON text hello: {"type": "hello", "protocol_version": 1, ...}
3. Server replies with JSON text ack: {"type": "hello_ack", "status": "ok", ...}
4. Client streams binary frames (length-prefixed: [json][rgb][depth][confidence]?)

Since P3 task 0.5 (Gate 4.5 plan) this module is a TRANSPORT ADAPTER: it owns
the handshake, the binary framing and the text messages, turns each binary
message into a ``RawFrame`` (rtsm/io/contracts.py) and hands it to the one
ingest front-end (rtsm/io/ingest_frontend.py), which runs the policy chain
every source shares -- tracking filter, receive-time pose mailbox, keyframe
rule, throttle, lane admission, decode on admit, FramePacket, enqueue,
callbacks, trace and ledgers. The replayer (rtsm/io/replayer.py) drives the
same framing + front-end from a recording. The decode functions that used to
live here are re-exported from rtsm/io/codecs.py so existing imports work.
"""

from __future__ import annotations
import asyncio
import json
import struct
import time
import threading
import logging
from typing import Any, Callable, Optional, Tuple

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
    TS_NORMAL, TS_NOT_AVAILABLE, PoseEvent, ReceiverEvent,
)
from rtsm.core.datamodel import IngestMeta
from rtsm.io.ingest_lanes import KF_MINTED
# Re-exported for existing imports (`from rtsm.io.websocket import decode_rgb, ...`).
from rtsm.io.codecs import (  # noqa: F401
    _ARKIT_TO_OPENCV, decode_depth, decode_rgb, depth_valid_fraction, forward_clearance_from_depth,
    normalize_matrix_convention, normalize_pose_convention, parse_arkit_pose, parse_pose, rescale_intrinsics,
)
from rtsm.io.contracts import (
    CONVENTION_ARKIT, CONVENTION_OPENCV, EncodedImage, FrameCorrection, FrameHeader, RawFrame,
)
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd

logger = logging.getLogger(__name__)

# Current handshake protocol version
PROTOCOL_VERSION = 1

# Supported format values
_RGB_FORMATS = {"jpeg", "png", "bgra", "nv12"}
_DEPTH_FORMATS = {"uint16_mm", "float32_m", "png_uint16"}
_POSE_FORMATS = {"matrix4x4_col_major", "quat_translation"}


# ─────────────────── Lens binary framing (the transport) ───────────────────


class LensFramingError(ValueError):
    """A truncated / unparseable binary message (a `malformed` drop). Carries
    the header ids when the header itself parsed."""

    def __init__(self, what: str, seq: Any = None, ts: Any = None) -> None:
        super().__init__(what)
        self.seq = seq
        self.ts = ts


def lens_raw_frame(data: bytes, *, source: str, apply_camera_flip: bool,
                   session_id: Optional[str] = None, hdr_ids: Optional[list] = None) -> RawFrame:
    """Steps 1-4 of the Lens protocol: [json_len][json][rgb_len][rgb][depth_len]
    [depth]([conf_len][conf])? -> RawFrame. Nothing is decoded here. Raises
    LensFramingError on truncation; any other exception (a non-object header,
    a missing field) propagates as a parse error, exactly as before. ``hdr_ids``
    (a list) receives ``[seq, ts]`` the moment the header parses so a later
    failure can still be traced with the ids."""
    offset = 0
    n = len(data)

    # 1. JSON header
    if n < 4:
        raise LensFramingError("message too short for json_len")
    (json_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if n < offset + json_len:
        raise LensFramingError("message truncated at JSON payload")
    header = json.loads(data[offset : offset + json_len].decode("utf-8"))
    offset += json_len
    hdr_seq = hdr_ts = None
    if isinstance(header, dict):
        hdr_seq = header.get("frame_id")
        hdr_ts = header.get("timestamp_ns")
    if hdr_ids is not None:
        hdr_ids[:] = [hdr_seq, hdr_ts]

    # 2. RGB payload
    if n < offset + 4:
        raise LensFramingError("message truncated at rgb_len", hdr_seq, hdr_ts)
    (rgb_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if n < offset + rgb_len:
        raise LensFramingError("message truncated at RGB payload", hdr_seq, hdr_ts)
    rgb_bytes = data[offset : offset + rgb_len]
    offset += rgb_len

    # 3. Depth payload
    if n < offset + 4:
        raise LensFramingError("message truncated at depth_len", hdr_seq, hdr_ts)
    (depth_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if n < offset + depth_len:
        raise LensFramingError("message truncated at depth payload", hdr_seq, hdr_ts)
    depth_bytes = data[offset : offset + depth_len]
    offset += depth_len

    # 4. Confidence payload (optional, backward compatible)
    confidence = None
    confidence_fmt = header.get("confidence_format")          # a non-dict header raises here -> parse_error, as before
    if confidence_fmt is not None and (n - offset) >= 4:
        (conf_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if conf_len > 0 and (n - offset) >= conf_len:
            conf_bytes = data[offset : offset + conf_len]
            offset += conf_len
            conf_w = int(header.get("confidence_width", 0) or 0)
            conf_h = int(header.get("confidence_height", 0) or 0)
            if conf_w > 0 and conf_h > 0 and len(conf_bytes) == conf_w * conf_h:
                confidence = EncodedImage(conf_bytes, "uint8", conf_w, conf_h)

    rgb_w = int(header.get("rgb_width", 0))
    rgb_h = int(header.get("rgb_height", 0))
    depth_w = int(header.get("depth_width", 0) or 0)
    depth_h = int(header.get("depth_height", 0) or 0)
    depth_scale = float(header.get("depth_scale", 0.001) or 0.001)
    # Intrinsics: declared at intrinsics_width/height, expressed at RGB resolution.
    # A header without fx/fy/cx/cy fails at packet build (after the tracking
    # filter / throttle / admission), where the old parser read them.
    intr = None
    intr_error = None
    try:
        intr = rescale_intrinsics(
            float(header["fx"]), float(header["fy"]), float(header["cx"]), float(header["cy"]),
            from_wh=(int(header.get("intrinsics_width", rgb_w)), int(header.get("intrinsics_height", rgb_h))),
            to_wh=(rgb_w, rgb_h),
        )
    except Exception as e:  # noqa: BLE001
        intr_error = f"{type(e).__name__}: {e}"
    hdr = FrameHeader(
        source=source,
        seq=hdr_seq,
        t_sensor_ns=hdr_ts,
        t_wall_utc_s=header.get("unix_timestamp"),
        tracking_state=header.get("tracking_state", TS_NOT_AVAILABLE),
        keyframe_hint=None,
        rgb=EncodedImage(rgb_bytes, header.get("rgb_format", "jpeg"), rgb_w, rgb_h),
        depth=EncodedImage(depth_bytes, header.get("depth_format"), depth_w, depth_h, depth_scale),
        intrinsics=intr,
        pose_raw=header.get("T_wc"),
        pose_format=header.get("pose_format", "matrix4x4_col_major"),
        pose_convention=(CONVENTION_ARKIT if apply_camera_flip else CONVENTION_OPENCV),
        confidence=confidence,
        session_id=session_id,
        pose_frame_id="arkit",
        keep_encoded_rgb=True,
        extra=({"intrinsics_error": intr_error} if intr_error else {}),
    )
    return RawFrame(header=hdr)


def parse_lens_message(fe: IngestFrontEnd, data: bytes, *, apply_camera_flip: bool,
                       latency_analytics: Any = None, session_id: Optional[str] = None) -> Optional[FramePacket]:
    """One Lens binary message through the front-end: mint the rx_seq, count the
    receive, frame it, admit it. A truncated message is one `malformed` line
    (None returned); any other failure leaves one `parse_error` line with the
    header ids and is RE-RAISED (the caller logs and continues)."""
    rx_seq = fe.next_rx_seq()
    if latency_analytics:
        latency_analytics.record_frame_received()
    ids: list = [None, None]
    try:
        raw = lens_raw_frame(data, source=fe.source, apply_camera_flip=apply_camera_flip,
                             session_id=session_id, hdr_ids=ids)
    except LensFramingError as e:
        logger.warning(f"[{fe.source}] {e}")
        fe.reject(RX_MALFORMED, seq=e.seq, ts=e.ts, rx_seq=rx_seq, consume_seq=False)
        return None
    except Exception:
        fe.reject(RX_PARSE_ERROR, seq=ids[0], ts=ids[1], frame_count=fe.frame_count, rx_seq=rx_seq,
                  consume_seq=False)
        raise
    return fe.admit(raw, rx_seq=rx_seq)


def corrections_pose_to_4x4(pose_data: list) -> Optional[np.ndarray]:
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


def handle_lens_text_message(text: str, *, apply_camera_flip: bool, session_id: Optional[str] = None,
                             on_pose_corrections: Optional[Callable[..., Any]] = None,
                             on_pose_corrections_batch: Optional[Callable[..., Any]] = None,
                             on_frame_correction: Optional[Callable[[FrameCorrection], None]] = None,
                             source: str = "websocket") -> None:
    """A JSON text message (e.g. pose_corrections from RTAB-Map / a loop closure):
    the batch is built in the engine convention and delivered to the batch
    callback (TSDF), else per correction, and as a FrameCorrection event."""
    msg = json.loads(text)
    msg_type = msg.get("type")
    if msg_type == "pose_corrections":
        corrections = msg.get("corrections", {})
        if not corrections:
            return
        batch = {}
        for kf_id, pose_data in corrections.items():
            pose_4x4 = corrections_pose_to_4x4(pose_data)
            if pose_4x4 is not None:
                if apply_camera_flip:
                    pose_4x4 = normalize_matrix_convention(pose_4x4, CONVENTION_ARKIT)
                batch[kf_id] = pose_4x4
        if not batch:
            return
        if on_frame_correction is not None:
            try:
                on_frame_correction(FrameCorrection(t_wall_utc_s=time.time(), session_id=session_id,
                                                    kind="loop_closure", corrections=dict(batch)))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{source}] on_frame_correction callback error: {e}")
        # Prefer batch callback (TSDF), fall back to per-correction
        if on_pose_corrections_batch is not None:
            on_pose_corrections_batch(batch)
        elif on_pose_corrections is not None:
            for kf_id, pose in batch.items():
                on_pose_corrections(kf_id, pose)
        else:
            logger.debug(f"[{source}] pose_corrections received but no callback registered")
            return
        logger.info(f"[{source}] Applied {len(batch)} pose corrections (loop closure)")
    else:
        logger.debug(f"[{source}] Ignoring text message type: {msg_type}")


# ─────────────────── WebSocket Receiver Class ───────────────────


class WebSocketReceiver:
    """
    WebSocket receiver for Calabi Lens ARKit iOS client.

    Accepts binary frames over ``ws://host:port/stream``, frames them into
    ``RawFrame``s and hands them to the ingest front-end, which decides, decodes
    and enqueues canonical ``FramePacket`` objects.
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
        ledger_sink: Optional[callable] = None,
        on_frame_correction: Optional[callable] = None,
    ) -> None:
        self.ingest_q = ingest_queue
        # The one ingest chain (rtsm/io/ingest_frontend.py). Frame-flow trace,
        # ledgers, pose / clearance sinks, throttle clock and admission queue
        # all live on it.
        self._fe = IngestFrontEnd(
            source=str(event_source), policy=WEBSOCKET_POLICY, ingest_queue=ingest_queue,
            admission_queue=admission_queue, throttle_clock=throttle_clock,
            keyframe_every_n=keyframe_every_n, nonkf_min_interval_s=nonkf_min_interval_s,
            require_tracking_normal=require_tracking_normal, confidence_threshold=confidence_threshold,
            pose_sink=pose_sink, clearance_sink=clearance_sink, event_sink=event_sink, ledger_sink=ledger_sink,
            latency_analytics=latency_analytics, on_camera_frame=on_camera_frame, on_keyframe=on_keyframe,
        )
        self.name = str(event_source)
        self._host = host
        self._port = port
        self._apply_camera_flip = apply_camera_flip
        self._on_pose_corrections = on_pose_corrections
        self._on_pose_corrections_batch = on_pose_corrections_batch
        self._on_frame_correction = on_frame_correction
        self._on_raw_message = on_raw_message
        self._on_handshake_done = on_handshake_done
        self._latency_analytics = latency_analytics
        self._active_session_id: Optional[str] = None

        # Threading
        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_event = asyncio.Event()

    # ── Compatibility delegates: the state the chain used to keep on this
    #    object lives on the front-end; tests and the watchdog read it here. ──

    @property
    def frontend(self) -> IngestFrontEnd:
        return self._fe

    def _fe_prop(name):  # noqa: N805 -- tiny descriptor factory, deleted below
        return property(lambda self: getattr(self._fe, name), lambda self, v: setattr(self._fe, name, v))

    _frame_count = _fe_prop("frame_count")
    _frame_epoch = _fe_prop("frame_epoch")
    _rx_seq = _fe_prop("rx_seq")
    _cur_rx_seq = _fe_prop("cur_rx_seq")
    _last_enq_ts_ns = _fe_prop("last_enq_ts_ns")
    _epoch_session_id = _fe_prop("_epoch_session_id")
    _event_sink = _fe_prop("event_sink")
    _ledger_sink = _fe_prop("ledger_sink")
    _pose_sink = _fe_prop("pose_sink")
    _clearance_sink = _fe_prop("clearance_sink")
    _admission_queue = _fe_prop("admission_queue")
    _require_tracking_normal = _fe_prop("require_tracking_normal")
    _confidence_threshold = _fe_prop("confidence_threshold")
    _keyframe_every_n = _fe_prop("keyframe_every_n")
    _on_camera_frame = _fe_prop("on_camera_frame")
    _on_keyframe = _fe_prop("on_keyframe")
    _hdr_seq = _fe_prop("_hdr_seq")
    _hdr_ts = _fe_prop("_hdr_ts")
    tracking_drops = _fe_prop("tracking_drops")
    last_rx_mono = _fe_prop("last_rx_mono")
    last_enqueue_mono = _fe_prop("last_enqueue_mono")
    del _fe_prop

    @property
    def _event_source(self) -> str:
        return self._fe.source

    @property
    def _throttle_clock(self) -> str:
        return self._fe.throttle.clock

    @property
    def _nonkf_min_interval_s(self) -> float:
        return self._fe.throttle.interval_s

    @_nonkf_min_interval_s.setter
    def _nonkf_min_interval_s(self, v: float) -> None:
        self._fe.throttle.interval_s = float(v)

    @property
    def _last_nonkf_enq_mono(self) -> float:
        return self._fe.throttle.last_admit_mono

    @_last_nonkf_enq_mono.setter
    def _last_nonkf_enq_mono(self, v: float) -> None:
        self._fe.throttle.last_admit_mono = float(v)

    @property
    def _last_nonkf_admit_sensor_ns(self) -> Optional[int]:
        return self._fe.throttle.last_admit_sensor_ns

    @_last_nonkf_admit_sensor_ns.setter
    def _last_nonkf_admit_sensor_ns(self, v: Optional[int]) -> None:
        self._fe.throttle.last_admit_sensor_ns = v

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
        one (a same-id reconnect keeps the epoch). Returns the epoch."""
        return self._fe.new_session(session_id)

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

        # Reset per-session state (frame counter, throttle stamps, last enqueued stamp)
        self._fe.reset_session_state()
        frames_received = 0
        frames_enqueued = 0
        t_start = time.monotonic()

        # ── Receive loop (binary frames + text messages) ──
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.receive":
                    self._fe.last_rx_mono = time.monotonic()
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
                            if pkt is not None and self._fe.enqueue(pkt):
                                frames_enqueued += 1
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
        handle_lens_text_message(
            text, apply_camera_flip=self._apply_camera_flip, session_id=self._active_session_id,
            on_pose_corrections=self._on_pose_corrections, on_pose_corrections_batch=self._on_pose_corrections_batch,
            on_frame_correction=self._on_frame_correction, source=self._fe.source,
        )

    @staticmethod
    def _corrections_pose_to_4x4(pose_data: list) -> Optional[np.ndarray]:
        """Convert pose correction data to a 4x4 row-major matrix."""
        return corrections_pose_to_4x4(pose_data)

    # ── Non-keyframe throttle (delegate; the algorithm is NonKfThrottle) ──

    def _admit_nonkf(self, sensor_ts_ns: Any) -> bool:
        """Decide whether a non-keyframe passes the min-interval throttle, and
        stamp the decision if it does (see ingest_frontend.NonKfThrottle)."""
        return self._fe.throttle.admit(sensor_ts_ns)

    # ── Frame-flow trace (delegate) ──

    def _trace_rx(self, decision: str, reason: str = "", *, pkt: Optional[FramePacket] = None,
                  seq: Any = None, ts: Any = None, is_kf: Optional[bool] = None,
                  frame_count: Optional[int] = None, queue_depth: Optional[int] = None,
                  depth_valid_frac: Optional[float] = None, lane: Optional[str] = None,
                  rx_seq: Optional[int] = None) -> None:
        """Emit one ReceiverEvent (see IngestFrontEnd.trace). Returns None."""
        self._fe.trace(decision, reason, pkt=pkt, seq=seq, ts=ts, is_kf=is_kf, frame_count=frame_count,
                       queue_depth=queue_depth, depth_valid_frac=depth_valid_frac, lane=lane, rx_seq=rx_seq)
        return None

    # ── Pose parse (delegates to the codec layer) ──

    def _pose_from_header(self, header: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Header T_wc -> (t_wc, q_wc_xyzw) in the engine convention: parse, then
        (when apply_camera_flip) the ARKit -> OpenCV camera flip."""
        t_wc, q_xyzw = parse_arkit_pose(
            T_wc_data=header["T_wc"],
            pose_format=header.get("pose_format", "matrix4x4_col_major"),
        )
        return normalize_pose_convention(t_wc, q_xyzw, CONVENTION_ARKIT if self._apply_camera_flip else CONVENTION_OPENCV)

    @staticmethod
    def _wall_and_clock(header: dict) -> Tuple[float, str]:
        """(wall stamp, pose_clock): the header's unix_timestamp tagged
        "sender", or this process's time.time() tagged "server"."""
        raw = header.get("unix_timestamp")
        return (float(raw), "sender") if raw else (time.time(), "server")

    # ── Binary message parsing ──

    def _parse_binary_message(self, data: bytes) -> Optional[FramePacket]:
        """Parse a single binary WebSocket message into a FramePacket, or None
        when the chain dropped it (malformed / tracking state / throttle /
        refused before decode -- each leaves its trace line). A failure after
        the header leaves one 'parse_error' line with the header ids and is
        re-raised (the stream loop logs and continues; the replayer likewise).
        The caller enqueues the packet (``self._fe.enqueue``)."""
        return parse_lens_message(self._fe, data, apply_camera_flip=self._apply_camera_flip,
                                  latency_analytics=self._latency_analytics, session_id=self._active_session_id)

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
            "last_rx_mono": self._fe.last_rx_mono,
            "last_enqueue_mono": self._fe.last_enqueue_mono,
            "tracking_drops": self._fe.tracking_drops,
        }

    def stop(self) -> None:
        """Signal shutdown (best-effort; daemon thread exits with process)."""
        logger.info("[websocket] Stop requested")
