"""
Ingest contracts (Gate 4.5 plan, P3 task 0.5) -- the public seam between a
transport adapter and the one ingest front-end.

CONTRACT_VERSION 1 (2026-09-24). Additive changes bump the minor number in
the docs; a renamed or removed field bumps CONTRACT_VERSION.

An adapter's whole job is bytes -> ``RawFrame`` (header fields + still-encoded
payloads + arrival stamps) plus pose / correction / tracking-state events. It
never decodes pixels, never applies keyframe or throttle logic, never touches
the ingest queue. The front-end (``rtsm/io/ingest_frontend.py``) does the rest,
identically for every source, which is what keeps the determinism anchors
(G1-A / G1-B) meaningful across sources.

Pose conventions: ``PoseSample`` / ``FrameHeader.pose_*`` carry the pose in the
SOURCE convention plus a ``pose_convention`` tag; the codec layer normalises
to the OpenCV camera convention the engine uses (ARKit's Y-up / Z-back camera
axes are flipped once at ingest, as today). The world frame is never touched.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from rtsm.core.datamodel import PinholeIntrinsics

CONTRACT_VERSION = 1

# Pose formats the codec layer parses (rtsm/io/codecs.py)
POSE_FMT_ARKIT_MATRIX = "matrix4x4_col_major"
POSE_FMT_ARKIT_QUAT = "quat_translation"
POSE_FMT_RTABMAP_EULER = "rtabmap_euler"          # [x, y, z, roll, pitch, yaw]
POSE_FMT_PREPARED = "prepared"                    # (t_wc, q_wc_xyzw) already parsed by the adapter
# Camera conventions
CONVENTION_ARKIT = "arkit"                        # Y-up, Z-back camera axes -> flipped to OpenCV
CONVENTION_OPENCV = "opencv"                      # engine convention, identity


@dataclass(slots=True)
class EncodedImage:
    """One still-encoded payload as the transport delivered it. ``data`` may
    also be an already-decoded ``np.ndarray`` with encoding ``raw_bgr`` /
    ``raw_depth_m`` (in-process producers, tests)."""
    data: Any                                     # bytes | memoryview | np.ndarray
    encoding: str                                 # rgb: jpeg|png|bgra|nv12|raw_bgr; depth: uint16_mm|float32_m|png_uint16|png_uint16_raw|raw_depth_m; confidence: uint8
    width: int = 0
    height: int = 0
    scale: float = 1.0                            # depth: metres per unit


@dataclass(slots=True)
class FrameHeader:
    """What the transport knows about a frame before anything is decoded."""
    source: str                                   # websocket | replay | zeromq | <adapter name>
    seq: Optional[int]                            # source frame id (None when the source has none)
    t_sensor_ns: Optional[int]                    # sensor stamp; 0 / missing -> None
    t_wall_utc_s: Optional[float]                 # the sender's wall stamp when it sends one, else None (-> server clock)
    tracking_state: str                           # verbatim source string; "not_available" when the source has none
    keyframe_hint: Optional[bool]                 # the source's own keyframe flag (ZeroMQ kf_pose); None when the source mints none
    rgb: EncodedImage
    depth: Optional[EncodedImage]
    intrinsics: Optional[PinholeIntrinsics]       # at RGB resolution (adapters call codecs.rescale_intrinsics); None when the source has none
    pose_raw: Any                                 # as the transport delivered it (list, dict, or (t, q) when prepared)
    pose_format: str = POSE_FMT_ARKIT_MATRIX
    pose_convention: str = CONVENTION_OPENCV
    confidence: Optional[EncodedImage] = None
    session_id: Optional[str] = None
    decode_key: Optional[Any] = None              # memo key when two frames may carry the same pixels (ZeroMQ camera stamp)
    pose_frame_id: str = "world"                  # PoseStamped.frame_id the packet carries ("arkit" on the Lens path)
    keep_encoded_rgb: bool = True                 # FramePacket.rgb_jpeg = the JPEG bytes (zero-copy viz); the ZeroMQ path never did
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawFrame:
    """A frame offered to the front-end. Nothing decoded, nothing policy-derived."""
    header: FrameHeader
    t_arrival_mono: float = field(default_factory=time.monotonic)
    t_arrival_wall: float = field(default_factory=time.time)


class TrackingStatus(str, enum.Enum):
    """The total, lossy mapping every SLAM's tracking state fits into
    (slam-multisession-decision-2026-09.md, field 2)."""
    OK = "ok"
    DEGRADED = "degraded"
    LOST = "lost"
    UNKNOWN = "unknown"

    @classmethod
    def from_arkit(cls, state: Optional[str]) -> "TrackingStatus":
        s = (state or "").lower()
        if s == "normal":
            return cls.OK
        if s.startswith("limited"):
            return cls.DEGRADED
        if s in ("not_available", "notavailable", "unavailable"):
            return cls.LOST
        return cls.UNKNOWN

    @classmethod
    def from_rtabmap(cls, _state: Optional[str] = None) -> "TrackingStatus":
        return cls.UNKNOWN                        # the bridge publishes no tracking state


@dataclass(slots=True)
class PoseSample:
    """A receive-time pose event (field 1 of the PoseSource contract): what the
    pose mailbox and the pose ledger consume. ``t_wc`` / ``q_wc_xyzw`` are in
    the ENGINE convention (adapters normalise through the codec layer)."""
    t_sensor_ns: Optional[int]
    t_wall_utc_s: Optional[float]                 # None -> the front-end stamps time.time() and tags "server"
    t_wc: np.ndarray
    q_wc_xyzw: np.ndarray
    tracking: TrackingStatus = TrackingStatus.UNKNOWN
    tracking_raw: str = "not_available"
    covariance: Optional[np.ndarray] = None       # 6x6 when the source publishes one
    session_id: Optional[str] = None
    map_ref: Optional[str] = None


@dataclass(slots=True)
class FrameCorrection:
    """A retroactive pose correction event (field 5): loop closure, relocalisation
    in a prior map, or a reset. ``corrections`` maps a frame id (the memory's
    ``last_update_frame_id`` key, e.g. ``ws_<seq>``) to the corrected 4x4
    ``T_wc`` in the engine convention."""
    t_wall_utc_s: float
    session_id: Optional[str]
    kind: str                                     # loop_closure | relocalized_in_prior_map | reset | unknown
    corrections: Dict[str, np.ndarray]
    map_ref: Optional[str] = None


@runtime_checkable
class Source(Protocol):
    """What the runners need from any ingest source (a transport adapter)."""
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def liveness(self) -> dict: ...


@dataclass
class SourceContext:
    """Everything the runner wires into every source: the queue, the ingest
    settings, the receive-time sinks and the callbacks. Built once by
    ``rtsm/run.py`` / ``rtsm/demo.py`` and handed to ``sources.make_source``."""
    ingest_queue: Any
    clock_mode: str = "wall"                      # wall | sensor (resolved)
    keyframe_every_n: int = 30
    nonkf_min_interval_s: float = 0.5
    require_tracking_normal: bool = True
    confidence_threshold: int = 1
    apply_camera_flip: bool = False
    pose_sink: Optional[Callable[..., Any]] = None
    clearance_sink: Optional[Callable[..., Any]] = None
    event_sink: Optional[Callable[[Any], None]] = None
    ledger_sink: Optional[Callable[[Any], None]] = None
    latency_analytics: Any = None
    on_camera_frame: Optional[Callable[[Any], None]] = None
    on_keyframe: Optional[Callable[[Any], None]] = None
    on_pose_corrections: Optional[Callable[..., Any]] = None
    on_pose_corrections_batch: Optional[Callable[..., Any]] = None
    on_frame_correction: Optional[Callable[[FrameCorrection], None]] = None
    on_kf_packet: Optional[Callable[..., Any]] = None
    on_kf_pose_update: Optional[Callable[..., Any]] = None
    on_raw_message: Optional[Callable[..., Any]] = None
    on_handshake_done: Optional[Callable[..., Any]] = None
