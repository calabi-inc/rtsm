"""
Lens recording -> rosbag2 (MCAP storage) converter, plus a strict reader for
the layout it writes (Gate 4.5 plan, P3 task 0).

The eval CLI reads bags. session1 -- the anchored dataset -- is a Lens
recording (``messages.bin`` + ``index.jsonl`` + ``meta.json``), so the bag path
must reproduce what the websocket receiver produced from it, array for array.
This module therefore decodes with the RECEIVER'S code (``lens_raw_frame`` +
``rtsm.io.codecs``) and writes exactly what it decoded:

    /camera/color/image_raw      sensor_msgs/Image  rgb8    NV12/JPEG -> BGR by the receiver's codec, channel-swapped (lossless)
    /camera/depth/image_rect_raw sensor_msgs/Image  16UC1   the wire bytes verbatim (uint16 millimetres, 0 = invalid)
    /camera/confidence/image_raw sensor_msgs/Image  mono8   the ARKit confidence map verbatim (0/1/2), when the frame carries one
    /camera/color/camera_info    sensor_msgs/CameraInfo     at RGB resolution, per frame (ARKit intrinsics vary per frame)
    /tf                          tf2_msgs/TFMessage         map -> camera_optical at the image stamp, ARKit flip BAKED IN (opencv convention)
    /arkit/frame_seq             std_msgs/UInt32            the source frame id (session1 has gaps: 0..405 over 240 frames)
    /arkit/tracking_state        std_msgs/String            verbatim ("normal", "limited...")

``header.stamp`` = the sensor stamp (``timestamp_ns``, ARKit's device-uptime
clock -- the stamp the sensor clock and the throttle run on); the rosbag2
message time (MCAP ``log_time``) = the sender's wall stamp (``unix_timestamp``)
in nanoseconds. Bag ``custom_data`` records the layout version, the session id
and whether the flip was baked in (then the reader runs
``apply_camera_flip=False``).

The pose convention rule (CLAUDE.md): the receiver right-multiplies ARKit's
``T_wc`` by ``diag(1,-1,-1,1)`` once at ingest. The converter goes through the
same two codec calls, so the bag's ``T_wc`` IS the receiver's ``T_wc``; the
parity test (``tests/evaluation/test_recording_mcap.py``) pins it bit for bit.

Requires the ``[eval]`` extra (``rosbags``); imported lazily so ``rtsm`` never
needs it.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from rtsm.core.datamodel import PinholeIntrinsics
from rtsm.io import codecs
from rtsm.io.codecs import UnsupportedEncoding
from rtsm.io.websocket import LensFramingError, lens_raw_frame

logger = logging.getLogger(__name__)

BAG_LAYOUT_VERSION = 1

TOPIC_RGB = "/camera/color/image_raw"
TOPIC_DEPTH = "/camera/depth/image_rect_raw"
TOPIC_CONF = "/camera/confidence/image_raw"
TOPIC_INFO = "/camera/color/camera_info"
TOPIC_TF = "/tf"
TOPIC_SEQ = "/arkit/frame_seq"
TOPIC_TRACKING = "/arkit/tracking_state"
FRAME_WORLD = "map"
FRAME_CAMERA = "camera_optical"

# wire depth encoding -> ROS image encoding (verbatim bytes); anything else is refused explicitly
_DEPTH_ROS_ENCODING = {"uint16_mm": "16UC1", "float32_m": "32FC1"}
_DEPTH_WIRE_ENCODING = {v: k for k, v in _DEPTH_ROS_ENCODING.items()}
_DEPTH_BYTES_PER_PX = {"16UC1": 2, "32FC1": 4}

_PIP_HINT = "install the eval extra: pip install \"rtsm[eval]\"  (rosbags, mcap, mcap-ros2-support)"


# ───────────────────────────── the recording ─────────────────────────────

def read_recording_meta(recording_dir: str | os.PathLike) -> dict:
    p = Path(recording_dir) / "meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def iter_recording(recording_dir: str | os.PathLike) -> Iterator[Tuple[dict, bytes]]:
    """(index entry, binary message bytes) in recording order."""
    d = Path(recording_dir)
    idx_path, bin_path = d / "index.jsonl", d / "messages.bin"
    if not idx_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"not a Lens recording (index.jsonl + messages.bin): {d}")
    entries = [json.loads(line) for line in idx_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(bin_path, "rb") as f:
        for e in entries:
            f.seek(int(e["offset"]))
            data = f.read(int(e["length"]))
            if len(data) != int(e["length"]):
                raise EOFError(f"messages.bin truncated at seq {e.get('seq')}: wanted {e['length']} bytes, got {len(data)}")
            yield e, data


@dataclass
class DecodedFrame:
    """One frame as the RECEIVER would see it (before the confidence filter):
    the comparison unit of the parity test. Produced from the recording by
    ``decode_recording_frame`` and from a bag by ``iter_bag_frames``."""
    seq: Optional[int]
    t_sensor_ns: int
    t_wall_utc_s: Optional[float]
    tracking_state: str
    rgb_bgr: np.ndarray                     # (H, W, 3) uint8, BGR (the FramePacket contract)
    depth_raw: bytes                        # the wire bytes
    depth_encoding: str                     # uint16_mm | float32_m (wire names)
    depth_width: int
    depth_height: int
    depth_scale: float
    confidence: Optional[np.ndarray]        # (h, w) uint8 or None
    intrinsics: PinholeIntrinsics           # at RGB resolution
    t_wc: np.ndarray                        # float32 (3,)
    q_wc_xyzw: np.ndarray                   # float32 (4,)
    session_id: Optional[str] = None

    def depth_m(self) -> Optional[np.ndarray]:
        """The receiver's depth decode of the raw bytes (NaN where invalid), pre confidence filter."""
        return codecs.decode_depth(self.depth_raw, self.depth_encoding, self.depth_width, self.depth_height, self.depth_scale)


def decode_recording_frame(data: bytes, *, apply_camera_flip: bool, session_id: Optional[str] = None,
                           source: str = "recording") -> DecodedFrame:
    """The receiver's framing + codecs on one binary message. Raises
    LensFramingError on truncation, UnsupportedEncoding on a payload this
    layout cannot carry, and whatever the codecs raise on a bad pose."""
    raw = lens_raw_frame(data, source=source, apply_camera_flip=apply_camera_flip, session_id=session_id)
    h = raw.header
    if h.t_sensor_ns is None:
        raise ValueError("frame without a sensor stamp (timestamp_ns) cannot be placed in a bag")
    if h.intrinsics is None:
        raise KeyError(h.extra.get("intrinsics_error") or "frame without intrinsics")
    if h.depth is None or h.depth.encoding not in _DEPTH_ROS_ENCODING:
        raise UnsupportedEncoding(
            f"depth_format {getattr(h.depth, 'encoding', None)!r} is not carried by bag layout {BAG_LAYOUT_VERSION} "
            f"(supported: {sorted(_DEPTH_ROS_ENCODING)})")
    if not isinstance(h.depth.data, (bytes, bytearray, memoryview)):
        raise UnsupportedEncoding("depth payload must be the wire bytes")
    bgr = codecs.decode_rgb(h.rgb.data, h.rgb.encoding, h.rgb.width, h.rgb.height)
    t, q = codecs.parse_pose(h.pose_raw, h.pose_format)
    t, q = codecs.normalize_pose_convention(t, q, h.pose_convention)
    conf = codecs.decode_confidence(h.confidence.data, h.confidence.width, h.confidence.height) if h.confidence is not None else None
    return DecodedFrame(
        seq=(int(h.seq) if h.seq is not None else None), t_sensor_ns=int(h.t_sensor_ns),
        t_wall_utc_s=(float(h.t_wall_utc_s) if h.t_wall_utc_s else None), tracking_state=str(h.tracking_state),
        rgb_bgr=np.ascontiguousarray(bgr), depth_raw=bytes(h.depth.data), depth_encoding=h.depth.encoding,
        depth_width=int(h.depth.width), depth_height=int(h.depth.height), depth_scale=float(h.depth.scale),
        confidence=(np.ascontiguousarray(conf) if conf is not None else None), intrinsics=h.intrinsics,
        t_wc=np.asarray(t, dtype=np.float32), q_wc_xyzw=np.asarray(q, dtype=np.float32),
        session_id=(h.session_id or session_id),
    )


# ───────────────────────────── the converter ─────────────────────────────

@dataclass
class ConversionSummary:
    recording_dir: str
    out_dir: str
    frames: int = 0
    skipped: List[Tuple[Optional[int], str]] = field(default_factory=list)
    topics: Dict[str, int] = field(default_factory=dict)
    bytes_in: int = 0
    bytes_out: int = 0
    seconds: float = 0.0
    session_id: Optional[str] = None
    arkit_flip_baked: bool = True
    compression: Optional[str] = None
    first_stamp_ns: Optional[int] = None
    last_stamp_ns: Optional[int] = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["skipped"] = [list(x) for x in self.skipped]
        return d


def _rosbags():
    try:
        from rosbags.rosbag2 import CompressionFormat, CompressionMode, Reader, StoragePlugin, Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:  # pragma: no cover - environment
        raise RuntimeError(f"rosbags is not installed; {_PIP_HINT}") from e
    return Writer, Reader, StoragePlugin, CompressionMode, CompressionFormat, get_typestore(Stores.ROS2_HUMBLE)


def _split_ns(ns: int) -> Tuple[int, int]:
    ns = int(ns)
    return ns // 1_000_000_000, ns % 1_000_000_000


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def convert_recording_to_mcap(recording_dir: str | os.PathLike, out_dir: str | os.PathLike, *,
                              apply_camera_flip: bool = True, max_frames: Optional[int] = None,
                              compression: Optional[str] = "zstd", overwrite: bool = False) -> ConversionSummary:
    """Write ``out_dir`` (a rosbag2 directory: ``metadata.yaml`` + one ``.mcap``)
    from the recording. ``apply_camera_flip`` bakes the ARKit->OpenCV flip
    into ``/tf`` exactly as the receiver applies it (session1 was recorded
    and anchored with the flip on); the bag then runs with the flip OFF.
    A frame the receiver would refuse to parse (truncated, no stamp, a
    depth encoding this layout cannot carry) is skipped and listed in the
    summary -- except UnsupportedEncoding, which is raised: a whole
    recording in an unsupported format is a configuration error, not noise.
    """
    Writer, _Reader, StoragePlugin, CompressionMode, CompressionFormat, ts = _rosbags()
    rec = Path(recording_dir)
    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (overwrite=True to replace it)")
        shutil.rmtree(out)
    meta = read_recording_meta(rec)
    session_id = meta.get("session_id")
    summary = ConversionSummary(recording_dir=str(rec), out_dir=str(out), session_id=session_id,
                                arkit_flip_baked=bool(apply_camera_flip), compression=compression)
    t0 = time.perf_counter()
    w = Writer(out, version=9, storage_plugin=StoragePlugin.MCAP)
    if compression == "zstd":
        w.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)     # before open()
    elif compression not in (None, "none"):
        raise ValueError(f"unknown compression {compression!r} (zstd | none)")
    try:
        _write_bag(w, rec, meta, session_id, summary, ts, apply_camera_flip, max_frames)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)               # never leave a half-written bag behind
        raise
    summary.seconds = round(time.perf_counter() - t0, 3)
    summary.bytes_out = _dir_size(out)
    return summary


def _write_bag(w, rec: Path, meta: dict, session_id, summary: ConversionSummary, ts, apply_camera_flip: bool,
               max_frames: Optional[int]) -> None:
    T = ts.types
    Image, Header, Time = T["sensor_msgs/msg/Image"], T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    CameraInfo, ROI = T["sensor_msgs/msg/CameraInfo"], T["sensor_msgs/msg/RegionOfInterest"]
    TFMessage, TransformStamped, Transform = T["tf2_msgs/msg/TFMessage"], T["geometry_msgs/msg/TransformStamped"], T["geometry_msgs/msg/Transform"]
    Vector3, Quaternion, String, UInt32 = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"], T["std_msgs/msg/String"], T["std_msgs/msg/UInt32"]
    with w:
        for k, v in (("rtsm_bag_layout", str(BAG_LAYOUT_VERSION)), ("rtsm_source", "lens_recording"),
                     ("recording_dir", rec.name), ("session_id", str(session_id or "")),
                     ("device_name", str(meta.get("device_name") or "")),
                     ("arkit_flip_baked", "true" if apply_camera_flip else "false"),
                     ("pose_convention", "opencv" if apply_camera_flip else "arkit"),
                     ("world_frame", FRAME_WORLD), ("camera_frame", FRAME_CAMERA)):
            w.set_custom_data(k, v)
        conns = {
            TOPIC_RGB: w.add_connection(TOPIC_RGB, "sensor_msgs/msg/Image", typestore=ts),
            TOPIC_DEPTH: w.add_connection(TOPIC_DEPTH, "sensor_msgs/msg/Image", typestore=ts),
            TOPIC_CONF: w.add_connection(TOPIC_CONF, "sensor_msgs/msg/Image", typestore=ts),
            TOPIC_INFO: w.add_connection(TOPIC_INFO, "sensor_msgs/msg/CameraInfo", typestore=ts),
            TOPIC_TF: w.add_connection(TOPIC_TF, "tf2_msgs/msg/TFMessage", typestore=ts),
            TOPIC_SEQ: w.add_connection(TOPIC_SEQ, "std_msgs/msg/UInt32", typestore=ts),
            TOPIC_TRACKING: w.add_connection(TOPIC_TRACKING, "std_msgs/msg/String", typestore=ts),
        }
        counts = {k: 0 for k in conns}

        def put(topic: str, msg: Any, msgtype: str, log_ns: int) -> None:
            w.write(conns[topic], int(log_ns), ts.serialize_cdr(msg, msgtype))
            counts[topic] += 1

        for entry, data in iter_recording(rec):
            if max_frames is not None and summary.frames >= int(max_frames):
                break
            summary.bytes_in += len(data)
            seq_hint = entry.get("seq")
            try:
                fr = decode_recording_frame(data, apply_camera_flip=apply_camera_flip, session_id=session_id)
            except UnsupportedEncoding:
                raise
            except (LensFramingError, ValueError, KeyError) as e:
                summary.skipped.append((seq_hint, f"{type(e).__name__}: {e}"))
                logger.warning("[recording_mcap] skipping recording seq %s: %s", seq_hint, e)
                continue
            sec, nsec = _split_ns(fr.t_sensor_ns)
            stamp = Time(sec=sec, nanosec=nsec)
            log_ns = int(round(fr.t_wall_utc_s * 1e9)) if fr.t_wall_utc_s else int(fr.t_sensor_ns)
            hdr = Header(stamp=stamp, frame_id=FRAME_CAMERA)

            h, wd = fr.rgb_bgr.shape[:2]
            rgb = np.ascontiguousarray(fr.rgb_bgr[..., ::-1]).reshape(-1)
            put(TOPIC_RGB, Image(header=hdr, height=h, width=wd, encoding="rgb8", is_bigendian=0, step=3 * wd, data=rgb),
                "sensor_msgs/msg/Image", log_ns)

            ros_depth = _DEPTH_ROS_ENCODING[fr.depth_encoding]
            bpp = _DEPTH_BYTES_PER_PX[ros_depth]
            if len(fr.depth_raw) != fr.depth_width * fr.depth_height * bpp:
                summary.skipped.append((seq_hint, f"depth size {len(fr.depth_raw)} != {fr.depth_width}x{fr.depth_height}x{bpp}"))
                counts[TOPIC_RGB] -= 1                     # keep the frame atomic: nothing of it stays in the bag
                continue
            put(TOPIC_DEPTH, Image(header=hdr, height=fr.depth_height, width=fr.depth_width, encoding=ros_depth, is_bigendian=0,
                                   step=bpp * fr.depth_width, data=np.frombuffer(fr.depth_raw, dtype=np.uint8)),
                "sensor_msgs/msg/Image", log_ns)
            if fr.confidence is not None:
                ch, cw = fr.confidence.shape[:2]
                put(TOPIC_CONF, Image(header=hdr, height=ch, width=cw, encoding="mono8", is_bigendian=0, step=cw,
                                      data=np.ascontiguousarray(fr.confidence).reshape(-1)), "sensor_msgs/msg/Image", log_ns)
            i = fr.intrinsics
            K = np.array([i.fx, 0.0, i.cx, 0.0, i.fy, i.cy, 0.0, 0.0, 1.0], dtype=np.float64)
            P = np.array([i.fx, 0.0, i.cx, 0.0, 0.0, i.fy, i.cy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
            put(TOPIC_INFO, CameraInfo(header=hdr, height=int(i.height), width=int(i.width), distortion_model="plumb_bob",
                                       d=np.zeros(0, dtype=np.float64), k=K, r=np.eye(3, dtype=np.float64).reshape(-1), p=P,
                                       binning_x=0, binning_y=0, roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)),
                "sensor_msgs/msg/CameraInfo", log_ns)
            t, q = fr.t_wc.astype(np.float64), fr.q_wc_xyzw.astype(np.float64)
            put(TOPIC_TF, TFMessage(transforms=[TransformStamped(
                header=Header(stamp=stamp, frame_id=FRAME_WORLD), child_frame_id=FRAME_CAMERA,
                transform=Transform(translation=Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                                    rotation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))))]),
                "tf2_msgs/msg/TFMessage", log_ns)
            if fr.seq is not None:
                put(TOPIC_SEQ, UInt32(data=int(fr.seq)), "std_msgs/msg/UInt32", log_ns)
            put(TOPIC_TRACKING, String(data=fr.tracking_state), "std_msgs/msg/String", log_ns)

            summary.frames += 1
            summary.first_stamp_ns = fr.t_sensor_ns if summary.first_stamp_ns is None else summary.first_stamp_ns
            summary.last_stamp_ns = fr.t_sensor_ns
        summary.topics = counts
        w.set_custom_data("frames", str(summary.frames))


# ───────────────────────────── the strict reader ─────────────────────────────

def read_bag_custom_data(bag_dir: str | os.PathLike) -> Dict[str, str]:
    """``custom_data`` from ``metadata.yaml`` (rosbags' Reader does not expose it)."""
    import yaml
    p = Path(bag_dir) / "metadata.yaml"
    if not p.is_file():
        return {}
    info = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("rosbag2_bagfile_information") or {}
    return {str(k): str(v) for k, v in (info.get("custom_data") or {}).items()}


def iter_bag_frames(bag_dir: str | os.PathLike) -> Iterator[DecodedFrame]:
    """Frames of a bag in THIS layout, decoded the way the receiver decodes
    (``rgb8`` -> BGR is the channel-order rule: keyed on the message encoding,
    never on pixel statistics). Messages of one frame share the rosbag2
    message time (the wall stamp), which is the grouping key; a frame is
    emitted once a later time arrives, so at most two frames are buffered.
    Task 1's reader generalises this (topic discovery, TF composition,
    other encodings); this one refuses anything it does not know."""
    _Writer, Reader, _SP, _CM, _CF, ts = _rosbags()
    custom = read_bag_custom_data(bag_dir)
    session_id = custom.get("session_id") or None
    pending: Dict[int, dict] = {}

    def emit(bucket: dict) -> DecodedFrame:
        for need in ("rgb", "depth", "info", "tf"):
            if need not in bucket:
                raise ValueError(f"incomplete frame at log_time {bucket.get('_log')}: missing {need}")
        rgb, depth, info, tf = bucket["rgb"], bucket["depth"], bucket["info"], bucket["tf"]
        if rgb.encoding != "rgb8":
            raise UnsupportedEncoding(f"bag layout {BAG_LAYOUT_VERSION} reader expects rgb8, got {rgb.encoding!r}")
        if depth.encoding not in _DEPTH_WIRE_ENCODING:
            raise UnsupportedEncoding(f"depth encoding {depth.encoding!r} not supported (16UC1 | 32FC1)")
        arr = np.asarray(rgb.data, dtype=np.uint8).reshape(int(rgb.height), int(rgb.width), 3)
        bgr = np.ascontiguousarray(arr[..., ::-1])
        depth_raw = np.asarray(depth.data, dtype=np.uint8).tobytes()
        conf = bucket.get("conf")
        conf_arr = (np.asarray(conf.data, dtype=np.uint8).reshape(int(conf.height), int(conf.width)) if conf is not None else None)
        k = np.asarray(info.k, dtype=np.float64)
        intr = PinholeIntrinsics(width=int(info.width), height=int(info.height), fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))
        tr = tf.transforms[0].transform
        t = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=np.float32)
        q = np.array([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w], dtype=np.float32)
        stamp_ns = int(rgb.header.stamp.sec) * 1_000_000_000 + int(rgb.header.stamp.nanosec)
        log_ns = int(bucket["_log"])
        return DecodedFrame(
            seq=bucket.get("seq"), t_sensor_ns=stamp_ns, t_wall_utc_s=(log_ns / 1e9 if log_ns != stamp_ns else None),
            tracking_state=bucket.get("tracking", "not_available"), rgb_bgr=bgr, depth_raw=depth_raw,
            depth_encoding=_DEPTH_WIRE_ENCODING[depth.encoding], depth_width=int(depth.width), depth_height=int(depth.height),
            depth_scale=(0.001 if depth.encoding == "16UC1" else 1.0), confidence=conf_arr, intrinsics=intr,
            t_wc=t, q_wc_xyzw=q, session_id=session_id,
        )

    with Reader(Path(bag_dir)) as r:
        for conn, log_ns, raw in r.messages():
            # flush every buffered frame OLDER than this message time
            for older in sorted(k for k in pending if k < log_ns):
                yield emit(pending.pop(older))
            b = pending.setdefault(log_ns, {"_log": log_ns})
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            if conn.topic == TOPIC_RGB:
                b["rgb"] = msg
            elif conn.topic == TOPIC_DEPTH:
                b["depth"] = msg
            elif conn.topic == TOPIC_CONF:
                b["conf"] = msg
            elif conn.topic == TOPIC_INFO:
                b["info"] = msg
            elif conn.topic == TOPIC_TF:
                b["tf"] = msg
            elif conn.topic == TOPIC_SEQ:
                b["seq"] = int(msg.data)
            elif conn.topic == TOPIC_TRACKING:
                b["tracking"] = str(msg.data)
            else:
                logger.debug("[recording_mcap] ignoring topic %s", conn.topic)
        for k in sorted(pending):
            yield emit(pending.pop(k))
