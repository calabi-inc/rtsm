"""
Bag reader for the ``bag`` ingest source (Gate 4.5 plan, P3 task 1): ROS 1
``.bag`` files, rosbag2 directories (sqlite3 or MCAP storage) and bare MCAP
files -> ``RawFrame``s for the ingest front-end.

Three stages, each explicit about what it chose and what it lost:

1. ``open_bag``        what the file contains (kind, storage, topics, type definitions).
2. ``resolve_topics``  which topics play RGB / depth / CameraInfo / pose / tracking /
                       seq, by rules or by ``io.bag.topics`` overrides; every choice
                       carries the rule that made it.
3. ``iter_bag_frames`` a first pass builds the pose source (TF + TF static, or
                       odometry) and reads the camera infos; a second pass pairs RGB
                       with depth by header stamp (nearest within a tolerance, one
                       depth per RGB), composes the camera pose at the image stamp,
                       maps the ROS encodings onto the codec layer WITHOUT decoding
                       (admit-before-decode holds for bags too) and yields a
                       ``RawFrame`` in the OpenCV convention.

What v1 refuses, with the reason a customer can act on (``BagRefusal``): no
RGB / depth / CameraInfo topic, no pose source (no TF chain from the camera
frame to a root and no odometry), depth not registered to RGB (the depth K,
scaled to the RGB size, differs by more than 1 % -- unless ``assume_aligned``),
``ros2idl`` schemas, RVL-compressed depth, big-endian images.

Decoding: ``rosbags`` is the one CDR implementation. ROS 1 bags and rosbag2
directories go through ``rosbags.highlevel.AnyReader`` with a default typestore
(bags written before message definitions were embedded carry none); a bare
``.mcap`` goes through the ``mcap`` container reader with the stock typestore
plus any schema the MCAP defines that the store lacks.

Requires the ``[eval]`` extra; imported lazily.
"""
from __future__ import annotations

import bisect
import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from rtsm.core.datamodel import PinholeIntrinsics
from rtsm.io import codecs
from rtsm.io.codecs import UnsupportedEncoding
from rtsm.io.contracts import CONVENTION_OPENCV, POSE_FMT_PREPARED, EncodedImage, FrameHeader, RawFrame
from rtsm.io.detections import DETECTION_2D, DETECTION_3D, adapter_for
from rtsm.io.msgdefs import register_vision_msgs
from rtsm.io.tf_buffer import TfBuffer, TfLookupError, make_T, norm_frame

logger = logging.getLogger(__name__)

IMAGE_T = "sensor_msgs/msg/Image"
COMPRESSED_T = "sensor_msgs/msg/CompressedImage"
INFO_T = "sensor_msgs/msg/CameraInfo"
TF_TYPES = ("tf2_msgs/msg/TFMessage", "tf/msg/tfMessage", "tf/tfMessage")
ODOM_T = "nav_msgs/msg/Odometry"
POSE_T = "geometry_msgs/msg/PoseStamped"
STRING_T = "std_msgs/msg/String"
U32_T = "std_msgs/msg/UInt32"

PREFERRED_WORLD_FRAMES = ("map", "world", "odom")
REGISTRATION_TOLERANCE = 0.01
_PIP_HINT = "install the eval extra: pip install \"rtsm[eval]\""


class BagRefusal(ValueError):
    """The bag cannot be read as RGB-D + pose in v1. ``reasons`` = [(code, detail)]."""

    def __init__(self, reasons: Sequence[Tuple[str, str]]) -> None:
        self.reasons = list(reasons)
        self.codes = [c for c, _ in self.reasons]
        super().__init__("; ".join(f"{c}: {d}" for c, d in self.reasons))


def norm_topic(name: str) -> str:
    """rosbag2 allows relative names (``d455_1_rgb_image``); everything is compared with a leading slash."""
    return "/" + str(name).strip().lstrip("/")


# ───────────────────────────── stage 1: what the file contains ─────────────────────────────

@dataclass
class TopicInfo:
    name: str                     # normalised (leading slash)
    msgtype: str
    count: int
    raw_name: str                 # as stored


@dataclass
class BagInfo:
    path: str
    kind: str                     # ros1 | rosbag2 | mcap
    storage: str                  # bag | sqlite3 | mcap
    topics: Dict[str, TopicInfo]
    has_typedefs: bool
    message_count: int
    duration_s: float
    typestore: str
    custom_data: Dict[str, str] = field(default_factory=dict)

    def of_type(self, *types: str) -> List[TopicInfo]:
        return [t for t in self.topics.values() if t.msgtype in types]


def _typestore(name: str):
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:  # pragma: no cover - environment
        raise RuntimeError(f"rosbags is not installed; {_PIP_HINT}") from e
    key = {"humble": "ROS2_HUMBLE", "iron": "ROS2_IRON", "jazzy": "ROS2_JAZZY", "foxy": "ROS2_FOXY",
           "noetic": "ROS1_NOETIC"}.get(str(name).lower())
    if key is None or not hasattr(Stores, key):
        raise ValueError(f"unknown typestore {name!r} (humble | iron | jazzy | foxy | noetic)")
    return get_typestore(getattr(Stores, key))


def _bag_kind(path: Path) -> Tuple[str, str]:
    if path.is_dir():
        if (path / "metadata.yaml").is_file():
            return "rosbag2", ("mcap" if list(path.glob("*.mcap")) else "sqlite3")
        raise FileNotFoundError(f"{path} is a directory without metadata.yaml (not a rosbag2 bag)")
    if path.is_file():
        suf = path.suffix.lower()
        if suf == ".mcap":
            return "mcap", "mcap"
        if suf == ".bag":
            return "ros1", "bag"
        if suf == ".db3":
            raise FileNotFoundError(f"{path}: pass the rosbag2 DIRECTORY (with metadata.yaml), not the .db3 file")
    raise FileNotFoundError(f"not a bag: {path}")


def _read_custom_data(path: Path) -> Dict[str, str]:
    md = path / "metadata.yaml"
    if not md.is_file():
        return {}
    try:
        import yaml
        info = (yaml.safe_load(md.read_text(encoding="utf-8")) or {}).get("rosbag2_bagfile_information") or {}
        return {str(k): str(v) for k, v in (info.get("custom_data") or {}).items()}
    except Exception:  # noqa: BLE001 -- metadata extras are optional
        return {}


class _RosbagsStream:
    """ROS 1 bags and rosbag2 directories through rosbags' AnyReader."""

    def __init__(self, path: Path, typestore_name: str) -> None:
        from rosbags.highlevel import AnyReader
        self.kind, self.storage = _bag_kind(path)
        self._reader = AnyReader([path], default_typestore=_typestore(typestore_name))
        self._reader.open()
        # vision_msgs (external detections) are in no stock typestore: register the bundled definitions the bag lacks
        try:
            self.registered_types = register_vision_msgs(self._reader.typestore, "ros1" if self.kind == "ros1" else "ros2")
        except Exception as e:  # noqa: BLE001 -- a bag without detections never needs them
            logger.debug("[bag] vision_msgs registration skipped: %s", e)
            self.registered_types = []
        topics: Dict[str, TopicInfo] = {}
        has_defs = False
        for c in self._reader.connections:
            n = norm_topic(c.topic)
            if getattr(c, "msgdef", ""):
                has_defs = True
            if n in topics:
                topics[n].count += int(c.msgcount)
            else:
                topics[n] = TopicInfo(name=n, msgtype=c.msgtype, count=int(c.msgcount), raw_name=c.topic)
        self.info = BagInfo(path=str(path), kind=self.kind, storage=self.storage, topics=topics,
                            has_typedefs=(has_defs or self.kind == "ros1"), message_count=int(self._reader.message_count),
                            duration_s=round(self._reader.duration / 1e9, 3), typestore=typestore_name,
                            custom_data=_read_custom_data(path) if path.is_dir() else {})

    def messages(self, topics: Sequence[str]) -> Iterator[Tuple[str, str, int, Any]]:
        want = {norm_topic(t) for t in topics}
        conns = [c for c in self._reader.connections if norm_topic(c.topic) in want]
        if not conns:
            return
        for conn, log_ns, raw in self._reader.messages(connections=conns):
            yield norm_topic(conn.topic), conn.msgtype, int(log_ns), self._reader.deserialize(raw, conn.msgtype)

    def close(self) -> None:
        self._reader.close()


class _McapStream:
    """A bare .mcap through the mcap container reader + rosbags CDR."""

    def __init__(self, path: Path, typestore_name: str) -> None:
        try:
            from mcap.reader import make_reader
        except ImportError as e:  # pragma: no cover - environment
            raise RuntimeError(f"mcap is not installed; {_PIP_HINT}") from e
        from rosbags.typesys import get_types_from_msg
        self._fh = open(path, "rb")
        try:
            self._reader = make_reader(self._fh)
            summ = self._reader.get_summary()
            if summ is None:
                raise ValueError(f"{path}: MCAP without a summary section (unindexed); re-index it first")
            bad = sorted({s.encoding for s in summ.schemas.values()} - {"ros2msg", "ros1msg"})
            if bad:
                names = sorted(s.name for s in summ.schemas.values() if s.encoding in bad)
                raise BagRefusal([("unsupported_schema_encoding",
                                   f"MCAP schemas {names} use encoding {bad}; only ros2msg / ros1msg text definitions are supported")])
            self._ts = _typestore(typestore_name)
            add: Dict[str, Any] = {}
            for s in summ.schemas.values():
                if s.name not in self._ts.types:
                    add.update(get_types_from_msg(s.data.decode("utf-8"), s.name))
            if add:
                self._ts.register(add)
            flavour = "ros1" if any(s.encoding == "ros1msg" for s in summ.schemas.values()) else "ros2"
            self.registered_types = register_vision_msgs(self._ts, flavour)
            stats = summ.statistics
            counts = dict(stats.channel_message_counts) if stats is not None else {}
            topics: Dict[str, TopicInfo] = {}
            self._raw_names: Dict[str, str] = {}
            for ch in summ.channels.values():
                n = norm_topic(ch.topic)
                topics[n] = TopicInfo(name=n, msgtype=summ.schemas[ch.schema_id].name, count=int(counts.get(ch.id, 0)), raw_name=ch.topic)
                self._raw_names[n] = ch.topic
            dur = ((stats.message_end_time - stats.message_start_time) / 1e9) if (stats is not None and stats.message_count) else 0.0
            self.kind, self.storage = "mcap", "mcap"
            self.info = BagInfo(path=str(path), kind="mcap", storage="mcap", topics=topics, has_typedefs=True,
                                message_count=int(stats.message_count) if stats is not None else 0, duration_s=round(dur, 3),
                                typestore=typestore_name)
        except BaseException:
            self._fh.close()
            raise

    def messages(self, topics: Sequence[str]) -> Iterator[Tuple[str, str, int, Any]]:
        raw = [self._raw_names[norm_topic(t)] for t in topics if norm_topic(t) in self._raw_names]
        if not raw:
            return
        for schema, channel, message in self._reader.iter_messages(topics=raw, log_time_order=True):
            yield norm_topic(channel.topic), schema.name, int(message.log_time), self._ts.deserialize_cdr(message.data, schema.name)

    def close(self) -> None:
        self._fh.close()


def _open_stream(path: str | os.PathLike, typestore: str):
    p = Path(path)
    kind, _storage = _bag_kind(p)
    return _McapStream(p, typestore) if kind == "mcap" else _RosbagsStream(p, typestore)


def open_bag(path: str | os.PathLike, *, typestore: str = "humble") -> BagInfo:
    s = _open_stream(path, typestore)
    try:
        return s.info
    finally:
        s.close()


# ───────────────────────────── stage 2: which topics play which role ─────────────────────────────

@dataclass
class TopicMap:
    rgb: Optional[str] = None
    depth: Optional[str] = None
    rgb_info: Optional[str] = None
    depth_info: Optional[str] = None
    confidence: Optional[str] = None
    tf: Optional[str] = None
    tf_static: Optional[str] = None
    odom: Optional[str] = None
    tracking: Optional[str] = None
    seq: Optional[str] = None
    detections: Optional[str] = None       # vision_msgs Detection2DArray / Detection3DArray (external detections, optional)
    rules: Dict[str, str] = field(default_factory=dict)

    ROLES = ("rgb", "depth", "rgb_info", "depth_info", "confidence", "tf", "tf_static", "odom", "tracking", "seq", "detections")

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.ROLES}


_NOT_RGB = re.compile(r"depth|infra|/ir_|_ir_|_ir/|/ir/|ir_image|mono|left|right|confidence|thermal|disparity", re.I)
_RGB_HINT = re.compile(r"color|colour|rgb|image_raw|image_color|image_rect|/image$|/image_compressed$", re.I)
_DEPTH_HINT = re.compile(r"depth", re.I)
_ALIGNED_HINT = re.compile(r"aligned_depth_to_color|depth_registered|depth/image_rect", re.I)
_IMAGE_SUFFIX = re.compile(r"[/_]?(image_raw|image_rect_raw|image_rect_color|image_rect|image_color|image|compressed)$", re.I)


def _prefix(topic: str) -> str:
    inner = topic.strip("/")
    return topic.rsplit("/", 1)[0] if "/" in inner else ""


def _stem(topic: str) -> str:
    return _IMAGE_SUFFIX.sub("", topic.rstrip("/"))


def resolve_topics(info: BagInfo, overrides: Optional[Dict[str, str]] = None) -> TopicMap:
    """Rules, first match wins, overrides first; every choice records its rule."""
    ov = {k: norm_topic(v) for k, v in (overrides or {}).items() if v}
    unknown = sorted(set(ov) - set(TopicMap.ROLES))
    if unknown:
        raise ValueError(f"io.bag.topics: unknown roles {unknown} (known: {list(TopicMap.ROLES)})")
    tm = TopicMap()
    topics = info.topics

    def pick(role: str, candidates: List[str], rule: str) -> None:
        if role in ov:
            if ov[role] not in topics:
                raise BagRefusal([(f"no_{role}_topic", f"override {ov[role]!r} is not in the bag (topics: {sorted(topics)})")])
            setattr(tm, role, ov[role])
            tm.rules[role] = "override"
            return
        if candidates:
            setattr(tm, role, candidates[0])
            tm.rules[role] = rule

    names = [t.name for t in info.of_type(IMAGE_T, COMPRESSED_T)]
    depth_c = sorted((n for n in names if _DEPTH_HINT.search(n) and "confidence" not in n.lower()),
                     key=lambda n: (0 if _ALIGNED_HINT.search(n) else 1, n))
    conf_c = sorted(n for n in names if "confidence" in n.lower())
    rgb_c = [n for n in names if not _NOT_RGB.search(n) and _RGB_HINT.search(n)]
    if not rgb_c:
        rgb_c = [n for n in names if not _NOT_RGB.search(n) and n not in depth_c and n not in conf_c]
    rgb_c.sort(key=lambda n: (0 if re.search(r"color|rgb", n, re.I) else 1, 0 if topics[n].msgtype == IMAGE_T else 1, n))
    pick("rgb", rgb_c, "image topic matching color|rgb|image_raw and not depth|ir|mono|left|right")
    pick("depth", [n for n in depth_c if n != tm.rgb], "image topic matching depth (aligned_depth_to_color preferred)")
    pick("confidence", [n for n in conf_c if n not in (tm.rgb, tm.depth)], "image topic matching confidence")

    infos = [t.name for t in info.of_type(INFO_T)]

    def info_for(image_topic: Optional[str]) -> List[str]:
        if not image_topic:
            return []
        stem = _stem(image_topic)
        stem_hits = [n for n in infos if "camera_info" in n.lower() and _stem(n.replace("camera_info", "image")) == stem]
        pre = _prefix(image_topic)
        same = [n for n in infos if pre and _prefix(n) == pre]
        ordered = list(dict.fromkeys(stem_hits + same))
        if not ordered and len(infos) == 1:
            ordered = infos
        return ordered

    pick("rgb_info", info_for(tm.rgb), "CameraInfo with the RGB topic's stem or prefix (or the only one)")
    pick("depth_info", [n for n in info_for(tm.depth) if n != tm.rgb_info], "CameraInfo with the depth topic's stem or prefix")

    tfs = [t.name for t in info.of_type(*TF_TYPES)]
    pick("tf", sorted(n for n in tfs if not n.endswith("tf_static")), "TF message topic")
    pick("tf_static", sorted(n for n in tfs if n.endswith("tf_static")), "tf_static topic")
    pick("odom", sorted(t.name for t in info.of_type(ODOM_T, POSE_T)), "nav_msgs/Odometry or PoseStamped topic")
    pick("tracking", sorted(t.name for t in info.of_type(STRING_T) if "tracking" in t.name.lower()), "std_msgs/String named tracking_state")
    pick("seq", sorted(t.name for t in info.of_type(U32_T) if "seq" in t.name.lower()), "std_msgs/UInt32 named frame_seq")
    dets = sorted((t.name for t in info.of_type(DETECTION_2D, DETECTION_3D)), key=lambda n: (0 if topics[n].msgtype == DETECTION_2D else 1, n))
    pick("detections", dets, "vision_msgs Detection2DArray (preferred) / Detection3DArray topic")
    return tm


# ───────────────────────────── stage 3: frames ─────────────────────────────

@dataclass
class BagStats:
    frames_seen: int = 0            # RGB messages with a usable stamp
    paired: int = 0
    unpaired_rgb: int = 0
    unpaired_depth: int = 0
    pose_missing: int = 0
    skipped_zero_stamp: int = 0
    decode_errors: int = 0
    yielded: int = 0
    pair_dt_ms_max: float = 0.0
    pose_kind: str = ""
    # external detections (a vision_msgs topic paired to the frames; all zero without one)
    detections_topic: Optional[str] = None
    detections_msgtype: Optional[str] = None
    detections_paired: int = 0               # messages attached to a frame
    detections_unpaired: int = 0             # messages no frame matched within the tolerance
    frames_without_detections: int = 0       # frames that went out without a detections message
    detections_errors: int = 0               # messages the adapter could not convert
    detections_dropped: Dict[str, int] = field(default_factory=dict)      # adapter drop reasons -> count
    detections_scoring: Dict[str, int] = field(default_factory=dict)      # present | absent | mixed -> messages
    world_frame: str = ""
    camera_frame: str = ""
    tf_chain: List[str] = field(default_factory=list)
    registration: str = ""
    topics: Dict[str, Any] = field(default_factory=dict)
    bag: Dict[str, Any] = field(default_factory=dict)
    refusal: Optional[List[Tuple[str, str]]] = None

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["refusal"] = [list(r) for r in self.refusal] if self.refusal else None
        return d


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _k_of(info_msg) -> Tuple[float, float, float, float]:
    k = np.asarray(info_msg.k if hasattr(info_msg, "k") else info_msg.K, dtype=np.float64).reshape(-1)
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def _intrinsics_for(info_msg, rgb_w: int, rgb_h: int) -> PinholeIntrinsics:
    fx, fy, cx, cy = _k_of(info_msg)
    return codecs.rescale_intrinsics(fx, fy, cx, cy, from_wh=(int(info_msg.width), int(info_msg.height)), to_wh=(rgb_w, rgb_h))


def _registration_check(rgb_info, depth_info) -> Tuple[bool, str]:
    """Depth registered to RGB iff the depth K, scaled to the RGB size, matches the
    RGB K within REGISTRATION_TOLERANCE. Frame ids are reported, not decisive:
    TUM's depth CameraInfo names the depth optical frame although its images
    are registered to the RGB frame."""
    if depth_info is None:
        return True, "no depth CameraInfo: registration assumed"
    fx, fy, cx, cy = _k_of(rgb_info)
    dfx, dfy, dcx, dcy = _k_of(depth_info)
    sx = int(rgb_info.width) / max(1, int(depth_info.width))
    sy = int(rgb_info.height) / max(1, int(depth_info.height))
    scaled = (dfx * sx, dfy * sy, dcx * sx, dcy * sy)
    rel = max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(scaled, (fx, fy, cx, cy)))
    frames = f"frames rgb {norm_frame(rgb_info.header.frame_id)!r} / depth {norm_frame(depth_info.header.frame_id)!r}"
    if rel <= REGISTRATION_TOLERANCE:
        return True, f"depth K matches rgb K within {rel * 100:.2f} % ({frames})"
    return False, (f"depth K scaled to the RGB size {tuple(round(v, 2) for v in scaled)} vs rgb K "
                   f"{(round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))}: max rel diff {rel * 100:.1f} % ({frames})")


class _Pairer:
    """RGB <-> depth by header stamp: nearest within ``tol_ns``, each depth used
    once. Messages arrive in log order; an RGB is resolved once a depth newer
    than ``rgb + tol`` has been seen (every candidate is then present) or at the end."""

    def __init__(self, tol_ns: int) -> None:
        self.tol = int(tol_ns)
        self.rgb: Deque[Tuple[int, Any, int]] = deque()          # (stamp, msg, log_ns)
        self.depth_stamps: List[int] = []                         # sorted, unused
        self.depth_msgs: List[Any] = []
        self.newest_depth = -1
        self.dropped_depth = 0

    def add_depth(self, stamp: int, msg: Any) -> None:
        i = bisect.bisect_right(self.depth_stamps, stamp)
        self.depth_stamps.insert(i, stamp)
        self.depth_msgs.insert(i, msg)
        self.newest_depth = max(self.newest_depth, stamp)

    def add_rgb(self, stamp: int, msg: Any, log_ns: int) -> None:
        self.rgb.append((stamp, msg, log_ns))

    def _take_nearest(self, stamp: int) -> Optional[Tuple[int, Any]]:
        i = bisect.bisect_left(self.depth_stamps, stamp)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.depth_stamps):
                d = abs(self.depth_stamps[j] - stamp)
                if d <= self.tol and (best is None or d < best[0]):
                    best = (d, j)
        if best is None:
            return None
        j = best[1]
        return self.depth_stamps.pop(j), self.depth_msgs.pop(j)

    def _prune_depth(self) -> None:
        horizon = (self.rgb[0][0] - self.tol) if self.rgb else (self.newest_depth - self.tol)
        while self.depth_stamps and self.depth_stamps[0] < horizon:
            self.depth_stamps.pop(0)
            self.depth_msgs.pop(0)
            self.dropped_depth += 1

    def resolve(self, final: bool = False) -> Iterator[Tuple[int, Any, int, Optional[Tuple[int, Any]]]]:
        while self.rgb and (final or self.newest_depth > self.rgb[0][0] + self.tol):
            stamp, msg, log_ns = self.rgb.popleft()
            yield stamp, msg, log_ns, self._take_nearest(stamp)
            self._prune_depth()
        if final:
            self.dropped_depth += len(self.depth_stamps)
            self.depth_stamps.clear()
            self.depth_msgs.clear()


def _build_pose_source(stream, tm: TopicMap, *, world_frame: Optional[str], camera_frame: Optional[str],
                       extrapolation_s: float, stats: BagStats):
    """First pass: TF / TF static / odometry into one TfBuffer, plus the first
    RGB and depth CameraInfo (for registration and the camera frame). Returns
    (buf, world | None, cam, rgb_info, depth_info, pose_reasons)."""
    buf = TfBuffer(extrapolation_s=extrapolation_s)
    want = [t for t in (tm.tf, tm.tf_static, tm.odom, tm.rgb_info, tm.depth_info) if t]
    rgb_info = depth_info = None
    odom_frames: Optional[Tuple[str, str]] = None
    n_tf = 0
    for topic, _mt, _log, msg in stream.messages(want):
        if topic == tm.rgb_info:
            rgb_info = rgb_info or msg
        elif topic == tm.depth_info:
            depth_info = depth_info or msg
        elif topic in (tm.tf, tm.tf_static):
            static = topic == tm.tf_static
            for tr in msg.transforms:
                t, q = tr.transform.translation, tr.transform.rotation
                try:
                    buf.add(tr.header.frame_id, tr.child_frame_id, _stamp_ns(tr.header), (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), static=static)
                    n_tf += 1
                except ValueError:
                    continue
        elif topic == tm.odom:
            pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
            parent = norm_frame(msg.header.frame_id) or "odom"
            child = norm_frame(getattr(msg, "child_frame_id", "") or "") or "base_link"
            odom_frames = (parent, child)
            p, o = pose.position, pose.orientation
            try:
                buf.add(parent, child, _stamp_ns(msg.header), (p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
                n_tf += 1
            except ValueError:
                continue
    reasons: List[Tuple[str, str]] = []
    if rgb_info is None:
        reasons.append(("no_camera_info", f"no CameraInfo message on {tm.rgb_info!r}"))
        return buf, None, "", None, depth_info, reasons
    cam = norm_frame(camera_frame) or norm_frame(rgb_info.header.frame_id)
    if not cam:
        reasons.append(("no_camera_frame", "CameraInfo carries no frame_id; set io.bag.camera_frame"))
        return buf, None, "", rgb_info, depth_info, reasons
    world, chain, pose_kind, why = choose_world_frame(buf, cam, world_frame=world_frame, odom_frames=odom_frames,
                                                       tf_topic=tm.tf, odom_topic=tm.odom, setting="io.bag", what="bag")
    if world is None:
        reasons.extend(why)
        return buf, None, cam, rgb_info, depth_info, reasons
    stats.pose_kind = pose_kind
    stats.world_frame, stats.camera_frame = world, cam
    stats.tf_chain = [f"{p} -> {c}" for p, c in chain]
    logger.info("[bag] pose source: %s, chain %s (%d transforms)", stats.pose_kind, " | ".join(stats.tf_chain), n_tf)
    return buf, world, cam, rgb_info, depth_info, reasons


def choose_world_frame(buf: TfBuffer, cam: str, *, world_frame: Optional[str], odom_frames: Optional[Tuple[str, str]],
                       tf_topic: Optional[str] = None, odom_topic: Optional[str] = None, setting: str = "io.bag",
                       what: str = "bag") -> Tuple[Optional[str], List[Tuple[str, str]], str, List[Tuple[str, str]]]:
    """The one rule for the pose source, shared by the bag reader and the live
    ``ros2`` source: the configured world frame, else the odometry parent when
    it reaches the camera, else the first reachable root in PREFERRED_WORLD_FRAMES
    order, and the chain must contain a moving hop. Returns
    ``(world | None, chain, pose_kind, reasons)``; ``reasons`` holds the refusal
    when ``world`` is None (``setting`` names the config block in the hint)."""
    world = norm_frame(world_frame) if world_frame else ""
    if not world:
        candidates = []
        for r in buf.roots():
            try:
                buf.chain(r, cam)
                candidates.append(r)
            except TfLookupError:
                continue
        if odom_frames and odom_frames[0] in candidates:
            world = odom_frames[0]
        else:
            candidates.sort(key=lambda r: (PREFERRED_WORLD_FRAMES.index(r) if r in PREFERRED_WORLD_FRAMES else 99, r))
            world = candidates[0] if candidates else ""
    if not world:
        return None, [], "", [("no_pose_source",
                               f"no TF chain from camera frame {cam!r} to a root (roots {buf.roots()}, hops {[(p, c) for p, c, _s in buf.hops()]}, "
                               f"odometry topic {odom_topic!r}); set {setting}.topics.tf / odom or {setting}.world_frame")]
    try:
        chain = buf.chain(world, cam)
    except TfLookupError as e:
        return None, [], "", [("no_pose_source", f"world frame {world!r}: {e}")]
    if all(buf.span((p, c)) is None for p, c in chain):
        return None, chain, "", [("no_pose_source", f"the TF chain {world} -> {cam} has only static hops: no moving pose in this {what} "
                                                     f"(tf {tf_topic!r}, odometry {odom_topic!r})")]
    pose_kind = "odometry" if (odom_frames and odom_frames in chain) else "tf"
    return world, chain, pose_kind, []


def _image_size(msg) -> Tuple[int, int]:
    return int(msg.width), int(msg.height)


def _rgb_encoded(msg, msgtype: str) -> EncodedImage:
    """Still encoded; row padding stripped here (cheap slice) so the codec needs no step."""
    if msgtype == COMPRESSED_T:
        return EncodedImage(bytes(msg.data), f"rosc:{msg.format}", 0, 0)
    w, h = _image_size(msg)
    enc = str(msg.encoding)
    ch = codecs._ROS_COLOR_CHANNELS.get(enc.lower())
    if ch is None:
        raise UnsupportedEncoding(f"Unsupported ROS image encoding: {enc!r}")
    if int(msg.is_bigendian or 0):
        raise UnsupportedEncoding("big-endian images are not supported")
    data = np.asarray(msg.data, dtype=np.uint8)
    step = int(msg.step) or w * ch
    if step != w * ch:
        data = np.ascontiguousarray(data[: step * h].reshape(h, step)[:, : w * ch]).reshape(-1)
    return EncodedImage(data.tobytes(), f"ros:{enc}", w, h)


def _png_size(png: bytes) -> Tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("compressedDepth payload is not a PNG")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _depth_encoded(msg, msgtype: str) -> EncodedImage:
    if msgtype == COMPRESSED_T:
        raw, wire, scale = codecs.ros_compressed_depth(str(msg.format), bytes(msg.data))
        w, h = _png_size(bytes(msg.data)[12:])
        return EncodedImage(raw, wire, w, h, scale)
    w, h = _image_size(msg)
    raw, wire, scale = codecs.ros_depth_wire(str(msg.encoding), np.asarray(msg.data, dtype=np.uint8).tobytes(), w, h,
                                             is_bigendian=int(msg.is_bigendian or 0))
    return EncodedImage(raw, wire, w, h, scale)


def _confidence_encoded(msg) -> Optional[EncodedImage]:
    if msg is None or str(msg.encoding).lower() not in ("mono8", "8uc1"):
        return None
    w, h = _image_size(msg)
    return EncodedImage(np.asarray(msg.data, dtype=np.uint8).tobytes()[: w * h], "uint8", w, h)


def _info_at(infos: List[Tuple[int, Any]], stamp: int):
    """The CameraInfo at exactly ``stamp`` if present, else the latest at or before it, else None."""
    best = None
    for s, m in infos:
        if s == stamp:
            return m
        if s <= stamp and (best is None or s > best[0]):
            best = (s, m)
    return best[1] if best else None


def probe_bag(path: str | os.PathLike, *, topics: Optional[Dict[str, str]] = None, tf_extrapolation_s: float = 0.05,
              world_frame: Optional[str] = None, camera_frame: Optional[str] = None, assume_aligned: bool = False,
              typestore: str = "humble") -> BagStats:
    """Everything ``iter_bag_frames`` decides before it reads a single image:
    what the bag contains, which topics play which role, the pose source and
    the registration check. Returns the stats (``refusal`` set when the bag
    would be refused; nothing is raised) -- the runner calls this above the
    GPU check so a refusal costs seconds, not a model load."""
    st = BagStats()
    try:
        stream = _open_stream(path, typestore)
    except BagRefusal as e:
        st.refusal = list(e.reasons)
        return st
    try:
        _prepare(stream, st, topics=topics, tf_extrapolation_s=tf_extrapolation_s, world_frame=world_frame,
                 camera_frame=camera_frame, assume_aligned=assume_aligned)
    except BagRefusal:
        pass                                            # st.refusal carries the reasons
    finally:
        stream.close()
    return st


def _prepare(stream, st: BagStats, *, topics, tf_extrapolation_s, world_frame, camera_frame, assume_aligned):
    """Stages 1-2 and the first pass. Returns (tm, buf, world, cam, rgb_info, session_id); raises BagRefusal
    (with st.refusal filled) when the bag cannot be read."""
    info = stream.info
    st.bag = {"kind": info.kind, "storage": info.storage, "typestore": info.typestore, "has_typedefs": info.has_typedefs,
              "message_count": info.message_count, "duration_s": info.duration_s}
    if not info.has_typedefs:
        logger.warning("[bag] %s carries no message definitions; assuming the %s typestore (io.bag.typestore)", info.path, info.typestore)
    try:
        tm = resolve_topics(info, topics)
    except BagRefusal as e:                       # an override naming a topic the bag lacks
        st.refusal = list(e.reasons)
        raise
    st.topics = {**tm.as_dict(), "rules": dict(tm.rules)}
    if tm.detections:
        st.detections_topic = tm.detections
        st.detections_msgtype = info.topics[tm.detections].msgtype
    reasons: List[Tuple[str, str]] = []
    image_names = [t.name for t in info.of_type(IMAGE_T, COMPRESSED_T)]
    for role in ("rgb", "depth"):
        if getattr(tm, role) is None:
            reasons.append((f"no_{role}_topic", f"no {role} topic found; image topics: {image_names}"))
    if tm.rgb_info is None:
        reasons.append(("no_camera_info", f"no CameraInfo for the RGB topic; CameraInfo topics: {[t.name for t in info.of_type(INFO_T)]}"))
    if reasons:
        st.refusal = reasons
        raise BagRefusal(reasons)
    buf, world, cam, rgb_info, depth_info, pose_reasons = _build_pose_source(
        stream, tm, world_frame=world_frame, camera_frame=camera_frame, extrapolation_s=tf_extrapolation_s, stats=st)
    reasons.extend(pose_reasons)
    if rgb_info is not None:
        ok, why = _registration_check(rgb_info, depth_info)
        st.registration = why + ("" if ok else (" (assumed aligned by config)" if assume_aligned else ""))
        if not ok and not assume_aligned:
            reasons.append(("unaligned_depth", why + "; set io.bag.assume_aligned: true only if the depth IS registered to the RGB"))
    if reasons:
        st.refusal = reasons
        raise BagRefusal(reasons)
    logger.info("[bag] registration: %s", st.registration)
    return tm, buf, world, cam, rgb_info, (info.custom_data.get("session_id") or None)



# ───────────────────────────── external detections ─────────────────────────────

DETECTIONS_WAIT_S = 1.0     # a frame waits this long (log time) for its detections message before it goes out without one


class _DetectionsJoiner:
    """Attaches to each emitted frame the detections message nearest in stamp
    (within the pair tolerance). Detectors publish after the image, so a frame
    waits: it is released with its detections when they arrive, without them
    when a detections message with a later stamp arrives or the log time passes
    the frame's by ``DETECTIONS_WAIT_S``, and at the end of the bag. Frame order
    is preserved; every outcome is counted in the BagStats."""

    def __init__(self, adapter, msgtype: str, tol_ns: int, wait_ns: int, st: "BagStats", tf_lookup, source: str) -> None:
        self.adapter, self.msgtype, self.tol, self.wait, self.st, self.tf_lookup, self.source = adapter, msgtype, int(tol_ns), int(wait_ns), st, tf_lookup, source
        self.pending: Deque[Tuple[RawFrame, int]] = deque()
        self.dets: List[Tuple[int, Any]] = []
        self.last_released: Optional[int] = None

    def add_frame(self, fr: RawFrame, log_ns: int):
        self.pending.append((fr, int(log_ns)))
        yield from self._release(log_ns=int(log_ns))

    def add_detections(self, msg: Any, log_ns: int):
        try:
            stamp = _stamp_ns(msg.header)
        except Exception:  # noqa: BLE001
            stamp = 0
        if stamp <= 0:
            self.st.detections_unpaired += 1
            return
        self.dets.append((int(stamp), msg))
        yield from self._release(log_ns=int(log_ns))

    def advance(self, log_ns: int):
        yield from self._release(log_ns=int(log_ns))

    def flush(self):
        yield from self._release(final=True)

    def _attach(self, fr: RawFrame, msg: Any, stamp: int) -> None:
        h = fr.header
        try:
            det = self.adapter.convert(msg, self.msgtype, rgb_hw=tuple(h.extra.get("rgb_hw") or (h.rgb.height, h.rgb.width)),
                                       intrinsics=h.intrinsics, tf_lookup=self.tf_lookup, t_sensor_ns=stamp, source=self.source)
        except Exception as e:  # noqa: BLE001 -- a bad message is counted, never fatal
            logger.warning("[bag] detections at %d not converted: %s", stamp, e)
            self.st.detections_errors += 1
            return
        h.extra["detections"] = det
        h.extra["detections_dt_ms"] = round(abs(stamp - int(h.t_sensor_ns)) / 1e6, 3)
        self.st.detections_paired += 1
        self.st.detections_scoring[det.scoring] = self.st.detections_scoring.get(det.scoring, 0) + 1
        for k, v in det.n_dropped.items():
            self.st.detections_dropped[k] = self.st.detections_dropped.get(k, 0) + int(v)

    def _release(self, log_ns: Optional[int] = None, final: bool = False):
        while self.pending:
            fr, fr_log = self.pending[0]
            fs = int(fr.header.t_sensor_ns)
            best = None
            for i, (ds, _m) in enumerate(self.dets):
                d = abs(ds - fs)
                if d <= self.tol and (best is None or d < best[0]):
                    best = (d, i)
            if best is not None:
                ds, msg = self.dets.pop(best[1])
                self._attach(fr, msg, ds)
            else:
                later = any(ds > fs + self.tol for ds, _m in self.dets)
                timed_out = log_ns is not None and (log_ns - fr_log) > self.wait
                if not (final or later or timed_out):
                    break
                self.st.frames_without_detections += 1
            self.pending.popleft()
            self.last_released = fs
            yield fr
        # detections that no frame can match any more (older than every frame still to come)
        floor = (int(self.pending[0][0].header.t_sensor_ns) if self.pending else self.last_released)
        if floor is not None:
            keep = [(ds, m) for ds, m in self.dets if ds >= floor - self.tol]
            self.st.detections_unpaired += len(self.dets) - len(keep)
            self.dets = keep
        if final:
            self.st.detections_unpaired += len(self.dets)
            self.dets = []


def iter_bag_frames(path: str | os.PathLike, *, topics: Optional[Dict[str, str]] = None, pair_tolerance_s: float = 0.02,
                    tf_extrapolation_s: float = 0.05, world_frame: Optional[str] = None, camera_frame: Optional[str] = None,
                    assume_aligned: bool = False, typestore: str = "humble", stats: Optional[BagStats] = None,
                    source: str = "bag", max_frames: Optional[int] = None) -> Iterator[RawFrame]:
    """The frames of a bag as ``RawFrame``s (module docstring). ``stats`` (a
    BagStats) is filled as it goes and holds the refusal reasons when the bag
    is refused (a BagRefusal is raised as well)."""
    st = stats if stats is not None else BagStats()
    stream = _open_stream(path, typestore)
    try:
        info = stream.info
        tm, buf, world, cam, rgb_info, session_id = _prepare(stream, st, topics=topics, tf_extrapolation_s=tf_extrapolation_s,
                                                              world_frame=world_frame, camera_frame=camera_frame,
                                                              assume_aligned=assume_aligned)

        want = [t for t in (tm.rgb, tm.depth, tm.rgb_info, tm.confidence, tm.tracking, tm.seq, tm.detections) if t]
        pairer = _Pairer(int(pair_tolerance_s * 1e9))
        infos: List[Tuple[int, Any]] = []
        confs: Dict[int, Any] = {}
        tracking_at: Dict[int, str] = {}
        latest_tracking = "not_available"
        seq_at: Dict[int, int] = {}
        index_of: Dict[int, int] = {}
        rgb_index = 0
        rgb_type = info.topics[tm.rgb].msgtype
        depth_type = info.topics[tm.depth].msgtype
        n_out = 0
        joiner: Optional[_DetectionsJoiner] = None
        if tm.detections:
            det_type = info.topics[tm.detections].msgtype
            adapter = adapter_for(det_type)
            if adapter is None:
                st.refusal = [("unsupported_detections_type", f"{tm.detections}: no detections adapter for {det_type}")]
                raise BagRefusal(st.refusal)

            def _tf_lookup(frame_id: str, t_ns: int, _buf=buf, _world=world, _cam=cam) -> np.ndarray:
                """T_cam_from_frame at t: the camera pose inverted, composed with world <- frame."""
                fid = norm_frame(frame_id)
                if fid == _cam:
                    return np.eye(4)
                T_wc = make_T(*_buf.lookup(_world, _cam, t_ns))
                return np.linalg.inv(T_wc) @ _buf.lookup_T(_world, fid, t_ns)

            joiner = _DetectionsJoiner(adapter, det_type, int(pair_tolerance_s * 1e9), int(DETECTIONS_WAIT_S * 1e9), st,
                                       _tf_lookup, source=f"{tm.detections}")

        def emit(stamp: int, rgb_msg, log_ns: int, depth_pair) -> Optional[RawFrame]:
            if depth_pair is None:
                st.unpaired_rgb += 1
                return None
            st.paired += 1
            st.pair_dt_ms_max = max(st.pair_dt_ms_max, abs(depth_pair[0] - stamp) / 1e6)
            try:
                t_wc, q_wc = buf.lookup(world, cam, stamp)
            except TfLookupError as e:
                st.pose_missing += 1
                logger.debug("[bag] no pose at %d: %s", stamp, e)
                return None
            try:
                rgb_img = _rgb_encoded(rgb_msg, rgb_type)
                depth_img = _depth_encoded(depth_pair[1], depth_type)
            except UnsupportedEncoding as e:
                st.decode_errors += 1
                st.refusal = [("unsupported_depth_encoding" if "depth" in str(e).lower() else "unsupported_encoding", str(e))]
                raise BagRefusal(st.refusal) from None
            except ValueError as e:
                st.decode_errors += 1
                logger.warning("[bag] frame at %d skipped: %s", stamp, e)
                return None
            ci = _info_at(infos, stamp) or rgb_info
            rgb_w, rgb_h = (rgb_img.width, rgb_img.height) if rgb_img.width else (int(ci.width), int(ci.height))
            intr = _intrinsics_for(ci, rgb_w, rgb_h)
            conf = _confidence_encoded(confs.pop(stamp, None))
            seq = seq_at.pop(log_ns, None)
            if seq is None:
                seq = index_of.get(stamp, 0)
            tracking = tracking_at.pop(log_ns, None) or (latest_tracking if tm.tracking else "normal")
            return RawFrame(header=FrameHeader(
                source=source, seq=int(seq), t_sensor_ns=int(stamp), t_wall_utc_s=(log_ns / 1e9 if log_ns else None),
                tracking_state=str(tracking), keyframe_hint=None, rgb=rgb_img, depth=depth_img, intrinsics=intr,
                pose_raw=(t_wc, q_wc), pose_format=POSE_FMT_PREPARED, pose_convention=CONVENTION_OPENCV, confidence=conf,
                session_id=session_id, pose_frame_id=world, keep_encoded_rgb=False,
                extra={"pair_dt_ms": round(abs(depth_pair[0] - stamp) / 1e6, 3), "rgb_hw": (int(rgb_h), int(rgb_w))},
            ))

        def frames():
            """The frames in emission order; with a detections topic, through the joiner (which may hold a frame
            until its detections arrive)."""
            nonlocal rgb_index, latest_tracking
            for topic, _mt, log_ns, msg in stream.messages(want):
                if topic == tm.rgb_info:
                    infos.append((_stamp_ns(msg.header), msg))
                    if len(infos) > 64:
                        del infos[:-64]
                    continue
                if topic == tm.tracking:
                    tracking_at[log_ns] = latest_tracking = str(msg.data)
                    continue
                if topic == tm.seq:
                    seq_at[log_ns] = int(msg.data)
                    continue
                if topic == tm.confidence:
                    confs[_stamp_ns(msg.header)] = msg
                    if len(confs) > 64:
                        for k in sorted(confs)[:-64]:
                            confs.pop(k, None)
                    continue
                if joiner is not None and topic == tm.detections:
                    yield from joiner.add_detections(msg, log_ns)
                    continue
                stamp = _stamp_ns(msg.header)
                if stamp <= 0:
                    st.skipped_zero_stamp += 1
                    continue
                if topic == tm.rgb:
                    st.frames_seen += 1
                    index_of[stamp] = rgb_index
                    rgb_index += 1
                    pairer.add_rgb(stamp, msg, log_ns)
                elif topic == tm.depth:
                    pairer.add_depth(stamp, msg)
                for stamp_r, rgb_msg, log_r, dp in pairer.resolve():
                    fr = emit(stamp_r, rgb_msg, log_r, dp)
                    if fr is None:
                        continue
                    if joiner is None:
                        yield fr
                    else:
                        yield from joiner.add_frame(fr, log_r)
                if joiner is not None:
                    yield from joiner.advance(log_ns)
            for stamp_r, rgb_msg, log_r, dp in pairer.resolve(final=True):
                fr = emit(stamp_r, rgb_msg, log_r, dp)
                if fr is None:
                    continue
                if joiner is None:
                    yield fr
                else:
                    yield from joiner.add_frame(fr, log_r)
            if joiner is not None:
                yield from joiner.flush()

        for fr in frames():
            n_out += 1
            st.yielded = n_out
            yield fr
            if max_frames is not None and n_out >= int(max_frames):
                break
        st.unpaired_depth = pairer.dropped_depth
    finally:
        stream.close()
