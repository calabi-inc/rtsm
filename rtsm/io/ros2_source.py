"""
The ``ros2`` ingest source: a minimal live rclpy subscriber as a transport
adapter on the one ingest front-end.

It is the bag source's twin. The topic rules, the ``sensor_msgs`` encodings
onto the codec layer, the RGB-depth pairing by header stamp, the TF buffer,
the world-frame rule and the registration check are the bag reader's
functions, imported here; rclpy message objects expose the same fields as the
``rosbags``-deserialised ones (``header.stamp``, ``data``, ``k``,
``transforms``), so a frame built live is the frame the bag reader would build
from a recording of the same stream (``tests/test_ros2_source.py`` pins that).

What a live stream needs and a bag does not:

* **discovery with a timeout** instead of a table of contents
  (``get_topic_names_and_types``, polled until RGB, depth and CameraInfo have
  a topic each, or the timeout refuses with the bag reader's codes);
* **QoS chosen per topic** from the publishers' offers (``qos: auto`` matches
  reliability; ``tf_static`` is transient-local), because a mismatch is a
  callback that never fires;
* **two threads**: the executor only appends to a deque; a worker pairs,
  looks the pose up, admits and enqueues, so a blocking lossless lane never
  stalls the executor;
* **head-of-line waiting for TF**: a resolved pair whose TF has not arrived
  waits up to ``tf_wait_s`` (newer frames wait behind it, order preserved),
  then counts as ``pose_missing``;
* **readiness buffering**: pairs resolved before the camera info, the world
  frame and the registration check are known are held (bounded) and flushed
  in order once they are.

Everything that touches rclpy lives in ``Ros2Source`` and ``probe``; the
``Ros2Ingress`` core is plain Python and is what the CPU suite tests.

rclpy exists only inside a sourced ROS 2 environment on Linux (Humble on
22.04, Jazzy on 24.04). Elsewhere ``start()`` refuses with RCLPY_HINT.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from rtsm.io.bag_reader import (
    COMPRESSED_T, IMAGE_T, INFO_T, ODOM_T, POSE_T, STRING_T, TF_TYPES, U32_T,
    BagInfo, BagRefusal, BagStats, TopicInfo, TopicMap, choose_world_frame, resolve_topics,
    _Pairer, _confidence_encoded, _depth_encoded, _info_at, _intrinsics_for, _registration_check, _rgb_encoded, _stamp_ns,
)
from rtsm.io.contracts import CONVENTION_OPENCV, POSE_FMT_PREPARED, FrameHeader, RawFrame, SourceContext
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd
from rtsm.io.tf_buffer import TfBuffer, TfLookupError, norm_frame

logger = logging.getLogger(__name__)

RCLPY_HINT = ("rclpy is not importable: the ros2 source runs inside a sourced ROS 2 environment on Linux "
              "(for example `source /opt/ros/jazzy/setup.bash`, then a venv created with --system-site-packages); "
              "see docs/guides/ingest-sources.md#ros-2-live")

QOS_MODES = ("auto", "reliable", "best_effort")
READY_BUFFER = 256          # resolved pairs held until the stream is ready (camera info + world frame + registration)
DEFAULT_TF_WAIT_S = 0.5     # how long a resolved pair waits for its TF (measured on the newest TF stamp seen)
_LIVE_ROLES = ("rgb", "depth", "rgb_info", "depth_info", "confidence", "tf", "tf_static", "odom", "tracking", "seq")


class Ros2Unavailable(RuntimeError):
    """rclpy cannot be imported here."""


# ───────────────────────────── pure helpers (no rclpy) ─────────────────────────────

def topics_to_info(names_and_types: List[Tuple[str, List[str]]]) -> BagInfo:
    """``get_topic_names_and_types()`` -> the ``BagInfo`` the bag reader's topic
    rules read. A topic advertised with several types keeps the first."""
    topics: Dict[str, TopicInfo] = {}
    for name, types in names_and_types:
        if not types:
            continue
        n = name if name.startswith("/") else "/" + name
        topics[n] = TopicInfo(name=n, msgtype=str(types[0]), count=0, raw_name=name)
    return BagInfo(path="<ros2 graph>", kind="ros2", storage="live", topics=topics, has_typedefs=True,
                   message_count=0, duration_s=0.0, typestore="live")


@dataclass
class QosChoice:
    reliability: str            # reliable | best_effort
    durability: str             # volatile | transient_local
    depth: int
    offered: List[Dict[str, str]] = field(default_factory=list)   # what the publishers offer, for the probe


def choose_qos(role: str, publishers: List[Any], mode: str = "auto") -> QosChoice:
    """Pick a subscription QoS that a publisher on the topic will match.

    ``publishers`` are rclpy ``TopicEndpointInfo`` objects (or anything with a
    ``qos_profile`` carrying ``reliability`` / ``durability`` enums whose
    ``name`` is e.g. ``RELIABLE``). Reliability: ``auto`` follows the
    publishers (reliable when any publisher is reliable, else best effort);
    ``reliable`` / ``best_effort`` force it. Durability: ``tf_static`` is
    transient-local (latched), everything else volatile. Depth: images 100 so
    a slow consumer under a reliable lane loses nothing short of a stall;
    TF 200 (bursts at 100 Hz); the rest 10."""
    if mode not in QOS_MODES:
        raise ValueError(f"io.ros2.qos must be one of {QOS_MODES}, got {mode!r}")
    offered: List[Dict[str, str]] = []
    any_reliable = False
    for p in publishers or []:
        q = getattr(p, "qos_profile", None)
        rel = str(getattr(getattr(q, "reliability", None), "name", "")).lower() or "unknown"
        dur = str(getattr(getattr(q, "durability", None), "name", "")).lower() or "unknown"
        offered.append({"node": str(getattr(p, "node_name", "") or ""), "reliability": rel, "durability": dur})
        any_reliable = any_reliable or rel == "reliable"
    if mode == "auto":
        reliability = "reliable" if any_reliable else "best_effort"
    else:
        reliability = mode
    durability = "transient_local" if role == "tf_static" else "volatile"
    depth = 100 if role in ("rgb", "depth", "confidence") else 200 if role in ("tf", "tf_static") else 10
    return QosChoice(reliability=reliability, durability=durability, depth=depth, offered=offered)


# ───────────────────────────── the rclpy-free core ─────────────────────────────

@dataclass
class _RgbPack:
    msg: Any
    seq: Optional[int]
    tracking: Optional[str]
    index: int = 0                      # arrival index: the seq fallback when the stream has no seq topic


class Ros2Ingress:
    """Message intake -> pairing -> pose -> ``RawFrame`` -> front-end.

    ``on_message(role, msgtype, msg, rx_wall_ns)`` may be called from any
    thread (the executor); ``run(stop_event)`` is the worker loop; ``step()``
    processes whatever is queued and returns (what the tests drive)."""

    def __init__(self, fe: IngestFrontEnd, stats: BagStats, tm: TopicMap, msgtypes: Dict[str, str], *,
                 pair_tolerance_s: float = 0.02, tf_extrapolation_s: float = 0.05, tf_wait_s: float = DEFAULT_TF_WAIT_S,
                 world_frame: Optional[str] = None, camera_frame: Optional[str] = None, assume_aligned: bool = False,
                 require_tracking_cfg: bool = True, session_id: Optional[str] = None, setting: str = "io.ros2") -> None:
        self._fe = fe
        self.stats = stats
        self.tm = tm
        self._types = msgtypes                                   # role -> msgtype of the chosen topic
        self._pairer = _Pairer(int(float(pair_tolerance_s) * 1e9))
        self._buf = TfBuffer(extrapolation_s=float(tf_extrapolation_s))
        self._tf_wait_ns = int(float(tf_wait_s) * 1e9)
        self._world_cfg = world_frame
        self._cam_cfg = camera_frame
        self._assume_aligned = bool(assume_aligned)
        self._require_tracking_cfg = bool(require_tracking_cfg)
        self._setting = setting
        self.session_id = session_id or f"ros2-{time.strftime('%Y%m%d-%H%M%S')}"
        # intake
        self._q: Deque[Tuple[str, Any, int]] = deque()
        self._cv = threading.Condition()
        # state
        self._infos: List[Tuple[int, Any]] = []
        self._rgb_info = None
        self._depth_info = None
        self._confs: Dict[int, Any] = {}
        self._latest_tracking: Optional[str] = None       # last tracking state seen (the fallback, as in the bag reader)
        self._fresh_seq: Optional[int] = None             # a seq / tracking message that arrived BEFORE its RGB waits here ...
        self._fresh_tracking: Optional[str] = None
        self._awaiting: Optional[_RgbPack] = None         # ... and an RGB that arrived before them waits here (our converter writes the image first)
        self._odom_frames: Optional[Tuple[str, str]] = None
        self._newest_tf_ns = -1
        self._rgb_index = 0
        self._world: Optional[str] = None
        self._cam: str = ""
        self._ready = False
        self._registration_done = False
        self._pending: Deque[Tuple[int, _RgbPack, int, Tuple[int, Any]]] = deque()   # resolved pairs awaiting TF / readiness
        self.enqueued = 0
        self.admit_errors = 0
        self.before_ready_dropped = 0
        self.error: Optional[BaseException] = None
        self._session_announced = False
        stats.topics = {**tm.as_dict(), "rules": dict(tm.rules)}

    # ── intake (any thread) ──

    def on_message(self, role: str, msg: Any, rx_wall_ns: Optional[int] = None) -> None:
        with self._cv:
            self._q.append((role, msg, int(rx_wall_ns if rx_wall_ns is not None else time.time_ns())))
            self._cv.notify()

    def queued(self) -> int:
        with self._cv:
            return len(self._q)

    # ── worker ──

    def run(self, stop_event: threading.Event, idle_s: float = 0.05) -> None:
        while not stop_event.is_set():
            with self._cv:
                if not self._q:
                    self._cv.wait(timeout=idle_s)
            try:
                self.step()
            except BagRefusal as e:
                self.error = e
                logger.error("[ros2] refused: %s", e)
                return
            except Exception as e:  # noqa: BLE001 -- surfaced through stats()/error, the loop goes on
                self.admit_errors += 1
                logger.exception("[ros2] worker error: %s", e)

    def step(self, final: bool = False) -> int:
        """Drain the intake, resolve pairs, emit what can be emitted. Returns the number of frames enqueued by this call."""
        before = self.enqueued
        while True:
            with self._cv:
                if not self._q:
                    break
                role, msg, rx_ns = self._q.popleft()
            self._dispatch(role, msg, rx_ns)
        for stamp, pack, rx_ns, dp in self._pairer.resolve(final=final):
            if dp is None:
                self.stats.unpaired_rgb += 1
                continue
            self.stats.paired += 1
            self.stats.pair_dt_ms_max = max(self.stats.pair_dt_ms_max, abs(dp[0] - stamp) / 1e6)
            self._pending.append((stamp, pack, rx_ns, dp))
            while len(self._pending) > READY_BUFFER:
                self._pending.popleft()
                self.before_ready_dropped += 1
        self._flush_pending(final=final)
        if final:
            self.stats.unpaired_depth = self._pairer.dropped_depth
        return self.enqueued - before

    # ── dispatch ──

    def _dispatch(self, role: str, msg: Any, rx_ns: int) -> None:
        if role in ("tf", "tf_static"):
            static = role == "tf_static"
            for tr in msg.transforms:
                t, q = tr.transform.translation, tr.transform.rotation
                try:
                    s = _stamp_ns(tr.header)
                    self._buf.add(tr.header.frame_id, tr.child_frame_id, s, (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), static=static)
                    if not static:
                        self._newest_tf_ns = max(self._newest_tf_ns, s)
                except ValueError:
                    continue
            self._maybe_ready()
            return
        if role == "odom":
            pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
            parent = norm_frame(msg.header.frame_id) or "odom"
            child = norm_frame(getattr(msg, "child_frame_id", "") or "") or "base_link"
            self._odom_frames = (parent, child)
            p, o = pose.position, pose.orientation
            try:
                s = _stamp_ns(msg.header)
                self._buf.add(parent, child, s, (p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
                self._newest_tf_ns = max(self._newest_tf_ns, s)
            except ValueError:
                pass
            self._maybe_ready()
            return
        if role == "rgb_info":
            if self._rgb_info is None:
                self._rgb_info = msg
            self._infos.append((_stamp_ns(msg.header), msg))
            if len(self._infos) > 64:
                del self._infos[:-64]
            self._maybe_ready()
            return
        if role == "depth_info":
            if self._depth_info is None:
                self._depth_info = msg
            self._maybe_ready()
            return
        if role == "tracking":
            self._latest_tracking = str(msg.data)
            if self._awaiting is not None and self._awaiting.tracking is None:
                self._awaiting.tracking = str(msg.data)
            else:
                self._fresh_tracking = str(msg.data)
            self._settle_awaiting()
            return
        if role == "seq":
            if self._awaiting is not None and self._awaiting.seq is None:
                self._awaiting.seq = int(msg.data)
            else:
                self._fresh_seq = int(msg.data)
            self._settle_awaiting()
            return
        if role == "confidence":
            self._confs[_stamp_ns(msg.header)] = msg
            if len(self._confs) > 64:
                for k in sorted(self._confs)[:-64]:
                    self._confs.pop(k, None)
            return
        stamp = _stamp_ns(msg.header)
        if stamp <= 0:
            self.stats.skipped_zero_stamp += 1
            return
        if role == "rgb":
            self.stats.frames_seen += 1
            # per-frame seq / tracking: the message may precede the image (fresh) or follow it (awaiting); either way the
            # pair cannot resolve before the next depth arrives, by which time both have been seen
            pack = _RgbPack(msg, self._fresh_seq, self._fresh_tracking, self._rgb_index)
            self._fresh_seq = self._fresh_tracking = None
            self._awaiting = pack if (self.tm.seq and pack.seq is None) or (self.tm.tracking and pack.tracking is None) else None
            self._rgb_index += 1
            self._pairer.add_rgb(stamp, pack, rx_ns)
        elif role == "depth":
            self._pairer.add_depth(stamp, msg)

    def _settle_awaiting(self) -> None:
        a = self._awaiting
        if a is not None and (not self.tm.seq or a.seq is not None) and (not self.tm.tracking or a.tracking is not None):
            self._awaiting = None

    # ── readiness: camera info, world frame, registration ──

    def _maybe_ready(self) -> None:
        if self._ready or self._rgb_info is None:
            return
        cam = norm_frame(self._cam_cfg) or norm_frame(self._rgb_info.header.frame_id)
        if not cam:
            raise BagRefusal([("no_camera_frame", f"CameraInfo carries no frame_id; set {self._setting}.camera_frame")])
        self._cam = cam
        world, chain, pose_kind, _why = choose_world_frame(self._buf, cam, world_frame=self._world_cfg, odom_frames=self._odom_frames,
                                                           tf_topic=self.tm.tf, odom_topic=self.tm.odom, setting=self._setting, what="stream")
        if world is None:
            return                                       # not yet: more TF may arrive (the source enforces the deadline)
        if not self._registration_done:
            if self.tm.depth_info and self._depth_info is None:
                return                                   # a depth CameraInfo topic exists: wait for its first message
            ok, why = _registration_check(self._rgb_info, self._depth_info)
            self.stats.registration = why + ("" if ok else (" (assumed aligned by config)" if self._assume_aligned else ""))
            if not ok and not self._assume_aligned:
                raise BagRefusal([("unaligned_depth", why + f"; set {self._setting}.assume_aligned: true only if the depth IS registered to the RGB")])
            self._registration_done = True
        self._world = world
        self.stats.pose_kind = pose_kind
        self.stats.world_frame, self.stats.camera_frame = world, cam
        self.stats.tf_chain = [f"{p} -> {c}" for p, c in chain]
        has_tracking = bool(self.tm.tracking)
        self._fe.require_tracking_normal = self._require_tracking_cfg and has_tracking
        self._fe.new_session(self.session_id)
        self._ready = True
        logger.info("[ros2] ready: pose %s, chain %s | registration: %s | tracking filter %s", pose_kind, " | ".join(self.stats.tf_chain),
                    self.stats.registration, "on" if self._fe.require_tracking_normal else "off (no tracking topic)")

    @property
    def ready(self) -> bool:
        return self._ready

    def not_ready_reasons(self) -> List[Tuple[str, str]]:
        """Why the stream is not ready yet, in the bag reader's codes (for the deadline and the probe)."""
        reasons: List[Tuple[str, str]] = []
        if self._rgb_info is None:
            reasons.append(("no_camera_info", f"no CameraInfo message received on {self.tm.rgb_info!r}"))
            return reasons
        cam = norm_frame(self._cam_cfg) or norm_frame(self._rgb_info.header.frame_id)
        world, _chain, _kind, why = choose_world_frame(self._buf, cam, world_frame=self._world_cfg, odom_frames=self._odom_frames,
                                                       tf_topic=self.tm.tf, odom_topic=self.tm.odom, setting=self._setting, what="stream")
        if world is None:
            reasons.extend(why)
        if self.tm.depth_info and self._depth_info is None:
            reasons.append(("no_camera_info", f"no CameraInfo message received on {self.tm.depth_info!r}"))
        return reasons

    # ── emit ──

    def _flush_pending(self, final: bool = False) -> None:
        if not self._ready:
            return
        while self._pending:
            stamp, pack, rx_ns, dp = self._pending[0]
            try:
                t_wc, q_wc = self._buf.lookup(self._world, self._cam, stamp)
            except TfLookupError as e:
                # TF may trail the image: wait (head of line) until TF newer than stamp + tf_wait has been seen
                if not final and self._newest_tf_ns < stamp + self._tf_wait_ns:
                    return
                self._pending.popleft()
                self.stats.pose_missing += 1
                logger.debug("[ros2] no pose at %d: %s", stamp, e)
                continue
            self._pending.popleft()
            raw = self._build(stamp, pack, rx_ns, dp, t_wc, q_wc)
            if raw is None:
                continue
            self._fe.last_rx_mono = time.monotonic()
            try:
                pkt = self._fe.admit(raw)
            except Exception as e:  # noqa: BLE001 -- parse_error line written inside admit(); one bad frame never ends the stream
                self.admit_errors += 1
                logger.warning("[ros2] frame seq %s skipped: %s", raw.header.seq, e)
                continue
            if pkt is not None and self._fe.enqueue(pkt):
                self.enqueued += 1
            self.stats.yielded += 1

    def _build(self, stamp: int, pack: _RgbPack, rx_ns: int, dp: Tuple[int, Any], t_wc, q_wc) -> Optional[RawFrame]:
        try:
            rgb_img = _rgb_encoded(pack.msg, self._types["rgb"])
            depth_img = _depth_encoded(dp[1], self._types["depth"])
        except ValueError as e:                          # UnsupportedEncoding is a ValueError too: refuse like the bag reader
            self.stats.decode_errors += 1
            if "encoding" in str(e).lower() or "big-endian" in str(e).lower():
                raise BagRefusal([("unsupported_depth_encoding" if "depth" in str(e).lower() else "unsupported_encoding", str(e))]) from None
            logger.warning("[ros2] frame at %d skipped: %s", stamp, e)
            return None
        ci = _info_at(self._infos, stamp) or self._rgb_info
        rgb_w, rgb_h = (rgb_img.width, rgb_img.height) if rgb_img.width else (int(ci.width), int(ci.height))
        intr = _intrinsics_for(ci, rgb_w, rgb_h)
        conf = _confidence_encoded(self._confs.pop(stamp, None))
        if self.tm.tracking:
            tracking = pack.tracking if pack.tracking is not None else (self._latest_tracking or "not_available")
        else:
            tracking = "normal"
        seq = pack.seq if pack.seq is not None else pack.index
        return RawFrame(header=FrameHeader(
            source=self._fe.source, seq=int(seq), t_sensor_ns=int(stamp), t_wall_utc_s=(rx_ns / 1e9 if rx_ns else None),
            tracking_state=str(tracking), keyframe_hint=None, rgb=rgb_img, depth=depth_img, intrinsics=intr,
            pose_raw=(t_wc, q_wc), pose_format=POSE_FMT_PREPARED, pose_convention=CONVENTION_OPENCV, confidence=conf,
            session_id=self.session_id, pose_frame_id=self._world or "", keep_encoded_rgb=False,
            extra={"pair_dt_ms": round(abs(dp[0] - stamp) / 1e6, 3), "rgb_hw": (int(rgb_h), int(rgb_w))},
        ))

    def as_dict(self) -> Dict[str, Any]:
        d = self.stats.as_dict()
        d.update({"ready": self._ready, "enqueued": self.enqueued, "admit_errors": self.admit_errors,
                  "before_ready_dropped": self.before_ready_dropped, "pending": len(self._pending), "queued": self.queued(),
                  "session_id": self.session_id, "error": (str(self.error) if self.error else None),
                  "refusal_codes": (list(self.error.codes) if isinstance(self.error, BagRefusal) else None)})
        return d


# ───────────────────────────── the rclpy shell ─────────────────────────────

def _import_rclpy():
    try:
        import rclpy  # noqa: F401
        from rclpy.executors import SingleThreadedExecutor  # noqa: F401
        from rclpy.qos import QoSProfile  # noqa: F401
        from rosidl_runtime_py.utilities import get_message  # noqa: F401
    except Exception as e:  # noqa: BLE001 -- ImportError on the dev box, RuntimeError on a half-sourced environment
        raise Ros2Unavailable(f"{RCLPY_HINT} ({type(e).__name__}: {e})") from None
    return rclpy


def _rclpy_init(rclpy) -> None:
    """``rclpy.init`` WITHOUT rclpy's own signal handlers: by default they swallow
    SIGINT (the context shuts down, no KeyboardInterrupt reaches the main
    thread) and ``python -m rtsm`` never leaves ``run_forever`` on Ctrl-C --
    found by the WSL gate. Our ``stop()`` shuts the context down instead."""
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):                 # very old rclpy: no option; accept the default
        rclpy.init(args=None)


def _qos_profile(choice: QosChoice):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=int(choice.depth),
        reliability=ReliabilityPolicy.RELIABLE if choice.reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if choice.durability == "transient_local" else DurabilityPolicy.VOLATILE,
    )


class _Graph:
    """Discovery on a live node: the topic map, the chosen QoS per role, the
    refusal when the stream lacks the essentials. Shared by the source and the probe."""

    def __init__(self, node, *, overrides: Optional[Dict[str, str]], qos_mode: str, timeout_s: float) -> None:
        self.node = node
        self.overrides = overrides or None
        self.qos_mode = qos_mode
        self.timeout_s = float(timeout_s)
        self.tm: Optional[TopicMap] = None
        self.info: Optional[BagInfo] = None
        self.qos: Dict[str, QosChoice] = {}
        self.reasons: List[Tuple[str, str]] = []

    def discover(self, stop: Optional[threading.Event] = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        last_err: Optional[BagRefusal] = None
        while True:
            self.info = topics_to_info(self.node.get_topic_names_and_types())
            try:
                self.tm = resolve_topics(self.info, self.overrides)
                last_err = None
            except BagRefusal as e:                      # an override naming a topic that is not (yet) advertised
                last_err = e
                self.tm = None
            if self.tm is not None and self.tm.rgb and self.tm.depth and self.tm.rgb_info:
                break
            if time.monotonic() >= deadline or (stop is not None and stop.is_set()):
                break
            time.sleep(0.25)
        if last_err is not None:
            self.reasons = list(last_err.reasons)
            return False
        tm = self.tm
        image_names = [t.name for t in self.info.of_type(IMAGE_T, COMPRESSED_T)]
        for role in ("rgb", "depth"):
            if getattr(tm, role) is None:
                self.reasons.append((f"no_{role}_topic", f"no {role} topic advertised within {self.timeout_s:.0f} s; image topics: {image_names}"))
        if tm.rgb_info is None:
            self.reasons.append(("no_camera_info", f"no CameraInfo topic for the RGB topic; CameraInfo topics: {[t.name for t in self.info.of_type(INFO_T)]}"))
        if self.reasons:
            return False
        for role in _LIVE_ROLES:
            topic = getattr(tm, role)
            if topic:
                pubs = self.node.get_publishers_info_by_topic(topic)
                self.qos[role] = choose_qos(role, pubs, self.qos_mode)
        return True

    def msgtypes(self) -> Dict[str, str]:
        return {role: self.info.topics[getattr(self.tm, role)].msgtype for role in _LIVE_ROLES if getattr(self.tm, role)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "topics": ({**self.tm.as_dict(), "rules": dict(self.tm.rules)} if self.tm else None),
            "msgtypes": (self.msgtypes() if self.tm and self.info else None),
            "qos": {r: {"chosen": {"reliability": q.reliability, "durability": q.durability, "depth": q.depth}, "offered": q.offered}
                    for r, q in self.qos.items()},
            "advertised": (sorted(self.info.topics) if self.info else []),
            "refusal": self.reasons or None,
        }


class Ros2Source:
    name = "ros2"

    def __init__(self, cfg: dict, ctx: SourceContext, *, topics: Optional[Dict[str, str]] = None, world_frame: Optional[str] = None,
                 camera_frame: Optional[str] = None, assume_aligned: bool = False, pair_tolerance_s: float = 0.02,
                 tf_extrapolation_s: float = 0.05, tf_wait_s: float = DEFAULT_TF_WAIT_S, qos: str = "auto", node_name: str = "rtsm",
                 discovery_timeout_s: float = 10.0, ready_timeout_s: float = 10.0, session_id: Optional[str] = None,
                 source_name: str = "ros2") -> None:
        if qos not in QOS_MODES:
            raise ValueError(f"io.ros2.qos must be one of {QOS_MODES}, got {qos!r}")
        self._ctx = ctx
        self._opts = dict(topics=topics, world_frame=world_frame, camera_frame=camera_frame, assume_aligned=bool(assume_aligned),
                          pair_tolerance_s=float(pair_tolerance_s), tf_extrapolation_s=float(tf_extrapolation_s), tf_wait_s=float(tf_wait_s))
        self._qos_mode = qos
        self._node_name = node_name
        self._discovery_timeout_s = float(discovery_timeout_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._session_id = session_id
        if ctx.apply_camera_flip:
            logger.info("[ros2] visualization.apply_camera_flip is ignored for ROS 2: optical frames are already the OpenCV convention")
        self._fe = IngestFrontEnd(
            source=source_name, policy=WEBSOCKET_POLICY, ingest_queue=ctx.ingest_queue,
            throttle_clock=ctx.clock_mode, keyframe_every_n=ctx.keyframe_every_n, keyframe_interval_s=ctx.keyframe_interval_s,
            nonkf_min_interval_s=ctx.nonkf_min_interval_s, require_tracking_normal=False,   # decided once the stream is known
            confidence_threshold=ctx.confidence_threshold, pose_sink=ctx.pose_sink, clearance_sink=ctx.clearance_sink,
            event_sink=ctx.event_sink, ledger_sink=ctx.ledger_sink, latency_analytics=ctx.latency_analytics,
            on_camera_frame=ctx.on_camera_frame, on_keyframe=ctx.on_keyframe,
        )
        self._stats = BagStats()
        self._ingress: Optional[Ros2Ingress] = None
        self._graph: Optional[_Graph] = None
        self._error: Optional[BaseException] = None
        self._stop_event = threading.Event()
        self._done = threading.Event()
        self._threads: List[threading.Thread] = []
        self._rclpy = None
        self._node = None
        self._executor = None
        self._we_initialised = False
        self._subs: List[Any] = []

    # ── Source protocol ──

    @property
    def frontend(self) -> IngestFrontEnd:
        return self._fe

    def start(self) -> None:
        """Import rclpy here (not at module import), then run discovery and subscriptions on a setup thread so the
        runner's start() never blocks; errors surface through stats()/error and the log."""
        self._rclpy = _import_rclpy()                    # raises Ros2Unavailable with the hint
        t = threading.Thread(target=self._setup_and_run, name="ros2-source", daemon=True)
        self._threads.append(t)
        t.start()
        logger.info("[ros2] starting node %r (qos %s, discovery timeout %.0f s)", self._node_name, self._qos_mode, self._discovery_timeout_s)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal every loop, then wait (bounded) for the setup thread's teardown:
        node destroyed and the rclpy context shut down BEFORE the interpreter
        finalises, or the DDS threads die under the C++ runtime and the process
        aborts on exit."""
        self._stop_event.set()
        q = self._ctx.ingest_queue
        if getattr(q, "policy", None) == "lossless":
            close = getattr(q, "close", None)
            if callable(close):
                close()
        try:
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001
            pass
        if self._threads and not self._done.wait(timeout=timeout):
            logger.warning("[ros2] teardown did not finish within %.0f s", timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout=timeout)

    def liveness(self) -> dict:
        alive = any(t.is_alive() for t in self._threads)
        return {"alive": alive, **self._fe.liveness()}

    def stats(self) -> Dict[str, Any]:
        d = self._ingress.as_dict() if self._ingress is not None else self._stats.as_dict()
        d.update({"node": self._node_name, "graph": (self._graph.as_dict() if self._graph else None), "done": self._done.is_set(),
                  "error": (str(self._error) if self._error else d.get("error")),
                  "refusal_codes": (list(self._error.codes) if isinstance(self._error, BagRefusal) else d.get("refusal_codes"))})
        return d

    @property
    def error(self) -> Optional[BaseException]:
        return self._error or (self._ingress.error if self._ingress else None)

    # ── setup + loops ──

    def _setup_and_run(self) -> None:
        rclpy = self._rclpy
        try:
            if not rclpy.ok():
                _rclpy_init(rclpy)
                self._we_initialised = True
            from rclpy.executors import SingleThreadedExecutor
            from rosidl_runtime_py.utilities import get_message
            self._node = rclpy.create_node(self._node_name)
            self._graph = _Graph(self._node, overrides=self._opts["topics"], qos_mode=self._qos_mode, timeout_s=self._discovery_timeout_s)
            if not self._graph.discover(self._stop_event):
                self._stats.refusal = list(self._graph.reasons)
                raise BagRefusal(self._graph.reasons)
            tm = self._graph.tm
            self._stats.topics = {**tm.as_dict(), "rules": dict(tm.rules)}
            self._ingress = Ros2Ingress(
                self._fe, self._stats, tm, self._graph.msgtypes(), pair_tolerance_s=self._opts["pair_tolerance_s"],
                tf_extrapolation_s=self._opts["tf_extrapolation_s"], tf_wait_s=self._opts["tf_wait_s"], world_frame=self._opts["world_frame"],
                camera_frame=self._opts["camera_frame"], assume_aligned=self._opts["assume_aligned"],
                require_tracking_cfg=bool(self._ctx.require_tracking_normal), session_id=self._session_id)
            ingress = self._ingress
            for role in _LIVE_ROLES:
                topic = getattr(tm, role)
                if not topic:
                    continue
                msg_cls = get_message(self._graph.msgtypes()[role])
                choice = self._graph.qos[role]
                cb = (lambda r: (lambda m: ingress.on_message(r, m, time.time_ns())))(role)
                self._subs.append(self._node.create_subscription(msg_cls, topic, cb, _qos_profile(choice)))
                logger.info("[ros2] %-10s %s (%s) qos %s/%s depth %d; publishers offer %s", role, topic, self._graph.msgtypes()[role],
                            choice.reliability, choice.durability, choice.depth,
                            [f"{o['node']}:{o['reliability']}/{o['durability']}" for o in choice.offered] or "none yet")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            spin = threading.Thread(target=self._spin, name="ros2-spin", daemon=True)
            self._threads.append(spin)
            spin.start()
            # readiness deadline: camera info + a moving TF chain + registration, or refuse with the reasons
            deadline = time.monotonic() + self._ready_timeout_s
            while not ingress.ready and not self._stop_event.is_set():
                ingress.step()
                if time.monotonic() >= deadline:
                    reasons = ingress.not_ready_reasons() or [("no_pose_source", "stream not ready before the deadline")]
                    self._stats.refusal = reasons
                    raise BagRefusal(reasons)
                time.sleep(0.05)
            ingress.run(self._stop_event)
            if ingress.error is not None:
                self._error = ingress.error
            else:
                ingress.step(final=True)
        except BagRefusal as e:
            self._error = e
            logger.error("[ros2] refused: %s", e)
        except Exception as e:  # noqa: BLE001
            self._error = e
            logger.exception("[ros2] source failed")
        finally:
            self._teardown()
            self._done.set()
            st = self._stats
            logger.info("[ros2] stopped: %s frames seen, %s paired, %s enqueued, %s unpaired rgb, %s without pose",
                        st.frames_seen, st.paired, (self._ingress.enqueued if self._ingress else 0), st.unpaired_rgb, st.pose_missing)

    def _spin(self) -> None:
        try:
            while not self._stop_event.is_set() and self._rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "ExternalShutdownException":   # the rclpy context was shut down from outside: a stop, not a fault
                logger.info("[ros2] rclpy context shut down; executor leaving")
                self._stop_event.set()
            elif not self._stop_event.is_set():
                self._error = e
                logger.exception("[ros2] executor stopped")

    def _teardown(self) -> None:
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._we_initialised and self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ───────────────────────────── probe ─────────────────────────────

def probe(*, topics: Optional[Dict[str, str]] = None, world_frame: Optional[str] = None, camera_frame: Optional[str] = None,
          assume_aligned: bool = False, qos: str = "auto", seconds: float = 5.0, node_name: str = "rtsm_probe") -> Dict[str, Any]:
    """What the node would subscribe to, and whether the stream is usable: the
    topic map with its rules, the publishers' QoS versus the chosen QoS, the
    CameraInfo and TF state after ``seconds`` of listening, the registration
    check, and the refusal in the bag reader's codes. No model loads."""
    rclpy = _import_rclpy()
    from rclpy.executors import SingleThreadedExecutor
    from rosidl_runtime_py.utilities import get_message
    we_init = False
    if not rclpy.ok():
        _rclpy_init(rclpy)
        we_init = True
    node = rclpy.create_node(node_name)
    out: Dict[str, Any] = {"seconds": seconds, "ok": False}
    try:
        g = _Graph(node, overrides=topics, qos_mode=qos, timeout_s=seconds)
        found = g.discover()
        out.update(g.as_dict())
        if not found:
            return out
        fe = IngestFrontEnd(source="ros2-probe", policy=WEBSOCKET_POLICY, ingest_queue=None, require_tracking_normal=False)
        st = BagStats()
        ingress = Ros2Ingress(fe, st, g.tm, g.msgtypes(), world_frame=world_frame, camera_frame=camera_frame,
                              assume_aligned=assume_aligned, require_tracking_cfg=False)
        subs = []
        for role in ("rgb_info", "depth_info", "tf", "tf_static", "odom"):
            topic = getattr(g.tm, role)
            if topic:
                cb = (lambda r: (lambda m: ingress.on_message(r, m, time.time_ns())))(role)
                subs.append(node.create_subscription(get_message(g.msgtypes()[role]), topic, cb, _qos_profile(g.qos[role])))
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        deadline = time.monotonic() + seconds
        refusal: Optional[List[Tuple[str, str]]] = None
        while time.monotonic() < deadline:
            ex.spin_once(timeout_sec=0.1)
            try:
                ingress.step()
            except BagRefusal as e:
                refusal = list(e.reasons)
                break
            if ingress.ready:
                break
        out.update({
            "ready": ingress.ready,
            "camera_info_seen": ingress._rgb_info is not None,
            "depth_info_seen": ingress._depth_info is not None,
            "tf": {"roots": ingress._buf.roots(), "hops": [{"parent": p, "child": c, "static": s} for p, c, s in ingress._buf.hops()],
                   "chain": st.tf_chain, "world_frame": st.world_frame or None, "camera_frame": st.camera_frame or None, "pose_kind": st.pose_kind or None},
            "registration": st.registration or None,
            "refusal": refusal or (None if ingress.ready else ingress.not_ready_reasons()),
        })
        out["ok"] = bool(ingress.ready and not refusal)
        return out
    finally:
        try:
            node.destroy_node()
        finally:
            if we_init and rclpy.ok():
                rclpy.shutdown()


def probe_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="rtsm ros2 probe",
                                 description="Show what the ros2 source would subscribe to on this ROS 2 graph, with the publishers' QoS "
                                             "and the TF / CameraInfo / registration state, before any model loads.")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to listen for CameraInfo and TF (default 5)")
    ap.add_argument("--qos", choices=QOS_MODES, default="auto")
    ap.add_argument("--topic", action="append", default=[], metavar="ROLE=TOPIC", help="override a role (rgb, depth, rgb_info, depth_info, tf, tf_static, odom, tracking, seq, confidence)")
    ap.add_argument("--world-frame", default=None)
    ap.add_argument("--camera-frame", default=None)
    ap.add_argument("--assume-aligned", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in args.topic if "=" in kv) or None
    try:
        res = probe(topics=overrides, world_frame=args.world_frame, camera_frame=args.camera_frame, assume_aligned=args.assume_aligned,
                    qos=args.qos, seconds=args.seconds)
    except Ros2Unavailable as e:
        print(f"rtsm ros2 probe: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        _print_probe(res)
    return 0 if res.get("ok") else 1


def _print_probe(res: Dict[str, Any]) -> None:
    print(f"ros2 probe ({res.get('seconds')} s): {'OK' if res.get('ok') else 'NOT USABLE'}")
    topics = res.get("topics") or {}
    rules = topics.get("rules", {}) if topics else {}
    print("topics:")
    for role in _LIVE_ROLES:
        t = topics.get(role) if topics else None
        if t:
            q = (res.get("qos") or {}).get(role, {})
            ch = q.get("chosen", {})
            off = ", ".join(f"{o['node'] or '?'}:{o['reliability']}" for o in q.get("offered", [])) or "no publisher seen"
            print(f"  {role:10s} {t}  [{rules.get(role, '')}]  subscribe {ch.get('reliability')}/{ch.get('durability')} depth {ch.get('depth')}  (offered: {off})")
    missing = [r for r in ("rgb", "depth", "rgb_info") if not (topics or {}).get(r)]
    if missing:
        print(f"  missing: {missing}; advertised: {res.get('advertised')}")
    tf = res.get("tf") or {}
    print(f"camera info: rgb {'seen' if res.get('camera_info_seen') else 'NOT seen'}, depth {'seen' if res.get('depth_info_seen') else 'not seen'}")
    print(f"tf: roots {tf.get('roots')}; {len(tf.get('hops') or [])} hops; chain {' | '.join(tf.get('chain') or []) or '-'}; pose {tf.get('pose_kind') or '-'}")
    print(f"registration: {res.get('registration') or '-'}")
    if res.get("refusal"):
        print("refusal:")
        for code, why in res["refusal"]:
            print(f"  {code}: {why}")
