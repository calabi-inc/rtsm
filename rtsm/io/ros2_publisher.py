"""The ``ros2`` publisher: RTSM's objects as a periodic state snapshot on two ROS 2 topics.

Off unless ``io.ros2.publish.enable`` is true. Two standard message types, so no ``ament``
package and RTSM stays pip-installable:

* ``vision_msgs/Detection3DArray`` on ``objects_topic``: the objects;
* ``visualization_msgs/MarkerArray`` on ``markers_topic``: the same objects for RViz (``""`` = off).

Every tick (``rate_hz``) publishes the WHOLE object list (confirmed only by default) as it
stands: idempotent, so a subscriber that joins late, drops a message or restarts is right again
at the next tick; with ``latched`` (transient-local durability, depth 1) the last snapshot
reaches anyone who subscribes afterwards. No per-event stream, no services: queries stay on
REST and MCP. The publisher is independent of the ingest source (frames may come from Lens, a
bag or a ROS 2 graph) and keeps its own node, ``<io.ros2.node_name>_publisher``.

The field mapping is the contract (docs/api/ros2.md):

    Detection3DArray.header         frame_id = the world frame; stamp = the node clock at the tick
    Detection3D.id                  the object id
    Detection3D.results[k]          one ObjectHypothesisWithPose per label score, best first:
                                    hypothesis.class_id = the label, hypothesis.score = its EWMA score,
                                    pose.pose.position = the object's world position (identity orientation)
    Detection3D.bbox.center         the world position; bbox.size = ``marker_size_m`` on every axis
                                    (RTSM estimates no extent)
    Marker (ns "rtsm/objects")      SPHERE at the position, ``marker_size_m``, id = crc32 of the object id
    Marker (ns "rtsm/labels")       TEXT_VIEW_FACING with the primary label above the sphere, same id
    MarkerArray                     a DELETEALL marker first, then the current objects

``/stats.ros2_publish`` carries the counters (ticks, objects and markers in the last message,
errors). rclpy and the two message packages are imported at ``start()``, never at module import.
"""
from __future__ import annotations

import logging
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rtsm.io.ros2_source import QosChoice, Ros2Unavailable, _import_rclpy, _qos_profile, _rclpy_init

logger = logging.getLogger(__name__)

QOS_MODES = ("reliable", "best_effort")
MSGS_HINT = ("the ros2 publisher needs vision_msgs and visualization_msgs: "
             "sudo apt install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-visualization-msgs")
_SPHERE, _TEXT_VIEW_FACING = 2, 9              # visualization_msgs/Marker.type
_ADD, _DELETEALL = 0, 3                         # visualization_msgs/Marker.action
NS_OBJECTS, NS_LABELS = "rtsm/objects", "rtsm/labels"


# ───────────────────────────── configuration ─────────────────────────────

@dataclass
class PublishConfig:
    """``io.ros2.publish``. Defaults are the documented ones; ``enable`` is False."""
    enable: bool = False
    objects_topic: str = "/rtsm/objects"
    markers_topic: str = "/rtsm/markers"        # "" disables the markers
    frame_id: str = "auto"                      # auto = the ros2 source's world frame, else "map"
    rate_hz: float = 5.0
    confirmed_only: bool = True
    qos: str = "reliable"                       # reliable | best_effort
    latched: bool = True                        # transient-local durability, depth 1
    marker_size_m: float = 0.15
    max_hypotheses: int = 5

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "PublishConfig":
        p = (((cfg or {}).get("io") or {}).get("ros2") or {}).get("publish") or {}
        c = cls(enable=bool(p.get("enable", False)),
                objects_topic=str(p.get("objects_topic", cls.objects_topic) or ""),
                markers_topic=str(p.get("markers_topic", cls.markers_topic) or ""),
                frame_id=str(p.get("frame_id", cls.frame_id) or "auto"),
                rate_hz=float(p.get("rate_hz", cls.rate_hz)),
                confirmed_only=bool(p.get("confirmed_only", cls.confirmed_only)),
                qos=str(p.get("qos", cls.qos)).lower(),
                latched=bool(p.get("latched", cls.latched)),
                marker_size_m=float(p.get("marker_size_m", cls.marker_size_m)),
                max_hypotheses=int(p.get("max_hypotheses", cls.max_hypotheses)))
        c.validate()
        return c

    def validate(self) -> None:
        if self.qos not in QOS_MODES:
            raise ValueError(f"io.ros2.publish.qos must be one of {QOS_MODES}, got {self.qos!r}")
        if not (self.rate_hz > 0):
            raise ValueError(f"io.ros2.publish.rate_hz must be > 0, got {self.rate_hz}")
        if not self.objects_topic:
            raise ValueError("io.ros2.publish.objects_topic must be a topic name")
        if not (self.marker_size_m > 0):
            raise ValueError(f"io.ros2.publish.marker_size_m must be > 0, got {self.marker_size_m}")
        if self.max_hypotheses < 1:
            raise ValueError(f"io.ros2.publish.max_hypotheses must be >= 1, got {self.max_hypotheses}")

    def as_dict(self) -> Dict[str, Any]:
        return {"enable": self.enable, "objects_topic": self.objects_topic, "markers_topic": self.markers_topic or None,
                "frame_id": self.frame_id, "rate_hz": self.rate_hz, "confirmed_only": self.confirmed_only, "qos": self.qos,
                "latched": self.latched, "marker_size_m": self.marker_size_m, "max_hypotheses": self.max_hypotheses}


# ───────────────────────────── snapshot (pure) ─────────────────────────────

def snapshot_objects(wm: Any, *, confirmed_only: bool = True, max_hypotheses: int = 5) -> List[Dict[str, Any]]:
    """The working memory's objects as plain records: id, xyz (3 floats), label (primary), labels (label, score)
    best first, confirmed, hits, stability. ``wm.iter_objects()`` copies the list under the WM lock."""
    out: List[Dict[str, Any]] = []
    for o in wm.iter_objects():
        if confirmed_only and not bool(getattr(o, "confirmed", False)):
            continue
        xyz = np.asarray(getattr(o, "xyz_world"), dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            continue
        scores = dict(getattr(o, "label_scores", None) or {})
        labels = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda kv: (-kv[1], kv[0]))[:max_hypotheses]
        primary = getattr(o, "label_primary", None)
        if primary is not None and all(l != primary for l, _ in labels):
            labels.insert(0, (str(primary), float(scores.get(primary, 0.0))))
        out.append({"id": str(getattr(o, "id")), "xyz": [float(v) for v in xyz], "label": (str(primary) if primary else None),
                    "labels": labels, "confirmed": bool(getattr(o, "confirmed", False)),
                    "hits": int(getattr(o, "hits", 0) or 0), "stability": float(getattr(o, "stability", 0.0) or 0.0)})
    out.sort(key=lambda r: r["id"])
    return out


def marker_id(object_id: str) -> int:
    """A stable int32 for an object id (RViz keys markers by namespace + int id)."""
    return zlib.crc32(object_id.encode("utf-8")) & 0x7FFFFFFF


# ───────────────────────────── messages (pure given the types) ─────────────────────────────

@dataclass
class MsgTypes:
    """The message classes, resolved from the ROS packages at start() or injected (tests)."""
    Detection3DArray: Any
    Detection3D: Any
    ObjectHypothesisWithPose: Any
    MarkerArray: Any
    Marker: Any

    @classmethod
    def resolve(cls) -> "MsgTypes":
        try:
            from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
            from visualization_msgs.msg import Marker, MarkerArray
        except Exception as e:  # noqa: BLE001
            raise Ros2Unavailable(f"{MSGS_HINT} ({type(e).__name__}: {e})") from None
        return cls(Detection3DArray, Detection3D, ObjectHypothesisWithPose, MarkerArray, Marker)


def _set_xyz(p: Any, xyz: List[float]) -> None:
    p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])


def build_detections(mt: MsgTypes, objects: List[Dict[str, Any]], *, frame_id: str, stamp: Any, size_m: float) -> Any:
    arr = mt.Detection3DArray()
    arr.header.frame_id = frame_id
    arr.header.stamp = stamp
    for r in objects:
        d = mt.Detection3D()
        d.header.frame_id = frame_id
        d.header.stamp = stamp
        d.id = r["id"]
        for label, score in r["labels"]:
            h = mt.ObjectHypothesisWithPose()
            h.hypothesis.class_id = label
            h.hypothesis.score = float(score)
            _set_xyz(h.pose.pose.position, r["xyz"])
            h.pose.pose.orientation.w = 1.0
            d.results.append(h)
        _set_xyz(d.bbox.center.position, r["xyz"])
        d.bbox.center.orientation.w = 1.0
        d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = float(size_m)
        arr.detections.append(d)
    return arr


def build_markers(mt: MsgTypes, objects: List[Dict[str, Any]], *, frame_id: str, stamp: Any, size_m: float) -> Any:
    arr = mt.MarkerArray()
    wipe = mt.Marker()
    wipe.header.frame_id = frame_id
    wipe.header.stamp = stamp
    wipe.action = _DELETEALL
    arr.markers.append(wipe)
    for r in objects:
        mid = marker_id(r["id"])
        s = mt.Marker()
        s.header.frame_id = frame_id
        s.header.stamp = stamp
        s.ns, s.id, s.type, s.action = NS_OBJECTS, mid, _SPHERE, _ADD
        _set_xyz(s.pose.position, r["xyz"])
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = float(size_m)
        s.color.r, s.color.g, s.color.b, s.color.a = (0.0, 0.94, 1.0, 0.85) if r["confirmed"] else (0.64, 0.64, 0.64, 0.5)
        arr.markers.append(s)
        t = mt.Marker()
        t.header.frame_id = frame_id
        t.header.stamp = stamp
        t.ns, t.id, t.type, t.action = NS_LABELS, mid, _TEXT_VIEW_FACING, _ADD
        _set_xyz(t.pose.position, [r["xyz"][0], r["xyz"][1], r["xyz"][2] + float(size_m)])
        t.pose.orientation.w = 1.0
        t.scale.z = float(size_m) * 0.8
        t.color.r, t.color.g, t.color.b, t.color.a = 0.93, 0.93, 0.93, 1.0
        t.text = r["label"] or r["id"][:8]
        arr.markers.append(t)
    return arr


# ───────────────────────────── the publisher ─────────────────────────────

class Ros2Publisher:
    """Owns a node (``<node_name>_publisher``), two publishers and a tick thread.

    ``world_frame`` is a callable returning the ros2 source's world frame (or None) for ``frame_id: auto``.
    ``start()`` raises ``Ros2Unavailable`` with the hint when rclpy or the message packages are missing: an
    explicitly enabled publisher that cannot publish is an error, not a warning.
    """

    def __init__(self, cfg: PublishConfig, wm: Any, *, node_name: str = "rtsm", world_frame: Optional[Callable[[], Optional[str]]] = None,
                 _rclpy: Any = None, _msg_types: Optional[MsgTypes] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self._wm = wm
        self._node_name = f"{node_name}_publisher"
        self._world_frame = world_frame
        self._rclpy = _rclpy
        self._mt = _msg_types
        self._node = None
        self._pub_objects = None
        self._pub_markers = None
        self._we_initialised = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._counters: Dict[str, Any] = {"ticks": 0, "published": 0, "last_objects": 0, "last_markers": 0, "errors": 0, "last_error": None,
                                          "last_tick_wall_utc": None}

    @classmethod
    def maybe(cls, cfg_root: Dict[str, Any], wm: Any, **kw: Any) -> Optional["Ros2Publisher"]:
        """None unless ``io.ros2.publish.enable``."""
        pc = PublishConfig.from_cfg(cfg_root)
        if not pc.enable:
            return None
        node_name = str((((cfg_root or {}).get("io") or {}).get("ros2") or {}).get("node_name", "rtsm") or "rtsm")
        return cls(pc, wm, node_name=node_name, **kw)

    # ── lifecycle ──
    def start(self) -> None:
        rclpy = self._rclpy or _import_rclpy()                       # raises Ros2Unavailable with the hint
        self._rclpy = rclpy
        mt = self._mt or MsgTypes.resolve()                          # raises Ros2Unavailable with the packages hint
        self._mt = mt
        if not rclpy.ok():
            _rclpy_init(rclpy)
            self._we_initialised = True
        self._node = rclpy.create_node(self._node_name)
        qos = _qos_profile(QosChoice(reliability=self.cfg.qos, durability=("transient_local" if self.cfg.latched else "volatile"),
                                     depth=1, offered=[]))
        self._pub_objects = self._node.create_publisher(mt.Detection3DArray, self.cfg.objects_topic, qos)
        if self.cfg.markers_topic:
            self._pub_markers = self._node.create_publisher(mt.MarkerArray, self.cfg.markers_topic, qos)
        self._thread = threading.Thread(target=self._loop, name="ros2-publisher", daemon=True)
        self._thread.start()
        logger.info("[ros2 publish] node %r: %s (Detection3DArray)%s at %.1f Hz, %s%s, frame %s, %s", self._node_name, self.cfg.objects_topic,
                    (f" + {self.cfg.markers_topic} (MarkerArray)" if self._pub_markers else ""), self.cfg.rate_hz, self.cfg.qos,
                    (" latched" if self.cfg.latched else ""), self.cfg.frame_id, ("confirmed objects" if self.cfg.confirmed_only else "all objects"))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        for attr in ("_pub_objects", "_pub_markers"):
            pub = getattr(self, attr)
            if pub is not None and self._node is not None:
                try:
                    self._node.destroy_publisher(pub)
                except Exception:  # noqa: BLE001
                    pass
            setattr(self, attr, None)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            self._node = None
        if self._we_initialised and self._rclpy is not None:
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._we_initialised = False
        logger.info("[ros2 publish] stopped after %d ticks", self._counters["ticks"])

    # ── one tick ──
    def frame_id(self) -> str:
        if self.cfg.frame_id != "auto":
            return self.cfg.frame_id
        try:
            wf = self._world_frame() if self._world_frame else None
        except Exception:  # noqa: BLE001
            wf = None
        return str(wf) if wf else "map"

    def _stamp(self) -> Any:
        try:
            return self._node.get_clock().now().to_msg()
        except Exception:  # noqa: BLE001
            from builtin_interfaces.msg import Time
            ns = time.time_ns()
            return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def tick(self) -> Dict[str, Any]:
        """Build and publish one snapshot; counters updated; exceptions counted, never raised into the loop."""
        try:
            objects = snapshot_objects(self._wm, confirmed_only=self.cfg.confirmed_only, max_hypotheses=self.cfg.max_hypotheses)
            frame_id, stamp = self.frame_id(), self._stamp()
            det = build_detections(self._mt, objects, frame_id=frame_id, stamp=stamp, size_m=self.cfg.marker_size_m)
            self._pub_objects.publish(det)
            n_markers = 0
            if self._pub_markers is not None:
                mk = build_markers(self._mt, objects, frame_id=frame_id, stamp=stamp, size_m=self.cfg.marker_size_m)
                self._pub_markers.publish(mk)
                n_markers = len(mk.markers)
            with self._lock:
                self._counters.update(ticks=self._counters["ticks"] + 1, published=self._counters["published"] + 1 + (1 if n_markers else 0),
                                      last_objects=len(objects), last_markers=n_markers, last_tick_wall_utc=time.time())
            return {"objects": len(objects), "markers": n_markers, "frame_id": frame_id}
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._counters["ticks"] += 1
                self._counters["errors"] += 1
                self._counters["last_error"] = f"{type(e).__name__}: {e}"
            logger.debug("[ros2 publish] tick failed", exc_info=True)
            return {"objects": 0, "markers": 0, "error": str(e)}

    def _loop(self) -> None:
        period = 1.0 / float(self.cfg.rate_hz)
        while not self._stop_event.is_set():
            if self._rclpy is not None and not self._rclpy.ok():
                logger.info("[ros2 publish] rclpy context is down; publisher leaving")
                return
            self.tick()
            self._stop_event.wait(period)

    # ── observability ──
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self._counters)
        return {**self.cfg.as_dict(), "node": self._node_name, "frame_id_resolved": (self.frame_id() if self._node is not None else None),
                "alive": bool(self._thread is not None and self._thread.is_alive()), **c}
