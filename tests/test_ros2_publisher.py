"""The ``ros2`` publisher without rclpy (the CPU suite): the configuration and its defaults (off), the snapshot
of the working memory, the two messages built against fake message types, and the publisher's lifecycle on a
fake node. The real packages are exercised by the WSL2 gate (rtsm-lab 2026-10-ros2-publisher)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from rtsm.io.ros2_publisher import (
    NS_LABELS, NS_OBJECTS, MsgTypes, PublishConfig, Ros2Publisher, build_detections, build_markers, marker_id, snapshot_objects,
)
from rtsm.io.ros2_source import Ros2Unavailable

REPO = Path(__file__).resolve().parents[1]


# ───────────────────────────── configuration ─────────────────────────────

def test_default_is_off_in_code_and_in_the_packaged_config():
    assert PublishConfig().enable is False
    assert PublishConfig.from_cfg({}).enable is False
    cfg = yaml.safe_load((REPO / "rtsm" / "cfg" / "rtsm.yaml").read_text(encoding="utf-8"))
    pc = PublishConfig.from_cfg(cfg)
    assert pc.enable is False and pc.objects_topic == "/rtsm/objects" and pc.markers_topic == "/rtsm/markers"
    assert pc.rate_hz == 5.0 and pc.confirmed_only and pc.qos == "reliable" and pc.latched and pc.frame_id == "auto"
    assert Ros2Publisher.maybe(cfg, wm=None) is None


def test_config_validation():
    base = {"io": {"ros2": {"publish": {"enable": True}}}}
    assert PublishConfig.from_cfg(base).enable is True
    for bad in ({"qos": "sometimes"}, {"rate_hz": 0}, {"objects_topic": ""}, {"marker_size_m": 0}, {"max_hypotheses": 0}):
        with pytest.raises(ValueError):
            PublishConfig.from_cfg({"io": {"ros2": {"publish": {"enable": True, **bad}}}})
    pc = PublishConfig.from_cfg({"io": {"ros2": {"publish": {"enable": True, "markers_topic": None, "qos": "BEST_EFFORT", "latched": False}}}})
    assert pc.markers_topic == "" and pc.qos == "best_effort" and pc.latched is False and pc.as_dict()["markers_topic"] is None


# ───────────────────────────── snapshot ─────────────────────────────

def _obj(oid, xyz, label, scores, confirmed=True, hits=3, stability=0.9):
    return NS(id=oid, xyz_world=np.asarray(xyz, dtype=np.float32), label_primary=label, label_scores=scores, confirmed=confirmed,
              hits=hits, stability=stability)


class _WM:
    def __init__(self, objs):
        self._objs = objs

    def iter_objects(self):
        return list(self._objs)


def test_snapshot_filters_sorts_and_caps_labels():
    wm = _WM([_obj("b" * 16, [1, 2, 3], "cup", {"cup": 0.6, "mug": 0.4, "bowl": 0.1}),
              _obj("a" * 16, [0, 0, 0], "box", {"box": 0.5}, confirmed=False),
              _obj("c" * 16, [np.nan, 0, 0], "ghost", {"ghost": 0.9})])
    s = snapshot_objects(wm, confirmed_only=True, max_hypotheses=2)
    assert [r["id"] for r in s] == ["b" * 16]                       # unconfirmed and non-finite dropped
    assert s[0]["labels"] == [("cup", 0.6), ("mug", 0.4)] and s[0]["label"] == "cup" and s[0]["xyz"] == [1.0, 2.0, 3.0]
    s2 = snapshot_objects(wm, confirmed_only=False, max_hypotheses=5)
    assert [r["id"] for r in s2] == ["a" * 16, "b" * 16] and s2[0]["confirmed"] is False
    # a primary label absent from the capped list is kept in front
    wm3 = _WM([_obj("d" * 16, [0, 0, 1], "rare", {"a": 0.9, "b": 0.8, "rare": 0.1})])
    assert snapshot_objects(wm3, max_hypotheses=2)[0]["labels"][0] == ("rare", 0.1)


def test_marker_id_is_stable_and_int32():
    a, b = marker_id("abc"), marker_id("abc")
    assert a == b and 0 <= a < 2 ** 31 and marker_id("abd") != a


# ───────────────────────────── messages on fake types ─────────────────────────────

class _Auto:
    """A message stand-in: attribute access creates nested stand-ins; list fields are lists."""
    _lists = ("detections", "results", "markers")

    def __init__(self):
        for name in self._lists:
            object.__setattr__(self, name, [])

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = _Auto()
        object.__setattr__(self, name, child)
        return child


def _types():
    return MsgTypes(Detection3DArray=_Auto, Detection3D=_Auto, ObjectHypothesisWithPose=_Auto, MarkerArray=_Auto, Marker=_Auto)


_OBJECTS = [{"id": "0123456789abcdef", "xyz": [1.0, -2.0, 0.5], "label": "cup", "labels": [("cup", 0.6), ("mug", 0.4)],
             "confirmed": True, "hits": 5, "stability": 0.9},
            {"id": "fedcba9876543210", "xyz": [0.0, 0.0, 0.0], "label": None, "labels": [], "confirmed": False, "hits": 1, "stability": 0.2}]


def test_detection3d_array_mapping():
    arr = build_detections(_types(), _OBJECTS, frame_id="map", stamp="T", size_m=0.2)
    assert arr.header.frame_id == "map" and arr.header.stamp == "T" and len(arr.detections) == 2
    d = arr.detections[0]
    assert d.id == "0123456789abcdef" and d.header.frame_id == "map"
    assert [(h.hypothesis.class_id, h.hypothesis.score) for h in d.results] == [("cup", 0.6), ("mug", 0.4)]
    assert (d.results[0].pose.pose.position.x, d.results[0].pose.pose.position.y, d.results[0].pose.pose.position.z) == (1.0, -2.0, 0.5)
    assert d.results[0].pose.pose.orientation.w == 1.0
    assert (d.bbox.center.position.x, d.bbox.center.position.z) == (1.0, 0.5) and d.bbox.size.x == d.bbox.size.y == d.bbox.size.z == 0.2
    assert arr.detections[1].results == []                             # no label scores -> no hypotheses, position still in bbox


def test_marker_array_mapping():
    arr = build_markers(_types(), _OBJECTS, frame_id="map", stamp="T", size_m=0.2)
    assert arr.markers[0].action == 3 and len(arr.markers) == 1 + 2 * len(_OBJECTS)      # DELETEALL first, sphere + text per object
    sphere, text = arr.markers[1], arr.markers[2]
    mid = marker_id("0123456789abcdef")
    assert sphere.ns == NS_OBJECTS and sphere.id == mid and sphere.type == 2 and sphere.action == 0
    assert (sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z) == (1.0, -2.0, 0.5) and sphere.scale.x == 0.2
    assert text.ns == NS_LABELS and text.id == mid and text.type == 9 and text.text == "cup" and text.pose.position.z == pytest.approx(0.7)
    assert arr.markers[4].text == "fedcba98" and arr.markers[3].color.a == 0.5           # unlabelled: id prefix; unconfirmed: dimmer


# ───────────────────────────── lifecycle on a fake node ─────────────────────────────

class _FakePublisher:
    def __init__(self, topic):
        self.topic, self.published = topic, []

    def publish(self, msg):
        self.published.append(msg)


class _FakeNode:
    def __init__(self, name):
        self.name, self.publishers, self.destroyed = name, [], False

    def create_publisher(self, msg_type, topic, qos):
        p = _FakePublisher(topic)
        self.publishers.append((msg_type, topic, qos, p))
        return p

    def destroy_publisher(self, pub):
        pass

    def destroy_node(self):
        self.destroyed = True

    def get_clock(self):
        raise RuntimeError("no clock on the fake node")


class _FakeRclpy:
    def __init__(self, ok=False):
        self._ok, self.nodes, self.inits, self.shutdowns = ok, [], 0, 0

    def ok(self):
        return self._ok

    def init(self, args=None, **kw):
        self._ok = True
        self.inits += 1

    def shutdown(self):
        self._ok = False
        self.shutdowns += 1

    def create_node(self, name):
        n = _FakeNode(name)
        self.nodes.append(n)
        return n


def _fake_builtin_time(monkeypatch):
    class Time:
        def __init__(self, sec=0, nanosec=0):
            self.sec, self.nanosec = sec, nanosec
    import types
    mod = types.ModuleType("builtin_interfaces.msg")
    mod.Time = Time
    pkg = types.ModuleType("builtin_interfaces")
    pkg.msg = mod
    monkeypatch.setitem(sys.modules, "builtin_interfaces", pkg)
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", mod)


def _qos_fake(monkeypatch):
    import rtsm.io.ros2_publisher as P
    monkeypatch.setattr(P, "_qos_profile", lambda choice: ("qos", choice.reliability, choice.durability, choice.depth))
    monkeypatch.setattr(P, "_rclpy_init", lambda rclpy: rclpy.init())


def test_publisher_lifecycle_on_a_fake_node(monkeypatch):
    _fake_builtin_time(monkeypatch)
    _qos_fake(monkeypatch)
    wm = _WM([_obj("b" * 16, [1, 2, 3], "cup", {"cup": 0.6}), _obj("a" * 16, [0, 0, 0], "box", {"box": 0.5}, confirmed=False)])
    rc = _FakeRclpy(ok=False)
    cfg = PublishConfig(enable=True, rate_hz=50.0, latched=True)
    pub = Ros2Publisher(cfg, wm, node_name="rtsm", world_frame=lambda: "odom", _rclpy=rc, _msg_types=_types())
    pub.start()
    assert rc.inits == 1 and pub._we_initialised and rc.nodes[0].name == "rtsm_publisher"
    (t1, topic1, qos1, p1), (t2, topic2, qos2, p2) = rc.nodes[0].publishers
    assert (topic1, topic2) == ("/rtsm/objects", "/rtsm/markers") and qos1 == ("qos", "reliable", "transient_local", 1)
    r = pub.tick()
    assert r == {"objects": 1, "markers": 3, "frame_id": "odom"}
    assert p1.published[-1].detections[0].id == "b" * 16 and len(p2.published[-1].markers) == 3
    assert p1.published[-1].header.stamp.sec > 0                       # wall-clock fallback stamp
    st = pub.stats()
    assert st["ticks"] >= 1 and st["last_objects"] == 1 and st["last_markers"] == 3 and st["errors"] == 0 and st["alive"]
    assert st["frame_id_resolved"] == "odom" and st["node"] == "rtsm_publisher" and st["enable"] is True
    pub.stop()
    assert rc.nodes[0].destroyed and rc.shutdowns == 1 and not pub.stats()["alive"]


def test_publisher_shares_a_live_context_and_leaves_it_up(monkeypatch):
    _fake_builtin_time(monkeypatch)
    _qos_fake(monkeypatch)
    rc = _FakeRclpy(ok=True)                                              # the ros2 source already initialised rclpy
    pub = Ros2Publisher(PublishConfig(enable=True, markers_topic="", frame_id="world"), _WM([]), _rclpy=rc, _msg_types=_types())
    pub.start()
    assert rc.inits == 0 and not pub._we_initialised and len(rc.nodes[0].publishers) == 1
    assert pub.tick() == {"objects": 0, "markers": 0, "frame_id": "world"}
    pub.stop()
    assert rc.shutdowns == 0 and rc.nodes[0].destroyed


def test_tick_errors_are_counted_not_raised(monkeypatch):
    _fake_builtin_time(monkeypatch)
    _qos_fake(monkeypatch)
    class _BadWM:
        def iter_objects(self):
            raise RuntimeError("wm exploded")
    pub = Ros2Publisher(PublishConfig(enable=True), _BadWM(), _rclpy=_FakeRclpy(ok=True), _msg_types=_types())
    pub.start()
    assert "error" in pub.tick()
    assert pub.stats()["errors"] >= 1 and "wm exploded" in pub.stats()["last_error"]      # the tick thread may have failed too
    pub.stop()


def test_preflight_is_a_noop_when_disabled_and_refuses_early_when_enabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    assert Ros2Publisher.preflight({}) is None                                     # disabled: nothing imported
    with pytest.raises(Ros2Unavailable):
        Ros2Publisher.preflight({"io": {"ros2": {"publish": {"enable": True}}}})
    with pytest.raises(ValueError):
        Ros2Publisher.preflight({"io": {"ros2": {"publish": {"enable": True, "rate_hz": -1}}}})


def test_start_refuses_with_the_hint_when_rclpy_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    pub = Ros2Publisher(PublishConfig(enable=True), _WM([]))
    with pytest.raises(Ros2Unavailable) as ei:
        pub.start()
    assert "sourced ROS 2 environment" in str(ei.value)


def test_start_refuses_with_the_packages_hint_when_vision_msgs_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "vision_msgs", None)
    monkeypatch.setitem(sys.modules, "vision_msgs.msg", None)
    pub = Ros2Publisher(PublishConfig(enable=True), _WM([]), _rclpy=_FakeRclpy(ok=True))
    with pytest.raises(Ros2Unavailable) as ei:
        pub.start()
    assert "vision-msgs" in str(ei.value)
