"""
The minimal ``ros2`` live source without rclpy (the CPU suite): the registry
entry and its refusal off a ROS environment, the QoS chooser, discovery on a
live-style topic list, the rclpy-free ingress on synthetic messages
(readiness buffering, head-of-line TF wait, pose_missing), and the parity
predicate: session1_bag's messages pushed through ``Ros2Ingress`` in log order
produce the frames ``iter_bag_frames`` produces from the same bag.
"""
from __future__ import annotations

import array
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from rtsm.io import sources
from rtsm.io.bag_reader import BagRefusal, BagStats, TopicMap, iter_bag_frames, resolve_topics
from rtsm.io.contracts import SourceContext
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.ros2_source import (
    QOS_MODES, RCLPY_HINT, Ros2Ingress, Ros2Source, Ros2Unavailable, choose_qos, topics_to_info,
)

sys.path.insert(0, os.path.dirname(__file__))
from _recordings import real_recording  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SESSION1_BAG = REPO / "recordings" / "session1_bag"


# ───────────────────────────── registry + refusal ─────────────────────────────

def test_ros2_is_a_builtin_source_and_constructs_without_rclpy():
    assert "ros2" in sources.available_sources()
    ctx = SourceContext(ingest_queue=IngestQueue(2), clock_mode="sensor", keyframe_every_n=4)
    src = sources.make_source("ros2", {"io": {"ros2": {"qos": "auto", "node_name": "t"}}}, ctx)
    assert isinstance(src, Ros2Source) and src.name == "ros2"
    assert src.frontend.source == "ros2"


def test_start_refuses_with_the_hint_when_rclpy_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)          # import rclpy -> ImportError
    ctx = SourceContext(ingest_queue=IngestQueue(2))
    src = Ros2Source({}, ctx)
    with pytest.raises(Ros2Unavailable) as ei:
        src.start()
    assert "sourced ROS 2 environment" in str(ei.value) and RCLPY_HINT.split(":")[0] in str(ei.value)


def test_bad_qos_mode_is_rejected_early():
    ctx = SourceContext(ingest_queue=IngestQueue(2))
    with pytest.raises(ValueError):
        Ros2Source({}, ctx, qos="sometimes")


# ───────────────────────────── QoS chooser ─────────────────────────────

def _pub(rel="RELIABLE", dur="VOLATILE", node="rosbag2_player"):
    return NS(node_name=node, qos_profile=NS(reliability=NS(name=rel), durability=NS(name=dur)))


def test_choose_qos_matches_publishers_and_knows_tf_static():
    assert set(QOS_MODES) == {"auto", "reliable", "best_effort"}
    q = choose_qos("rgb", [_pub("RELIABLE")], "auto")
    assert (q.reliability, q.durability, q.depth) == ("reliable", "volatile", 100)
    assert q.offered == [{"node": "rosbag2_player", "reliability": "reliable", "durability": "volatile"}]
    q = choose_qos("rgb", [_pub("BEST_EFFORT")], "auto")
    assert q.reliability == "best_effort"
    q = choose_qos("rgb", [_pub("BEST_EFFORT"), _pub("RELIABLE", node="camera")], "auto")
    assert q.reliability == "reliable"                       # any reliable publisher -> reliable
    assert choose_qos("depth", [], "auto").reliability == "best_effort"   # nobody yet: compatible with either
    assert choose_qos("depth", [_pub("RELIABLE")], "best_effort").reliability == "best_effort"
    assert choose_qos("depth", [], "reliable").reliability == "reliable"
    q = choose_qos("tf_static", [_pub("RELIABLE", "TRANSIENT_LOCAL")], "auto")
    assert (q.durability, q.depth) == ("transient_local", 200)
    assert choose_qos("tracking", [], "auto").depth == 10
    with pytest.raises(ValueError):
        choose_qos("rgb", [], "fast")


# ───────────────────────────── discovery on a live topic list ─────────────────────────────

LIVE_GRAPH = [
    ("/camera/color/image_raw", ["sensor_msgs/msg/Image"]),
    ("/camera/depth/image_rect_raw", ["sensor_msgs/msg/Image"]),
    ("/camera/confidence/image_raw", ["sensor_msgs/msg/Image"]),
    ("/camera/color/camera_info", ["sensor_msgs/msg/CameraInfo"]),
    ("/tf", ["tf2_msgs/msg/TFMessage"]),
    ("/tf_static", ["tf2_msgs/msg/TFMessage"]),
    ("/arkit/tracking_state", ["std_msgs/msg/String"]),
    ("/arkit/frame_seq", ["std_msgs/msg/UInt32"]),
    ("/rosout", ["rcl_interfaces/msg/Log"]),
    ("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]),
]


def test_topics_to_info_feeds_the_bag_readers_rules():
    info = topics_to_info(LIVE_GRAPH)
    assert info.kind == "ros2" and set(info.topics) >= {"/camera/color/image_raw", "/tf_static"}
    tm = resolve_topics(info)
    assert tm.rgb == "/camera/color/image_raw" and tm.depth == "/camera/depth/image_rect_raw"
    assert tm.rgb_info == "/camera/color/camera_info" and tm.confidence == "/camera/confidence/image_raw"
    assert tm.tf == "/tf" and tm.tf_static == "/tf_static"
    assert tm.tracking == "/arkit/tracking_state" and tm.seq == "/arkit/frame_seq"
    assert tm.rules["rgb"].startswith("image topic matching")
    tm2 = resolve_topics(info, {"rgb": "/camera/color/image_raw"})
    assert tm2.rules["rgb"] == "override"
    with pytest.raises(BagRefusal):
        resolve_topics(info, {"depth": "/not/advertised"})
    # a topic advertised without a type is ignored, names get their leading slash
    info2 = topics_to_info([("camera/rgb", ["sensor_msgs/msg/Image"]), ("/empty", [])])
    assert list(info2.topics) == ["/camera/rgb"]


# ───────────────────────────── synthetic messages ─────────────────────────────

def _stamp(t_s: float):
    sec = int(t_s)
    return NS(sec=sec, nanosec=int(round((t_s - sec) * 1e9)))


def _hdr(t_s: float, frame_id: str = ""):
    return NS(stamp=_stamp(t_s), frame_id=frame_id)


def _rgb(t_s: float, w: int = 8, h: int = 6, val: int = 7):
    data = array.array("B", np.full((h, w, 3), val, dtype=np.uint8).tobytes())   # rclpy delivers array('B')
    return NS(header=_hdr(t_s, "cam_optical"), width=w, height=h, encoding="rgb8", is_bigendian=0, step=w * 3, data=data)


def _depth(t_s: float, w: int = 8, h: int = 6, mm: int = 1500):
    data = array.array("B", np.full((h, w), mm, dtype=np.uint16).tobytes())
    return NS(header=_hdr(t_s, "cam_optical"), width=w, height=h, encoding="16UC1", is_bigendian=0, step=w * 2, data=data)


def _info(t_s: float, w: int = 8, h: int = 6, frame_id: str = "cam_optical", fx: float = 10.0):
    return NS(header=_hdr(t_s, frame_id), width=w, height=h, k=[fx, 0, w / 2, 0, fx, h / 2, 0, 0, 1])


def _tf(t_s: float, parent: str, child: str, x: float):
    tr = NS(header=_hdr(t_s, parent), child_frame_id=child,
            transform=NS(translation=NS(x=x, y=0.0, z=0.0), rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))
    return NS(transforms=[tr])


class _FakeFE:
    """Captures what the ingress would hand the front-end."""
    source = "ros2"

    def __init__(self):
        self.raws, self.sessions, self.require_tracking_normal, self.last_rx_mono = [], [], True, None

    def new_session(self, sid):
        self.sessions.append(sid)
        return 1

    def admit(self, raw, **_kw):
        self.raws.append(raw)
        return ("pkt", raw.header.seq)

    def enqueue(self, pkt):
        return True


def _ingress(fe=None, **kw):
    fe = fe or _FakeFE()
    tm = TopicMap(rgb="/rgb", depth="/depth", rgb_info="/info", tf="/tf")
    types_ = {"rgb": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image", "rgb_info": "sensor_msgs/msg/CameraInfo", "tf": "tf2_msgs/msg/TFMessage"}
    return fe, Ros2Ingress(fe, BagStats(), tm, types_, session_id="t", **kw)


def test_readiness_buffers_pairs_until_camera_info_and_a_moving_tf_chain_exist():
    fe, ing = _ingress()
    # two pairs arrive before anything else is known
    for t in (1.00, 1.10):
        ing.on_message("rgb", _rgb(t), 1)
        ing.on_message("depth", _depth(t + 0.003), 1)
    ing.on_message("depth", _depth(1.30), 1)           # a depth newer than rgb + tol resolves the pairs
    assert ing.step() == 0 and not ing.ready and ing.stats.paired == 2
    assert [c for c, _ in ing.not_ready_reasons()] == ["no_camera_info"]
    ing.on_message("rgb_info", _info(1.0), 1)
    assert ing.step() == 0 and not ing.ready
    assert ing.not_ready_reasons()[0][0] == "no_pose_source"
    # a moving chain world -> cam_optical (two samples) makes the stream ready; the held pairs flush in order
    ing.on_message("tf", _tf(0.90, "world", "cam_optical", 0.0), 1)
    ing.on_message("tf", _tf(1.40, "world", "cam_optical", 0.5), 1)
    assert ing.step() == 2 and ing.ready
    assert fe.sessions == ["t"] and [r.header.t_sensor_ns for r in fe.raws] == [1_000_000_000, 1_100_000_000]
    assert ing.stats.world_frame == "world" and ing.stats.camera_frame == "cam_optical" and ing.stats.pose_kind == "tf"
    assert fe.raws[0].header.pose_format == "prepared" and fe.raws[0].header.pose_frame_id == "world"
    np.testing.assert_allclose(fe.raws[0].header.pose_raw[0], [0.1, 0.0, 0.0], atol=1e-9)   # interpolated at t = 1.0
    assert fe.raws[0].header.rgb.encoding == "ros:rgb8" and fe.raws[0].header.depth.encoding == "uint16_mm"
    assert fe.raws[0].header.intrinsics.fx == 10.0 and fe.raws[0].header.extra["rgb_hw"] == (6, 8)
    assert fe.require_tracking_normal is False                # no tracking topic -> filter off


def test_head_of_line_wait_for_tf_then_pose_missing():
    fe, ing = _ingress(tf_wait_s=0.5)
    ing.on_message("rgb_info", _info(1.0), 1)
    ing.on_message("tf", _tf(0.90, "world", "cam_optical", 0.0), 1)
    ing.on_message("tf", _tf(1.00, "world", "cam_optical", 0.1), 1)
    ing.step()
    assert ing.ready
    # a pair at t = 1.5 whose TF has not arrived yet: it waits (nothing emitted, nothing lost)
    ing.on_message("rgb", _rgb(1.50), 1)
    ing.on_message("depth", _depth(1.50), 1)
    ing.on_message("depth", _depth(1.80), 1)
    assert ing.step() == 0 and ing.stats.paired == 1 and ing.stats.pose_missing == 0
    # TF catches up: the frame goes out with the interpolated pose
    ing.on_message("tf", _tf(1.60, "world", "cam_optical", 0.7), 1)
    assert ing.step() == 1 and ing.stats.pose_missing == 0
    # a pair whose TF never comes: it waits while the stream lives (nothing lost, nothing emitted) ...
    ing.on_message("rgb", _rgb(3.00), 1)
    ing.on_message("depth", _depth(3.00), 1)
    ing.on_message("depth", _depth(3.30), 1)
    assert ing.step() == 0 and ing.stats.pose_missing == 0 and len(fe.raws) == 1
    # ... and counts as pose_missing at the final flush (TF ends at 1.6, beyond the 50 ms extrapolation)
    ing.step(final=True)
    assert ing.stats.pose_missing == 1 and len(fe.raws) == 1


def test_unpaired_rgb_and_zero_stamps_are_counted():
    fe, ing = _ingress()
    ing.on_message("rgb", _rgb(0.0), 1)                 # zero stamp: skipped
    ing.on_message("rgb", _rgb(2.0), 1)                 # no depth within 20 ms
    ing.on_message("depth", _depth(2.5), 1)
    ing.step()
    assert ing.stats.skipped_zero_stamp == 1 and ing.stats.unpaired_rgb == 1 and ing.stats.frames_seen == 1


def test_unaligned_depth_refuses_like_the_bag_reader():
    fe, ing = _ingress()
    ing.tm.depth_info = "/depth_info"
    ing.on_message("rgb_info", _info(1.0, fx=10.0), 1)
    ing.on_message("depth_info", _info(1.0, fx=12.0, frame_id="depth_optical"), 1)
    ing.on_message("tf", _tf(0.9, "world", "cam_optical", 0.0), 1)
    ing.on_message("tf", _tf(1.1, "world", "cam_optical", 0.1), 1)
    with pytest.raises(BagRefusal) as ei:
        ing.step()
    assert ei.value.codes == ["unaligned_depth"] and "io.ros2.assume_aligned" in str(ei.value)


# ───────────────────────────── parity with the bag reader ─────────────────────────────

@pytest.mark.skipif(not ((SESSION1_BAG / "metadata.yaml").is_file() and any(real_recording(p) for p in SESSION1_BAG.glob("*.mcap"))),
                    reason="recordings/session1_bag not available (LFS pointer or missing)")
def test_session1_bag_through_the_live_ingress_matches_the_bag_reader():
    """The determinism argument for the live path: the same messages, in log
    order, through Ros2Ingress produce the frames iter_bag_frames produces."""
    from rtsm.io.bag_reader import _open_stream
    expected = list(iter_bag_frames(SESSION1_BAG))         # the whole bag: 240 frames on session1
    stream = _open_stream(SESSION1_BAG, "humble")
    try:
        info = stream.info
        tm = resolve_topics(info)
        roles = {getattr(tm, r): r for r in TopicMap.ROLES if getattr(tm, r)}
        types_ = {r: info.topics[t].msgtype for t, r in roles.items()}
        fe = _FakeFE()
        ing = Ros2Ingress(fe, BagStats(), tm, types_, session_id="live")
        for topic, _mt, log_ns, msg in stream.messages(list(roles)):
            ing.on_message(roles[topic], msg, log_ns)
            ing.step()
        ing.step(final=True)
    finally:
        stream.close()
    got = fe.raws
    assert len(got) == len(expected) > 100
    for a, b in zip(got, expected):
        ha, hb = a.header, b.header
        assert (ha.seq, ha.t_sensor_ns, ha.tracking_state) == (hb.seq, hb.t_sensor_ns, hb.tracking_state)
        np.testing.assert_allclose(ha.pose_raw[0], hb.pose_raw[0], atol=1e-12)
        np.testing.assert_allclose(ha.pose_raw[1], hb.pose_raw[1], atol=1e-12)
        assert (ha.pose_format, ha.pose_convention, ha.pose_frame_id) == (hb.pose_format, hb.pose_convention, hb.pose_frame_id)
        assert (ha.rgb.encoding, ha.rgb.width, ha.rgb.height) == (hb.rgb.encoding, hb.rgb.width, hb.rgb.height)
        assert bytes(ha.rgb.data) == bytes(hb.rgb.data)
        assert (ha.depth.encoding, ha.depth.width, ha.depth.height, ha.depth.scale) == (hb.depth.encoding, hb.depth.width, hb.depth.height, hb.depth.scale)
        assert bytes(ha.depth.data) == bytes(hb.depth.data)
        assert (ha.confidence is None) == (hb.confidence is None)
        if ha.confidence is not None:
            assert bytes(ha.confidence.data) == bytes(hb.confidence.data)
        assert (ha.intrinsics.fx, ha.intrinsics.fy, ha.intrinsics.cx, ha.intrinsics.cy) == (hb.intrinsics.fx, hb.intrinsics.fy, hb.intrinsics.cx, hb.intrinsics.cy)
        assert ha.extra == hb.extra
    assert ing.stats.pose_missing == 0 and ing.before_ready_dropped == 0
    assert fe.require_tracking_normal is True                 # session1 carries a tracking topic
