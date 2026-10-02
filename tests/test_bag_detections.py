"""
A vision_msgs detections topic through the bag reader and the bag source (P3,
the detections adapter): discovery and overrides, the stamp join (detectors
publish after the image), the counts, the packet field, 3-D boxes through the
bag's own TF chain and intrinsics, and max_frames with a joiner in the loop.
Synthetic MCAP bags from the task-1 helpers (8x6 images).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rtsm.io.bag_reader import BagRefusal, BagStats, iter_bag_frames, probe_bag
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.msgdefs import DETECTION_2D, DETECTION_3D, register_vision_msgs
from rtsm.io.sources import make_source

from test_bag_reader import S, T, TS, _hdr, simple_bag_messages, write_bag
from test_bag_source import _ctx

register_vision_msgs(TS, "ros2")


# ───────────────────────────── builders ─────────────────────────────

def _hyp(label, score):
    OHP, OH, PWC, P, Pt, Q = (T["vision_msgs/msg/ObjectHypothesisWithPose"], T["vision_msgs/msg/ObjectHypothesis"], T["geometry_msgs/msg/PoseWithCovariance"],
                              T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/Point"], T["geometry_msgs/msg/Quaternion"])
    pose = PWC(pose=P(position=Pt(x=0.0, y=0.0, z=0.0), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)), covariance=np.zeros(36))
    return OHP(hypothesis=OH(class_id=label, score=(float(score) if score is not None else float("nan"))), pose=pose)


def det2d(stamp_ns, boxes, *, frame="cam_optical"):
    """boxes: [(cx, cy, w, h, label, score)] in the 8x6 image."""
    BB, P2, PT = T["vision_msgs/msg/BoundingBox2D"], T["vision_msgs/msg/Pose2D"], T["vision_msgs/msg/Point2D"]
    dets = [T["vision_msgs/msg/Detection2D"](header=_hdr(stamp_ns, frame), results=[_hyp(l, s)],
                                              bbox=BB(center=P2(position=PT(x=float(cx), y=float(cy)), theta=0.0), size_x=float(w), size_y=float(h)), id="")
            for cx, cy, w, h, l, s in boxes]
    return T["vision_msgs/msg/Detection2DArray"](header=_hdr(stamp_ns, frame), detections=dets)


def det3d(stamp_ns, boxes, *, frame="cam_optical"):
    """boxes: [((x, y, z), (sx, sy, sz), label, score)] in ``frame``."""
    BB3, V3, P, Pt, Q = T["vision_msgs/msg/BoundingBox3D"], T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/Point"], T["geometry_msgs/msg/Quaternion"]
    dets = [T["vision_msgs/msg/Detection3D"](header=_hdr(stamp_ns, frame), results=[_hyp(l, s)],
                                              bbox=BB3(center=P(position=Pt(x=c[0], y=c[1], z=c[2]), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)), size=V3(x=sz[0], y=sz[1], z=sz[2])), id="")
            for c, sz, l, s in boxes]
    return T["vision_msgs/msg/Detection3DArray"](header=_hdr(stamp_ns, frame), detections=dets)


def frame_stamp(i: int) -> int:
    return S + i * 100_000_000


def frame_log(i: int) -> int:
    return 1_700_000_000 * S + i * 100_000_000


def with_detections(msgs, dets, *, topic="/detections", msgtype=DETECTION_2D, publish_delay_ns=50_000_000):
    """Append detections messages published ``publish_delay_ns`` after their frame's log time."""
    out = list(msgs)
    for i, msg in dets:
        out.append((topic, msgtype, frame_log(i) + publish_delay_ns, msg))
    out.sort(key=lambda m: m[2])
    return out


def _frames(path, **kw):
    st = BagStats()
    frames = list(iter_bag_frames(path, stats=st, **kw))
    return frames, st


# ───────────────────────────── discovery + join ─────────────────────────────

def test_detections_discovered_joined_and_counted(tmp_path):
    n = 6
    dets = [(i, det2d(frame_stamp(i), [(4, 3, 4, 2, "cup", 0.9), (2, 2, 2, 2, "box", None)])) for i in (0, 1, 2, 4, 5)]
    msgs = with_detections(simple_bag_messages(n), dets)
    msgs.append(("/detections", DETECTION_2D, frame_log(n) + 10 * S, det2d(S + 10 * S, [(4, 3, 2, 2, "ghost", 0.5)])))   # matches no frame
    bag = write_bag(tmp_path / "b", msgs)
    probe = probe_bag(bag)
    assert probe.refusal is None and probe.detections_topic == "/detections" and probe.detections_msgtype == DETECTION_2D
    assert probe.topics["rules"]["detections"].startswith("vision_msgs")
    frames, st = _frames(bag)
    assert [fr.header.seq for fr in frames] == list(range(n)) and st.yielded == n
    assert st.detections_paired == 5 and st.detections_unpaired == 1 and st.frames_without_detections == 1 and st.detections_errors == 0
    assert st.detections_scoring == {"mixed": 5} and st.detections_dropped == {}
    for i, fr in enumerate(frames):
        d = fr.header.extra.get("detections")
        if i == 3:
            assert d is None
            continue
        assert d.count == 2 and d.labels == ["cup", "box"] and d.t_sensor_ns == frame_stamp(i) and d.source == "/detections"
        assert d.boxes_xyxy.tolist() == [[2.0, 2.0, 6.0, 4.0], [1.0, 1.0, 3.0, 3.0]] and fr.header.extra["detections_dt_ms"] == 0.0
        assert d.scores[0] == pytest.approx(0.9) and np.isnan(d.scores[1]) and d.scoring == "mixed"
        assert fr.header.extra["rgb_hw"] == (6, 8)


def test_without_a_detections_topic_nothing_changes(tmp_path):
    bag = write_bag(tmp_path / "b", simple_bag_messages(4))
    probe = probe_bag(bag)
    assert probe.detections_topic is None and probe.topics["detections"] is None
    frames, st = _frames(bag)
    assert len(frames) == 4 and st.detections_paired == 0 and st.frames_without_detections == 0
    assert all("detections" not in fr.header.extra for fr in frames)


def test_override_and_missing_override_topic(tmp_path):
    dets = [(i, det2d(frame_stamp(i), [(4, 3, 4, 2, "cup", 0.9)])) for i in range(3)]
    msgs = with_detections(simple_bag_messages(3), dets, topic="/my/dets")
    bag = write_bag(tmp_path / "b", msgs)
    frames, st = _frames(bag, topics={"detections": "/my/dets"})
    assert st.detections_topic == "/my/dets" and st.topics["rules"]["detections"] == "override" and st.detections_paired == 3
    with pytest.raises(BagRefusal):
        _frames(bag, topics={"detections": "/nope"})
    assert probe_bag(bag, topics={"detections": "/nope"}).refusal[0][0] == "no_detections_topic"


def test_stamps_matching_no_frame_leave_every_frame_without_detections(tmp_path):
    dets = [(i, det2d(frame_stamp(i) + 500_000_000, [(4, 3, 4, 2, "cup", 0.9)])) for i in range(4)]   # 0.5 s off every frame
    bag = write_bag(tmp_path / "b", with_detections(simple_bag_messages(4), dets))
    frames, st = _frames(bag, pair_tolerance_s=0.02)
    assert len(frames) == 4 and st.frames_without_detections == 4 and st.detections_unpaired == 4 and st.detections_paired == 0


def test_nearest_within_tolerance_wins_and_publish_delay_does_not_matter(tmp_path):
    # detections stamped 8 ms after the image (within the 20 ms tolerance), published 400 ms later
    dets = [(i, det2d(frame_stamp(i) + 8_000_000, [(4, 3, 4, 2, "cup", 0.9)])) for i in range(4)]
    bag = write_bag(tmp_path / "b", with_detections(simple_bag_messages(4), dets, publish_delay_ns=400_000_000))
    frames, st = _frames(bag)
    assert st.detections_paired == 4 and [fr.header.extra["detections_dt_ms"] for fr in frames] == [8.0] * 4
    assert [fr.header.seq for fr in frames] == [0, 1, 2, 3]                       # order preserved through the wait


def test_max_frames_counts_released_frames(tmp_path):
    dets = [(i, det2d(frame_stamp(i), [(4, 3, 4, 2, "cup", 0.9)])) for i in range(6)]
    bag = write_bag(tmp_path / "b", with_detections(simple_bag_messages(6), dets, publish_delay_ns=250_000_000))
    frames, st = _frames(bag, max_frames=3)
    assert len(frames) == 3 and st.yielded == 3 and all(fr.header.extra.get("detections") is not None for fr in frames)


# ───────────────────────────── 3-D ─────────────────────────────

def test_detection3d_through_the_bag_tf_chain_and_intrinsics(tmp_path):
    # a 0.2 m cube 2 m ahead of the camera, once in the camera frame and once in the base frame (+0.05 m x ahead of cam_optical)
    dets = [(0, det3d(frame_stamp(0), [((0.0, 0.0, 2.0), (0.2, 0.2, 0.2), "cube", 0.7)], frame="cam_optical")),
            (1, det3d(frame_stamp(1), [((0.05, 0.0, 2.0), (0.2, 0.2, 0.2), "cube", 0.7)], frame="base"))]
    bag = write_bag(tmp_path / "b", with_detections(simple_bag_messages(2), dets, topic="/dets3d", msgtype=DETECTION_3D))
    frames, st = _frames(bag)
    assert st.detections_msgtype == DETECTION_3D and st.detections_paired == 2 and st.detections_dropped == {}
    fx, cx, cy = 5.0, 4.0, 3.0                                                    # the synthetic CameraInfo at 8x6
    want = [cx - fx * 0.1 / 1.9, cy - fx * 0.1 / 1.9, cx + fx * 0.1 / 1.9, cy + fx * 0.1 / 1.9]
    for fr in frames:
        d = fr.header.extra["detections"]
        assert d.count == 1 and d.labels == ["cube"] and d.boxes_xyxy[0].tolist() == pytest.approx(want, abs=1e-5)


def test_2d_topic_preferred_over_3d_when_both_exist(tmp_path):
    msgs = with_detections(simple_bag_messages(2), [(0, det2d(frame_stamp(0), [(4, 3, 4, 2, "cup", 0.9)]))], topic="/d2")
    msgs = with_detections(msgs, [(0, det3d(frame_stamp(0), [((0.0, 0.0, 2.0), (0.2, 0.2, 0.2), "cube", 0.7)]))], topic="/d3", msgtype=DETECTION_3D)
    bag = write_bag(tmp_path / "b", msgs)
    assert probe_bag(bag).detections_topic == "/d2"
    assert probe_bag(bag, topics={"detections": "/d3"}).detections_msgtype == DETECTION_3D


# ───────────────────────────── the source + the packet ─────────────────────────────

def test_bag_source_carries_detections_on_the_packet(tmp_path):
    dets = [(i, det2d(frame_stamp(i), [(4, 3, 4, 2, "cup", 0.9)])) for i in (0, 2, 3)]
    bag = write_bag(tmp_path / "b", with_detections(simple_bag_messages(4), dets))
    q = IngestQueue(64)
    src = make_source("bag", {"io": {"bag": {}}}, _ctx(q, keyframe_every_n=2, nonkf_min_interval_s=0.0), path=str(bag))
    src.start(); assert src.wait(30)
    st = src.stats()
    assert st["yielded"] == 4 and st["detections_paired"] == 3 and st["frames_without_detections"] == 1
    pkts = []
    while True:
        p = q.get(timeout=0.0)
        if p is None:
            break
        pkts.append(p)
    assert [p.time.seq for p in pkts] == [0, 1, 2, 3]
    assert pkts[1].detections is None and all(pkts[i].detections is not None and pkts[i].detections.count == 1 for i in (0, 2, 3))
    assert pkts[0].detections.labels == ["cup"] and pkts[0].detections.boxes_xyxy.tolist() == [[2.0, 2.0, 6.0, 4.0]]
