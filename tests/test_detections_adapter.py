"""
The detections contract, the vision_msgs adapters, the depth-band masks and
the ``external`` segmentation backend (P3, the detections adapter). CPU only:
messages are built with rosbags' typestores from the bundled definitions,
no bag is written here (tests/test_bag_detections.py does that).

Contract under test:
  * ROS 2 and ROS 1 Detection2DArray layouts convert to the same boxes / labels;
    scores are None when the detector reports none, NaN per unscored detection
    in a mixed message; rotated boxes become their bounding box; ROS 1
    source_img rescales the boxes to the paired RGB; degenerate / outside
    boxes are dropped and counted;
  * Detection3DArray boxes project through TF + the intrinsics to the 2-D box
    of their corners; an unresolved frame or a box behind the camera is
    counted, never guessed;
  * depth-band masks keep the object and drop the wall behind it, at depth
    resolution under a larger RGB; the box is the fallback;
  * the external backend passes scores=None through, keeps None per unscored
    label, drops tiny boxes, and returns an empty result for a frame without
    detections.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")
from rosbags.typesys import Stores, get_typestore

from rtsm.io import detections as D
from rtsm.io.msgdefs import DETECTION_2D, DETECTION_3D, register_vision_msgs, vision_msgs_types

TS2 = get_typestore(Stores.ROS2_HUMBLE)
register_vision_msgs(TS2, "ros2")
T2 = TS2.types
TS1 = get_typestore(Stores.ROS1_NOETIC)
register_vision_msgs(TS1, "ros1")
T1 = TS1.types
S = 1_000_000_000


# ───────────────────────────── builders ─────────────────────────────

def _hdr2(ns: int, frame: str = "cam"):
    return T2["std_msgs/msg/Header"](stamp=T2["builtin_interfaces/msg/Time"](sec=ns // S, nanosec=ns % S), frame_id=frame)


def _pose2(x=0.0, y=0.0, z=0.0, q=(0.0, 0.0, 0.0, 1.0)):
    P, Pt, Q = T2["geometry_msgs/msg/Pose"], T2["geometry_msgs/msg/Point"], T2["geometry_msgs/msg/Quaternion"]
    return P(position=Pt(x=x, y=y, z=z), orientation=Q(x=q[0], y=q[1], z=q[2], w=q[3]))


def _hyp2(label: str, score):
    OHP, OH, PWC = T2["vision_msgs/msg/ObjectHypothesisWithPose"], T2["vision_msgs/msg/ObjectHypothesis"], T2["geometry_msgs/msg/PoseWithCovariance"]
    return OHP(hypothesis=OH(class_id=label, score=(float(score) if score is not None else float("nan"))), pose=PWC(pose=_pose2(), covariance=np.zeros(36)))


def det2d(cx, cy, w, h, hyps, *, theta=0.0, did="", ns=S):
    BB, P2, PT = T2["vision_msgs/msg/BoundingBox2D"], T2["vision_msgs/msg/Pose2D"], T2["vision_msgs/msg/Point2D"]
    return T2["vision_msgs/msg/Detection2D"](header=_hdr2(ns), results=[_hyp2(l, s) for l, s in hyps],
                                              bbox=BB(center=P2(position=PT(x=float(cx), y=float(cy)), theta=float(theta)), size_x=float(w), size_y=float(h)), id=did)


def arr2d(dets, ns=S, frame="cam"):
    return T2["vision_msgs/msg/Detection2DArray"](header=_hdr2(ns, frame), detections=list(dets))


def det3d(center, size, hyps, *, q=(0.0, 0.0, 0.0, 1.0), ns=S, frame="cam"):
    BB3, V3 = T2["vision_msgs/msg/BoundingBox3D"], T2["geometry_msgs/msg/Vector3"]
    return T2["vision_msgs/msg/Detection3D"](header=_hdr2(ns, frame), results=[_hyp2(l, s) for l, s in hyps],
                                              bbox=BB3(center=_pose2(*center, q=q), size=V3(x=size[0], y=size[1], z=size[2])), id="")


def arr3d(dets, ns=S, frame="cam"):
    return T2["vision_msgs/msg/Detection3DArray"](header=_hdr2(ns, frame), detections=list(dets))


def _hdr1(ns: int, frame: str = "cam"):
    return T1["std_msgs/msg/Header"](seq=0, stamp=T1["builtin_interfaces/msg/Time"](sec=ns // S, nanosec=ns % S), frame_id=frame)


def det2d_ros1(cx, cy, w, h, hyps, *, theta=0.0, img_wh=(0, 0), ns=S):
    OHP, PWC, P, Pt, Q = (T1["vision_msgs/msg/ObjectHypothesisWithPose"], T1["geometry_msgs/msg/PoseWithCovariance"],
                          T1["geometry_msgs/msg/Pose"], T1["geometry_msgs/msg/Point"], T1["geometry_msgs/msg/Quaternion"])
    pose = PWC(pose=P(position=Pt(x=0.0, y=0.0, z=0.0), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)), covariance=np.zeros(36))
    results = [OHP(id=int(i), score=float(s), pose=pose) for i, s in hyps]
    BB, P2D = T1["vision_msgs/msg/BoundingBox2D"], T1["geometry_msgs/msg/Pose2D"]
    img = T1["sensor_msgs/msg/Image"](header=_hdr1(ns), height=img_wh[1], width=img_wh[0], encoding="rgb8", is_bigendian=0, step=0, data=np.zeros(0, dtype=np.uint8))
    return T1["vision_msgs/msg/Detection2D"](header=_hdr1(ns), results=results, bbox=BB(center=P2D(x=float(cx), y=float(cy), theta=float(theta)), size_x=float(w), size_y=float(h)), source_img=img)


def arr2d_ros1(dets, ns=S):
    return T1["vision_msgs/msg/Detection2DArray"](header=_hdr1(ns), detections=list(dets))


class K:
    fx, fy, cx, cy = 100.0, 100.0, 80.0, 60.0


# ───────────────────────────── definitions + registry ─────────────────────────────

def test_bundled_definitions_register_once_and_round_trip():
    assert len(vision_msgs_types("ros2")) == 10 and len(vision_msgs_types("ros1")) == 7
    ts = get_typestore(Stores.ROS2_HUMBLE)
    added = register_vision_msgs(ts, "ros2")
    assert DETECTION_2D in added and DETECTION_3D in added and register_vision_msgs(ts, "ros2") == []
    msg = arr2d([det2d(100, 50, 40, 20, [("cup", 0.9)])])
    raw = TS2.serialize_cdr(msg, DETECTION_2D)
    back = TS2.deserialize_cdr(raw, DETECTION_2D)
    assert back.detections[0].results[0].hypothesis.class_id == "cup"
    with pytest.raises(ValueError):
        vision_msgs_types("ros3")


def test_registration_brings_the_standard_dependencies_a_bag_typestore_lacks():
    """A rosbag2 with embedded definitions yields a typestore that knows only its own topics' types;
    registering vision_msgs there must also bring geometry_msgs/PoseWithCovariance and friends
    (found by the G3-5 gate on session1_bag: serialising a Detection2DArray failed on the missing type)."""
    bare = get_typestore(Stores.EMPTY)
    added = register_vision_msgs(bare, "ros2")
    assert "geometry_msgs/msg/PoseWithCovariance" in added and "std_msgs/msg/Header" in added and DETECTION_2D in added
    raw = bare.serialize_cdr(bare.types[DETECTION_2D](header=bare.types["std_msgs/msg/Header"](stamp=bare.types["builtin_interfaces/msg/Time"](sec=1, nanosec=0), frame_id="c"),
                                                      detections=[]), DETECTION_2D)
    assert bare.deserialize_cdr(raw, DETECTION_2D).header.frame_id == "c"
    assert register_vision_msgs(bare, "ros2") == []
    bare1 = get_typestore(Stores.EMPTY)
    assert "geometry_msgs/msg/PoseWithCovariance" in register_vision_msgs(bare1, "ros1") and DETECTION_3D in bare1.types


def test_registry_knows_both_types():
    assert D.adapter_for(DETECTION_2D).name == "vision_msgs_2d" and D.adapter_for(DETECTION_3D).name == "vision_msgs_3d"
    assert D.adapter_for("vision_msgs/msg/Classification") is None
    assert set(D.detection_msgtypes()) >= {DETECTION_2D, DETECTION_3D}


# ───────────────────────────── 2-D ─────────────────────────────

def test_ros2_detection2d_boxes_labels_scores_and_ids():
    msg = arr2d([det2d(100, 50, 40, 20, [("cup", 0.9), ("mug", 0.4)], did="t7"),
                 det2d(20, 20, 10, 10, [("box", 0.3)])], ns=S + 7)
    det = D.adapter_for(DETECTION_2D).convert(msg, DETECTION_2D, rgb_hw=(120, 160))
    assert det.count == 2 and det.scoring == "present"
    assert det.boxes_xyxy.tolist() == [[80.0, 40.0, 120.0, 60.0], [15.0, 15.0, 25.0, 25.0]]
    assert det.labels == ["cup", "box"] and det.hypotheses[0] == [("cup", 0.9), ("mug", 0.4)]
    assert det.scores.tolist() == pytest.approx([0.9, 0.3]) and det.ids == ["t7", None]
    assert det.t_sensor_ns == S + 7 and det.frame_id == "cam" and det.msgtype == DETECTION_2D and det.n_dropped == {}


def test_scores_none_when_the_detector_reports_none_and_nan_when_mixed():
    unscored = arr2d([det2d(100, 50, 40, 20, [("cup", None)]), det2d(20, 20, 10, 10, [("box", None)])])
    det = D.adapter_for(DETECTION_2D).convert(unscored, DETECTION_2D, rgb_hw=(120, 160))
    assert det.scores is None and det.scoring == "absent" and det.labels == ["cup", "box"] and det.hypotheses[0] == [("cup", None)]
    mixed = arr2d([det2d(100, 50, 40, 20, [("cup", 0.8)]), det2d(20, 20, 10, 10, [("box", None)])])
    det = D.adapter_for(DETECTION_2D).convert(mixed, DETECTION_2D, rgb_hw=(120, 160))
    assert det.scoring == "mixed" and det.n_scored == 1 and det.scores[0] == pytest.approx(0.8) and math.isnan(det.scores[1])
    # a scored hypothesis ranks above an unscored one whatever the message order
    ranked = arr2d([det2d(100, 50, 40, 20, [("maybe", None), ("sure", 0.2)])])
    det = D.adapter_for(DETECTION_2D).convert(ranked, DETECTION_2D, rgb_hw=(120, 160))
    assert det.labels == ["sure"] and det.hypotheses[0] == [("sure", 0.2), ("maybe", None)]
    # no hypothesis at all: a box without a label
    bare = arr2d([det2d(100, 50, 40, 20, [])])
    det = D.adapter_for(DETECTION_2D).convert(bare, DETECTION_2D, rgb_hw=(120, 160))
    assert det.count == 1 and det.labels == [None] and det.scores is None


def test_rotated_clipped_degenerate_and_outside_boxes():
    msg = arr2d([det2d(80, 60, 40, 20, [("a", 0.5)], theta=math.pi / 2),      # rotated 90 deg: 20 wide, 40 tall
                 det2d(150, 60, 40, 20, [("b", 0.5)]),                          # clipped at the right edge (160)
                 det2d(10, 10, 0, 20, [("c", 0.5)]),                            # degenerate
                 det2d(400, 400, 10, 10, [("d", 0.5)])])                        # outside
    det = D.adapter_for(DETECTION_2D).convert(msg, DETECTION_2D, rgb_hw=(120, 160))
    assert det.count == 2
    assert det.boxes_xyxy[0].tolist() == pytest.approx([70.0, 40.0, 90.0, 80.0], abs=1e-6)
    assert det.boxes_xyxy[1].tolist() == [130.0, 50.0, 160.0, 70.0]
    assert det.n_dropped == {D.DROP_DEGENERATE: 1, D.DROP_OUTSIDE: 1}


def test_ros1_layout_int_ids_and_source_img_rescale():
    msg = arr2d_ros1([det2d_ros1(100, 50, 40, 20, [(3, 0.7), (5, 0.1)], img_wh=(320, 240))], ns=S + 3)
    det = D.adapter_for(DETECTION_2D).convert(msg, DETECTION_2D, rgb_hw=(120, 160))     # the RGB is half the source image
    assert det.count == 1 and det.labels == ["3"] and det.hypotheses[0] == [("3", 0.7), ("5", 0.1)]
    assert det.boxes_xyxy[0].tolist() == [40.0, 20.0, 60.0, 30.0] and det.ids is None and det.t_sensor_ns == S + 3
    # the explicit image_hw option does the same for a ROS 2 message
    msg2 = arr2d([det2d(100, 50, 40, 20, [("cup", 0.9)])])
    det2 = D.adapter_for(DETECTION_2D).convert(msg2, DETECTION_2D, rgb_hw=(120, 160), image_hw=(240, 320))
    assert det2.boxes_xyxy[0].tolist() == [40.0, 20.0, 60.0, 30.0]


# ───────────────────────────── 3-D ─────────────────────────────

def test_detection3d_projects_through_tf_and_intrinsics():
    # a 0.2 m cube 2 m ahead in the camera frame: corners at x in +-0.1, y in +-0.1, z in 1.9..2.1
    msg = arr3d([det3d((0.0, 0.0, 2.0), (0.2, 0.2, 0.2), [("cube", 0.6)])])
    ident = lambda fid, t: np.eye(4)
    det = D.adapter_for(DETECTION_3D).convert(msg, DETECTION_3D, rgb_hw=(120, 160), intrinsics=K, tf_lookup=ident)
    x0, y0, x1, y1 = det.boxes_xyxy[0]
    assert x0 == pytest.approx(80 - 100 * 0.1 / 1.9) and x1 == pytest.approx(80 + 100 * 0.1 / 1.9)
    assert y0 == pytest.approx(60 - 100 * 0.1 / 1.9) and y1 == pytest.approx(60 + 100 * 0.1 / 1.9)
    assert det.labels == ["cube"] and det.scores.tolist() == pytest.approx([0.6])
    # the same box expressed in a frame 1 m to the left of the camera: TF brings it back
    msg_l = arr3d([det3d((-1.0, 0.0, 2.0), (0.2, 0.2, 0.2), [("cube", 0.6)], frame="left")], frame="left")
    T_cam_from_left = np.eye(4); T_cam_from_left[0, 3] = 1.0
    det_l = D.adapter_for(DETECTION_3D).convert(msg_l, DETECTION_3D, rgb_hw=(120, 160), intrinsics=K, tf_lookup=lambda fid, t: T_cam_from_left if fid == "left" else np.eye(4))
    assert det_l.boxes_xyxy[0].tolist() == pytest.approx(det.boxes_xyxy[0].tolist())


def test_detection3d_counts_unresolved_frames_behind_camera_and_missing_intrinsics():
    msg = arr3d([det3d((0.0, 0.0, 2.0), (0.2, 0.2, 0.2), [("a", 0.5)]), det3d((0.0, 0.0, -2.0), (0.2, 0.2, 0.2), [("b", 0.5)])])
    def tf(fid, t):
        raise KeyError(fid)
    det = D.adapter_for(DETECTION_3D).convert(msg, DETECTION_3D, rgb_hw=(120, 160), intrinsics=K, tf_lookup=tf)
    assert det.count == 0 and det.n_dropped == {D.DROP_UNRESOLVED_FRAME: 2}
    det = D.adapter_for(DETECTION_3D).convert(msg, DETECTION_3D, rgb_hw=(120, 160), intrinsics=K, tf_lookup=lambda f, t: np.eye(4))
    assert det.count == 1 and det.labels == ["a"] and det.n_dropped == {D.DROP_BEHIND: 1}
    det = D.adapter_for(DETECTION_3D).convert(msg, DETECTION_3D, rgb_hw=(120, 160), intrinsics=None, tf_lookup=lambda f, t: np.eye(4))
    assert det.count == 0 and det.n_dropped == {D.DROP_NO_INTRINSICS: 2}


# ───────────────────────────── masks ─────────────────────────────

def test_depth_band_mask_keeps_the_object_and_drops_the_wall():
    H, W = 60, 80
    depth = np.full((H, W), 3.0, dtype=np.float32)          # a wall at 3 m
    depth[20:40, 30:50] = 1.0                                # an object at 1 m
    boxes = np.array([[25.0, 15.0, 55.0, 45.0]], dtype=np.float32)   # a loose box around the object
    masks, how = D.boxes_to_masks(boxes, (H, W), depth, band_m=0.2)
    assert how == {"depth_band": 1, "box": 0}
    m = masks[0]
    assert m[20:40, 30:50].all() and not m[16, 27] and m.sum() == 400
    # no depth -> the box; mask_from="box" -> the box
    masks_b, how_b = D.boxes_to_masks(boxes, (H, W), None)
    assert how_b == {"depth_band": 0, "box": 1} and masks_b[0].sum() == 30 * 30
    masks_c, how_c = D.boxes_to_masks(boxes, (H, W), depth, mask_from="box")
    assert how_c["box"] == 1 and masks_c[0].sum() == 30 * 30


def test_depth_band_mask_at_lower_depth_resolution_and_sparse_depth():
    H, W = 120, 160                                          # RGB
    depth = np.full((30, 40), 3.0, dtype=np.float32)        # depth at a quarter of the resolution
    depth[10:20, 15:25] = 1.0                                # object = RGB rows 40..80, cols 60..100
    boxes = np.array([[50.0, 30.0, 110.0, 90.0]], dtype=np.float32)
    masks, how = D.boxes_to_masks(boxes, (H, W), depth, band_m=0.2)
    assert how["depth_band"] == 1 and masks[0][40:80, 60:100].all() and masks[0].sum() == 40 * 40
    # too little valid depth under the box -> the box
    sparse = np.full((30, 40), np.nan, dtype=np.float32); sparse[12, 18] = 1.0
    masks_s, how_s = D.boxes_to_masks(boxes, (H, W), sparse, band_m=0.2)
    assert how_s == {"depth_band": 0, "box": 1} and masks_s[0].sum() == 60 * 60


def test_depth_band_keeps_the_component_nearest_the_median_not_the_largest():
    H, W = 40, 40
    depth = np.full((H, W), 3.0, dtype=np.float32)
    depth[5:35, 5:20] = 1.0           # a 1 m pole on the left of the box (450 px: the median of the box is 1.0)
    depth[15:25, 22:32] = 1.1         # the object at 1.1 m (100 px), separated from the pole by a wall column
    boxes = np.array([[5.0, 5.0, 35.0, 35.0]], dtype=np.float32)
    masks, how = D.boxes_to_masks(boxes, (H, W), depth, band_m=0.2)
    assert how["depth_band"] == 1
    # the band (|d - med| <= 0.2) holds both the pole and the object; the component containing the median pixel wins
    m = masks[0]
    assert m[5:35, 5:20].all() and not m[15:25, 22:32].any()


# ───────────────────────────── the external backend ─────────────────────────────

def test_external_backend_passes_scores_none_through_and_drops_tiny_boxes():
    pytest.importorskip("torch")
    from PIL import Image
    from rtsm.models.segmentation.external import ExternalDetectionsSegmenter
    seg = ExternalDetectionsSegmenter({"min_box_px": 64, "mask_from": "box", "class_names": {"3": "chair"}})
    img = Image.new("RGB", (160, 120))
    det = D.Detections(boxes_xyxy=np.array([[10, 10, 50, 50], [0, 0, 4, 4], [60, 60, 100, 100]], dtype=np.float32), scores=None,
                       labels=["cup", "dot", "3"], hypotheses=[[("cup", None)], [("dot", None)], [("3", None)]])
    r = seg.segment(img, detections=det, depth_m=None)
    assert r.count == 2 and r.scores is None and r.labels == ["cup", "chair"] and r.label_confidence == [None, None]
    assert r.confirmation_source == ["external", "external"] and r.masks.shape == (2, 120, 160) and bool(r.masks[0, 20, 20]) and not bool(r.masks[0, 55, 55])
    assert seg.stats()["dropped_small_boxes"] == 1 and seg.stats()["masks"]["box"] == 2
    # scored + mixed
    det_s = D.Detections(boxes_xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32), scores=np.array([0.9, 0.4], dtype=np.float32),
                         labels=["cup", "box"], hypotheses=[[("cup", 0.9)], [("box", 0.4)]])
    r = seg.segment(img, detections=det_s)
    assert r.scores.tolist() == pytest.approx([0.9, 0.4]) and r.label_confidence == pytest.approx([0.9, 0.4])
    det_m = D.Detections(boxes_xyxy=det_s.boxes_xyxy, scores=np.array([0.9, np.nan], dtype=np.float32), labels=["cup", "box"], hypotheses=[[("cup", 0.9)], [("box", None)]])
    r = seg.segment(img, detections=det_m)
    assert r.scores is None and r.label_confidence[0] == pytest.approx(0.9) and r.label_confidence[1] is None
    # a frame without detections: an empty result, counted
    r0 = seg.segment(img, detections=None)
    assert r0.count == 0 and r0.masks.shape == (0, 120, 160) and seg.stats()["frames_without_detections"] == 1
    assert seg.name == "external" and seg.provides_masks and not seg.provides_embeddings and not seg.supports_vocab
    with pytest.raises(ValueError):
        ExternalDetectionsSegmenter({"mask_from": "sam"})
    with pytest.raises(ValueError):
        ExternalDetectionsSegmenter({"refine": "sam2"})


def test_external_backend_is_the_registered_backend():
    pytest.importorskip("torch")
    from rtsm.models.segmentation import get_segmenter
    seg = get_segmenter({"segmentation": {"backend": "external", "external": {"depth_band_m": 0.1}}})
    assert seg.name == "external" and seg.consumes_detections is True and seg.depth_band_m == 0.1
