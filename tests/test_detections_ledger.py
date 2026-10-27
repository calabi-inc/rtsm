"""Raw detections (ledger kind det, B5): the backend's output and a paired detections topic are written per processed frame
before the filter, with the index every other ledger line uses; the live ROS 2 source pairs a detections topic to its frames."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from rtsm.core.datamodel import FramePacket, IngestMeta, PoseStamped, TimeBundle
from rtsm.core.pipeline import Pipeline, _mask_boxes, backend_detections, topic_detections
from rtsm.evaluation.event_log import DET_BACKEND, DET_TOPIC, KIND_DET, LEDGER_KINDS, LEDGER_SCHEMA, DetectionsEvent
from rtsm.io.detections import DETECTION_2D, Detections
from rtsm.io.ingest_queue import IngestQueue


def test_constants_and_schema():
    assert KIND_DET == "det" and KIND_DET in LEDGER_KINDS and LEDGER_SCHEMA == 5 and (DET_BACKEND, DET_TOPIC) == ("backend", "topic")
    e = DetectionsEvent(timestamp=1.0, frame_seq=1, t_sensor_ns=5, source="backend", name="dual", n=0, rgb_hw=[6, 8], detections=[])
    assert e.kind == "det" and e.scoring is None and e.msgtype is None


def test_backend_rows_from_boxes_labels_and_confirmation():
    seg = NS(boxes=torch.tensor([[1.2, 2.0, 5.6, 6.0], [0.0, 0.0, 0.0, 0.0]]), masks=None, labels=["cup", "box"], scores=torch.tensor([0.9, float("nan")]),
             detection_labels=None, label_confidence=None, confirmation_source=["dual", "fastsam_only"])
    rows = backend_detections(seg, [6, 8])
    assert rows[0] == {"idx": 0, "label": "cup", "score": pytest.approx(0.9), "bbox_rgb": [1, 2, 6, 6], "confirmation": "dual"}
    assert rows[1]["bbox_rgb"] is None and rows[1]["score"] is None and rows[1]["label"] == "box"    # a degenerate box with no mask to fall back on
    assert backend_detections(None, None) == [] and backend_detections(NS(boxes=None, masks=None, labels=None, scores=None, detection_labels=None, label_confidence=None), None) == []


def test_backend_rows_fall_back_to_the_masks_tight_box_scaled_to_rgb():
    m = torch.zeros(2, 3, 4, dtype=torch.bool)
    m[0, 1:3, 1:3] = True                                                  # rows 1-2, cols 1-2 of a 3x4 mask grid
    seg = NS(boxes=torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 3.0, 3.0]]), masks=m, labels=None, scores=None, detection_labels=["a", "b"],
             label_confidence=[0.5, None], confirmation_source=None)
    rows = backend_detections(seg, [6, 8])                                 # RGB is 2x the mask grid
    assert rows[0]["bbox_rgb"] == [2, 2, 6, 6] and rows[0]["label"] == "a" and rows[0]["score"] == 0.5 and "confirmation" not in rows[0]
    assert rows[1]["bbox_rgb"] == [2, 2, 3, 3] and rows[1]["score"] is None   # a real box stays the backend's
    assert _mask_boxes(m, [6, 8]) == [[2, 2, 6, 6], None] and _mask_boxes(np.zeros((2, 2), dtype=bool), None) == []
    only_masks = NS(boxes=None, masks=m, labels=None, scores=None, detection_labels=None, label_confidence=None, confirmation_source=None)
    assert [r["bbox_rgb"] for r in backend_detections(only_masks, None)] == [[1, 1, 3, 3], None]


def test_topic_rows_keep_unscored_as_none():
    det = Detections(boxes_xyxy=np.array([[1.4, 2.0, 5.5, 6.0], [0.0, 0.0, 3.0, 3.0]], dtype=np.float32), scores=np.array([0.7, float("nan")], dtype=np.float32),
                     labels=["cup", None], hypotheses=[[("cup", 0.7), ("mug", 0.2)], []], ids=["t1", None], source="/dets", msgtype=DETECTION_2D, t_sensor_ns=5)
    rows = topic_detections(det)
    assert rows == [{"idx": 0, "label": "cup", "score": pytest.approx(0.7), "bbox_rgb": [1, 2, 6, 6], "id": "t1", "n_hypotheses": 2},
                    {"idx": 1, "label": None, "score": None, "bbox_rgb": [0, 0, 3, 3], "id": None, "n_hypotheses": 0}]
    assert topic_detections(Detections.empty()) == []


def _packet(seq=7, ts_ns=123_000, detections=None):
    return FramePacket(rgb=np.zeros((6, 8, 3), dtype=np.uint8), depth_m=np.ones((6, 8), dtype=np.float32),
                       pose=PoseStamped(stamp_ns=ts_ns, frame_id="arkit", t_wc=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                        q_wc_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
                       intr=None, is_keyframe=True, frame_epoch=4,
                       time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=ts_ns, seq=seq),
                       ingest=IngestMeta(lane="keyframe", keyframe_origin="minted", rx_seq=9), detections=detections)


@pytest.mark.parametrize("ledgers", [True, False])
def test_pipeline_writes_a_backend_line_and_a_topic_line(tmp_path: Path, ledgers: bool):
    from rtsm.evaluation.event_log import EventLogWriter
    w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"), ledgers=ledgers)
    pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None, associator=None,
                    ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w)
    seg = NS(boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0]]), masks=None, labels=["cup"], scores=torch.tensor([0.8]), detection_labels=None, label_confidence=None, confirmation_source=None)
    det = Detections(boxes_xyxy=np.array([[1.0, 1.0, 4.0, 5.0]], dtype=np.float32), scores=None, labels=["cup"], hypotheses=[[("cup", None)]],
                     source="/dets", msgtype=DETECTION_2D, t_sensor_ns=126_000, n_dropped={"degenerate": 1})
    snap = NS(rgb=np.zeros((6, 8, 3), dtype=np.uint8))
    pipe._write_det_lines(_packet(detections=det), snap, seg, "grounded_sam2")
    pipe._write_det_lines(_packet(seq=8, ts_ns=200_000), snap, seg, "grounded_sam2")       # a frame without a topic message
    w.close()
    rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    det_rows = [r for r in rows if r["kind"] == "det"]
    if not ledgers:
        assert det_rows == []
        return
    assert [(r["source"], r["frame_seq"], r["n"]) for r in det_rows] == [("backend", 7, 1), ("topic", 7, 1), ("backend", 8, 1)]
    b, t = det_rows[0], det_rows[1]
    assert b["name"] == "grounded_sam2" and b["rgb_hw"] == [6, 8] and b["t_sensor_ns"] == 123_000 and b["scoring"] == "present"
    assert b["detections"] == [{"idx": 0, "label": "cup", "score": pytest.approx(0.8), "bbox_rgb": [1, 1, 4, 4]}]
    assert t["name"] == "/dets" and t["msgtype"] == DETECTION_2D and t["dt_ms"] == pytest.approx(0.003) and t["n_dropped"] == {"degenerate": 1} and t["scoring"] == "absent"
    assert t["detections"] == [{"idx": 0, "label": "cup", "score": None, "bbox_rgb": [1, 1, 4, 5], "id": None, "n_hypotheses": 1}]


# ───────────────────────── the live ROS 2 detections role ─────────────────────────

def _live_dets(t_s: float, boxes, frame: str = "cam_optical"):
    """A vision_msgs/Detection2DArray in the ROS 2 layout, as rclpy would hand it over. boxes: [(cx, cy, w, h, label, score)]."""
    sec = int(t_s)
    hdr = NS(stamp=NS(sec=sec, nanosec=int(round((t_s - sec) * 1e9))), frame_id=frame)
    dets = [NS(header=hdr, results=[NS(hypothesis=NS(class_id=l, score=float(s)), pose=None)],
               bbox=NS(center=NS(position=NS(x=float(cx), y=float(cy)), theta=0.0), size_x=float(w), size_y=float(h)), id="")
            for cx, cy, w, h, l, s in boxes]
    return NS(header=hdr, detections=dets)


def test_live_source_pairs_a_detections_topic_to_its_frames():
    from test_ros2_source import _FakeFE, _depth, _info, _rgb, _tf
    from rtsm.io.bag_reader import BagStats, TopicMap
    from rtsm.io.ros2_source import Ros2Ingress
    fe = _FakeFE()
    tm = TopicMap(rgb="/rgb", depth="/depth", rgb_info="/info", tf="/tf", detections="/dets")
    types_ = {"rgb": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image", "rgb_info": "sensor_msgs/msg/CameraInfo", "tf": "tf2_msgs/msg/TFMessage",
              "detections": DETECTION_2D}
    ing = Ros2Ingress(fe, BagStats(), tm, types_, session_id="t", detections_wait_s=10.0)
    ing.on_message("rgb_info", _info(1.0), 1)
    ing.on_message("tf", _tf(0.90, "world", "cam_optical", 0.0), 1)
    ing.on_message("tf", _tf(1.90, "world", "cam_optical", 0.5), 1)
    for t in (1.00, 1.10):
        ing.on_message("rgb", _rgb(t), 1)
        ing.on_message("depth", _depth(t + 0.003), 1)
    ing.on_message("depth", _depth(1.30), 1)                                # resolves both pairs
    ing.on_message("detections", _live_dets(1.004, [(4, 3, 2, 2, "cup", 0.9)]), 1)   # the first frame's, 4 ms after the image
    assert ing.step() == 1 and ing.ready                                     # the first frame goes out with its detections; the second waits
    assert len(fe.raws) == 1 and fe.raws[0].header.extra["detections"].count == 1 and fe.raws[0].header.extra["detections"].labels == ["cup"]
    assert fe.raws[0].header.extra["detections_dt_ms"] == pytest.approx(4.0) and ing.stats.detections_paired == 1
    ing.on_message("detections", _live_dets(1.30, [(1, 1, 1, 1, "box", 0.5)]), 1)   # a later message releases the second frame without one
    assert ing.step() == 1 and len(fe.raws) == 2 and "detections" not in fe.raws[1].header.extra and ing.stats.frames_without_detections == 1
    ing.on_message("detections", _live_dets(0.0, []), 1)                       # a zero stamp is counted, never paired
    ing.step(final=True)
    assert ing.stats.detections_unpaired >= 2 and ing.as_dict()["detections_buffered"] == 0
    # the wait: with no later message and no timeout the frame is held; with a zero wait it goes out at once
    fe2 = _FakeFE()
    ing2 = Ros2Ingress(fe2, BagStats(), tm, types_, session_id="t", detections_wait_s=0.0)
    ing2.on_message("rgb_info", _info(1.0), 1)
    ing2.on_message("tf", _tf(0.90, "world", "cam_optical", 0.0), 1)
    ing2.on_message("tf", _tf(1.90, "world", "cam_optical", 0.5), 1)
    ing2.on_message("rgb", _rgb(1.0), 1)
    ing2.on_message("depth", _depth(1.003), 1)
    ing2.on_message("depth", _depth(1.30), 1)
    assert ing2.step() == 1 and "detections" not in fe2.raws[0].header.extra and ing2.stats.frames_without_detections == 1


def test_live_source_refuses_an_unknown_detections_type():
    from test_ros2_source import _FakeFE
    from rtsm.io.bag_reader import BagRefusal, BagStats, TopicMap
    from rtsm.io.ros2_source import _LIVE_ROLES, Ros2Ingress, choose_qos
    assert "detections" in _LIVE_ROLES and choose_qos("detections", [], "auto").depth == 100
    tm = TopicMap(rgb="/rgb", depth="/depth", rgb_info="/info", tf="/tf", detections="/dets")
    types_ = {"rgb": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image", "rgb_info": "sensor_msgs/msg/CameraInfo", "tf": "tf2_msgs/msg/TFMessage",
              "detections": "my_msgs/msg/Boxes"}
    with pytest.raises(BagRefusal, match="unsupported_detections_type|no detections adapter"):
        Ros2Ingress(_FakeFE(), BagStats(), tm, types_, session_id="t")
