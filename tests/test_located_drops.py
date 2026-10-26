"""Located drops (trace schema_version 4, ledger schema 2): every candidate that did not reach association is on the
frame line with its stage, reason and RGB-space box; the obs mask block carries bbox_rgb / mask_hw; a FastSAM-only
mask under dual has no detection confidence."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from rtsm.core.datamodel import FramePacket, IngestMeta, PoseStamped, TimeBundle
from rtsm.core.pipeline import (Pipeline, ScoringTrace, ScoringTraceEntry, _bbox_to_rgb, _mask_stats_dict,
                                apply_detection_labels, detector_label_score, detector_lists)
from rtsm.evaluation.event_log import LEDGER_SCHEMA, SCHEMA_VERSION, EventLogWriter, FrameEvent
from rtsm.io.ingest_queue import IngestQueue
from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult
from rtsm.models.segmentation.dual_segmenter import DualConfirmationSegmenter
from rtsm.utils.mask_staging import FilterDiagnostics, MaskStats, run_heuristics


def _lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _stats(idx, bbox, *, border=0.0, cpx=None):
    x0, y0, x1, y1 = bbox
    return MaskStats(idx=idx, area_px=(x1 - x0) * (y1 - y0), bbox=tuple(bbox), coverage=0.1, border_fraction=border,
                     depth_valid=1.0, depth_p50=2.0, depth_spread=0.01,
                     centroid_px=(cpx if cpx is not None else ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))


def _pipe(tmp_path, cfg=None, ledgers=True):
    w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"), ledgers=ledgers)
    return Pipeline(cfg=cfg or {}, segmenter=None, clip=None, working_mem=None, proximity_index=None, associator=None,
                    ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w), w


def _packet(seq=7, ts_ns=123_000):
    return FramePacket(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=np.ones((8, 8), dtype=np.float32),
                       pose=PoseStamped(stamp_ns=ts_ns, frame_id="arkit", t_wc=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                        q_wc_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
                       intr=None, is_keyframe=True, frame_epoch=4,
                       time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=ts_ns, seq=seq),
                       ingest=IngestMeta(lane="keyframe", keyframe_origin="minted", rx_seq=9))


# ───────────────────────── the mask filter locates its drops ─────────────────────────

def _filter_cfg(track):
    return {"filters": {"min_area_px": 50, "depth": {"z_min_m": 0.2, "z_max_m": 5.0, "sigma_max_m": 0.35}},
            "camera": {"fx": 30.0, "fy": 30.0, "cx": 16.0, "cy": 16.0},
            "staging": {"depth_erode_px": 0, "depth_valid_min": 0.15},
            "diagnostics": {"track_drops": track}}


def _three_masks():
    ann = torch.zeros(3, 32, 32, dtype=torch.bool)
    ann[0, 3:5, 6:8] = True            # 4 px: area drop
    ann[1, 10:20, 2:12] = True         # 100 px, no depth under it: depth_valid drop
    ann[2, 20:30, 20:30] = True        # 100 px, good depth: kept
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    depth[10:20, 2:12] = np.nan
    return ann, depth


def test_filter_drops_carry_their_box_when_tracked():
    ann, depth = _three_masks()
    kept, stats, diag = run_heuristics(ann, depth, cfg=_filter_cfg(True))
    assert len(kept) == 1 and stats[0].idx == 2 and (diag.dropped_area, diag.dropped_depth_valid, diag.passed) == (1, 1, 1)
    by_idx = {d["idx"]: d for d in diag.drop_details}
    assert by_idx[0]["reason"] == "area" and by_idx[0]["bbox"] == [6, 3, 8, 5] and by_idx[0]["area_px"] == 4
    assert by_idx[1]["reason"] == "depth_valid" and by_idx[1]["bbox"] == [2, 10, 12, 20] and by_idx[1]["depth_valid"] == 0.0


def test_filter_counts_only_when_not_tracked():
    ann, depth = _three_masks()
    _, _, diag = run_heuristics(ann, depth, cfg=_filter_cfg(False))
    assert diag.drop_details == [] and diag.dropped_area == 1 and diag.dropped_depth_valid == 1


# ───────────────────────── the selection locates dedup and top-K drops ─────────────────────────

def _select(tmp_path):
    pipe, _ = _pipe(tmp_path, cfg={"staging": {"topk_preclip": 1}})
    # A and B are the same box (B duplicates A; A sorts first on the tie), C is a different box with a border penalty
    stats = [_stats(0, (0, 0, 10, 10)), _stats(1, (0, 0, 10, 10)), _stats(2, (20, 20, 30, 30), border=0.9)]
    masks = [torch.zeros(32, 32, dtype=torch.bool) for _ in stats]
    final = pipe._score_and_select(masks, stats, frame_hw=(32, 32))
    return pipe, final


def test_select_marks_dedup_and_topk_drops(tmp_path):
    pipe, final = _select(tmp_path)
    assert [c.stats.idx for c in final] == [0]
    st = pipe._last_scoring_trace
    e = {x.mask_idx: x for x in st.entries}
    assert e[0].selected_topk and e[0].dropped_by is None and e[0].bbox == (0, 0, 10, 10)
    assert not e[1].selected_topk and e[1].dropped_by == "dedup" and e[1].dup_of == 0
    assert not e[2].selected_topk and e[2].dropped_by == "topk" and e[2].dup_of is None and e[2].bbox == (20, 20, 30, 30)
    assert st.n_candidates == 3 and st.n_selected == 1


def test_dropped_rows_locate_every_lost_candidate(tmp_path):
    pipe, _ = _select(tmp_path)
    fd = FilterDiagnostics(total_input=4, dropped_area=1, passed=3,
                           drop_details=[{"idx": 3, "reason": "area", "bbox": [1, 1, 3, 3], "area_px": 4, "area_min": 50}])
    seg = SimpleNamespace(detection_labels=["cup", None, "box", "tray"], label_confidence=[0.9, None, float("nan"), 0.4], labels=None, scores=None)
    rows = pipe._dropped_rows(fd, pipe._last_scoring_trace, seg, 2.0, 2.0)
    assert [r["idx"] for r in rows] == [1, 2, 3]
    dedup, topk, area = rows
    assert dedup["stage"] == "select" and dedup["reason"] == "dedup" and dedup["dup_of"] == 0 and dedup["bbox"] == [0, 0, 20, 20]
    assert dedup["detector_label"] is None and dedup["detector_score"] is None and isinstance(dedup["priority"], float)
    assert topk["reason"] == "topk" and topk["bbox"] == [40, 40, 60, 60] and topk["detector_label"] == "box" and topk["detector_score"] is None
    assert area["stage"] == "filter" and area["reason"] == "area" and area["bbox"] == [2, 2, 6, 6] and area["priority"] is None
    assert area["detector_label"] == "tray" and area["detector_score"] == 0.4 and area["area_px"] == 4 and area["area_min"] == 50
    # the selected candidate is not a dropped row, and a frame with nothing dropped has none
    assert all(r["idx"] != 0 for r in rows)
    assert pipe._dropped_rows(FilterDiagnostics(), ScoringTrace(n_candidates=1, n_selected=1, entries=[
        ScoringTraceEntry(mask_idx=0, priority=1.0, coverage=0.1, border_fraction=0.0, depth_valid=1.0, depth_spread=0.0,
                          bbox_area_norm=0.1, confirmation_source=None, selected_topk=True)]), None) == []
    # a filter row without a box (an older drop_details entry) keeps bbox None
    old = pipe._dropped_rows(FilterDiagnostics(drop_details=[{"idx": 5, "reason": "depth_spread"}]), None, None)
    assert old == [{"idx": 5, "stage": "filter", "reason": "depth_spread", "bbox": None, "priority": None, "dup_of": None,
                    "detector_label": None, "detector_score": None}]


def test_detector_lists_fall_back_to_labels_and_scores():
    base = SimpleNamespace(detection_labels=None, label_confidence=None, labels=["a", "", "c"], scores=torch.tensor([0.5, 0.6, 0.7]))
    labels, conf = detector_lists(base)
    assert labels == ["a", "", "c"] and conf == [pytest.approx(0.5), pytest.approx(0.6), pytest.approx(0.7)]
    assert detector_label_score(labels, conf, 0) == ("a", pytest.approx(0.5))
    assert detector_label_score(labels, conf, 1) == (None, None)          # an empty label is no label
    assert detector_label_score(labels, conf, 7) == (None, None)          # out of range
    assert detector_label_score(labels, None, 2) == ("c", None)           # no confidences at all
    assert detector_label_score(["x"], ["bad"], 0) == ("x", None)         # unparsable
    assert detector_lists(None) == (None, None)


def test_bbox_to_rgb_is_the_identity_at_scale_one():
    assert _bbox_to_rgb((1, 2, 3, 4)) == [1, 2, 3, 4] and _bbox_to_rgb((1, 2, 3, 4), 1.5, 0.5) == [2, 1, 4, 2]


# ───────────────────────── the frame line and the obs mask block ─────────────────────────

def test_frame_line_serialises_the_located_drops(tmp_path):
    w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"))
    w.write(FrameEvent(timestamp=1.0, frame_seq=3, is_keyframe=True, n_masks_raw=5, t_sensor_ns=30, rgb_hw=[480, 640], mask_hw=[480, 640],
                       dropped=[{"idx": 1, "stage": "select", "reason": "topk", "bbox": [1, 2, 3, 4], "priority": 0.3, "dup_of": None,
                                 "detector_label": "cup", "detector_score": 0.8, "confirmation_source": "dual"}]))
    w.close()
    meta, frame = _lines(w.path)
    assert meta["schema_version"] == SCHEMA_VERSION == 4 and LEDGER_SCHEMA == 4
    assert frame["kind"] == "frame" and frame["rgb_hw"] == [480, 640] and frame["mask_hw"] == [480, 640]
    assert frame["dropped"][0]["reason"] == "topk" and frame["dropped"][0]["bbox"] == [1, 2, 3, 4]
    # an older-style write (no dropped rows) still serialises, with the defaults
    w2 = EventLogWriter(enabled=True, configured_path=str(tmp_path / "f.jsonl"))
    w2.write(FrameEvent(timestamp=1.0, frame_seq=3, is_keyframe=True, n_masks_raw=0))
    w2.close()
    assert _lines(w2.path)[1]["dropped"] == [] and _lines(w2.path)[1]["rgb_hw"] is None


def test_obs_mask_block_has_the_rgb_box_and_the_mask_grid(tmp_path):
    pipe, w = _pipe(tmp_path)
    cand = SimpleNamespace(stats=_stats(3, (1, 2, 3, 4), cpx=(8.0, 8.0)), priority=0.7, label_topk=[("cup", 0.9)])
    recs = [dict(cand=cand, outcome="no_p_cam")]
    pipe._write_obs_lines(_packet(), recs, mask_to_rgb=(2.0, 2.0), mask_hw=(4, 4))
    pipe._write_obs_lines(_packet(seq=8), recs)
    w.close()
    scaled, plain = [r for r in _lines(w.path) if r["kind"] == "obs"]
    assert scaled["mask"]["bbox"] == [1, 2, 3, 4] and scaled["mask"]["bbox_rgb"] == [2, 4, 6, 8] and scaled["mask"]["mask_hw"] == [4, 4]
    assert plain["mask"]["bbox"] == [1, 2, 3, 4] and plain["mask"]["bbox_rgb"] == [1, 2, 3, 4] and plain["mask"]["mask_hw"] is None
    assert scaled["mask"]["centroid_px"] == [8.0, 8.0] and "plane_normal_cam" not in scaled["mask"]
    assert _mask_stats_dict(None) is None


# ───────────────────────── dual: a FastSAM-only mask has no confidence ─────────────────────────

class _Fixed(SegmentationAdapter):
    def __init__(self, result):
        self._r = result

    def segment(self, image, vocab=None):
        return self._r

    def close(self):
        pass

    @property
    def name(self):
        return "fixed"


def _masks(*boxes, hw=(16, 16)):
    out = torch.zeros(len(boxes), *hw, dtype=torch.bool)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        out[k, y0:y1, x0:x1] = True
    return out


def test_dual_fastsam_only_mask_reports_no_confidence():
    img = Image.new("RGB", (16, 16))
    f = SegmentationResult(masks=_masks((0, 0, 8, 8), (10, 10, 16, 16)),
                           boxes=torch.tensor([[0, 0, 8, 8], [10, 10, 16, 16]], dtype=torch.float32), scores=torch.tensor([1.0, 1.0]))
    y = SegmentationResult(masks=_masks((0, 0, 8, 8)), boxes=torch.tensor([[0, 0, 8, 8]], dtype=torch.float32),
                           scores=torch.tensor([0.7]), labels=["cup"], label_source="vocab")
    r = DualConfirmationSegmenter(_Fixed(f), _Fixed(y), iou_confirm_threshold=0.4, prefer_mask="fastsam").segment(img)
    assert r.confirmation_source == ["dual", "fastsam_only"] and r.detection_labels == ["cup", None]
    assert r.label_confidence == [pytest.approx(0.7), None] and r.scores.tolist() == [pytest.approx(0.7), 0.0]
    c0, c1 = (SimpleNamespace(stats=SimpleNamespace(idx=i), label_topk=[("thing", 0.2)]) for i in (0, 1))
    apply_detection_labels([c0, c1], r, unscored_prior=1.0)
    assert (c0.detector_label, c0.detector_score) == ("cup", pytest.approx(0.7)) and c0.label_topk[0] == ("cup", pytest.approx(0.7))
    assert getattr(c1, "detector_label", None) is None and getattr(c1, "detector_score", None) is None and c1.label_topk == [("thing", 0.2)]
    # the located-drop helper reads the same lists
    labels, conf = detector_lists(r)
    assert detector_label_score(labels, conf, 1) == (None, None) and detector_label_score(labels, conf, 0) == ("cup", pytest.approx(0.7))
