"""The backend label merge (``rtsm.core.pipeline.apply_detection_labels``) and the dual segmenter's ``label_source``.

A label drawn from the supplied vocabulary (grounded phrases, prompted classes) goes first in the scored list with
its raw confidence. A label from a model's own built-in vocabulary (prompt-free YOLOE) stays out of the scored list,
so the CLIP vocabulary classifier's labels decide the primary label, and is kept as ``detector_label`` for the ledger.
"""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from rtsm.core.pipeline import apply_detection_labels
from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult
from rtsm.models.segmentation.dual_segmenter import DualConfirmationSegmenter


def _cand(idx, topk):
    return SimpleNamespace(stats=SimpleNamespace(idx=idx), label_topk=list(topk))


def _seg(labels, conf, source):
    return SimpleNamespace(detection_labels=labels, label_confidence=conf, labels=None, scores=None, label_source=source)


def test_vocab_label_goes_first_with_its_raw_confidence():
    c = _cand(0, [("box", 0.21), ("shelf", 0.19)])
    apply_detection_labels([c], _seg(["tissue box"], [0.62], "vocab"), unscored_prior=1.0)
    assert c.label_topk[0] == ("tissue box", 0.62) and c.label_topk[1:] == [("box", 0.21), ("shelf", 0.19)]
    assert c.detector_label == "tissue box" and c.detector_score == 0.62 and c.label_unscored is None


def test_builtin_label_stays_out_of_the_scored_list():
    c = _cand(0, [("shelf", 0.23), ("box", 0.20)])
    apply_detection_labels([c], _seg(["heat"], [0.71], "builtin"), unscored_prior=1.0)
    assert c.label_topk == [("shelf", 0.23), ("box", 0.20)]            # the vocabulary classifier's labels stand
    assert c.detector_label == "heat" and c.detector_score == 0.71     # kept for the observation ledger
    assert getattr(c, "label_unscored", None) is None


def test_unknown_source_behaves_as_vocab():
    c = _cand(0, [("cup", 0.3)])
    apply_detection_labels([c], _seg(["mug"], [0.5], None), unscored_prior=1.0)
    assert c.label_topk[0] == ("mug", 0.5)


def test_base_labels_are_used_when_detection_labels_are_absent():
    c = _cand(0, [("cup", 0.3)])
    seg = SimpleNamespace(detection_labels=None, label_confidence=None, labels=["heat"], scores=torch.tensor([0.9]), label_source="builtin")
    apply_detection_labels([c], seg, unscored_prior=1.0)
    assert c.label_topk == [("cup", 0.3)] and c.detector_label == "heat" and c.detector_score == pytest.approx(0.9)
    seg2 = SimpleNamespace(detection_labels=None, label_confidence=None, labels=["mug"], scores=torch.tensor([0.9]), label_source="vocab")
    apply_detection_labels([c], seg2, unscored_prior=1.0)
    assert c.label_topk[0][0] == "mug" and c.label_topk[0][1] == pytest.approx(0.9)


def test_unscored_label_takes_the_prior_and_is_flagged():
    c = _cand(0, [("cup", 0.3)])
    apply_detection_labels([c], _seg(["mug"], [float("nan")], "vocab"), unscored_prior=0.8)
    assert c.label_topk[0] == ("mug", 0.8) and c.label_unscored == "mug" and c.detector_score is None


def test_missing_or_empty_detection_label_leaves_the_candidate_alone():
    c = _cand(1, [("cup", 0.3)])
    apply_detection_labels([c], _seg([None, None], [0.0, 0.0], "vocab"), unscored_prior=1.0)
    assert c.label_topk == [("cup", 0.3)] and getattr(c, "detector_label", None) is None
    apply_detection_labels([c], None, unscored_prior=1.0)
    assert c.label_topk == [("cup", 0.3)]


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


def test_dual_segmenter_propagates_the_yoloe_label_source():
    img = Image.new("RGB", (16, 16))
    f = SegmentationResult(masks=_masks((0, 0, 8, 8), (10, 10, 16, 16)),
                           boxes=torch.tensor([[0, 0, 8, 8], [10, 10, 16, 16]], dtype=torch.float32), scores=torch.tensor([1.0, 1.0]))
    for source in ("builtin", "vocab"):
        y = SegmentationResult(masks=_masks((0, 0, 8, 8)), boxes=torch.tensor([[0, 0, 8, 8]], dtype=torch.float32),
                               scores=torch.tensor([0.7]), labels=["heat"], label_source=source)
        r = DualConfirmationSegmenter(_Fixed(f), _Fixed(y), iou_confirm_threshold=0.4, prefer_mask="fastsam").segment(img)
        assert r.label_source == source
        assert r.confirmation_source == ["dual", "fastsam_only"] and r.detection_labels == ["heat", None]
        only_y = DualConfirmationSegmenter(_Fixed(SegmentationResult(masks=_masks(hw=(16, 16)))), _Fixed(y)).segment(img)
        assert only_y.label_source == source and only_y.confirmation_source == ["yoloe_only"]
    # the result's default: unspecified
    assert SegmentationResult(masks=_masks((0, 0, 4, 4))).label_source is None
