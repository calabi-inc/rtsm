"""
The ``external`` segmentation backend: the frame's own detections become the
``SegmentationResult`` the pipeline consumes (P3, the detections adapter).

No model runs here. The pipeline hands the packet's ``Detections`` (see
``rtsm/io/detections.py``) and the frame's depth to ``segment``; the backend
turns the boxes into instance masks (the depth band inside each box, or the
box itself), passes the labels and the scores through -- ``scores=None`` stays
None, a detection without a score keeps ``None`` in ``label_confidence`` --
and marks every candidate ``confirmation_source = "external"``. CLIP
embeddings are computed by the pipeline from the crops exactly as for our own
detectors, so association, the memory, the ledgers and the report run
unchanged on the customer's detections.

Config (``segmentation.external``):
  mask_from: depth_band | box      how masks are made from boxes (default depth_band)
  depth_band_m: 0.20               +- metres around the box's median depth
  min_box_px: 64                   boxes with a smaller area are dropped (counted)
  class_names: {}                  optional id -> name map (ROS 1 int64 class ids)
  unscored_label_prior: 1.0        the label score stored for an unscored detection (pipeline label merge)
  refine: none                     reserved for a mask refiner (SAM box prompts); only ``none`` exists
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from rtsm.io.detections import SOURCE_EXTERNAL, Detections, boxes_to_masks
from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)

MASK_MODES = ("depth_band", "box")
REFINERS = ("none",)


class ExternalDetectionsSegmenter(SegmentationAdapter):
    consumes_detections = True

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = dict(cfg or {})
        self.mask_from = str(c.get("mask_from", "depth_band")).lower()
        if self.mask_from not in MASK_MODES:
            raise ValueError(f"segmentation.external.mask_from must be one of {MASK_MODES}, got {self.mask_from!r}")
        self.depth_band_m = float(c.get("depth_band_m", 0.20))
        self.min_box_px = float(c.get("min_box_px", 64))
        self.class_names = {str(k): str(v) for k, v in (c.get("class_names") or {}).items()}
        self.unscored_label_prior = float(c.get("unscored_label_prior", 1.0))
        refine = str(c.get("refine", "none")).lower()
        if refine not in REFINERS:
            raise ValueError(f"segmentation.external.refine: only {REFINERS} exist, got {refine!r}")
        self.refine = refine
        self.frames = 0
        self.frames_without = 0
        self.mask_how = {"depth_band": 0, "box": 0}
        self.dropped_small = 0

    # ── SegmentationAdapter ──
    def segment(self, image: Image.Image, vocab: Optional[List[str]] = None, *, detections: Optional[Detections] = None,
                depth_m: Optional[np.ndarray] = None) -> SegmentationResult:
        H, W = int(image.height), int(image.width)
        self.frames += 1
        if detections is None or detections.count == 0:
            self.frames_without += 1
            return self._empty(H, W)
        boxes = np.asarray(detections.boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = np.where(area >= self.min_box_px)[0]
        self.dropped_small += int(boxes.shape[0] - keep.size)
        if keep.size == 0:
            self.frames_without += 1
            return self._empty(H, W)
        boxes = boxes[keep]
        if detections.masks is not None and detections.masks.shape[1:] == (H, W):
            masks = np.asarray(detections.masks, dtype=bool)[keep]
            self.mask_how["box"] += 0
        else:
            masks, how = boxes_to_masks(boxes, (H, W), depth_m, band_m=self.depth_band_m, mask_from=self.mask_from)
            for k, v in how.items():
                self.mask_how[k] = self.mask_how.get(k, 0) + v
        labels = [self._name(detections.labels[i]) for i in keep]
        if detections.scores is None:
            conf: List[Optional[float]] = [None] * len(keep)
            scores_t: Optional[torch.Tensor] = None
        else:
            raw = np.asarray(detections.scores, dtype=np.float32)[keep]
            conf = [(float(s) if np.isfinite(s) else None) for s in raw]
            scores_t = torch.from_numpy(raw) if np.all(np.isfinite(raw)) else None   # a mixed message is unscored for the monitor
        return SegmentationResult(
            masks=torch.from_numpy(masks), boxes=torch.from_numpy(boxes.copy()), scores=scores_t,
            labels=[l if l is not None else "" for l in labels], detection_labels=labels, label_confidence=conf,
            confirmation_source=[SOURCE_EXTERNAL] * len(keep), vocab=None,
        )

    def _name(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        return self.class_names.get(str(label), str(label))

    @staticmethod
    def _empty(H: int, W: int) -> SegmentationResult:
        return SegmentationResult(masks=torch.zeros((0, H, W), dtype=torch.bool), boxes=torch.zeros((0, 4), dtype=torch.float32),
                                  scores=None, labels=[], detection_labels=[], label_confidence=[], confirmation_source=[], vocab=None)

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "external"

    @property
    def supports_vocab(self) -> bool:
        return False

    @property
    def provides_embeddings(self) -> bool:
        return False

    @property
    def provides_masks(self) -> bool:
        return True

    def stats(self) -> Dict[str, Any]:
        return {"frames": self.frames, "frames_without_detections": self.frames_without, "masks": dict(self.mask_how),
                "dropped_small_boxes": self.dropped_small, "mask_from": self.mask_from, "depth_band_m": self.depth_band_m}
