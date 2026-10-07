"""
Dual-confirmation segmentation: FastSAM + YOLOE with IoU merge.

Runs both models sequentially on each frame, then merges masks by IoU:
  - Dual-confirmed (IoU >= threshold): FastSAM mask + YOLOE label + priority boost
  - FastSAM-only (unmatched):          FastSAM mask, no detection label, falls back to CLIP
  - YOLOE-only (unmatched):            YOLOE mask + label, lower priority (no FastSAM confirmation)
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from PIL import Image
import torch
import numpy as np
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)


def _pairwise_iou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of masks.

    Args:
        masks_a: [Na, H, W] bool
        masks_b: [Nb, H, W] bool

    Returns:
        [Na, Nb] float IoU matrix
    """
    Na, H, W = masks_a.shape
    Nb = masks_b.shape[0]
    if Na == 0 or Nb == 0:
        return torch.zeros(Na, Nb)

    # Flatten to [N, H*W] float for matrix multiply
    a = masks_a.reshape(Na, -1).float()  # [Na, HW]
    b = masks_b.reshape(Nb, -1).float()  # [Nb, HW]

    intersection = a @ b.T  # [Na, Nb]
    area_a = a.sum(dim=1, keepdim=True)  # [Na, 1]
    area_b = b.sum(dim=1, keepdim=True)  # [Nb, 1]
    union = area_a + area_b.T - intersection  # [Na, Nb]

    return intersection / (union + 1e-6)


def _resize_masks_nearest(masks: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize masks to target resolution via nearest-neighbor."""
    if masks.shape[0] == 0:
        return torch.empty(0, target_h, target_w, dtype=torch.bool)
    # F.interpolate expects [N, 1, H, W] float
    resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(),
        size=(target_h, target_w),
        mode="nearest",
    )
    return resized.squeeze(1).bool()


class DualConfirmationSegmenter(SegmentationAdapter):
    """
    Composite segmenter that runs FastSAM and YOLOE in sequence,
    then merges their masks via IoU-based dual confirmation.

    Dual-confirmed masks get YOLOE's detection label attached.
    All masks are returned in a single SegmentationResult with
    confirmation_source metadata for downstream priority boosting.
    """

    def __init__(
        self,
        fastsam: SegmentationAdapter,
        yoloe: SegmentationAdapter,
        iou_confirm_threshold: float = 0.40,
        prefer_mask: str = "fastsam",
    ):
        """
        Args:
            fastsam: FastSAM segmentation adapter (class-agnostic)
            yoloe: YOLOE segmentation adapter (open-vocabulary)
            iou_confirm_threshold: IoU above which a pair is considered dual-confirmed
            prefer_mask: Which mask to keep for dual-confirmed pairs ("fastsam" or "yoloe")
        """
        self.fastsam = fastsam
        self.yoloe = yoloe
        self.iou_threshold = iou_confirm_threshold
        self.prefer_mask = prefer_mask

        logger.info(
            f"DualConfirmationSegmenter initialized: "
            f"iou_threshold={iou_confirm_threshold}, prefer_mask={prefer_mask}"
        )

    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Run both FastSAM and YOLOE, merge by IoU.

        Returns merged SegmentationResult with confirmation_source metadata.
        """
        # Run both models sequentially
        result_f = self.fastsam.segment(image, vocab=None)  # FastSAM ignores vocab
        result_y = self.yoloe.segment(image, vocab=vocab)

        masks_f = result_f.masks if result_f.has_masks else torch.empty(0, image.height, image.width, dtype=torch.bool)
        masks_y = result_y.masks if result_y.has_masks else torch.empty(0, image.height, image.width, dtype=torch.bool)

        Nf = masks_f.shape[0]
        Ny = masks_y.shape[0]

        logger.debug(f"dual: fastsam={Nf} masks, yoloe={Ny} masks")

        # Handle empty cases
        if Nf == 0 and Ny == 0:
            return self._empty_result(image.size)
        if Nf == 0:
            return self._yoloe_only_result(result_y)
        if Ny == 0:
            return self._fastsam_only_result(result_f)

        # Normalize resolutions: resize YOLOE masks to match FastSAM if they differ
        Hf, Wf = masks_f.shape[1], masks_f.shape[2]
        Hy, Wy = masks_y.shape[1], masks_y.shape[2]
        if Hy != Hf or Wy != Wf:
            logger.debug(f"dual: resizing YOLOE masks {Wy}x{Hy} -> {Wf}x{Hf}")
            masks_y = _resize_masks_nearest(masks_y, Hf, Wf)

        # Compute pairwise IoU
        iou_matrix = _pairwise_iou(masks_f, masks_y)  # [Nf, Ny]

        # Greedy bipartite matching: for each YOLOE mask, find best FastSAM match
        matched_f = set()
        matched_y = set()
        matches = []  # (f_idx, y_idx, iou)

        # Sort all pairs by IoU descending for greedy matching
        if iou_matrix.numel() > 0:
            flat = iou_matrix.flatten()
            sorted_indices = flat.argsort(descending=True)
            for flat_idx in sorted_indices:
                fi = int(flat_idx) // Ny
                yi = int(flat_idx) % Ny
                iou_val = float(flat[flat_idx])
                if iou_val < self.iou_threshold:
                    break  # All remaining are below threshold
                if fi in matched_f or yi in matched_y:
                    continue
                matches.append((fi, yi, iou_val))
                matched_f.add(fi)
                matched_y.add(yi)

        # Build merged output
        merged_masks = []
        merged_labels = []       # detection_labels
        merged_conf = []         # label_confidence
        merged_source = []       # confirmation_source
        merged_scores = []       # detection scores (YOLOE conf or 0)
        merged_boxes = []

        # 1) Dual-confirmed pairs
        for fi, yi, iou_val in matches:
            mask = masks_f[fi] if self.prefer_mask == "fastsam" else masks_y[yi]
            merged_masks.append(mask)
            merged_labels.append(result_y.labels[yi] if result_y.labels else None)
            merged_conf.append(float(result_y.scores[yi]) if result_y.scores is not None else 0.0)
            merged_source.append("dual")
            merged_scores.append(float(result_y.scores[yi]) if result_y.scores is not None else 0.0)
            # Use box from preferred mask source
            if self.prefer_mask == "fastsam" and result_f.boxes is not None and fi < result_f.boxes.shape[0]:
                merged_boxes.append(result_f.boxes[fi])
            elif result_y.boxes is not None and yi < result_y.boxes.shape[0]:
                merged_boxes.append(result_y.boxes[yi])
            else:
                merged_boxes.append(None)
            logger.debug(
                f"dual: confirmed f[{fi}]<->y[{yi}] iou={iou_val:.2f} "
                f"label={result_y.labels[yi] if result_y.labels else '?'}"
            )

        # 2) FastSAM-only (unmatched)
        for fi in range(Nf):
            if fi in matched_f:
                continue
            merged_masks.append(masks_f[fi])
            merged_labels.append(None)
            merged_conf.append(0.0)
            merged_source.append("fastsam_only")
            merged_scores.append(0.0)
            if result_f.boxes is not None and fi < result_f.boxes.shape[0]:
                merged_boxes.append(result_f.boxes[fi])
            else:
                merged_boxes.append(None)

        # 3) YOLOE-only (unmatched)
        for yi in range(Ny):
            if yi in matched_y:
                continue
            merged_masks.append(masks_y[yi])
            merged_labels.append(result_y.labels[yi] if result_y.labels else None)
            merged_conf.append(float(result_y.scores[yi]) if result_y.scores is not None else 0.0)
            merged_source.append("yoloe_only")
            merged_scores.append(float(result_y.scores[yi]) if result_y.scores is not None else 0.0)
            if result_y.boxes is not None and yi < result_y.boxes.shape[0]:
                merged_boxes.append(result_y.boxes[yi])
            else:
                merged_boxes.append(None)

        # Stack into tensors
        N = len(merged_masks)
        if N == 0:
            return self._empty_result(image.size)

        masks_tensor = torch.stack(merged_masks, dim=0)  # [N, H, W] bool
        scores_tensor = torch.tensor(merged_scores, dtype=torch.float32)

        # Build boxes tensor (handle missing boxes)
        boxes_tensor = None
        if any(b is not None for b in merged_boxes):
            box_list = []
            for b in merged_boxes:
                if b is not None:
                    t = b if isinstance(b, torch.Tensor) else torch.tensor(b)
                    box_list.append(t.cpu())
                else:
                    box_list.append(torch.zeros(4))
            boxes_tensor = torch.stack(box_list, dim=0)

        n_dual = sum(1 for s in merged_source if s == "dual")
        n_fsam = sum(1 for s in merged_source if s == "fastsam_only")
        n_yoloe = sum(1 for s in merged_source if s == "yoloe_only")
        logger.info(f"dual merge: {n_dual} dual, {n_fsam} fastsam_only, {n_yoloe} yoloe_only")

        return SegmentationResult(
            masks=masks_tensor,
            boxes=boxes_tensor,
            scores=scores_tensor,
            labels=merged_labels,
            vocab=vocab,
            confirmation_source=merged_source,
            detection_labels=merged_labels,
            label_confidence=merged_conf,
            fastsam_raw_count=Nf,
            yoloe_raw_count=Ny,
            label_source=result_y.label_source,
        )

    def warmup(self) -> None:
        """Eagerly load both sub-models."""
        self.fastsam.warmup()
        self.yoloe.warmup()

    def _empty_result(self, image_size: tuple) -> SegmentationResult:
        W, H = image_size
        return SegmentationResult(
            masks=torch.empty(0, H, W, dtype=torch.bool),
            boxes=torch.empty(0, 4),
            scores=torch.empty(0),
            labels=[],
            confirmation_source=[],
            detection_labels=[],
            label_confidence=[],
        )

    def _fastsam_only_result(self, result_f: SegmentationResult) -> SegmentationResult:
        """Wrap FastSAM-only result with confirmation metadata."""
        N = result_f.count
        return SegmentationResult(
            masks=result_f.masks,
            boxes=result_f.boxes,
            scores=result_f.scores,
            labels=result_f.labels,
            vocab=result_f.vocab,
            confirmation_source=["fastsam_only"] * N,
            detection_labels=[None] * N,
            label_confidence=[0.0] * N,
        )

    def _yoloe_only_result(self, result_y: SegmentationResult) -> SegmentationResult:
        """Wrap YOLOE-only result with confirmation metadata."""
        N = result_y.count
        return SegmentationResult(
            masks=result_y.masks,
            boxes=result_y.boxes,
            scores=result_y.scores,
            labels=result_y.labels,
            vocab=result_y.vocab,
            confirmation_source=["yoloe_only"] * N,
            detection_labels=result_y.labels,
            label_confidence=[float(result_y.scores[i]) if result_y.scores is not None else 0.0 for i in range(N)],
            label_source=result_y.label_source,
        )

    def close(self) -> None:
        self.fastsam.close()
        self.yoloe.close()

    @property
    def name(self) -> str:
        return "dual"

    @property
    def supports_vocab(self) -> bool:
        return self.yoloe.supports_vocab

    @property
    def provides_embeddings(self) -> bool:
        return False

    @property
    def provides_masks(self) -> bool:
        return True
