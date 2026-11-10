"""
FastSAM segmentation backend.

Uses pip-installed ultralytics FastSAM (v8.4+). No bundled fork needed.
"""
from __future__ import annotations
from typing import List, Optional
from PIL import Image
import torch
import numpy as np
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)


class FastSAMSegmenter(SegmentationAdapter):
    """
    Open-world segmentation using FastSAM.

    Segments "everything" in the image without requiring a vocabulary.
    Does not provide embeddings — CLIP encoding is done separately in the pipeline.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.4,
        iou: float = 0.9,
        retina_masks: bool = True,
        max_det: int = 300,
    ):
        self._model = None
        self._model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks
        self.max_det = max_det
        logger.info(f"FastSAMSegmenter initialized: device={device}, imgsz={imgsz}")

    def _load_model(self):
        """Lazy load FastSAM from pip-installed ultralytics."""
        if self._model is not None:
            return
        from ultralytics import FastSAM
        self._model = FastSAM(self._model_path)
        logger.info(f"FastSAM model loaded: {self._model_path}")

    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Segment all objects in the image.

        Args:
            image: PIL RGB image
            vocab: Ignored — FastSAM doesn't use text prompts

        Returns:
            SegmentationResult with masks (no labels, no embeddings)
        """
        self._load_model()

        # retina_masks=True ensures masks are at the original image resolution,
        # which is required for correct centroid backprojection with per-frame
        # intrinsics (especially when RGB is 1920x1440 from ARKit).
        results = self._model(
            image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )

        if not results or len(results) == 0:
            return self._empty_result(image.size)

        result = results[0]

        # Extract masks directly from results (modern ultralytics, no FastSAMPrompt needed)
        masks = None
        row_counts = None
        if hasattr(result, "masks") and result.masks is not None:
            mask_data = result.masks.data
            if isinstance(mask_data, torch.Tensor):
                mask_bool = mask_data.bool()
                if mask_bool.is_cuda and mask_bool.ndim == 3:
                    # Mask pixels per row, counted while the stack is still on the GPU: the mask statistics
                    # start from this count and would otherwise make it with a pass over the stack on the CPU.
                    row_counts = mask_bool.sum(dim=2).cpu().numpy().astype(np.int64, copy=False)
                masks = mask_bool.cpu()
            else:
                masks = torch.from_numpy(np.asarray(mask_data)).bool()

        if masks is None or masks.numel() == 0:
            return self._empty_result(image.size)

        # Extract boxes if available, otherwise derive from masks
        boxes = None
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes)
            boxes = boxes.cpu()
        else:
            boxes = self._masks_to_boxes(masks)

        # Extract scores if available
        scores = None
        if hasattr(result, "boxes") and result.boxes is not None and result.boxes.conf is not None:
            scores = result.boxes.conf
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores)
            scores = scores.cpu()

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=None,
            embeddings=None,
            mask_row_counts=row_counts,
        )

    def warmup(self) -> None:
        """Eagerly load FastSAM model weights."""
        self._load_model()

    def _empty_result(self, image_size: tuple) -> SegmentationResult:
        W, H = image_size
        return SegmentationResult(
            masks=torch.empty(0, H, W, dtype=torch.bool),
            boxes=torch.empty(0, 4),
            scores=torch.empty(0),
        )

    def _masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert masks [N, H, W] to bounding boxes [N, 4] in xyxy format."""
        if masks.numel() == 0:
            return torch.empty(0, 4)

        N = masks.shape[0]
        boxes = []
        for i in range(N):
            mask = masks[i]
            if not mask.any():
                boxes.append([0, 0, 0, 0])
                continue
            rows = torch.where(mask.any(dim=1))[0]
            cols = torch.where(mask.any(dim=0))[0]
            y0, y1 = rows[0].item(), rows[-1].item() + 1
            x0, x1 = cols[0].item(), cols[-1].item() + 1
            boxes.append([x0, y0, x1, y1])
        return torch.tensor(boxes, dtype=torch.float32)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "fastsam"

    @property
    def supports_vocab(self) -> bool:
        return False

    @property
    def provides_embeddings(self) -> bool:
        return False

    @property
    def provides_masks(self) -> bool:
        return True
