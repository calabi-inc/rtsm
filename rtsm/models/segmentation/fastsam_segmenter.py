"""
FastSAM segmentation backend.

Wraps the existing FastSAM implementation to conform to SegmentationAdapter interface.
"""
from __future__ import annotations
from typing import List, Optional
from PIL import Image
import torch
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult
from rtsm.models.fastsam.model import FastSAM
from rtsm.models.fastsam.prompt import FastSAMPrompt

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
    ):
        """
        Args:
            model_path: Path to FastSAM weights (e.g., FastSAM-x.pt)
            device: "cuda" or "cpu"
            imgsz: Input image size for inference
            conf: Confidence threshold
            iou: IoU threshold for NMS
        """
        self.model = FastSAM(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        logger.info(f"FastSAMSegmenter initialized: device={device}, imgsz={imgsz}")

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
        # Run FastSAM inference
        everything_results = self.model(
            image,
            device=self.device,
            retina_masks=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        # Process results to get masks
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        ann = prompt_process.everything_prompt()

        # Convert to standard format [N, H, W] bool tensor
        masks = self._convert_annotations(ann, image.size)

        # Extract bounding boxes from masks
        boxes = self._masks_to_boxes(masks) if masks.numel() > 0 else None

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=None,  # FastSAM doesn't provide per-mask scores after everything_prompt
            labels=None,  # No classification
            embeddings=None,  # Requires separate CLIP encoding
        )

    def _convert_annotations(
        self, ann: any, image_size: tuple
    ) -> torch.Tensor:
        """Convert FastSAM output to [N, H, W] bool tensor."""
        W, H = image_size

        if ann is None:
            return torch.empty(0, H, W, dtype=torch.bool)

        # Handle different annotation formats from FastSAM
        if isinstance(ann, torch.Tensor):
            if ann.ndim == 3:
                return ann.bool()
            elif ann.ndim == 2:
                return ann.unsqueeze(0).bool()

        if hasattr(ann, 'shape'):
            # numpy array
            import numpy as np
            if isinstance(ann, np.ndarray):
                if ann.ndim == 3:
                    return torch.from_numpy(ann).bool()
                elif ann.ndim == 2:
                    return torch.from_numpy(ann).unsqueeze(0).bool()

        # List of masks
        if isinstance(ann, (list, tuple)) and len(ann) > 0:
            masks = []
            for m in ann:
                if isinstance(m, torch.Tensor):
                    masks.append(m.bool())
                else:
                    masks.append(torch.from_numpy(m).bool())
            return torch.stack(masks, dim=0)

        return torch.empty(0, H, W, dtype=torch.bool)

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
        """Release model resources."""
        if hasattr(self.model, 'model'):
            del self.model.model
        del self.model
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
