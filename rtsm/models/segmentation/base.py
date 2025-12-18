"""
Abstract base class for segmentation/detection models.

This abstraction allows RTSM to swap between different backends:
- FastSAM (open-world segmentation)
- YOLO-World (open-vocabulary detection)
- Future: 3D-aware segmentation, Grounding DINO + SAM, etc.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import torch
import numpy as np


@dataclass
class SegmentationResult:
    """
    Unified output format for all segmentation/detection models.

    All fields except `masks` are optional — different models provide different data.
    The pipeline adapts based on what's available.
    """
    # Core outputs (at least one of masks or boxes should be present)
    masks: Optional[torch.Tensor] = None        # [N, H, W] bool — instance masks
    boxes: Optional[torch.Tensor] = None        # [N, 4] float — xyxy format

    # Confidence and classification
    scores: Optional[torch.Tensor] = None       # [N] float — detection confidence
    labels: Optional[List[str]] = None          # [N] str — class names (open-vocab models)
    class_ids: Optional[torch.Tensor] = None    # [N] int — class indices into vocabulary

    # Embeddings (if model provides, can skip CLIP)
    embeddings: Optional[torch.Tensor] = None   # [N, D] float — visual embeddings

    # Metadata
    vocab: Optional[List[str]] = None           # vocabulary used for detection

    @property
    def count(self) -> int:
        """Number of detected instances."""
        if self.masks is not None:
            return self.masks.shape[0]
        if self.boxes is not None:
            return self.boxes.shape[0]
        return 0

    @property
    def has_masks(self) -> bool:
        return self.masks is not None and self.masks.numel() > 0

    @property
    def has_boxes(self) -> bool:
        return self.boxes is not None and self.boxes.numel() > 0

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None and self.embeddings.numel() > 0


class SegmentationAdapter(ABC):
    """
    Abstract base class for segmentation/detection models.

    Implementations:
    - FastSAMSegmenter: Open-world "segment everything"
    - YOLOWorldSegmenter: Open-vocabulary detection with optional masks
    - Future: GroundingDINOSegmenter, SAM2Segmenter, etc.
    """

    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Segment/detect objects in the image.

        Args:
            image: PIL RGB image
            vocab: Optional list of class names for open-vocabulary models.
                   Ignored by models that don't support text prompts.

        Returns:
            SegmentationResult with detected instances
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release model resources (GPU memory, etc.)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for logging/config."""
        pass

    @property
    def supports_vocab(self) -> bool:
        """Does this model support open-vocabulary text prompts?"""
        return False

    @property
    def provides_embeddings(self) -> bool:
        """Does this model provide visual embeddings (can skip CLIP)?"""
        return False

    @property
    def provides_masks(self) -> bool:
        """Does this model provide instance masks (vs boxes only)?"""
        return True

    # Convenience method for backward compatibility
    def segment_everything(self, image: Image.Image) -> torch.Tensor:
        """
        Legacy interface: segment all objects, return masks only.

        Returns:
            Boolean mask tensor [N, H, W]
        """
        result = self.segment(image, vocab=None)
        if result.masks is not None:
            return result.masks
        return torch.empty(0, image.height, image.width, dtype=torch.bool)
