"""
Abstract base class for segmentation/detection models.

This abstraction allows RTSM to swap between different backends:
- FastSAM (open-world segmentation)
- YOLOE (open-vocabulary detection + segmentation in one pass)
- DualConfirmation (FastSAM + YOLOE with IoU merge)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
from PIL import Image
import numpy as np

if TYPE_CHECKING:
    import torch


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

    # Dual-confirmation metadata (populated by DualConfirmationSegmenter)
    confirmation_source: Optional[List[str]] = None   # [N] "dual" | "fastsam_only" | "yoloe_only"
    detection_labels: Optional[List[str]] = None      # [N] YOLOE label per mask (None if fastsam_only)
    label_confidence: Optional[List[float]] = None    # [N] YOLOE detection confidence per mask

    # Where the detection labels come from (2026-10-05). "vocab": drawn from the vocabulary rtsm supplied (grounded
    # phrases, prompted classes, an external detector's names). "builtin": the model's own built-in vocabulary
    # (prompt-free YOLOE, 1200+ categories including abstract words). None: unspecified, treated as "vocab".
    # The pipeline keeps a "builtin" label out of the scored label list (see rtsm.core.pipeline.apply_detection_labels).
    label_source: Optional[str] = None

    # Pre-merge raw model output counts (populated by DualConfirmationSegmenter)
    fastsam_raw_count: Optional[int] = None
    yoloe_raw_count: Optional[int] = None

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
    - YOLOESegmenter: Open-vocabulary detection + segmentation
    - DualConfirmationSegmenter: FastSAM + YOLOE with IoU merge
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

    def warmup(self) -> None:
        """Eagerly load model weights and download any missing assets.

        Called at startup before the pipeline begins processing frames.
        Default implementation is a no-op; subclasses with lazy loading
        should override to trigger their ``_load_model()`` path.
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
        import torch

        result = self.segment(image, vocab=None)
        if result.masks is not None:
            return result.masks
        return torch.empty(0, image.height, image.width, dtype=torch.bool)
