"""
YOLOE segmentation/detection backend.

Open-vocabulary detection + instance segmentation in a single pass using YOLOE.
Supports both prompted (set_classes) and prompt-free (-pf) model variants.

YOLOE26 provides:
  - Instance masks at native resolution
  - Open-vocabulary labels from text prompts or 1200+ built-in categories (PF)
  - Detection confidence scores
  - Single-pass inference (no separate SAM head needed)
"""
from __future__ import annotations
from typing import List, Optional
from PIL import Image
import torch
import numpy as np
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)


class YOLOESegmenter(SegmentationAdapter):
    """
    Open-vocabulary segmentation using YOLOE / YOLOE26.

    Model variants:
    - yoloe-26s-seg.pt   — small, fast iteration (~8-12ms on consumer GPU)
    - yoloe-26l-seg.pt   — large, best mask quality (36.8% LVIS mAP, ~2-3ms on RTX 5090)
    - yoloe-26s-seg-pf.pt — prompt-free (1200+ built-in categories, no vocab setup)

    Vocabulary can be:
    - Set at initialization (default_vocab)
    - Overridden per-call via segment(vocab=...)
    - Skipped entirely for prompt-free (-pf) models
    """

    def __init__(
        self,
        model_path: str = "model_store/yolo/yoloe-26s-seg.pt",
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        default_vocab: Optional[List[str]] = None,
        retina_masks: bool = True,
        max_det: int = 300,
    ):
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks
        self.max_det = max_det
        self._model_path = model_path
        self._is_prompt_free = "-pf" in model_path.lower()
        self.default_vocab = None if self._is_prompt_free else (default_vocab or self._default_indoor_vocab())
        self._current_vocab: Optional[List[str]] = None

        self._model = None

        logger.info(
            f"YOLOESegmenter initialized: model={model_path}, device={device}, "
            f"imgsz={imgsz}, prompt_free={self._is_prompt_free}"
        )

    @staticmethod
    def _default_indoor_vocab() -> List[str]:
        """Default vocabulary for indoor/warehouse scenarios."""
        return [
            # Furniture
            "chair", "table", "desk", "couch", "sofa", "bed", "shelf", "cabinet",
            # Electronics
            "monitor", "laptop", "keyboard", "mouse", "phone", "tv", "speaker",
            # Containers
            "box", "cardboard box", "bin", "basket", "crate", "pallet",
            # Common objects
            "bottle", "cup", "mug", "book", "bag", "backpack", "plant", "lamp",
            # People/vehicles
            "person", "forklift",
        ]

    def _load_model(self):
        """Lazy load the YOLOE model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLOE
            self._model = YOLOE(self._model_path)
            self._model.to(self.device)
            logger.info(f"YOLOE model loaded: {self._model_path}")
        except ImportError:
            raise ImportError(
                "YOLOE requires ultralytics >= 8.3. "
                "Install with: pip install -U ultralytics"
            )

    def _set_vocab(self, vocab: List[str]) -> None:
        """Set the detection vocabulary (only if changed). No-op for prompt-free models."""
        if self._is_prompt_free:
            return
        if self._current_vocab == vocab:
            return
        self._model.set_classes(vocab)
        self._current_vocab = vocab
        logger.debug(f"YOLOE vocab set: {len(vocab)} classes")

    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Detect and segment objects in the image.

        Args:
            image: PIL RGB image
            vocab: List of class names to detect. If None, uses default_vocab.
                   Ignored for prompt-free models.

        Returns:
            SegmentationResult with masks, boxes, labels, and scores
        """
        self._load_model()

        # Set vocabulary (no-op for prompt-free)
        if not self._is_prompt_free:
            vocab = vocab or self.default_vocab
            self._set_vocab(vocab)

        results = self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=self.retina_masks,
            max_det=self.max_det,
            verbose=False,
        )

        if not results or len(results) == 0:
            return self._empty_result(image.size)

        result = results[0]

        # Extract detections
        boxes = result.boxes.xyxy if result.boxes is not None else None
        scores = result.boxes.conf if result.boxes is not None else None
        class_ids = result.boxes.cls if result.boxes is not None else None

        # Map class IDs to labels
        labels = None
        if class_ids is not None:
            if self._is_prompt_free:
                # Prompt-free: use model's built-in names dict
                names = result.names or {}
                labels = [names.get(int(cid), f"class_{int(cid)}") for cid in class_ids]
            else:
                labels = [vocab[int(cid)] for cid in class_ids]

        # Extract masks (YOLOE seg models always provide masks)
        masks = None
        if hasattr(result, "masks") and result.masks is not None:
            mask_data = result.masks.data
            if isinstance(mask_data, torch.Tensor):
                masks = mask_data.bool().cpu()
            else:
                masks = torch.from_numpy(np.asarray(mask_data)).bool()

        # Ensure CPU tensors (consistent with FastSAM output)
        if boxes is not None:
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes)
            boxes = boxes.cpu()
        if scores is not None:
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores)
            scores = scores.cpu()
        if class_ids is not None:
            if not isinstance(class_ids, torch.Tensor):
                class_ids = torch.tensor(class_ids, dtype=torch.int64)
            class_ids = class_ids.cpu()

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_ids=class_ids,
            embeddings=None,
            vocab=vocab if not self._is_prompt_free else None,
            label_source=("builtin" if self._is_prompt_free else "vocab"),
        )

    def warmup(self) -> None:
        """Eagerly load YOLOE model and set up vocabulary (downloads MobileCLIP if needed)."""
        self._load_model()
        if not self._is_prompt_free and self.default_vocab:
            self._set_vocab(self.default_vocab)

    def _empty_result(self, image_size: tuple) -> SegmentationResult:
        """Return empty result for no detections."""
        W, H = image_size
        return SegmentationResult(
            masks=torch.empty(0, H, W, dtype=torch.bool),
            boxes=torch.empty(0, 4),
            scores=torch.empty(0),
            labels=[],
            class_ids=torch.empty(0, dtype=torch.int64),
        )

    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "yoloe"

    @property
    def supports_vocab(self) -> bool:
        return not self._is_prompt_free

    @property
    def provides_embeddings(self) -> bool:
        return False

    @property
    def provides_masks(self) -> bool:
        return True
