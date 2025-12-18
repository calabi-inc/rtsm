"""
YOLO-World segmentation/detection backend.

Open-vocabulary object detection with optional SAM-based mask generation.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from PIL import Image
import torch
import numpy as np
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)


class YOLOWorldSegmenter(SegmentationAdapter):
    """
    Open-vocabulary detection using YOLO-World.

    Unlike FastSAM (segment everything), YOLO-World detects objects
    from a user-defined vocabulary. This is more reliable for task-specific
    applications like warehouses where you know what objects to detect.

    Vocabulary can be:
    - Set at initialization (default_vocab)
    - Overridden per-call via segment(vocab=...)
    - Loaded from config
    """

    def __init__(
        self,
        model_path: str = "yolov8x-worldv2.pt",
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        default_vocab: Optional[List[str]] = None,
        with_masks: bool = True,
    ):
        """
        Args:
            model_path: Path to YOLO-World weights or model name
            device: "cuda" or "cpu"
            imgsz: Input image size for inference
            conf: Confidence threshold
            iou: IoU threshold for NMS
            default_vocab: Default vocabulary if not provided at inference time
            with_masks: If True, generate masks via SAM head (slower but needed for RTSM)
        """
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.with_masks = with_masks
        self.default_vocab = default_vocab or self._default_indoor_vocab()
        self._current_vocab: Optional[List[str]] = None

        # Lazy load model
        self._model = None
        self._model_path = model_path

        logger.info(
            f"YOLOWorldSegmenter initialized: device={device}, imgsz={imgsz}, "
            f"vocab_size={len(self.default_vocab)}, with_masks={with_masks}"
        )

    def _default_indoor_vocab(self) -> List[str]:
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
            # People/vehicles (optional)
            "person", "forklift",
        ]

    def _load_model(self):
        """Lazy load the YOLO-World model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(self._model_path)
            self._model.to(self.device)
            logger.info(f"YOLO-World model loaded: {self._model_path}")
        except ImportError:
            raise ImportError(
                "YOLO-World requires ultralytics package. "
                "Install with: pip install ultralytics"
            )

    def _set_vocab(self, vocab: List[str]) -> None:
        """Set the detection vocabulary (only if changed)."""
        if self._current_vocab == vocab:
            return
        self._model.set_classes(vocab)
        self._current_vocab = vocab
        logger.debug(f"YOLO-World vocab set: {len(vocab)} classes")

    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Detect objects from vocabulary in the image.

        Args:
            image: PIL RGB image
            vocab: List of class names to detect. If None, uses default_vocab.

        Returns:
            SegmentationResult with boxes, labels, scores, and optionally masks
        """
        self._load_model()

        # Set vocabulary
        vocab = vocab or self.default_vocab
        self._set_vocab(vocab)

        # Run inference
        results = self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
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
            labels = [vocab[int(cid)] for cid in class_ids]

        # Extract masks if available
        masks = None
        if self.with_masks and hasattr(result, 'masks') and result.masks is not None:
            masks = torch.from_numpy(result.masks.data).bool()

        # Convert to tensors
        if boxes is not None and not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if scores is not None and not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        if class_ids is not None and not isinstance(class_ids, torch.Tensor):
            class_ids = torch.tensor(class_ids, dtype=torch.int64)

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_ids=class_ids,
            embeddings=None,  # YOLO-World doesn't expose CLIP-compatible embeddings
            vocab=vocab,
        )

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
        return "yoloworld"

    @property
    def supports_vocab(self) -> bool:
        return True

    @property
    def provides_embeddings(self) -> bool:
        # YOLO-World uses CLIP text encoder but doesn't expose visual embeddings
        return False

    @property
    def provides_masks(self) -> bool:
        return self.with_masks
