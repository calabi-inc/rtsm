"""
Grounded SAM2 segmentation backend.

Open-vocabulary detection + instance segmentation using:
  1. Grounding DINO (text prompt → bounding boxes + labels)
  2. SAM2 (box prompts → pixel-precise masks)

Both models are Apache 2.0 licensed — no AGPL dependency.

Pipeline:
  vocab ["chair", "table"] → GDINO detects boxes → SAM2 segments each box → masks + labels
"""
from __future__ import annotations
from typing import List, Optional
from PIL import Image
import torch
import numpy as np
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)


class GroundedSAM2Segmenter(SegmentationAdapter):
    """
    Open-vocabulary detection + segmentation using Grounding DINO + SAM2.

    Grounding DINO finds objects by text description (truly open-vocabulary —
    any natural language phrase). SAM2 then segments each detected box into
    a pixel-precise mask.

    This is the Apache 2.0-licensed replacement for YOLOE.
    """

    def __init__(
        self,
        gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        default_vocab: Optional[List[str]] = None,
    ):
        self._gdino_model_id = gdino_model_id
        self._sam2_model_id = sam2_model_id
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self.default_vocab = default_vocab or self._default_indoor_vocab()

        self._gdino_model = None
        self._gdino_processor = None
        self._sam2_predictor = None

        logger.info(
            f"GroundedSAM2Segmenter initialized: gdino={gdino_model_id}, "
            f"sam2={sam2_model_id}, device={device}"
        )

    @staticmethod
    def _default_indoor_vocab() -> List[str]:
        """Default vocabulary for indoor/warehouse/home scenarios."""
        return [
            # Furniture
            "chair", "table", "desk", "couch", "sofa", "bed", "shelf", "cabinet",
            # Electronics
            "monitor", "laptop", "keyboard", "mouse", "phone", "tv", "speaker",
            "tablet", "remote", "headphones", "charger",
            # Containers
            "box", "cardboard box", "bin", "basket", "crate", "pallet",
            "tissue box", "storage box",
            # Home objects
            "bottle", "cup", "mug", "book", "bag", "backpack", "plant", "lamp",
            "pillow", "cushion", "blanket", "curtain", "clock", "picture frame",
            # Toys / plush
            "doll", "stuffed animal", "toy", "figurine", "plush",
            # People / vehicles
            "person", "forklift",
        ]

    def _load_models(self):
        """Lazy-load Grounding DINO and SAM2 models."""
        if self._gdino_model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            raise ImportError(
                "Grounding DINO requires the 'transformers' package. "
                "Install with: pip install transformers>=4.44.0"
            )

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 requires the 'sam2' package. "
                "Install with: pip install sam2"
            )

        logger.info(f"Loading Grounding DINO: {self._gdino_model_id} ...")
        self._gdino_processor = AutoProcessor.from_pretrained(self._gdino_model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._gdino_model_id
        ).to(self._device)

        logger.info(f"Loading SAM2: {self._sam2_model_id} ...")
        self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
            self._sam2_model_id, device=self._device,
        )
        logger.info("Grounded SAM2 models loaded")

    @staticmethod
    def _format_caption(vocab: List[str]) -> str:
        """Convert vocab list to GDINO text prompt: 'chair . table . person .'"""
        return " . ".join(vocab) + " ."

    @staticmethod
    def _match_label(phrase: str, vocab: List[str]) -> int:
        """Map GDINO output phrase back to vocab index.

        GDINO may return compound phrases like "laptop book" when multiple
        vocab terms match. We find the first vocab entry contained in the
        phrase.
        """
        phrase_lower = phrase.lower().strip()
        for i, v in enumerate(vocab):
            if v.lower() in phrase_lower:
                return i
        return -1

    def segment(
        self,
        image: Image.Image,
        vocab: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Detect and segment objects using GDINO + SAM2.

        Args:
            image: PIL RGB image
            vocab: List of class names to detect. If None, uses default_vocab.

        Returns:
            SegmentationResult with masks, boxes, labels, and scores
        """
        self._load_models()

        vocab = vocab or self.default_vocab
        caption = self._format_caption(vocab)

        # Step 1: Grounding DINO detection
        inputs = self._gdino_processor(
            images=image, text=caption, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes = results["boxes"]        # [N, 4] xyxy float, pixel coords, on GPU
        scores = results["scores"]      # [N] float, on GPU
        text_labels = results.get("text_labels", results.get("labels", []))

        N = len(boxes)
        if N == 0:
            return self._empty_result(image.size)

        # Step 2: SAM2 box-prompted segmentation
        np_image = np.array(image)
        self._sam2_predictor.set_image(np_image)

        masks_np, iou_scores, _ = self._sam2_predictor.predict(
            box=boxes.cpu().numpy(),
            multimask_output=False,
        )
        # masks_np shape: [N, 1, H, W] if multimask_output=False
        # or [N, H, W] — handle both
        masks_tensor = torch.from_numpy(masks_np)
        if masks_tensor.ndim == 4:
            masks_tensor = masks_tensor.squeeze(1)
        masks_tensor = masks_tensor.bool()  # [N, H, W] bool CPU

        # Step 3: Map labels to vocab indices
        labels = list(text_labels) if not isinstance(text_labels, list) else text_labels
        class_ids = torch.tensor(
            [self._match_label(label, vocab) for label in labels],
            dtype=torch.int64,
        )
        # Normalize span-merged compound phrases ("water bottle tissue
        # box") to the single matched vocabulary entry (2026-08-28):
        # downstream stores labels verbatim and searches them by
        # substring, so a compound phrase would make the WRONG class
        # searchable on this object. _match_label picks the first
        # contained entry — inherently ambiguous for compounds, but one
        # clean class (arbitrated later by image) beats a multi-class
        # string. Unmatched phrases stay raw.
        labels = [vocab[int(ci)] if int(ci) >= 0 else lbl
                  for lbl, ci in zip(labels, class_ids)]

        # Ensure CPU tensors
        boxes_cpu = boxes.cpu().float()
        scores_cpu = scores.cpu().float()

        return SegmentationResult(
            masks=masks_tensor,
            boxes=boxes_cpu,
            scores=scores_cpu,
            labels=labels,
            class_ids=class_ids,
            vocab=vocab,
        )

    def warmup(self) -> None:
        """Eagerly load both models (downloads from HuggingFace if needed)."""
        self._load_models()

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
        if self._gdino_model is not None:
            del self._gdino_model
            self._gdino_model = None
        if self._gdino_processor is not None:
            del self._gdino_processor
            self._gdino_processor = None
        if self._sam2_predictor is not None:
            del self._sam2_predictor
            self._sam2_predictor = None
        torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "grounded_sam2"

    @property
    def supports_vocab(self) -> bool:
        return True

    @property
    def provides_embeddings(self) -> bool:
        return False

    @property
    def provides_masks(self) -> bool:
        return True
