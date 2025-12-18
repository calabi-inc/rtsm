"""
Segmentation module with swappable backends.

Usage:
    from rtsm.models.segmentation import get_segmenter, SegmentationResult

    # Create segmenter from config
    segmenter = get_segmenter(cfg)

    # Run segmentation
    result = segmenter.segment(pil_image)

    # Access results
    print(f"Found {result.count} objects")
    if result.has_masks:
        masks = result.masks  # [N, H, W] bool tensor
    if result.labels:
        print(f"Labels: {result.labels}")
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import logging

from rtsm.models.segmentation.base import SegmentationAdapter, SegmentationResult

logger = logging.getLogger(__name__)

# Registry of available backends
_BACKENDS = {}


def register_backend(name: str):
    """Decorator to register a segmentation backend."""
    def decorator(cls):
        _BACKENDS[name] = cls
        return cls
    return decorator


def get_segmenter(cfg: Dict[str, Any]) -> SegmentationAdapter:
    """
    Factory function to create a segmenter from config.

    Config format:
        segmentation:
          backend: fastsam | yoloworld
          fastsam:
            model_path: model_store/fastsam/FastSAM-x.pt
            ...
          yoloworld:
            model_path: yolov8x-worldv2.pt
            vocab: [...]
            ...

    Args:
        cfg: Full RTSM config dict

    Returns:
        SegmentationAdapter instance
    """
    seg_cfg = cfg.get("segmentation", {})
    backend = seg_cfg.get("backend", "fastsam")

    logger.info(f"Creating segmenter: backend={backend}")

    if backend == "fastsam":
        return _create_fastsam(cfg, seg_cfg)
    elif backend == "yoloworld":
        return _create_yoloworld(cfg, seg_cfg)
    else:
        raise ValueError(
            f"Unknown segmentation backend: {backend}. "
            f"Available: fastsam, yoloworld"
        )


def _create_fastsam(cfg: Dict[str, Any], seg_cfg: Dict[str, Any]) -> SegmentationAdapter:
    """Create FastSAM segmenter from config."""
    from rtsm.models.segmentation.fastsam_segmenter import FastSAMSegmenter

    fastsam_cfg = seg_cfg.get("fastsam", {})

    return FastSAMSegmenter(
        model_path=fastsam_cfg.get("model_path", "model_store/fastsam/FastSAM-x.pt"),
        device=fastsam_cfg.get("device", "cuda"),
        imgsz=fastsam_cfg.get("imgsz", 640),
        conf=fastsam_cfg.get("conf", 0.4),
        iou=fastsam_cfg.get("iou", 0.9),
    )


def _create_yoloworld(cfg: Dict[str, Any], seg_cfg: Dict[str, Any]) -> SegmentationAdapter:
    """Create YOLO-World segmenter from config."""
    from rtsm.models.segmentation.yoloworld_segmenter import YOLOWorldSegmenter

    yolo_cfg = seg_cfg.get("yoloworld", {})

    return YOLOWorldSegmenter(
        model_path=yolo_cfg.get("model_path", "yolov8x-worldv2.pt"),
        device=yolo_cfg.get("device", "cuda"),
        imgsz=yolo_cfg.get("imgsz", 640),
        conf=yolo_cfg.get("conf", 0.25),
        iou=yolo_cfg.get("iou", 0.7),
        default_vocab=yolo_cfg.get("vocab", None),
        with_masks=yolo_cfg.get("with_masks", True),
    )


# Re-export for convenience
__all__ = [
    "SegmentationAdapter",
    "SegmentationResult",
    "get_segmenter",
]
