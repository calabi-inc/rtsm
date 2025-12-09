from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


# NOTE:
# - Keep this module lightweight: avoid importing heavy libs (torch, PIL, numpy).
# - Use "Any" for tensors/arrays/images to prevent import-time overhead.


@dataclass(frozen=True)
class FastSAMTask:
    """GPU segmentation task input.

    Attributes
    - frame_id: application-level identifier for correlating results
    - ts_ns: sensor timestamp in nanoseconds
    - rgb: RGB image (e.g., np.ndarray HxWx3 uint8 or PIL.Image)
    - is_keyframe: priority hint for admission/backpressure policies
    - submit_mono_s: monotonic time at enqueue (for latency/staleness checks)
    """

    frame_id: str
    ts_ns: int
    rgb: Any
    is_keyframe: bool = False
    submit_mono_s: float = 0.0


@dataclass(frozen=True)
class FastSAMResult:
    """Segmentation result from FastSAM.

    - ann: model-specific annotation/masks object (e.g., torch.Tensor or np.ndarray)
    """

    frame_id: str
    ts_ns: int
    ann: Any


@dataclass(frozen=True)
class CLIPTask:
    """GPU embedding task input for CLIP image encoder.

    - crops: list of image crops (np.ndarray/PIL.Image). Can be empty.
    - batch_size: optional override for batching policy.
    """

    frame_id: str
    ts_ns: int
    crops: List[Any]
    batch_size: Optional[int] = None


@dataclass(frozen=True)
class CLIPResult:
    """Batched CLIP embedding output.

    - feats: [K, D] tensor/array (e.g., torch.Tensor), normalized if caller configured so.
    """

    frame_id: str
    ts_ns: int
    feats: Any


@dataclass(frozen=True)
class Shutdown:
    """Sentinel message to ask workers to exit gracefully."""

    reason: str = ""


__all__ = [
    "FastSAMTask",
    "FastSAMResult",
    "CLIPTask",
    "CLIPResult",
    "Shutdown",
]


