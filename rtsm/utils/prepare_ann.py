import torch
from typing import Union, List

def prepare_ann(ann: Union[torch.Tensor, List], thr: float = 0.5) -> torch.Tensor:
    """
    Ensure masks are [N,H,W] torch.bool on CPU and contiguous.
    - Accepts float32/uint8/bool, [N,H,W] or [N,1,H,W], CPU or CUDA.
    - Accepts empty list (returns empty [0,H,W] tensor).
    - Thresholds once if needed; no work if already bool.
    """
    # Handle empty list case (no detections)
    if isinstance(ann, list):
        if len(ann) == 0:
            return torch.zeros((0, 1, 1), dtype=torch.bool)
        # Convert list to tensor if needed
        ann = torch.stack([torch.as_tensor(a) for a in ann])

    t = ann
    if t.ndim == 4 and t.size(1) == 1:
        t = t[:, 0]
    if t.is_cuda:
        t = t.cpu()
    if t.dtype is not torch.bool:
        t = (t > thr) if t.dtype.is_floating_point else (t != 0)
    return t.contiguous()  # [N,H,W], bool, CPU