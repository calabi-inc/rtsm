from __future__ import annotations
import os
from typing import List, Tuple, Union

import torch
from PIL import Image
import numpy as np

ImageLike = Union[str, Image.Image, np.ndarray]

def _to_pil_rgb(x: ImageLike) -> Image.Image:
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    # numpy array branch
    arr = x
    if arr.dtype != np.uint8:
        # assume [0,1] float -> uint8
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    # assume already RGB; if you know it's BGR, swap here:
    # arr = arr[..., ::-1]  # uncomment if your crops are BGR
    return Image.fromarray(arr, mode="RGB")

@torch.no_grad()
def encode_image(
    image_path: Union[str, Image.Image],
    model: torch.nn.Module,
    preprocess,
    device: str = "cuda",
    normalize: bool = True,
    keep_on_device: bool = True,
) -> torch.Tensor:
    """
    Compute a single-image CLIP embedding. Returns a 1D tensor on CPU.
    """
    image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
    image_t = preprocess(image)  # [C,H,W], float32
    # Match model dtype (fp16 on CUDA if the model is half)
    model_dtype = next(model.parameters()).dtype
    image_t = image_t.to(device=device, dtype=model_dtype).unsqueeze(0)  # [1,C,H,W]

    feats = model.encode_image(image_t)
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = feats[0].detach()
    return out if keep_on_device else out.to("cpu")  # [D]

@torch.no_grad()
def encode_images_batch(
    images: List[ImageLike],
    model: torch.nn.Module,
    preprocess,
    device: str = "cuda",
    normalize: bool = True,
    batch_size: int = 16,
    keep_on_device: bool = False,
) -> torch.Tensor:
    """
    Batched CLIP embeddings for a list of images (paths, PIL, or np.ndarray).
    Returns [N, D] on CPU.
    """
    model_dtype = next(model.parameters()).dtype
    all_feats: List[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        pil_imgs = [_to_pil_rgb(im) for im in chunk]
        batch = torch.stack([preprocess(im) for im in pil_imgs])  # [B,C,H,W]
        batch = batch.to(device=device, dtype=model_dtype)
        feats = model.encode_image(batch)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        all_feats.append(feats if keep_on_device else feats.to("cpu"))
    if not all_feats:
        return torch.empty(0, device=(device if keep_on_device else "cpu"))
    out = torch.cat(all_feats, dim=0)
    return out
