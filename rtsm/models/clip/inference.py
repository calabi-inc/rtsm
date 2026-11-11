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

def _probe_image(size: int) -> np.ndarray:
    """A size x size RGB probe in which every channel holds every byte value: scaling and normalising act on
    one value at a time, so equality on this image is equality on every image of that size."""
    probe = np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)
    flat = probe.reshape(-1, 3)
    n = min(256, flat.shape[0])
    flat[:n] = np.arange(n, dtype=np.uint8)[:, None]
    return probe


def tensor_preprocess_params(preprocess, device: str = "cpu"):
    """(size, scale, mean, std) when ``preprocess`` is, for an image that already has the model's input size,
    nothing but a scaling to [0, 1] and a normalisation; None otherwise.

    It is decided by running it: a probe image of that size (every byte value in every channel) goes through
    ``preprocess`` on the CPU and through the tensor operations on ``device``, and the two outputs must be
    bit-identical. A transform that resizes to another size, crops differently or converts colours fails the
    probe and keeps the per-image path; so does a device whose arithmetic differs from the CPU's.
    scale (255), mean and std come back as float32 tensors on ``device``.
    """
    size = mean = std = None
    for t in getattr(preprocess, "transforms", None) or []:
        kind = t.__class__.__name__
        if kind == "Normalize":
            mean, std = t.mean, t.std
        elif kind in ("Resize", "CenterCrop"):
            s = getattr(t, "size", None)
            if isinstance(s, (tuple, list)):
                s = int(s[0]) if len(s) in (1, 2) and all(int(v) == int(s[0]) for v in s) else None
            if s is not None:
                size = int(s)
    if size is None or size < 16 or mean is None or std is None or len(mean) != 3 or len(std) != 3:
        return None
    params = (size,
              torch.tensor(255.0, dtype=torch.float32, device=device),
              torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
              torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1))
    probe = _probe_image(size)
    ref = preprocess(Image.fromarray(probe))
    fast = _tensor_batch([probe], params, device)
    if fast is None or not isinstance(ref, torch.Tensor) or ref.dtype != torch.float32 or ref.shape != fast.shape[1:]:
        return None
    if not torch.equal(ref, fast[0].to("cpu")):
        return None
    return params


def _tensor_batch(chunk, tensor_pre, device: str):
    """[B,3,S,S] float32 on ``device`` for a chunk made only of SxSx3 uint8 arrays (S = the model's input size);
    None for anything else. The values are those of the per-image transform (tensor_preprocess_params proves it
    for the model and the device at hand), without a PIL image and a transform call per crop.

    The scaling divides by a tensor on the device, not by the Python number: on CUDA a division by a scalar is
    computed as a multiplication by its reciprocal, which differs from the division in the last bit for about
    half of the byte values."""
    size, scale, mean, std = tensor_pre
    for im in chunk:
        if not (isinstance(im, np.ndarray) and im.dtype == np.uint8 and im.shape == (size, size, 3)):
            return None
    arr = np.stack(chunk)                                   # [B,S,S,3] uint8, one contiguous block
    x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).contiguous().to(torch.float32).div(scale)
    return x.sub_(mean).div_(std)


@torch.no_grad()
def encode_images_batch(
    images: List[ImageLike],
    model: torch.nn.Module,
    preprocess,
    device: str = "cuda",
    normalize: bool = True,
    batch_size: int = 16,
    keep_on_device: bool = False,
    tensor_pre=None,
) -> torch.Tensor:
    """
    Batched CLIP embeddings for a list of images (paths, PIL, or np.ndarray).
    Returns [N, D] on CPU.

    tensor_pre: the result of tensor_preprocess_params for this model's ``preprocess``. With it, a chunk of
    uint8 arrays that already have the model's input size is scaled and normalised as one tensor on the device
    (the same values, see there); every other input goes through ``preprocess`` image by image, as before.
    """
    model_dtype = next(model.parameters()).dtype
    all_feats: List[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        batch = _tensor_batch(chunk, tensor_pre, device) if tensor_pre is not None else None
        if batch is None:
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
