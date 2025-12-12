from __future__ import annotations
# rtsm/models/clip/adapter.py
from dataclasses import dataclass
import torch
import numpy as np
from .loader import load_clip
from .inference import (
    encode_image as _encode_image,
    encode_images_batch as _encode_images_batch,
)

@dataclass
class ClipArtifacts:
    model: object
    preprocess: object
    tokenizer: object | None = None

class CLIPAdapter:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", local_dir="model_store/clip", device="cuda"):
        model, preprocess, tokenizer = load_clip(model_name, pretrained, local_dir, device=device)
        self.artifacts = ClipArtifacts(model=model, preprocess=preprocess, tokenizer=tokenizer)
        self.device = device

    def encode_image(self, image):  # image path, PIL, or np.ndarray
        return _encode_image(image, self.artifacts.model, self.artifacts.preprocess, device=self.device, keep_on_device=True)

    def encode_images_batch(self, images, batch_size=16):  # images: paths, PIL, or np.ndarrays
        return _encode_images_batch(images, self.artifacts.model, self.artifacts.preprocess, device=self.device, batch_size=batch_size, keep_on_device=True)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to L2-normalized embedding for semantic search."""
        tokens = self.artifacts.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.artifacts.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].astype(np.float32)

