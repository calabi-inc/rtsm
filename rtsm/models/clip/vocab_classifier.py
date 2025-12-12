from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import yaml
from PIL import Image
import logging
logger = logging.getLogger(__name__)

__all__ = ["LabelParams", "ClipVocabClassifier"]

@dataclass
class LabelParams:
    """Thresholds for accepting a label."""
    min_top: float = 0.30     # minimum cosine score for top-1
    min_margin: float = 0.05  # (top1 - top2) minimum margin
    topk: int = 5             # how many classes to return in .topk

class ClipVocabClassifier:
    """
    Zero-shot classifier on top of CLIP.

    - At startup, build & cache text features from a small vocab YAML.
    - Per keyframe, batch-encode masked crops with CLIP image encoder.
    - Classify via cosine against cached text features.

    Usage:
        model, preprocess, tokenizer, _ = load_clip_model(cfg.clip)
        clf = ClipVocabClassifier(model, tokenizer, preprocess, "config/clip/vocab.yaml", device="cuda")

        results = clf.classify_feats(rgb_pil_image, list_of_masks, return_embeddings=True)
        # results[i].label, .score, .topk, .embedding (512-d), .bbox
    """

    def __init__(
        self,
        model,
        tokenizer,
        preprocess,
        vocab_yaml: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        drop_model_after_cache: bool = False,
        text_on_cpu: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = torch.device(device)
        if dtype is None:
            try:
                dtype = next(self.model.parameters()).dtype
            except StopIteration:
                dtype = torch.float16
        self.dtype = dtype

        self.text_feats, self.class_ids, self.params_yaml = self._build_text_features(vocab_yaml)
        # Place text feats
        if text_on_cpu:
            self.text_feats = self.text_feats.to("cpu", dtype=torch.float32).contiguous()
        else:
            self.text_feats = self.text_feats.to(self.device, dtype=self.dtype).contiguous()

        if drop_model_after_cache:
            self.release_clip()

    # --------------------------- text side (once) ---------------------------

    @torch.no_grad()
    def _build_text_features(self, cfg_path: str) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Build per-class text embeddings by averaging template×synonym prompts.
        Returns:
            text_feats: [C, D] normalized (float32 on CPU; caller moves to device)
            class_ids: list[str] class ids in order
            params_yaml: dict with 'min_top' and 'min_margin' defaults
        """
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        templates: List[str] = cfg.get("templates", ["a photo of a {}"])
        classes = cfg["classes"]
        class_ids = [c["id"] for c in classes]
        params_yaml = {
            "min_top": cfg.get("unknown", {}).get("min_top", 0.30),
            "min_margin": cfg.get("unknown", {}).get("min_margin", 0.05),
        }

        # Build expanded prompts per class and remember offsets
        prompts: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for c in classes:
            names = [c["id"]] + c.get("synonyms", [])
            start = len(prompts)
            for name in names:
                for t in templates:
                    prompts.append(t.format(name))
            offsets.append((start, len(prompts)))

        # Tokenize & encode text
        toks = self.tokenizer(prompts)
        if isinstance(toks, torch.Tensor):
            toks = toks.to(self.device)
        with torch.no_grad():
            t = self.model.encode_text(toks)  # [P, D]
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Average prompts per class
        per_class = []
        for s, e in offsets:
            v = t[s:e].float().mean(dim=0)
            v = v / v.norm().clamp_min(1e-12)
            per_class.append(v.cpu())  # keep CPU float32; move to device later
        text_feats = torch.stack(per_class, dim=0)  # [C, D]
        return text_feats, class_ids, params_yaml

    @torch.no_grad()
    def classify_feats(
        self,
        img_feats: torch.Tensor,  # [K,D], normalized
        params: Optional[LabelParams] = None,
        return_topk: Optional[int] = None,
    ) -> List[Tuple[str, float, int, List[Tuple[str, float, int]]]]:
        """
        Compare image feats against cached text feats, return label tuples.
        """
        if img_feats.numel() == 0:
            return []

        # Accept a single embedding [D] by expanding to [1,D]
        if img_feats.dim() == 1:
            img_feats = img_feats.unsqueeze(0)

        # Ensure text features are on the same device/dtype as img_feats
        if self.text_feats.device != img_feats.device or self.text_feats.dtype != img_feats.dtype:
            text_feats = self.text_feats.to(device=img_feats.device, dtype=img_feats.dtype, non_blocking=True)
        else:
            text_feats = self.text_feats

        C = text_feats.shape[0]
        k = int(min(return_topk or 5, C))

        logits = img_feats @ text_feats.T  # cosine since both normalized → [K,C]

        topv, topi = logits.max(dim=1)
        masked = logits.clone()
        masked.scatter_(1, topi.unsqueeze(1), float("-inf"))
        secondv, _ = masked.max(dim=1)

        tkv, tki = logits.topk(k=k, dim=1)

        # default thresholds from YAML if none provided
        if params is None:
            params = LabelParams(
                min_top=float(self.params_yaml["min_top"]),
                min_margin=float(self.params_yaml["min_margin"]),
                topk=k,
            )

        out = []
        for i in range(logits.size(0)):
            topk_list = [
                (self.class_ids[int(tki[i, j])], float(tkv[i, j]), int(tki[i, j]))
                for j in range(k)
            ]
            tv = float(topv[i].item())
            margin = float((topv[i] - secondv[i]).item())
            if tv >= params.min_top and margin >= params.min_margin:
                out.append((self.class_ids[int(topi[i].item())], tv, int(topi[i].item()), topk_list))
            else:
                out.append(("unknown", tv, -1, topk_list))
        return out

    def release_clip(self):
        """
        Release references to the CLIP model and preprocess to free memory.
        """
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def ensure_text_on_device(self) -> None:
        """Move text features to current device/dtype if they are on CPU."""
        if self.text_feats.device.type != self.device.type:
            self.text_feats = self.text_feats.to(self.device, dtype=self.dtype, non_blocking=True).contiguous()
