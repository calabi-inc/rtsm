from __future__ import annotations
import os, torch, open_clip

def load_clip(model_name: str, pretrained: str, local_dir: str | None = None, device="cuda"):
    """
    Try to load CLIP from a local checkpoint saved by scripts/fetch_clip.py.
    Fallback to open_clip's built-in downloader if not found.
    """
    arch = f"{model_name}-quickgelu" if pretrained == "openai" else model_name
    if local_dir:
        ckpt = os.path.join(local_dir, f"{model_name}-{pretrained}", "model.pt")
        if os.path.isfile(ckpt):
            model = open_clip.create_model(arch, pretrained=pretrained, device=device)
            sd = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            model.eval()
            if device == "cuda": model = model.half()
            preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer

    # Fallback: will download to cache (~/.cache/)
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device
    )
    if device == "cuda": model = model.half()
    tokenizer = open_clip.get_tokenizer(arch)
    return model, preprocess, tokenizer
