"""
Fetch all RTSM models into model_store/.

Usage:
    python scripts/fetch_models.py              # fetch all
    python scripts/fetch_models.py --only clip   # fetch only clip
    python scripts/fetch_models.py --only yolo   # fetch only yolo
    python scripts/fetch_models.py --only fastsam # fetch only fastsam
    python scripts/fetch_models.py --verify      # sha256 of the FastSAM / YOLOE files against the validated files

The FastSAM and YOLOE files are fetched by NAME through ultralytics, and the file behind a name changes upstream in
place (yoloe-26s-seg-pf.pt was replaced on 2026-09-17 by an export that labels differently). Every fetched file is
therefore checked by sha256 against rtsm/models/weights.py; a mismatch is printed, the file is kept, and every run's
provenance records it. Exit code 1 from --verify when any validated file is missing or differs.

Directory layout:
    model_store/
      clip/ViT-B-32-openai/model.pt
      fastsam/FastSAM-x.pt
      yolo/yoloe-26s-seg.pt
      yolo/yoloe-26s-seg-pf.pt
"""
import argparse
import os
import sys
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_STORE = os.path.join(ROOT, "model_store")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VALIDATED_LAYOUT = {"FastSAM-x.pt": "fastsam", "yoloe-26s-seg.pt": "yolo", "yoloe-26s-seg-pf.pt": "yolo"}


def _report(path: str, tag: str) -> None:
    """Print the validation line for a fetched file; a mismatch gets a visible block."""
    from rtsm.models.weights import check_weight_file
    st = check_weight_file(path)
    if st.state == "mismatch":
        print(f"[{tag}] WARNING: {st.line()}")
        print(f"[{tag}]          this is not the file rtsm's documented numbers were produced with; the file served under this")
        print(f"[{tag}]          name changed upstream. The file is kept; every run records its sha256 in resolved.json.")
    else:
        print(f"[{tag}] {st.line()}")


def verify_all(model_store: str = None):
    """Status of every validated FastSAM / YOLOE file under model_store (missing files count)."""
    from rtsm.models.weights import check_weight_file
    store = model_store or MODEL_STORE
    return [check_weight_file(os.path.join(store, sub, name)) for name, sub in VALIDATED_LAYOUT.items()]


def fetch_clip():
    """Fetch CLIP ViT-B-32 (openai) into model_store/clip/."""
    import torch
    import open_clip

    out = os.path.join(MODEL_STORE, "clip", "ViT-B-32-openai")
    ckpt = os.path.join(out, "model.pt")
    if os.path.isfile(ckpt):
        print(f"[clip] Already present: {ckpt}")
        return

    os.makedirs(out, exist_ok=True)
    print("[clip] Downloading ViT-B-32 (openai) ...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cpu"
    )
    torch.save(model.state_dict(), ckpt)
    print(f"[clip] Saved: {ckpt}")


def fetch_fastsam():
    """Fetch FastSAM-x into model_store/fastsam/."""
    out = os.path.join(MODEL_STORE, "fastsam")
    ckpt = os.path.join(out, "FastSAM-x.pt")
    if os.path.isfile(ckpt):
        print(f"[fastsam] Already present: {ckpt}")
        return

    os.makedirs(out, exist_ok=True)
    print("[fastsam] Downloading FastSAM-x via ultralytics ...")
    from ultralytics import FastSAM
    # ultralytics auto-downloads to CWD or cache; load then copy
    model = FastSAM("FastSAM-x.pt")
    src = model.ckpt_path if hasattr(model, "ckpt_path") else "FastSAM-x.pt"
    if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(ckpt):
        shutil.move(src, ckpt)
    print(f"[fastsam] Saved: {ckpt}")
    _report(ckpt, "fastsam")


def fetch_yolo():
    """Fetch YOLOE-26s-seg and YOLOE-26s-seg-pf into model_store/yolo/."""
    out = os.path.join(MODEL_STORE, "yolo")
    os.makedirs(out, exist_ok=True)

    from ultralytics import YOLOE

    variants = [
        "yoloe-26s-seg.pt",       # prompted (text/visual vocab)
        "yoloe-26s-seg-pf.pt",    # prompt-free (4 585 built-in categories)
    ]
    for variant in variants:
        ckpt = os.path.join(out, variant)
        if os.path.isfile(ckpt):
            print(f"[yolo] Already present: {ckpt}")
            continue

        print(f"[yolo] Downloading {variant} via ultralytics ...")
        model = YOLOE(variant)
        src = model.ckpt_path if hasattr(model, "ckpt_path") else variant
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(ckpt):
            shutil.move(src, ckpt)
        print(f"[yolo] Saved: {ckpt}")
        _report(ckpt, "yolo")


def fetch_sam2():
    """Pre-download SAM2 weights via HuggingFace Hub into local cache."""
    model_id = "facebook/sam2.1-hiera-small"
    out = os.path.join(MODEL_STORE, "sam2")
    marker = os.path.join(out, ".downloaded")
    if os.path.isfile(marker):
        print(f"[sam2] Already cached: {model_id}")
        return

    os.makedirs(out, exist_ok=True)
    print(f"[sam2] Downloading {model_id} via HuggingFace Hub ...")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2ImagePredictor.from_pretrained(model_id, device="cpu")
    with open(marker, "w") as f:
        f.write(model_id + "\n")
    print(f"[sam2] Cached: {model_id}")


def fetch_gdino():
    """Pre-download Grounding DINO via HuggingFace Hub."""
    model_id = "IDEA-Research/grounding-dino-tiny"
    out = os.path.join(MODEL_STORE, "gdino")
    marker = os.path.join(out, ".downloaded")
    if os.path.isfile(marker):
        print(f"[gdino] Already cached: {model_id}")
        return

    os.makedirs(out, exist_ok=True)
    print(f"[gdino] Downloading {model_id} via HuggingFace Hub ...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    AutoProcessor.from_pretrained(model_id)
    AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    with open(marker, "w") as f:
        f.write(model_id + "\n")
    print(f"[gdino] Cached: {model_id}")


FETCHERS = {
    "clip": fetch_clip,
    "fastsam": fetch_fastsam,
    "yolo": fetch_yolo,
    "sam2": fetch_sam2,
    "gdino": fetch_gdino,
}


def main(argv=None) -> int:
    global MODEL_STORE
    ap = argparse.ArgumentParser(description="Fetch RTSM models into model_store/")
    ap.add_argument("--only", choices=list(FETCHERS.keys()), help="Fetch only this model")
    ap.add_argument("--verify", action="store_true", help="check the FastSAM / YOLOE files against the validated sha256s and exit")
    ap.add_argument("--model-store", default=None, help=f"model store directory (default: {MODEL_STORE})")
    args = ap.parse_args(argv)
    if args.model_store:
        MODEL_STORE = args.model_store

    if args.verify:
        statuses = verify_all(MODEL_STORE)
        for st in statuses:
            print(("[verify] " if st.state == "validated" else "[verify] !! ") + st.line())
        bad = [st for st in statuses if st.state != "validated"]
        print(f"[verify] {len(statuses) - len(bad)} of {len(statuses)} validated files present and matching")
        return 1 if bad else 0

    targets = [args.only] if args.only else list(FETCHERS.keys())
    for name in targets:
        try:
            FETCHERS[name]()
        except Exception as e:
            print(f"[{name}] ERROR: {e}", file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
