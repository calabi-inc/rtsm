#!/usr/bin/env python3
"""
Measure the ACTUAL mask tensor shape each segmentation backend produces.
=======================================================================
The datasheet's mask resolution came from the `retina_masks` config flag —
but only FastSAM/YOLOE consume it. GroundedSAM2/SAM2 call `set_image(full RGB)`
with no resize, so their masks come back at the native input resolution, which
dominates the heuristics (depth-stats) cost. This probe segments one real
session1 frame per backend and prints `result.masks.shape` — ground truth,
reusable on x86 and Jetson.

    python scripts/probe_mask_shapes.py                # all backends
    python scripts/probe_mask_shapes.py fastsam grounded_sam2
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rtsm.io.websocket import decode_rgb  # noqa: E402
from rtsm.models.segmentation import get_segmenter  # noqa: E402

RECORDING = ROOT / "recordings" / "session1" / "messages.bin"
CONFIG = ROOT / "rtsm" / "cfg" / "rtsm.yaml"
ALL = ["fastsam", "yoloe", "dual", "grounded_sam2", "sam2"]


def load_frame0() -> Image.Image:
    """Decode RGB of the first binary message into a PIL image."""
    with open(RECORDING, "rb") as f:
        data = f.read(20_000_000)  # first message is ~4.3 MB
    off = 0
    (json_len,) = struct.unpack_from("<I", data, off); off += 4
    hdr = json.loads(data[off:off + json_len].decode("utf-8")); off += json_len
    (rgb_len,) = struct.unpack_from("<I", data, off); off += 4
    rgb_bytes = data[off:off + rgb_len]
    rgb = decode_rgb(rgb_bytes, hdr["rgb_format"], hdr["rgb_width"], hdr["rgb_height"])
    print(f"input RGB: {hdr['rgb_width']}x{hdr['rgb_height']} ({hdr['rgb_format']}) "
          f"-> decoded {rgb.shape}")
    return Image.fromarray(rgb)


def main():
    backends = [b for b in (sys.argv[1:] or ALL)]
    img = load_frame0()
    base = yaml.safe_load(open(CONFIG))

    print(f"\n{'backend':16s} {'mask tensor shape':22s} {'masks':>6} {'px/mask':>10}")
    print("-" * 60)
    for name in backends:
        cfg = json.loads(json.dumps(base))  # deep copy
        cfg["segmentation"]["backend"] = name
        try:
            seg = get_segmenter(cfg)
            res = seg.segment(img)
            shp = tuple(res.masks.shape)
            n = shp[0] if len(shp) >= 1 else 0
            px = (shp[1] * shp[2]) if len(shp) == 3 else 0
            print(f"{name:16s} {str(shp):22s} {n:>6} {px:>10,}")
            if hasattr(seg, "close"):
                seg.close()
        except Exception as e:
            print(f"{name:16s} ERROR: {type(e).__name__}: {e}")
        finally:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
