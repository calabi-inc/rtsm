#!/usr/bin/env python3
"""
Cross-arch torch installer for RTSM.
====================================
torch/torchvision cannot be installed identically on x86 and ARM:
  - x86-64 desktop  -> PyTorch's CUDA wheel index (download.pytorch.org)
  - aarch64 / Jetson -> NVIDIA's Jetson wheel index (pypi.jetson-ai-lab.io),
    matched to the device's JetPack/CUDA — stock PyPI torch has no working
    aarch64+CUDA build.

This script detects the architecture and runs the right `pip install`, so the
same command works on either machine:

    python scripts/install_torch.py            # detect + install
    python scripts/install_torch.py --dry-run  # print the command only

NOTE (Jetson): the index URL encodes the JetPack CUDA version. Defaults below
target JetPack 6.2 (CUDA 12.6). VERIFY your device first:
    cat /etc/nv_tegra_release      # JetPack/L4T version
    nvcc --version                 # bundled CUDA  -> sets jp6/cuXXX
If you're on JetPack 6.1 (cu124) or a different rev, override with --jetson-cu.
After install, confirm GPU: python -c "import torch; print(torch.cuda.is_available())"
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys

# ── x86-64 desktop (matches the 5090 dev box: CUDA 12.8) ──
X86_INDEX = "https://download.pytorch.org/whl/cu128"
X86_PKGS = ["torch", "torchvision"]

# ── aarch64 / Jetson (JetPack 6.2 = CUDA 12.6, verified working versions) ──
JETSON_INDEX_TMPL = "https://pypi.jetson-ai-lab.io/jp6/{cu}"
JETSON_CU_DEFAULT = "cu126"
JETSON_PKGS = ["torch==2.8.0", "torchvision==0.23.0"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Install torch for the current architecture.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the pip command without running it.")
    ap.add_argument("--jetson-cu", default=JETSON_CU_DEFAULT,
                    help="Jetson CUDA tag for the index URL (e.g. cu126, cu124). "
                         "Match your JetPack: cat /etc/nv_tegra_release / nvcc --version.")
    args = ap.parse_args()

    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        index = JETSON_INDEX_TMPL.format(cu=args.jetson_cu)
        pkgs = JETSON_PKGS
        target = f"aarch64 / Jetson ({args.jetson_cu})"
    else:
        index = X86_INDEX
        pkgs = X86_PKGS
        target = f"x86-64 desktop ({X86_INDEX.rsplit('/', 1)[-1]})"

    cmd = [sys.executable, "-m", "pip", "install", *pkgs, "--index-url", index]
    print(f"Detected arch : {platform.machine()}  ->  {target}")
    print("Command       : " + " ".join(cmd))

    if args.dry_run:
        print("(dry-run: not executed)")
        return 0

    print("Installing...\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
