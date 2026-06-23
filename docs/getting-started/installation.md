# Installation

## Prerequisites

- Python 3.10+ (3.12 on desktop; 3.10 on Jetson / JetPack 6.x)
- CUDA-capable GPU (tested on RTX 3080, RTX 4090, RTX 5090; Jetson Orin via the edge profile)
- For live capture: iPhone with [Calabi Lens](https://github.com/calabi-inc/rtsm-arkit-client) or Intel RealSense D435i + RTAB-Map
- For demo/replay: no hardware needed

---

## Install RTSM

### Option A: pip install (recommended)

```bash
# Core only (query client, REST API, data contracts — no GPU needed)
pip install rtsm

# With GPU perception pipeline (default: Grounding DINO + SAM2, Apache 2.0)
pip install "rtsm[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# With 3D visualization
pip install "rtsm[gpu,viz]" --extra-index-url https://download.pytorch.org/whl/cu128

# Everything
pip install "rtsm[all]" --extra-index-url https://download.pytorch.org/whl/cu128
```

!!! tip "CUDA Version"
    The `--extra-index-url` flag tells pip which PyTorch build to use. Replace `cu128` with your CUDA version:

    - CUDA 11.8: `https://download.pytorch.org/whl/cu118`
    - CUDA 12.1: `https://download.pytorch.org/whl/cu121`
    - CUDA 12.8: `https://download.pytorch.org/whl/cu128`

### Option B: from source (development)

```bash
git clone https://github.com/calabi-inc/rtsm.git
cd rtsm
pip install -e ".[gpu,viz]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### Option C: faster backends (AGPL, opt-in)

The default backends (Grounding DINO + SAM2) are Apache 2.0 licensed. For faster inference using FastSAM + YOLOE (AGPL-3.0 via ultralytics), install the opt-in extra:

```bash
pip install "rtsm[gpu-ultralytics]" --extra-index-url https://download.pytorch.org/whl/cu128
```

Then set `backend: dual` in your config. See [Configuration](configuration.md) for details.

### Option D: edge / Jetson (ARM, aarch64)

On NVIDIA Jetson (Orin), stock PyPI `torch` has no working aarch64+CUDA build — it must come from NVIDIA's Jetson wheel index, matched to the device's JetPack/CUDA. The `edge` profile runs **FastSAM-only** and deliberately excludes `torch`/`torchvision` (installed separately) and the heavy SAM2/GDINO CUDA backends.

```bash
# 0. verify the device first
cat /etc/nv_tegra_release      # JetPack/L4T version
nvcc --version                 # CUDA 12.6 → cu126 (default), 12.4 → cu124
python3 --version              # 3.10 on JetPack 6.x

# 1. torch — arch-detecting helper (x86 → PyTorch index, Jetson → NVIDIA index)
python scripts/install_torch.py            # add --jetson-cu cu124 if JetPack 6.1
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. RTSM + lean edge deps (no torch re-pull, no SAM2/transformers)
pip install -e ".[edge]"
```

Then set `backend: fastsam` in your config. FastSAM-only is the recommended edge default — lighter than `dual` with comparable object discovery.

---

## Install Extras

| Extra | What it adds | License |
|-------|-------------|---------|
| `gpu` | Grounding DINO, SAM2, CLIP, torch, FAISS | Apache 2.0 |
| `gpu-ultralytics` | FastSAM, YOLOE (via ultralytics) | AGPL-3.0 |
| `edge` | FastSAM + YOLOE + SigLIP + FAISS (Jetson/ARM, no torch/SAM2) | AGPL-3.0 |
| `viz` | Open3D, matplotlib (3D visualization) | MIT |
| `mcp` | MCP server + httpx (AI agent integration) | Apache 2.0 |
| `all` | gpu + viz + mcp | Mixed |

---

## Model Weights

### Permissive backends (default)

All models **auto-download on first run** via HuggingFace Hub. No manual setup needed.

| Model | HuggingFace ID | Size | Auto-download |
|-------|---------------|------|:---:|
| Grounding DINO (tiny) | `IDEA-Research/grounding-dino-tiny` | ~340MB | Yes |
| SAM2.1 Hiera (small) | `facebook/sam2.1-hiera-small` | ~160MB | Yes |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` (via open_clip) | ~340MB | Yes |

First run will download ~840MB of model weights to your HuggingFace cache (`~/.cache/huggingface/`). Subsequent runs use the cache.

### AGPL backends (opt-in)

| Model | Source | Size | Auto-download |
|-------|--------|------|:---:|
| FastSAM-x | ultralytics (auto-downloads) | ~140MB | Yes |
| YOLOE-26s-seg | ultralytics (auto-downloads) | ~50MB | Yes |

### Pre-download (optional)

To avoid download delays on first run:

```python
# Pre-download permissive models
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

from sam2.build_sam import build_sam2
# SAM2 downloads automatically via sam2 package

import open_clip
open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
```

---

## Verify Installation

```bash
# Check import works
python -c "import rtsm; print('RTSM ready')"

# Check GPU pipeline loads (requires [gpu] extra)
python -c "from rtsm.models.segmentation import get_segmenter; print('Segmentation ready')"
```

---

## Docker

```bash
# GPU pipeline (requires nvidia-docker)
docker run --gpus all calabi/rtsm demo
```

!!! note "Docker Images"
    Pre-built Docker images are planned for a future release. For now, use the Dockerfiles in `tests/` for reference:

    - `tests/Dockerfile.deptest` — Dependency verification (core, GPU, full)
    - `tests/Dockerfile.gpu-test` — GPU pipeline replay test

---

## Install MCP Support (Optional)

To expose RTSM as an MCP tool server for AI agents (Claude, Cursor, LangGraph, etc.):

```bash
pip install "rtsm[gpu,mcp]" --extra-index-url https://download.pytorch.org/whl/cu128
```

See [MCP (AI Agents)](../api/mcp.md) for setup and configuration.

---

## Next Steps

- [Quick Start](quick-start.md) — Run your first session
- [Configuration](configuration.md) — Choose backends and tune parameters
- [MCP (AI Agents)](../api/mcp.md) — Connect your AI agent to RTSM
