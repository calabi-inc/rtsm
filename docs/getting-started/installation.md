# Installation

## Prerequisites

- Python 3.10+ (3.12 on desktop; 3.10 on Jetson / JetPack 6.x)
- CUDA-capable GPU (tested on RTX 3080, RTX 4090, RTX 5090; Jetson Orin via the edge profile)
- For live capture: iPhone with [Calabi Lens](https://github.com/calabi-inc/rtsm-arkit-client) or Intel RealSense D435i + RTAB-Map
- For demo/replay: no hardware needed

---

## Install RTSM

!!! note "What's new"
    0.2.0 (October 2026) is the first release since April: `rtsm eval` on bags and recordings with the single-bag report, the bag readers, the Python client, the detections adapter, and headless by default (`--viz` turns the dashboard on). The full list, with the behaviour changes a 0.1.1 user should read first, is in the [changelog](https://github.com/calabi-inc/rtsm/blob/main/CHANGELOG.md); `rtsm version` prints what you have.

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

A CUDA-ready image is published with every release: `ghcr.io/calabi-inc/rtsm:<version>` and `:latest`. Python 3.12, PyTorch cu128, the `gpu`, `eval` and `mcp` extras, no model weights (the default tier, about 2.5 GB, downloads on first use into the `/models` volume). It needs an NVIDIA driver on the host and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for `--gpus`.

```bash
# the report on a bag or recording in the current directory
docker run --rm --gpus all -v rtsm-models:/models -v "$PWD":/data   ghcr.io/calabi-inc/rtsm eval my_session.bag --repeats 3

# the headless server: REST + MCP on 8002, the Calabi Lens WebSocket on 8765
docker run --rm --gpus all -v rtsm-models:/models -p 8002:8002 -p 8765:8765 ghcr.io/calabi-inc/rtsm

# with the 3-D dashboard on 8083
docker run --rm --gpus all -v rtsm-models:/models -p 8002:8002 -p 8765:8765 -p 8083:8083 ghcr.io/calabi-inc/rtsm --viz

# the bundled demo (replays a packaged clip; dashboard on 8083)
docker run --rm --gpus all -v rtsm-models:/models -p 8002:8002 -p 8083:8083 ghcr.io/calabi-inc/rtsm demo

# any subcommand works the same way
docker run --rm ghcr.io/calabi-inc/rtsm version
```

The container's working directory is `/data`, so paths in commands are relative to what you mount there; output of `rtsm eval` lands next to the input. Runs inside the image are deterministic (three repeats of the TUM fr1 bag gave identical fingerprints, every floor 0), but a Linux container and a Windows install do not give identical numbers: the same bag ended with 203 objects in the image and 217 on a Windows install, with the same re-identification rate to within 0.3 points. Compare runs made on the same platform. `docker compose -f docker/docker-compose.yml up` runs the server with the same ports and volume. To build the image yourself from a checkout: `docker build -f docker/Dockerfile -t rtsm .`

!!! note "What is not in the image"
    The AGPL backends (`dual`, `fastsam`, `yoloe`) and their weights: `pip install ultralytics` inside the container and mount the weights at `/data/model_store` if you opt in. The Jetson / ARM image is a separate build and is not published yet.

!!! note "What CI verifies"
    Every push and pull request runs the core CPU suite on Linux and Windows (Python 3.12, CPU torch, no model weights, no LFS) and installs the built wheel into a clean environment without torch (`.github/workflows/ci.yml`). The GPU gates (the session1 anchor, the eval runs) stay local.

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
