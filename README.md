# RTSM — Real-Time Spatial Memory for Robots

[![PyPI](https://img.shields.io/pypi/v/rtsm)](https://pypi.org/project/rtsm/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/rtsm/)

![RTSM Demo](repo_media/rtsm-demo-gif.gif)

Turns RGB-D streams into a **persistent, queryable 3D object map** — objects get stable IDs, 3D positions, CLIP embeddings, and semantic labels, updated in real time.

```bash
pip install "rtsm[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
rtsm demo
```

**246 ms/frame** · **74 objects tracked** · **Apache 2.0** · Python 3.12+ · RTX 3080–5090

**[Demo Video](https://youtu.be/abhXsbvOLQg)** · **[Docs](https://calabi-inc.github.io/rtsm)** · **[PyPI](https://pypi.org/project/rtsm/)**

---

## What It Does

- Builds a **live 3D object map** from RGB-D + pose streams (ARKit, RealSense, or recorded sessions)
- Assigns **persistent IDs** to objects across viewpoints and time — not per-frame detection, real tracking
- Stores spatial, semantic, and temporal metadata per object (position, CLIP embedding, label confidence, view history)
- Supports **semantic + spatial queries** (e.g. *"red bin near dock 3"*) via REST API and MCP
- **SLAM-agnostic** — sits above any perception stack that provides RGB-D + pose

---

## Try It

`rtsm demo` runs a pre-recorded 50-frame indoor scene through the full pipeline with 3D visualization:

```bash
rtsm demo              # full pipeline + 3D viewer (opens browser)
rtsm demo --no-viz     # headless — API only at localhost:8002
```

No hardware needed — replay uses a bundled recording.

**Try searching for these objects** (type in the search bar or use the API):
`tissue box` · `doll` · `laptop` · `pillow` · `curtain` · `lamp` · `humidifier`

> `rtsm demo` runs a short 50-frame clip. For the full room sweep (240 frames), clone the repo with `git lfs install && git clone` then run `rtsm --replay recordings/session1`.

**[Watch the full demo on YouTube](https://youtu.be/abhXsbvOLQg)**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 RTSM — Real-Time Spatio-Semantic Memory                  │
└──────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  Calabi Lens     │   │   D435i + SLAM   │   │  Recorded        │
  │  (ARKit iOS)     │   │   (RTABMap)      │   │  Session         │
  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
           │ WebSocket            │ ZeroMQ               │ --replay
           ▼                      ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  I/O Layer                                                               │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  WebSocket  │  │  ZMQ Bridge │  │   Replay     │  │  Recorder    │   │
│  │  Receiver   │  │  (sensors)  │  │  Receiver    │  │  (--record)  │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────────────┘   │
│         └────────────────┴────────────────┘                              │
│                          │                                               │
│                   ┌──────▼───────┐     ┌──────────────┐                  │
│                   │ IngestQueue  │────>│ FramePacket  │                  │
│                   │  (buffer)    │     │ (RGB,D,Pose) │                  │
│                   └──────────────┘     └──────┬───────┘                  │
│                                               │                          │
└───────────────────────────────────────────────┼──────────────────────────┘
                                                │
                          ┌─────────────────────▼───────────────────────┐
                          │              Ingest Gate                    │
                          │   (keyframe priority, sweep-based skip)     │
                          └─────────────────────┬───────────────────────┘
                                                │
┌───────────────────────────────────────────────▼──────────────────────────┐
│  Perception Pipeline                                                     │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐                                   │
│  │ Grounding DINO │  │     SAM2       │    Default: grounded_sam2          │
│  │ (detection +   │─>│ (box-prompted  │    GDINO detects → SAM2 segments   │
│  │  labels)       │  │  masks)        │    (Apache 2.0, no AGPL)           │
│  └────────────────┘  └───────┬────────┘                                   │
│                              │                                            │
│                 ▼                                                        │
│  ┌───────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Mask Staging  │────>│ Top-K Select │────>│ CLIP Encode  │             │
│  │ (heuristics)  │     │  (priority)  │     │(224x224 crop)│             │
│  └───────────────┘     └──────────────┘     └──────┬───────┘             │
│                                                    │                     │
│                     ┌───────────────┐        ┌─────▼────────┐            │
│                     │ Vocab Classify│<───────│  Embeddings  │            │
│                     │ (label + conf)│        │  (512-D L2)  │            │
│                     └───────┬───────┘        └──────────────┘            │
│                             │                                            │
└─────────────────────────────┼────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Association                                                             │
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────────┐               │
│  │  Proximity  │────>│  Embedding  │────>│  Score Fusion │               │
│  │   Query     │     │  Cosine Sim │     │ (match/create)│               │
│  └─────────────┘     └─────────────┘     └───────────────┘               │
│                                                                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Working Memory                                                          │
│                                                                          │
│  ObjectState:                                                            │
│    - id, xyz_world (3D position)                                         │
│    - emb_mean, emb_gallery (CLIP embeddings)                             │
│    - view_bins (multi-view fusion)                                       │
│    - label_scores (EWMA label confidence)                                │
│    - stability, hits, confirmed                                          │
│    - image_crops (JPEG snapshots)                                        │
│                                                                          │
│  Proto -> Confirmed (hits >= 2, stability >= 0.55, views >= 1)           │
│                                                                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Long-Term Memory (FAISS / Milvus)                                       │
│                                                                          │
│  Semantic Search: query(text) -> CLIP -> top-k objects                   │
│                                                                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  API & Visualization                                                     │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │    REST API     │  │    WebSocket    │  │     3D Demo     │           │
│  │    /objects     │  │  point clouds   │  │    (Three.js)   │           │
│  │    /search      │  │  objects_update │  │                 │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### From PyPI (recommended)

```bash
# GPU — permissive license (SAM2 + Grounding DINO, Apache 2.0)
pip install "rtsm[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# GPU — ultralytics backends (FastSAM + YOLOE, AGPL-3.0)
pip install "rtsm[gpu-ultralytics]" --extra-index-url https://download.pytorch.org/whl/cu128

# Everything (GPU + visualization)
pip install "rtsm[all]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### From Source

```bash
git clone https://github.com/calabi-inc/rtsm.git
cd rtsm
pip install ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### Edge / Jetson (ARM)

On NVIDIA Jetson (Orin), `torch` ships from NVIDIA's Jetson wheel index — not PyPI — and the lean `edge` profile runs FastSAM-only (skips the heavy SAM2/GDINO CUDA backends). A helper picks the right torch source per architecture:

```bash
# 1. torch for this machine (x86 → PyTorch CUDA index, Jetson → NVIDIA index)
python scripts/install_torch.py                 # add --jetson-cu cu124 if on JetPack 6.1
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. RTSM + FastSAM-only deps (no torch re-pull, no SAM2/transformers)
pip install -e ".[edge]"

# 3. set `segmentation.backend: fastsam` in config, then run
rtsm --replay recordings/session1
```

> **Jetson note:** targets JetPack 6.x (Python 3.10, CUDA 12.6). Verify your CUDA with `nvcc --version` and pass `--jetson-cu cu124`/`cu126` to match. The `edge` extra installs FastSAM + YOLOE + SigLIP + FAISS only.

### Download Models

```bash
python scripts/fetch_models.py                # all default models (SAM2, GDINO, CLIP)
python scripts/fetch_models.py --only sam2    # or individually
```

> **License note:** `rtsm[gpu]` uses only Apache 2.0 / MIT dependencies. `rtsm[gpu-ultralytics]` adds the `ultralytics` package (AGPL-3.0) for FastSAM and YOLOE backends.
>
> **CUDA version:** Use `cu128` for most GPUs (RTX 3080–5090). For Blackwell-only features use `cu130`. On **Jetson/ARM**, torch comes from NVIDIA's index — use `python scripts/install_torch.py` (see [Edge / Jetson](#edge--jetson-arm)) instead of `--extra-index-url`. See [PyTorch install](https://pytorch.org/get-started/locally/) for other options.

---

## Usage

### Live — iPhone (ARKit over WebSocket)

```bash
rtsm                   # starts pipeline + API + visualization
```

### Live — RealSense D435i + RTAB-Map

```bash
# Set io.receiver: zeromq in config/rtsm.yaml first
rtsm
```

### Replay a Recorded Session

```bash
rtsm --replay recordings/session1
```

### Record & Replay

```bash
# Record only (no GPU needed — works with core-only install)
rtsm --record recordings/my_session --record-only

# Record while running pipeline
rtsm --record recordings/my_session

# Replay at original rate
rtsm --replay recordings/my_session
```

Recordings are self-contained directories with raw WebSocket data. Replay feeds the exact same bytes through the full pipeline, preserving all time-dependent behavior.

### API

```bash
curl http://localhost:8000/objects                                    # list all objects
curl "http://localhost:8000/search/semantic?query=red%20mug&top_k=5"  # semantic search
curl http://localhost:8000/stats/detailed                             # system stats
curl http://localhost:8000/stats/analytics                            # runtime analytics
```

---

## Segmentation Backends

RTSM supports multiple segmentation backends via `segmentation.backend` in `config/rtsm.yaml`:

| Backend | License | Description | Seg time* | Pipeline total* | Labels |
|---------|---------|-------------|-----------|-----------------|--------|
| `grounded_sam2` | Apache 2.0 | Grounding DINO detect + SAM2 segment | 217 ms | 531 ms | Open-vocab (text-prompted) |
| `sam2` | Apache 2.0 | SAM2 auto-mask (segment everything) | ~860 ms | ~1000 ms | None (class-agnostic) |
| `fastsam` | AGPL-3.0 | FastSAM (segment everything) | ~50 ms | ~200 ms | None (class-agnostic) |
| `yoloe` | AGPL-3.0 | YOLOE detection + segmentation | ~60 ms | ~210 ms | Open-vocab / 1200+ built-in |
| `dual` | AGPL-3.0 | FastSAM + YOLOE with IoU merge | 100 ms | 246 ms | Dual-confirmed labels |

*Mean on RTX 5090, 640x480 input.*

**Default:** `grounded_sam2` — permissive license, open-vocabulary, no AGPL dependency.

```yaml
segmentation:
  backend: grounded_sam2    # or: sam2, fastsam, yoloe, dual
```

> `fastsam`, `yoloe`, and `dual` require `pip install "rtsm[gpu-ultralytics]"`.

---

## Performance

Benchmarked on RTX 5090 (32 GB), iPhone ARKit recording (240 frames, 76s indoor scene), 640x480 RGB input.

| Metric | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|--------|------------------------|------------------------------|
| **Mean latency** | **246 ms** | **531 ms** |
| P50 latency | 213 ms | 486 ms |
| P95 latency | 604 ms | 942 ms |
| Masks/frame | 25.7 | 11.3 |
| Objects confirmed | 74 | 42 |
| Confirmation rate | 65.4% | 59.2% |
| License | AGPL-3.0 | Apache-2.0 |

> Full breakdown: **[Benchmarks](https://calabi-inc.github.io/rtsm/benchmarks/)** | [`reports/backend_comparison.md`](reports/backend_comparison.md)

---

## Configuration

See [`config/rtsm.yaml`](config/rtsm.yaml) for full configuration options:

- **Camera intrinsics** — focal length, resolution
- **I/O endpoints** — ZeroMQ addresses for camera and SLAM
- **Pipeline tuning** — mask filtering, association thresholds
- **Memory settings** — object promotion, expiry, vector store

---

## Project Structure

```
rtsm/
├── core/           # Pipeline, association, ingest gate, data models
├── models/         # SAM2, Grounding DINO, FastSAM, YOLOE, CLIP adapters
├── stores/         # Working memory, proximity index, sweep cache, vector stores
├── io/             # WebSocket + ZeroMQ receivers, recorder, replayer
├── analytics/      # Runtime analytics (latency, segmentation, congestion)
├── api/            # REST API server (FastAPI)
├── visualization/  # WebSocket server, TSDF fusion, 3D demo
└── utils/          # Mask staging, transforms, helpers
config/
├── rtsm.yaml       # Main configuration
└── clip/vocab.yaml  # CLIP vocabulary
scripts/
├── fetch_models.py          # Download models
├── debug_segmentation.py    # A/B segmentation viewer
└── benchmark_backends.py    # Backend benchmark
```

---

## Roadmap

- [x] Dual-confirmation segmentation (FastSAM + YOLOE)
- [x] AGPL-clean default (SAM2 + Grounding DINO, Apache 2.0)
- [x] YOLOE prompt-free (1200+ LVIS categories)
- [x] WebSocket receiver for Calabi Lens (ARKit iOS)
- [x] Record/replay system for offline testing
- [x] A/B segmentation debug tooling
- [x] Real-time analytics dashboard
- [x] Agent interface (MCP — 6 tools via SSE)
- [ ] Evaluation framework (ArUco ground truth, precision/recall)
- [ ] More protocols (ROS 2, gRPC)
- [ ] LLM integration for high-level queries
- [ ] Docker image

---

## Acknowledgments

RTSM builds on excellent open-source work:

- **SAM 2** — Ravi et al., 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) · [GitHub](https://github.com/facebookresearch/sam2)
- **Grounding DINO** — Liu et al., 2023. [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
- **FastSAM** — Zhao et al., 2023. [arXiv:2306.12156](https://arxiv.org/abs/2306.12156) · [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)
- **YOLOE** — THU-MIG, ICCV 2025. [GitHub](https://github.com/THU-MIG/yoloe) · [Ultralytics](https://docs.ultralytics.com/models/yoloe/)
- **CLIP** — Radford et al., 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [GitHub](https://github.com/openai/CLIP)
- **RTAB-Map** — Labb&eacute; & Michaud, 2019. [Paper](https://doi.org/10.1002/rob.21831) · [GitHub](https://github.com/introlab/rtabmap)

---

## Cite

```bibtex
@software{chang2025rtsm,
  author       = {Chang, Chi Feng},
  title        = {{RTSM}: Real-Time Spatio-Semantic Memory},
  year         = {2025},
  url          = {https://github.com/calabi-inc/rtsm},
  note         = {Object-centric queryable memory for spatial AI and robotics}
}
```

---

## License

Apache-2.0

---

Built by [Chi Feng, Chang](https://github.com/vipipi)
