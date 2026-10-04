# RTSM — Real-Time Spatial Memory for Robots

[![CI](https://github.com/calabi-inc/rtsm/actions/workflows/ci.yml/badge.svg)](https://github.com/calabi-inc/rtsm/actions/workflows/ci.yml)
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
- **Detector-agnostic** — runs on its own segmenters or on your detector's `vision_msgs` detections from a bag (`rtsm eval --set segmentation.backend=external`), and `rtsm eval` reports what the memory did with them

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

> `rtsm demo` runs a short 50-frame clip. For the full room sweep (240 frames), clone the repo with `git lfs install && git clone` then run `rtsm --viz --replay recordings/session1`.

**[Watch the full demo on YouTube](https://youtu.be/abhXsbvOLQg)**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RTSM — Real-Time Spatio-Semantic Memory: the path of one frame              │
└──────────────────────────────────────────────────────────────────────────────┘

 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ Calabi Lens    │  │ RealSense+SLAM │  │ ROS bag        │  │ Recording      │
 │ (ARKit iPhone) │  │ (RTAB-Map ...) │  │ ROS 1, rosbag2,│  │ (Lens session) │
 │                │  │                │  │ or MCAP        │  │                │
 │ WebSocket      │  │ ZeroMQ         │  │ `--bag`        │  │ `--replay`     │
 └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
          │                   │                   │                   │
          ▼                   ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  I/O layer  (rtsm/io)                                                        │
│                                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ Transport adapters   │  │ Ingest front-end     │  │ Ingest lanes         │ │
│ │ bytes -> RawFrame    │  │ sensor clock, pairing│  │ `latest`   (live)    │ │
│ │ (one per source)     │  │ keyframes, throttle, │  │ `lossless` (replay)  │ │
│ │                      │  │ admit before decode  │  │ bounded memory       │ │
│ └──────────────────────┘  └──────────────────────┘  └──────────────────────┘ │
│                                                                              │
│  ->  FramePacket: RGB, depth, intrinsics, pose, external detections          │
│      (the Recorder taps the raw stream: `--record`)                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │  same admission chain for every source
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Ingest gate                                                                 │
│                                                                              │
│  keyframes first; non-keyframes skipped while the camera sweeps;             │
│  dark, flat or depth-less frames rejected before any model runs              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Perception pipeline  (rtsm/core/pipeline.py)                                │
│                                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ Segment              │  │ Filter               │  │ Score + top-K        │ │
│ │ backend-swappable:   │  │ mask heuristics:     │  │ rank by coverage,    │ │
│ │ default grounded_sam2│  │ area, border contact,│  │ depth quality,       │ │
│ │ sam2 | dual | fastsam│  │ depth validity,      │  │ structure; keep K    │ │
│ │ yoloe | external     │  │ planarity            │  │ (bounds CLIP work)   │ │
│ └──────────────────────┘  └──────────────────────┘  └──────────────────────┘ │
│                                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ CLIP encode          │  │ Classify             │  │ `external` backend   │ │
│ │ 224x224 crop per mask│  │ cosine vs vocabulary │  │ your detector's boxes│ │
│ │ -> 512-D embedding   │  │ -> label + confidence│  │ stand in for Segment │ │
│ └──────────────────────┘  └──────────────────────┘  └──────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │  mask, 3-D point, embedding, label
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Association  (rtsm/core/associator.py)                                      │
│                                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ Proximity query      │  │ Embedding match      │  │ Fusion               │ │
│ │ nearby objects from  │  │ cosine similarity,   │  │ match an object, or  │ │
│ │ the spatial grid     │  │ gate 0.90            │  │ create a proto       │ │
│ └──────────────────────┘  └──────────────────────┘  └──────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Working memory  (rtsm/core/working_memory.py)                               │
│                                                                              │
│  ObjectState: id, xyz_world, emb_mean + gallery, view_bins, label scores,    │
│               stability, hits, confirmed, JPEG snapshots                     │
│                                                                              │
│  proto  ->  confirmed   (hits >= 2, stability >= 0.55, >= 1 view bin)        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │  confirmed objects, periodically
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Long-term memory  (FAISS, or Milvus)                                        │
│                                                                              │
│  semantic search:  text -> CLIP -> top-k objects                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Outputs                                                                     │
│                                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ REST API             │  │ MCP                  │  │ WebSocket + viewer   │ │
│ │ /objects, /search/*, │  │ tools for agents     │  │ `--viz`: 3-D clouds, │ │
│ │ /stats, /healthz     │  │ (SSE and stdio)      │  │ object updates       │ │
│ └──────────────────────┘  └──────────────────────┘  └──────────────────────┘ │
│                                                                              │
│  Diagnostics: the frame-flow trace and the pose / obs / view ledgers         │
│  feed `rtsm eval` -> metrics.json + report.md  (offline, on a bag)           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

> What changed since the April release: [CHANGELOG.md](CHANGELOG.md). Current version: `rtsm version`.

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
rtsm                   # headless: pipeline + REST API (+ MCP)
rtsm --viz             # plus the 3D dashboard (opens the browser)
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

> Full breakdown: **[Benchmarks](https://calabi-inc.github.io/rtsm/benchmarks/)**

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
├── core/           # Pipeline, association, ingest + frame-quality gates, sensor clock
├── models/         # SAM2, Grounding DINO, FastSAM, YOLOE, CLIP adapters
├── stores/         # Working memory, proximity index, sweep cache, vector stores
├── io/             # Ingest front-end; websocket / ZeroMQ / replay / bag sources; codecs; recorder; MCP
├── evaluation/     # Diagnostics event log, ledgers, `rtsm eval` runner, metrics, report
├── analytics/      # Runtime analytics (latency, segmentation, congestion)
├── api/            # REST API server (FastAPI)
├── visualization/  # Dashboard server, TSDF fusion, 3D demo
├── cfg/            # Packaged configuration (rtsm.yaml, demo_config.yaml), CLIP vocabulary, tuning controls
├── client.py       # Python client for the REST API (the SDK)
├── engine.py       # Model loading + runtime construction shared by every runner
└── utils/          # Mask staging, transforms, helpers
scripts/
├── fetch_models.py          # Download models
├── debug_segmentation.py    # A/B segmentation viewer
├── benchmark_backends.py    # Backend benchmark
└── recording_to_mcap.py     # Lens recording -> rosbag2 / MCAP
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
- [x] Diagnostics ledgers + `rtsm eval` (ROS 1 / rosbag2 / MCAP bags, headless runs, a report with same-input floors)
- [x] Python client (`rtsm.client`)
- [x] Your own detector's `vision_msgs` detections through the memory (`segmentation.backend: external`)
- [ ] More protocols (ROS 2 live node, gRPC)
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
