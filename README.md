![RTSM](repo_media/background.jpg)

# RTSM — Real-Time Spatio-Semantic Memory

RTSM is a real-time spatial memory system that turns RGB-D streams into a **persistent, queryable 3D object-centric world state**.

Instead of treating perception as disposable frames, RTSM maintains **stable object identities over time**, enabling robots and embodied agents to answer questions like:
- *What objects exist in this space?*
- *Where are they right now?*
- *What changed, and when?*

**[Watch Demo Video](https://youtu.be/abhXsbvOLQg)** · **[Documentation](https://calabi-inc.github.io/rtsm)**

---

## Why RTSM

Modern perception systems can detect objects, but they lack **memory**.
SLAM systems build geometry, vision models detect semantics, and language models reason abstractly—but there is no shared layer that connects **space, objects, and history**.

RTSM fills this gap by acting as an explicit **spatial memory layer**:
- SLAM provides geometry and poses
- Vision models provide object masks and semantics
- RTSM fuses them into a persistent world representation

This makes spatial state **inspectable, queryable, and reusable** across robots, agents, and applications.

---

## What RTSM Does

- Builds a live 3D map from RGB-D + pose streams
- Assigns **persistent IDs** to objects across viewpoints and time
- Stores spatial, semantic, and temporal metadata per object
- Supports semantic + spatial queries (e.g. *"red bin near dock 3"*)
- Exposes a programmatic API and real-time 3D visualizer

RTSM is **SLAM-agnostic** and designed to sit above existing perception stacks.

---

## Who This Is For

- Robotics and embodied AI researchers
- Developers building agentic or world-model-based systems
- Teams exploring persistent perception, spatial reasoning, or digital twins

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 RTSM — Real-Time Spatio-Semantic Memory                  │
└──────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────────┐
                            │   RGB-D Sensor   │
                            │   + SLAM (Pose)  │
                            └────────┬─────────┘
                                     │ ZeroMQ
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  I/O Layer                                                               │
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌──────────────┐                │
│  │ ZMQ Bridge  │────>│ IngestQueue │────>│ FramePacket  │                │
│  │  (sensors)  │     │  (buffer)   │     │ (RGB,D,Pose) │                │
│  └─────────────┘     └─────────────┘     └──────┬───────┘                │
│                                                 │                        │
└─────────────────────────────────────────────────┼────────────────────────┘
                                                  │
                            ┌─────────────────────▼───────────────────────┐
                            │              Ingest Gate                    │
                            │   (keyframe priority, sweep-based skip)     │
                            └─────────────────────┬───────────────────────┘
                                                  │
┌─────────────────────────────────────────────────▼────────────────────────┐
│  Perception Pipeline                                                     │
│                                                                          │
│  ┌────────────┐     ┌───────────────┐     ┌──────────────┐               │
│  │  FastSAM   │────>│ Mask Staging  │────>│ Top-K Select │               │
│  │ (segment)  │     │ (heuristics)  │     │  (priority)  │               │
│  └────────────┘     └───────────────┘     └──────┬───────┘               │
│                                                  │                       │
│                     ┌───────────────┐     ┌──────▼───────┐               │
│                     │ Vocab Classify│<────│ CLIP Encode  │               │
│                     │ (label + conf)│     │(224x224 crop)│               │
│                     └───────┬───────┘     └──────────────┘               │
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
│  Proto -> Confirmed (hits >= 2, stability >= 0.5, views >= 2)            │
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

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (tested on RTX 3080)
- RGB-D camera (Intel RealSense D435i tested)
- SLAM system providing poses (RTAB-Map, ORB-SLAM3)

> **Tested on:** WSL2 Ubuntu 22.04 with RTAB-Map.
> Note: WSL2 has USB passthrough limitations — you may need [usbipd-win](https://github.com/dorssel/usbipd-win) for camera access.

### Installation

```bash
git clone https://github.com/calabi-inc/rtsm.git
cd rtsm
# Install dependencies
pip install -r requirements.txt

# Install RTSM in editable mode
pip install -e .
```

### Download Models

```bash
# FastSAM weights
mkdir -p model_store/fastsam
# Download FastSAM-x.pt to model_store/fastsam/

# CLIP weights (auto-downloaded on first run)
python scripts/fetch_clip.py
```

### Run

```bash
# Start RTSM (expects RGB-D + pose stream via ZeroMQ)
python -m rtsm.run
```

RTSM will start:
- **Perception pipeline** — processing frames
- **REST API** — `http://localhost:8000`
- **Visualization WebSocket** — `ws://localhost:8081`

### API Examples

```bash
# List all objects
curl http://localhost:8000/objects

# Semantic search
curl "http://localhost:8000/search/semantic?query=red%20mug&top_k=5"

# Get system stats
curl http://localhost:8000/stats/detailed
```

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
├── core/           # Pipeline, association, data models
├── models/         # FastSAM, CLIP adapters
├── stores/         # Working memory, proximity index, vector stores
├── io/             # ZeroMQ ingestion, frame buffering
├── api/            # REST API server
├── visualization/  # WebSocket server, 3D demo
└── utils/          # Helpers, transforms
```

---

## Performance

*Benchmarks on RTX 5090 (your mileage may vary):*

| Stage | Metric |
|-------|--------|
| Input throttling | 30 Hz raw → 5–7 Hz processed (keyframe gating) |
| Mask filtering | Heuristic filter rejects 10–15% area masks as insignificant |
| Proto-object yield | >90% of static object masks successfully accumulate via associator |
| Frame latency | <30 ms end-to-end (FastSAM + CLIP stack) |
| LTM upsert rate | 5 s default interval |

---

## Roadmap

- [ ] More adapters (YOLO-World, ORB-SLAM3)
- [ ] Direct plugin for Isaac Sim
- [ ] More communication protocols (ROS 2, MQTT, Kafka)
- [ ] LLM integration for high-level queries (agentic mode)
- [ ] Dockerization

---

## Acknowledgments

RTSM builds on excellent open-source work:

- **FastSAM** — Zhao et al., *Fast Segment Anything*, 2023.
  [arXiv:2306.12156](https://arxiv.org/abs/2306.12156) · [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)

- **CLIP** — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.
  [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [GitHub](https://github.com/openai/CLIP)

- **RTAB-Map** — Labbé & Michaud, *RTAB-Map as an Open-Source Lidar and Visual SLAM Library for Large-Scale and Long-Term Online Operation*, Journal of Field Robotics, 2019.
  [Paper](https://doi.org/10.1002/rob.21831) · [GitHub](https://github.com/introlab/rtabmap)

---

## Cite This

If you use RTSM in your research, please cite:

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

## Community

This project is under active development. If you have questions or run into issues, feel free to open an issue — I'm happy to help.

If you find RTSM useful, please consider giving it a star! I'm also looking for design partners — reach out to [Calabi](https://github.com/calabi-team) if you're interested in collaborating.

---

## License

Apache-2.0

---

## Author

Built by [Chi Feng, Chang](https://github.com/vipipi)
