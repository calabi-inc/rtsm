# Architecture

RTSM processes RGB-D frames through a 10-stage pipeline that extracts, tracks, and stores objects in a queryable spatial memory. The system is **segmentation-model-agnostic** — any backend that produces instance masks can feed the pipeline.

---

## System Overview

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Calabi Lens     │   │   D435i + SLAM   │   │  Recorded        │
│  (ARKit iOS)     │   │   (RTABMap)      │   │  Session         │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │ WebSocket            │ ZeroMQ               │ --replay
         ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  I/O Layer                                                      │
│  WebSocket / ZMQ / Replay → ingest lanes → FramePacket          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    Ingest Gate      │
                    │ (keyframe priority, │
                    │  sweep-based skip)  │
                    └─────────┬──────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Perception Pipeline                                             │
│                                                                  │
│  Segmentation → Heuristics → Scoring → CLIP Encode → Vocab      │
│  (swappable)   (depth,       (top-K)   (ViT-B/32)   Classify    │
│                 border,                                          │
│                 planarity)                                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Association                                                     │
│  Proximity Query → Embedding Cosine Sim → Match / Create         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Working Memory                                                  │
│  Proto-Objects → Confirmed Objects (hits, stability, view bins)  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Long-Term Memory (FAISS / Milvus)                               │
│  Semantic search: query(text) → CLIP → top-k objects             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  API & Visualization                                             │
│  REST API  |  MCP (agents)  |  WebSocket  |  3D Demo (Three.js)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### I/O Layer

Receives RGB-D frames and camera poses from multiple sources:

- **WebSocket** — Calabi Lens (ARKit, iPhone)
- **ZeroMQ** — Intel RealSense D435i + RTAB-Map
- **Replay** — Recorded sessions for deterministic benchmarking

Frames wait in the **ingest lanes** (`rtsm/io/ingest_lanes.py`; `ingest.policy`): live, a small keyframe FIFO drained first plus one non-keyframe slot a newer frame supersedes (`latest`); under replay and evaluation a producer-paced FIFO that never drops (`lossless`); the previous 512-deep tail-drop `IngestQueue` remains as the `legacy` rollback. The **Ingest Gate** selects which frames to process based on keyframe priority and sweep-cache novelty, throttling 30 Hz input to ~1-5 Hz processing.

### Perception Pipeline

1. **Segmentation** — Extract instance masks from RGB (backend-swappable, see below)
2. **Heuristics** — Filter masks by area, border contact, depth validity, planarity
3. **Scoring** — Rank surviving masks by priority (coverage, depth quality, structure)
4. **Top-K Selection** — Limit to 15 candidates per frame (bounds CLIP compute)
5. **CLIP Encode** — 224x224 crop → ViT-B/32 → 512-dim embedding
6. **Vocab Classify** — Cosine similarity to text embeddings → label + confidence

### Segmentation Backends

The segmentation stage is a pluggable adapter. RTSM ships with five backends:

| Backend | Architecture | License | Mean seg time |
|---------|-------------|---------|---------------|
| `grounded_sam2` (default) | Transformer (Swin + Hiera ViT) | Apache-2.0 | 222 ms |
| `sam2` | Transformer (Hiera ViT) | Apache-2.0 | ~860 ms |
| `dual` | CNN (YOLOv8) | AGPL-3.0 | 116 ms |
| `fastsam` | CNN (YOLOv8) | AGPL-3.0 | ~50 ms |
| `yoloe` | CNN (YOLOv8) | AGPL-3.0 | ~60 ms |

The pipeline stages downstream of segmentation (heuristics, CLIP, association, memory) are identical regardless of backend. See [Benchmarks](../benchmarks.md) for measured performance.

### Association

Matches new observations to existing objects in working memory:

1. **Proximity Query** — Find nearby objects via spatial grid index
2. **Embedding Similarity** — Cosine similarity of CLIP vectors (threshold: 0.90)
3. **Score Fusion** — Weighted combination → match existing or create new proto

### Working Memory

Holds `ObjectState` records with position, embeddings, view history, and labels. Objects follow a lifecycle:

```
New observation → Proto-object → Confirmed object → Long-term memory
```

Promotion requires repeated observation (hits >= 2), embedding stability, and multi-view coverage.

### Long-Term Memory

Confirmed objects are periodically upserted to FAISS (or Milvus) for semantic search. Text queries are encoded via CLIP and matched against stored embeddings.

### API Layer

- **REST API** — Query objects, semantic search, stats, analytics
- **MCP** — Model Context Protocol interface for AI agents
- **WebSocket** — Real-time point clouds and object updates
- **3D Demo** — Three.js visualization with TSDF mesh fusion

---

## Data Flow

Each frame passes through the full pipeline:

```
Frame → Gate → Segment → Filter → Score → Encode → Associate → Update → Index
```

Measured end-to-end latency: **210 ms** (dual) / **510 ms** (grounded_sam2) on RTX 5090.

---

## Next Steps

- [Perception Pipeline](perception-pipeline.md) — Deep dive into segmentation and encoding
- [Memory Model](memory-model.md) — How objects are tracked and promoted
- [Benchmarks](../benchmarks.md) — Full performance data
