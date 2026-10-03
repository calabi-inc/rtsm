# Architecture

RTSM processes RGB-D frames through a 10-stage pipeline that extracts, tracks, and stores objects in a queryable spatial memory. The system is **segmentation-model-agnostic** — any backend that produces instance masks can feed the pipeline.

---

## System Overview

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

## Components

### I/O Layer

Receives RGB-D frames and camera poses from multiple sources:

- **WebSocket** — Calabi Lens (ARKit, iPhone)
- **ZeroMQ** — Intel RealSense D435i + RTAB-Map
- **Replay** — Recorded sessions for deterministic benchmarking
- **Bag** — ROS 1 `.bag`, rosbag2 and MCAP files (`--bag`, `rtsm eval`), with an optional external-detections topic

Each source is a transport adapter that turns bytes into a `RawFrame` (still-encoded payloads + header) or a pose event; the **ingest front-end** (`rtsm/io/ingest_frontend.py`) then runs the one admission chain for all of them — tracking filter, receive-time pose mailbox, keyframe rule, non-keyframe throttle on the ingest clock, lane admission before any pixel is decoded, decode on admit, `FramePacket`, frame-flow trace. New sources register by name (`rtsm/io/sources.py`, the `rtsm.sources` entry-point group); see the [Ingest Sources guide](../guides/ingest-sources.md).

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
- **3D Demo** — Three.js visualization (opt-in: `--viz`; per-keyframe clouds by default, TSDF fusion opt-in)

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
