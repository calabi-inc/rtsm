# Architecture

RTSM processes RGB-D frames through a pipeline that extracts, tracks, and stores objects in a queryable spatial memory.

---

## System Overview

```
RGB-D Sensor + SLAM (Pose)
         |
         | ZeroMQ
         v
    I/O Layer
    ZMQ Bridge -> IngestQueue -> FramePacket
         |
         v
  Perception Pipeline
    FastSAM -> Mask Staging -> Top-K Select -> CLIP Encode -> Vocab Classify
         |
         v
   Association
    Proximity Query -> Embedding Cosine Sim -> Score Fusion (match/create)
         |
         v
      Memory
    Working Memory -> Long-Term Memory (FAISS)
         |
         v
  API & Visualization
    REST API | WebSocket | 3D Demo
```

---

## Components

### I/O Layer

Receives RGB-D frames and camera poses via ZeroMQ. Frames are buffered in an ingest queue with keyframe gating to throttle processing (30 Hz → 5-7 Hz).

### Perception Pipeline

1. **FastSAM** — Segments the RGB image into object masks
2. **Mask Staging** — Filters by area (rejects too small/large)
3. **Top-K Select** — Limits masks per frame for processing budget
4. **CLIP Encode** — Extracts 512-dim embedding from each mask crop
5. **Vocab Classify** — Assigns labels via cosine similarity to text embeddings

### Association

Matches new observations to existing objects:

1. **Proximity Query** — Find nearby objects in 3D space
2. **Embedding Similarity** — Compare CLIP vectors
3. **Score Fusion** — Weighted combination → match or create new

### Memory

- **Working Memory** — Active object states (position, embeddings, view history)
- **Long-Term Memory** — Confirmed objects indexed in FAISS for semantic search

### API Layer

- **REST API** — Query objects, search by text, get stats
- **WebSocket** — Stream point clouds and object updates
- **3D Demo** — Three.js visualization

---

## Data Flow

```
Frame → Segment → Encode → Associate → Update Memory → Index → Query
```

Each frame takes <30ms end-to-end on RTX 5090.

---

## Next Steps

- [Perception Pipeline](perception-pipeline.md) — Deep dive into segmentation and encoding
- [Memory Model](memory-model.md) — How objects are tracked and promoted
