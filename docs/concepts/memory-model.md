# Memory Model

RTSM maintains a two-tier memory system: **Working Memory** for active tracking and **Long-Term Memory** for persistent, queryable storage.

---

## Object Lifecycle

```
Observation → Proto-Object → Confirmed Object → Long-Term Memory
```

---

## Working Memory

Working memory holds `ObjectState` records for all tracked objects:

```python
ObjectState:
  id: str                    # unique identifier
  xyz_world: [x, y, z]       # 3D position in world frame
  emb_mean: [512]            # running mean of CLIP embeddings
  emb_gallery: [[512], ...]  # gallery of view embeddings
  view_bins: set             # viewpoint directions seen
  label_scores: dict         # EWMA label confidences
  stability: float           # embedding consistency score
  hits: int                  # observation count
  confirmed: bool            # promotion status
  image_crops: [bytes, ...]  # JPEG snapshots
  last_seen: timestamp
```

### Proto-Objects

New observations create **proto-objects** — tentative entries that may be noise or transient.

### Promotion Criteria

Proto-objects become **confirmed** when:

| Criterion | Default |
|-----------|---------|
| `hits` | ≥ 2 |
| `stability` | ≥ 0.5 |
| `view_bins` | ≥ 2 viewpoints |

This filters out:

- Single-frame noise
- Inconsistent detections
- Objects seen from only one angle

---

## Association

When a new observation arrives, we find the best matching object:

1. **Spatial proximity** — Query objects within radius (default: 0.3m)
2. **Embedding similarity** — Cosine similarity of CLIP vectors
3. **Score fusion** — Weighted combination

```python
score = α * spatial_score + (1-α) * embedding_score

if score > threshold:
    match → update existing object
else:
    create new proto-object
```

### Update on Match

When matched, the object state is updated:

- Position: Exponential moving average
- Embedding: Added to gallery, mean updated
- Hits: Incremented
- Stability: Recalculated from embedding variance

---

## Long-Term Memory

Confirmed objects are periodically upserted to **Long-Term Memory** (default: every 5 seconds).

### Storage

- **FAISS** (default) — Local vector index
- **Milvus** (optional) — Distributed vector database

### Indexed Fields

| Field | Purpose |
|-------|---------|
| `emb_mean` | Semantic search |
| `xyz_world` | Spatial queries |
| `label` | Filtered search |

### Semantic Search

Text queries are encoded via CLIP and matched against stored embeddings:

```
"red mug" → CLIP → query vector → FAISS top-k → ranked objects
```

---

## Expiry

Objects not seen for a configurable duration are:

1. Marked as stale in working memory
2. Retained in long-term memory (queryable but flagged)

This handles:

- Objects that moved
- Objects removed from scene
- Temporary occlusions

---

## Next Steps

- [REST API](../api/rest-api.md) — Query the memory
- [Architecture](architecture.md) — Full system overview
