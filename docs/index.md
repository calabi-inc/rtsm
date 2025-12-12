# RTSM — Real-Time Spatio-Semantic Memory

**Object-centric queryable memory for spatial AI and robotics.**

RTSM builds a persistent, searchable memory of objects in 3D space from RGB-D camera streams. Ask natural language queries like *"Where is the red mug?"* and get answers grounded in real-world coordinates.

<div style="text-align: center; margin: 2rem 0;">
  <a href="https://youtu.be/abhXsbvOLQg" target="_blank">
    <strong>▶ Watch Demo Video</strong>
  </a>
</div>

---

## Features

- **Real-time segmentation** — FastSAM extracts object instances from each frame
- **Semantic embeddings** — CLIP encodes visual features for natural language queries
- **Persistent memory** — Objects are tracked across views, fused, and promoted to long-term memory
- **Spatial indexing** — Fast proximity queries via 3D grid + vector search (FAISS)
- **Queryable** — REST API and semantic search: find objects by description

```json
// "Where is the red backpack?"
{ "id": "a3f2c1", "xyz": [1.2, 0.4, 2.1], "confidence": 0.87 }
```

---

## Quick Links

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — Get RTSM running
- :material-rocket-launch: **[Quick Start](getting-started/quick-start.md)** — Your first query in 5 minutes
- :material-cog: **[Configuration](getting-started/configuration.md)** — Tune for your setup
- :material-api: **[REST API](api/rest-api.md)** — API reference

</div>

---

## Performance

*Benchmarks on RTX 5090:*

| Stage | Metric |
|-------|--------|
| Input throttling | 30 Hz raw → 5–7 Hz processed |
| Proto-object yield | >90% of static masks accumulate |
| Frame latency | <30 ms (FastSAM + CLIP) |
| LTM upsert rate | 5 s default interval |

---

## License

Apache-2.0 — See [GitHub](https://github.com/calabi-inc/rtsm) for details.
