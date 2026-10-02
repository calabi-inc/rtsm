# RTSM — Real-Time Spatio-Semantic Memory

**Object-centric queryable memory for spatial AI and robotics.**

RTSM builds a persistent, searchable memory of objects in 3D space from RGB-D camera streams. Ask natural language queries like *"Where is the red mug?"* and get answers grounded in real-world coordinates.

<div style="text-align: center; margin: 2rem 0;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/abhXsbvOLQg" frameborder="0" allowfullscreen></iframe>
</div>

---

## Why RTSM

Vision models can detect objects. SLAM systems can map geometry. Language models can reason abstractly. But none of them **remember where things are**.

RTSM is the missing layer between perception and reasoning:

- SLAM provides geometry and poses
- Vision models provide object masks and semantics
- **RTSM fuses them into a persistent, queryable world state**

This makes spatial state inspectable, queryable, and reusable across robots, agents, and applications — regardless of which segmentation model or SLAM system you use.

---

## Features

- **Model-agnostic** — Swappable segmentation backends (CNN or transformer, permissive or AGPL)
- **Real-time** — 210 ms mean pipeline latency (dual backend, RTX 5090)
- **Persistent memory** — Objects tracked across views with stable IDs, promoted from proto to confirmed
- **Semantic search** — Find objects by natural language via CLIP embeddings + FAISS
- **Spatial search** — Find objects near 3D world coordinates or relative to other objects
- **MCP integration** — AI agents (Claude, Cursor, LangGraph) can query spatial memory via Model Context Protocol
- **Record & replay** — Capture live sessions for offline benchmarking and reproducible testing
- **Runtime analytics** — Per-stage latency, segmentation rates, and throughput dashboards
- **Queryable API** — REST endpoints for objects, search, stats, and analytics
- **Offline evaluation** — `rtsm eval` runs the pipeline on ROS 1 / rosbag2 / MCAP bags or recordings and writes a report with same-input floors; the ledgers are an open, documented format
- **Bring your own detector** — a `vision_msgs` detections topic in the bag becomes the memory's input; the report names the detector
- **Python client** — `rtsm.client` talks to the REST API without the perception stack installed

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
- :material-chart-bar: **[Benchmarks](benchmarks.md)** — Performance data
- :material-robot: **[MCP (AI Agents)](api/mcp.md)** — Connect AI agents to spatial memory

</div>

---

## Performance at a Glance

*Measured on RTX 5090, iPhone ARKit recording (162 frames, 458s indoor scene). [Full benchmarks](benchmarks.md).*

| Metric | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|--------|------------------------|------------------------------|
| Mean latency | 210 ms | 510 ms |
| P95 latency | 509 ms | 721 ms |
| Masks/frame | 28.8 | 13.4 |
| Objects confirmed | 60 | 35 |
| License | AGPL-3.0 | Apache-2.0 |

RTSM's 10-stage pipeline is backend-agnostic — swap between CNN and transformer segmenters with a single config change, same memory layer, same API.

---

## License

Apache-2.0 — See [GitHub](https://github.com/calabi-inc/rtsm) for details.
