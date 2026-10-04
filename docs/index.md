---
title: RTSM
hide:
  - navigation
  - toc
---

<div class="rtsm-hero" markdown>

<img class="rtsm-hero__mark" src="assets/logo.png" alt="">

# Real-Time Spatio-Semantic Memory

<p class="rtsm-hero__tagline">A persistent, queryable memory of the objects a robot has seen, built from RGB-D frames and poses. Ask <em>"where is the red mug?"</em> and get world coordinates back. Apache-2.0.</p>

[Install](getting-started/installation.md){ .md-button .md-button--primary }
[Evaluate a bag](guides/eval.md){ .md-button }
[GitHub](https://github.com/calabi-inc/rtsm){ .md-button }

</div>

<img class="rtsm-demo" src="https://raw.githubusercontent.com/calabi-inc/rtsm/main/repo_media/rtsm-demo-gif.gif" alt="RTSM placing objects in 3-D as a handheld camera moves through a room">

## The missing layer

Vision models detect objects. SLAM maps geometry. Language models reason. None of them remember where things are. RTSM sits between perception and reasoning: SLAM supplies poses, a segmentation model supplies masks and labels, and RTSM fuses them into a world state that stays inspectable, queryable and reusable across robots, agents and applications, whichever model or SLAM you use.

<div class="grid cards" markdown>

-   :material-cube-scan:{ .lg .middle } **Persistent object memory**

    ---

    Objects keep stable ids across views, promoted from proto to confirmed on repeated, consistent observation; a long-term index outlives the session.

-   :material-magnify:{ .lg .middle } **Semantic and spatial search**

    ---

    Natural-language queries through CLIP embeddings and FAISS; coordinate and radius queries on the same memory.

-   :material-swap-horizontal:{ .lg .middle } **Model-agnostic**

    ---

    Swappable segmentation backends, permissive or AGPL. Already run a detector? Its `vision_msgs` detections become the memory's input.

-   :material-clipboard-check-outline:{ .lg .middle } **Offline evaluation**

    ---

    `rtsm eval` runs the whole pipeline on ROS 1, rosbag2 or MCAP bags and writes a report where every number carries its same-input floor.

-   :material-robot-outline:{ .lg .middle } **Built for agents**

    ---

    REST, a Python client, and six MCP tools so Claude, Cursor or a LangGraph agent can ask where things are.

-   :material-record-rec:{ .lg .middle } **Record and replay**

    ---

    Capture a live session, replay it deterministically on the sensor clock, compare runs with their floors.

</div>

## Start here

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **[Installation](getting-started/installation.md)**

    ---

    pip with the CUDA index, the Docker image, Jetson.

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](getting-started/quick-start.md)**

    ---

    The bundled demo and a first query in five minutes.

-   :material-package-variant:{ .lg .middle } **[Evaluating a Bag](guides/eval.md)**

    ---

    Your own recording through the pipeline, with the report.

-   :material-flask-outline:{ .lg .middle } **[Use Cases](use-cases/index.md)**

    ---

    What the report says about public recordings nobody at Calabi made, numbers and all.

-   :material-api:{ .lg .middle } **[API](api/index.md)**

    ---

    REST, the Python client, MCP, the dashboard WebSocket.

-   :material-sitemap:{ .lg .middle } **[Architecture](concepts/architecture.md)**

    ---

    The path of one frame, top to bottom.

</div>

## Measured, not promised

<div class="rtsm-stats" markdown>
<div markdown><span class="rtsm-stats__value">210 ms</span><span class="rtsm-stats__label">mean pipeline latency, dual backend, RTX 5090</span></div>
<div markdown><span class="rtsm-stats__value">510 ms</span><span class="rtsm-stats__label">mean latency, grounded_sam2 (Apache-2.0 default)</span></div>
<div markdown><span class="rtsm-stats__value">0</span><span class="rtsm-stats__label">floor on every scalar over three same-input runs of a public bag</span></div>
<div markdown><span class="rtsm-stats__value">1 s</span><span class="rtsm-stats__label">to refuse a bag that lacks poses or registered depth, before any model loads</span></div>
</div>

| | dual (FastSAM + YOLOE) | grounded_sam2 (Grounding DINO + SAM2) |
|---|---|---|
| Mean latency | 210 ms | 510 ms |
| P95 latency | 509 ms | 721 ms |
| Masks per frame | 28.8 | 13.4 |
| Objects confirmed | 60 | 35 |
| License | AGPL-3.0 | Apache-2.0 |

*RTX 5090, an iPhone ARKit recording of an indoor scene (162 frames). [Full benchmarks](benchmarks.md); the [use-case pages](use-cases/index.md) show the same code on recordings we did not make, including where it falls short.*

```json
// "Where is the red backpack?"
{ "id": "a3f2c1", "xyz": [1.2, 0.4, 2.1], "confidence": 0.87 }
```

!!! info "A short video"
    A two-minute walkthrough is on [YouTube](https://www.youtube.com/watch?v=abhXsbvOLQg).

Apache-2.0. Source, issues and releases on [GitHub](https://github.com/calabi-inc/rtsm); the package on [PyPI](https://pypi.org/project/rtsm/).
