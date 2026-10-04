---
title: Home
hide:
  - navigation
  - toc
---

<div class="rtsm-hero" markdown>

<div class="rtsm-hero__pill"><span class="dot"></span><span><strong>RTSM 0.2.0</strong> on PyPI — <em>pip install rtsm</em></span></div>

# Real-Time Spatio-Semantic <span class="shine">Memory</span>

<p class="rtsm-hero__tagline">RTSM maintains a persistent, queryable memory of objects from an RGB-D stream with camera poses: for each object a world-frame position, a visual embedding and a label distribution, updated as the camera moves and queried over REST or MCP. Open source, Apache-2.0.</p>

<div class="rtsm-terminal"><div class="label">&gt;_ Terminal</div><code>pip install "rtsm[gpu]" &amp;&amp; rtsm demo</code></div>

[Install](getting-started/installation.md){ .md-button .md-button--primary }
[Evaluate a bag](guides/eval.md){ .md-button }
[Source](https://github.com/calabi-inc/rtsm){ .md-button }

<div class="rtsm-hero__meta"><span class="ok">rtsm 0.2.0</span><span>Apache-2.0</span><span>Python 3.10–3.13</span><span>5 segmentation backends + external detections</span><span>REST + MCP</span><span>ROS 1 / rosbag2 / MCAP in</span></div>

</div>

<img class="rtsm-demo" src="https://raw.githubusercontent.com/calabi-inc/rtsm/main/repo_media/rtsm-demo-gif.gif" alt="The dashboard during a replay: a point cloud of a room with detected objects labelled in place">

## What RTSM is

A SLAM system or a tracking camera supplies poses; a segmentation model supplies masks and labels per frame. RTSM takes both, lifts each mask to a 3-D point with the depth, and associates it with the objects it already holds by position and visual similarity. A match updates an existing object; a miss creates a candidate that is promoted on repeated, consistent observation. The result is a world state that can be queried while the camera is still moving, by text, by coordinate, or by an agent through MCP.

It is not a SLAM system and not a detector: it depends on both, and the report it writes attributes its numbers to the detector in use.

<div class="grid cards" markdown>

-   :material-cube-scan:{ .lg .middle } **Object memory**

    ---

    Stable ids across views; proto objects promoted to confirmed on repeated observation; a long-term vector index that outlives the session.

-   :material-magnify:{ .lg .middle } **Semantic and spatial queries**

    ---

    Text queries through CLIP embeddings and FAISS; coordinate and radius queries on the same memory.

-   :material-swap-horizontal:{ .lg .middle } **Pluggable perception**

    ---

    Five segmentation backends (Apache-2.0 or AGPL), or the output of a detector you already run, read from a `vision_msgs` topic.

-   :material-clipboard-check-outline:{ .lg .middle } **Offline evaluation**

    ---

    `rtsm eval` runs the pipeline on ROS 1, rosbag2 or MCAP bags and writes a report; each number carries its spread over repeated runs of the same input.

-   :material-robot-outline:{ .lg .middle } **Interfaces**

    ---

    A REST API, a Python client with no perception dependencies, and six MCP tools for agent frameworks.

-   :material-record-rec:{ .lg .middle } **Record and replay**

    ---

    Live sessions recorded to disk and replayed on the sensor clock, so two runs of one recording are comparable.

</div>

## Known limitations

- **Poses come from outside.** ARKit, RTAB-Map, or a bag's TF or odometry. RTSM does no localisation and cannot recover from a pose source that drifts or resets.
- **Re-identification is the weak point.** On the 2012 TUM RGB-D sequences the appearance gate (cosine 0.90) rejects most returns to an already-known object: 95 % of created objects are duplicates of something in memory. The [use-case pages](use-cases/index.md) report this in full.
- **One session, one world frame.** Objects carry no session identity yet; a second session into the same server adds to the same map in whatever frame its poses arrive in.
- **Numbers depend on the platform.** The same bag gives slightly different object counts under Linux and Windows PyTorch builds; each is deterministic on its own. Compare runs made on the same platform.

## Start here

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **[Installation](getting-started/installation.md)**

    ---

    pip with the CUDA index, the Docker image, Jetson.

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](getting-started/quick-start.md)**

    ---

    The bundled demo and a first query.

-   :material-package-variant:{ .lg .middle } **[Evaluating a Bag](guides/eval.md)**

    ---

    Your own recording through the pipeline, with the report.

-   :material-flask-outline:{ .lg .middle } **[Use Cases](use-cases/index.md)**

    ---

    Results on public datasets we did not record, TUM RGB-D and NVIDIA r2b, every number included.

-   :material-api:{ .lg .middle } **[API](api/index.md)**

    ---

    REST, the Python client, MCP, the dashboard WebSocket.

-   :material-sitemap:{ .lg .middle } **[Architecture](concepts/architecture.md)**

    ---

    The path of one frame, top to bottom.

</div>

## Measurements

The figures below come from the [benchmark page](benchmarks.md) and the [use-case runs](use-cases/index.md). Each use-case number is reported with its spread over three runs of the same input; a difference smaller than that spread means nothing.

<div class="rtsm-stats" markdown>
<div markdown><span class="rtsm-stats__value">210 ms</span><span class="rtsm-stats__label">mean pipeline latency per processed frame, dual backend, RTX 5090</span></div>
<div markdown><span class="rtsm-stats__value">510 ms</span><span class="rtsm-stats__label">mean latency, grounded_sam2 (the Apache-2.0 default)</span></div>
<div markdown><span class="rtsm-stats__value">31 %</span><span class="rtsm-stats__label">re-identification of in-view objects on the TUM fr3 office loop (2012 Kinect)</span></div>
<div markdown><span class="rtsm-stats__value">0</span><span class="rtsm-stats__label">spread on every reported scalar over three same-input runs</span></div>
</div>

| | dual (FastSAM + YOLOE) | grounded_sam2 (Grounding DINO + SAM2) |
|---|---|---|
| Mean latency | 210 ms | 510 ms |
| P95 latency | 509 ms | 721 ms |
| Masks per frame | 28.8 | 13.4 |
| Objects confirmed | 60 | 35 |
| License | AGPL-3.0 | Apache-2.0 |

*RTX 5090, an iPhone ARKit recording of an indoor scene (162 frames). The two backends differ in what they detect, so the object counts are not comparable as accuracy; they describe the backends' behaviour on this recording.*

```json
// "Where is the red backpack?"
{ "id": "a3f2c1", "xyz": [1.2, 0.4, 2.1], "confidence": 0.87 }
```

!!! info "A short video"
    A two-minute walkthrough of the dashboard is on [YouTube](https://www.youtube.com/watch?v=abhXsbvOLQg).

Apache-2.0. Source and issues on [GitHub](https://github.com/calabi-inc/rtsm), releases on [PyPI](https://pypi.org/project/rtsm/). Developed by [Calabi Inc.](https://www.calabi.com/)
