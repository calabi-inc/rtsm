# Performance Benchmarks

Measured on RTX 5090 (32 GB VRAM), replaying an iPhone ARKit recording (162 frames, 458s indoor scene) at original rate. Both backends process the same deterministic input through the identical 10-stage pipeline.

Benchmark script: [`scripts/benchmark_backends.py`](https://github.com/calabi-inc/rtsm/blob/main/scripts/benchmark_backends.py)

---

## Backend Comparison

| Metric | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|--------|------------------------|------------------------------|
| **Mean latency** | **210 ms** | **510 ms** |
| P50 (median) | 170 ms | 502 ms |
| P95 | 509 ms | 721 ms |
| Max | 539 ms | 830 ms |
| Masks/frame | 28.8 | 13.4 |
| Objects confirmed | 60 | 35 |
| Confirmation rate | 52.2% | 45.5% |
| License | AGPL-3.0 | Apache-2.0 |

The dual backend is **2.4x faster** and discovers **71% more confirmed objects**, primarily because it runs two CNN-based models versus transformer-based models, and operates prompt-free (4 585 built-in categories vs 30-class text prompt).

---

## Per-Stage Breakdown

### Mean Latency (ms)

| Stage | dual | grounded_sam2 | Notes |
|-------|------|---------------|-------|
| Segmentation | 116 (55%) | 222 (44%) | Model inference — CNN vs transformer |
| Heuristics | 60 (29%) | 239 (47%) | Depth, border, planarity filtering |
| Scoring | 0.2 (<1%) | 0.2 (<1%) | Priority ranking |
| CLIP encode | 23 (11%) | 24 (5%) | ViT-B/32, top-15 candidates |
| Association | 6 (3%) | 4 (1%) | Proximity + cosine matching |
| **Total** | **210** | **510** | |

### P95 Latency (ms)

| Stage | dual | grounded_sam2 |
|-------|------|---------------|
| Segmentation | 394 | 433 |
| Heuristics | 85 | 367 |
| Scoring | 0.4 | 0.3 |
| CLIP encode | 30 | 29 |
| Association | 8 | 7 |
| **Total** | **509** | **721** |

!!! note "Heuristics bottleneck"
    For grounded_sam2, the heuristics stage (239 ms) is the largest cost — **not** the model itself. SAM2 produces higher-fidelity masks that are more expensive to filter through depth validation and planarity checks. Optimizing heuristics for SAM2 mask characteristics is the fastest path to reducing total latency.

---

## Segmentation Analysis

### Mask Generation

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Raw masks/frame | 28.8 | 13.4 |
| Top-K candidates (CLIP input) | 13.6 | 9.0 |
| Overall survival (candidate/raw) | 47.0% | 66.9% |

The dual backend generates more masks but with lower individual survival rate. Grounded SAM2 produces fewer but higher-quality masks that pass heuristics more reliably.

### Dual Confirmation Breakdown

In dual mode, FastSAM (class-agnostic) and YOLOE (open-vocab) masks are cross-validated via IoU (threshold 0.40):

| Source | % of all masks | % selected into top-K |
|--------|----------------|----------------------|
| Dual-confirmed (both agree) | 22.4% | 32.8% |
| FastSAM-only | 61.4% | 58.1% |
| YOLOE-only | 16.1% | 9.1% |

Dual-confirmed masks are preferentially promoted into top-K (1.5x selection rate vs presence rate), indicating the pipeline correctly prioritizes cross-validated detections.

---

## Object Discovery

### Final Working Memory State

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Total objects | 115 | 77 |
| Confirmed | 60 | 35 |
| Proto (unconfirmed) | 55 | 42 |
| Avg hits/object | 3.0 | 2.8 |
| Confirmation rate | 52.2% | 45.5% |

The dual backend discovers 49% more total objects and 71% more confirmed objects from the same input, driven by higher mask-per-frame count and prompt-free vocabulary coverage.

---

## Throughput

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Frames processed | 38 | 38 |
| Processing Hz | 1.13 | 1.13 |
| Mean queue depth | 0.1 | 0.3 |
| Warmup frames skipped | 5 | 5 |

Both backends process the same 38 frames (out of 162 total) — the remainder are filtered by the ingest gate (keyframe gating, non-keyframe throttle, sweep policy). Processing Hz is identical because the replay feeds frames at real-time rate with natural gaps between keyframes.

---

## Architecture Trade-offs

| Dimension | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|-----------|------------------------|------------------------------|
| **Architecture** | CNN (YOLOv8 backbone) | Transformer (Swin + Hiera ViT) |
| **License** | AGPL-3.0 | Apache-2.0 |
| **Vocabulary** | 4 585 built-in tags (prompt-free) | Configurable text prompt |
| **Mask quality** | Two-model IoU consensus | SAM2 high-quality masks |
| **Latency** | 210 ms mean | 510 ms mean |
| **Detection strategy** | Class-agnostic + open-vocab merge | Text-prompted detection |
| **Best for** | Research, prototyping, max recall | Commercial deployment, permissive licensing |

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| Python | 3.12 |
| Recording | `recordings/session1` (iPhone ARKit, 162 frames, 458.9s) |
| RGB resolution | 640x480 |
| Mask resolution | 640x640 (`retina_masks: false`) |
| CLIP model | ViT-B/32 (OpenAI) |
| Top-K pre-CLIP | 15 |
| Association `cos_min` | 0.90 |
| Promote hits | 2 |
| Proto TTL | 10.0s |
| Replay rate | Real-time (original recording rate) |

---

## Reproducing

```bash
# Run the benchmark (requires both backend dependencies)
pip install "rtsm[all]" --extra-index-url https://download.pytorch.org/whl/cu128
python scripts/benchmark_backends.py
```

Results are written to `reports/` (`backend_comparison.md`, one `raw_<backend>.json` per backend); the directory is not tracked.

---

## Accuracy Evaluation

!!! info "Coming soon"
    Precision, recall, and spatial position accuracy metrics will be added when the ArUco marker ground-truth evaluation framework is complete. See [Roadmap](index.md).
