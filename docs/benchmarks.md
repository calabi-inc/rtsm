# Performance Benchmarks

Two machines, each measured its own way:

- **[Workstation](#workstation-rtx-5090)** (RTX 5090): the two full backends side by side on one recording replayed at its own rate, with [`scripts/benchmark_backends.py`](https://github.com/calabi-inc/rtsm/blob/main/scripts/benchmark_backends.py).
- **[Edge](#edge-jetson-orin-nano-super)** (Jetson Orin Nano Super): the two edge backends on five recordings, with [`rtsm eval`](guides/eval.md).

All figures are from the code after release 0.3.1 unless a row says 0.3.1. That release computed the per-mask statistics over the whole frame for every mask and prepared every crop for the image tower through PIL; both sections below show what changing that is worth. The stored memory is the same in both versions: the lossless replays give identical memory fingerprints on every recording and backend measured here.

---

## Workstation: RTX 5090

`recordings/session1` (an iPhone recording made with Calabi Lens: 240 frames over 75.8 s, RGB 1920x1440, LiDAR depth 256x192) replayed at its original rate. Both backends process the same 53 frames through the same pipeline; the first five are left out of the statistics.

### Backend Comparison

| Metric | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|--------|------------------------|------------------------------|
| **Mean latency** | **126 ms** | **365 ms** |
| P50 (median) | 130 ms | 378 ms |
| P95 | 146 ms | 436 ms |
| Max | 147 ms | 457 ms |
| Masks/frame | 25.9 | 29.2 |
| Objects in memory | 124 | 149 |
| Objects confirmed | 65 | 71 |
| Confirmation rate | 52.4% | 47.7% |
| License | AGPL-3.0 | Apache-2.0 |

The dual backend is **2.9x faster**: two CNN-based models against two transformer-based ones. The two backends differ in what they detect (a prompt-free model with 4 585 built-in categories against a text-prompted one), so the object counts describe their behaviour on this recording and are not comparable as accuracy.

### Against 0.3.1

Same script, same machine, same day, the two versions one after the other:

| | dual | grounded_sam2 |
|---|---|---|
| Mean latency, 0.3.1 | 179 ms | 835 ms |
| Mean latency, after 0.3.1 | **126 ms** (1.4x) | **365 ms** (2.3x) |
| P95, 0.3.1 | 216 ms | 1 031 ms |
| P95, after 0.3.1 | 146 ms | 436 ms |
| Segmentation, mean | 61 ms in both | 215 → 236 ms |
| Mask statistics, mean | 55 → 26 ms | 553 → 91 ms |
| Embedding stage (crops, image tower, classifier), mean | 48 → 24 ms | 48 → 21 ms |
| Objects in memory / confirmed | 124 / 65 in both | 149 / 71 in both |
| Processing rate | 1.16 Hz in both | 1.08 → 1.16 Hz |

!!! note "Mask statistics on a high-resolution recording"
    SAM2 returns its masks at the resolution of the frame, here 1920x1440. In 0.3.1 the statistics of every mask (area, bounding box, border contact, depth quantiles, centroid, planarity) were computed with about a dozen passes over that whole frame, which made the filter the largest stage of the default backend: 553 of 835 ms, and the pipeline fell behind the recording (mean queue depth 4.9). They are now computed on each mask's bounding box, with the same values, and the model is the largest stage again.

The segmentation stage of `grounded_sam2` reads about 20 ms higher than on 0.3.1 although its code is the same. Called alone on the frames of this recording the segmenter takes 212 ms in either version; inside the faster pipeline, the copy of the detector's input to the GPU takes 30 to 60 ms instead of 3 in part of the frames. The cause is not identified (Windows, one machine); the totals above include it.

---

## Per-Stage Breakdown

### Mean Latency (ms)

| Stage | dual | grounded_sam2 | Notes |
|-------|------|---------------|-------|
| Segmentation | 61 (49%) | 236 (65%) | Model inference: CNN vs transformer |
| Heuristics | 26 (21%) | 91 (25%) | Per-mask statistics and the depth filters |
| Scoring | 0.3 (<1%) | 0.4 (<1%) | Priority ranking |
| CLIP encode | 24 (19%) | 21 (6%) | SigLIP ViT-B-16, top-15 candidates |
| Association | 5.1 (4%) | 5.7 (2%) | Proximity + cosine matching |
| **Total** | **126** | **365** | |

The stages do not add up to the total: the remaining 9 to 10 ms is the work between the timed stages.

### P95 Latency (ms)

| Stage | dual | grounded_sam2 |
|-------|------|---------------|
| Segmentation | 72 | 289 |
| Heuristics | 35 | 164 |
| Scoring | 0.4 | 0.5 |
| CLIP encode | 36 | 33 |
| Association | 6.5 | 7.5 |
| **Total** | **146** | **436** |

---

## Segmentation Analysis

### Mask Generation

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Raw masks/frame | 25.9 | 29.2 |
| Top-K candidates (CLIP input) | 13.1 | 14.2 |
| Overall survival (candidate/raw) | 50.7% | 48.6% |

### Dual Confirmation Breakdown

In dual mode, FastSAM (class-agnostic) and YOLOE (open-vocab) masks are cross-validated via IoU (threshold 0.40):

| Source | % of all masks | % selected into top-K |
|--------|----------------|----------------------|
| Dual-confirmed (both agree) | 27.8% | 31.4% |
| FastSAM-only | 59.0% | 59.1% |
| YOLOE-only | 13.2% | 9.5% |

Dual-confirmed masks are selected into top-K slightly more often than they occur (1.1x), YOLOE-only masks less often (0.7x).

---

## Object Discovery

### Final Working Memory State

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Total objects | 124 | 149 |
| Confirmed | 65 | 71 |
| Proto (unconfirmed) | 59 | 78 |
| Avg hits/object | 4.0 | 3.4 |
| Confirmation rate | 52.4% | 47.7% |

---

## Throughput

| Metric | dual | grounded_sam2 |
|--------|------|---------------|
| Frames processed | 53 | 53 |
| Processing Hz | 1.16 | 1.16 |
| Mean queue depth | 0.0 | 0.2 |
| Warmup frames skipped | 5 | 5 |

Both backends process the same 53 frames (out of 240 total); the remainder are filtered by the ingest gate (keyframe gating, non-keyframe throttle, sweep policy). Processing Hz is identical because the replay feeds frames at real-time rate with natural gaps between keyframes.

---

## Architecture Trade-offs

| Dimension | dual (FastSAM + YOLOE) | grounded_sam2 (GDINO + SAM2) |
|-----------|------------------------|------------------------------|
| **Architecture** | CNN (YOLOv8 backbone) | Transformer (Swin + Hiera ViT) |
| **License** | AGPL-3.0 | Apache-2.0 |
| **Vocabulary** | 4 585 built-in tags (prompt-free) | Configurable text prompt |
| **Mask quality** | Two-model IoU consensus | SAM2 high-quality masks |
| **Latency** | 126 ms mean | 365 ms mean |
| **Detection strategy** | Class-agnostic + open-vocab merge | Text-prompted detection |
| **Best for** | Research, prototyping, max recall | Commercial deployment, permissive licensing |

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| Python | 3.12 |
| Recording | `recordings/session1` (iPhone, Calabi Lens, 240 frames, 75.8 s) |
| RGB resolution | 1920x1440 (depth 256x192) |
| Mask resolution | 480x640 for dual (`retina_masks: false`), frame resolution for SAM2 |
| CLIP model | SigLIP ViT-B-16 (webli) |
| Top-K pre-CLIP | 15 |
| Association `cos_min` | 0.90 |
| Promote hits | 2 |
| Proto TTL | 10.0s |
| Replay rate | Real-time (original recording rate) |
| Measured | 2026-11-10, each version twice: after 0.3.1 dual 126 and 123 ms, grounded_sam2 365 and 368 ms; 0.3.1 dual 179 and 177 ms, grounded_sam2 835 and 831 ms. The tables show the two runs made one right after the other (179 / 835 and 126 / 365). |

---

## Reproducing

```bash
# Run the benchmark (requires both backend dependencies)
pip install "rtsm[all]" --extra-index-url https://download.pytorch.org/whl/cu128
python scripts/benchmark_backends.py
```

Results are written to `reports/` (`backend_comparison.md`, one `raw_<backend>.json` per backend); the directory is not tracked.

---

## Edge: Jetson Orin Nano Super

The same engine and the same PyTorch models on a Jetson Orin Nano Super developer kit (8 GB): no TensorRT engine, no ONNX Runtime, nothing exported. The two backends of the [edge install](getting-started/installation.md#option-d-edge-jetson-arm-aarch64) are measured, `fastsam` and `dual`; `grounded_sam2` was not run on this board.

Five recordings, 650 processed frames per backend, each replayed with [`rtsm eval`](guides/eval.md) in two ways:

- **Lossless**: every frame the ingest gate admits is processed, however long it takes. This gives the time per frame, and a fingerprint of the stored memory that repeats exactly from run to run.
- **At the recording's own rate** (`--speed 1.0 --set ingest.policy=latest`): the recording plays as the sensor produced it, and a frame that arrives while the pipeline is busy replaces the one that waits, as on a robot. This gives what the board keeps.

| Recording | What it is | Frames a lossless run processes |
|---|---|---|
| `fr1` | TUM RGB-D `freiburg1_desk`, a handheld Kinect, 640x480 | 45 |
| `fr3` | TUM RGB-D `freiburg3_long_office_household`, a handheld Kinect, 640x480 | 194 |
| `pioneer` | TUM RGB-D `freiburg2_pioneer_slam`, a Kinect on a wheeled robot, 640x480 | 237 |
| `office` | a synthetic office rendered in Isaac Sim, 640x480 | 121 |
| `session1` | the iPhone recording of the workstation section, 1920x1440 | 53 |

### Time per Frame

Median per processed frame in the lossless replay, in ms; the first five frames of a run are left out. The last two columns are means over the five recordings.

| Backend, version | fr1 | fr3 | pioneer | office | session1 | Mean | P95, mean |
|---|---|---|---|---|---|---|---|
| `fastsam`, 0.3.1 | 715 | 789 | 397 | 740 | 786 | 685 | 823 |
| `fastsam`, after 0.3.1 | 390 | 430 | 310 | 379 | 538 | **409** (1.7x) | 478 |
| `dual`, 0.3.1 | 926 | 992 | 589 | 922 | 999 | 885 | 1 046 |
| `dual`, after 0.3.1 | 578 | 691 | 484 | 634 | 733 | **624** (1.4x) | 746 |

The stored memory is the same in both versions: on each of the five recordings and for both backends the memory fingerprint of the newer code equals 0.3.1's, so the ten pairs hold the same objects, positions, labels and embeddings.

### Where the Time Goes

Median per stage on `fr3`, in ms:

| Stage | `fastsam`, 0.3.1 | `fastsam`, after 0.3.1 | `dual`, 0.3.1 | `dual`, after 0.3.1 |
|---|---|---|---|---|
| Segmentation | 176 | 148 | 370 | 350 |
| Mask statistics and filters | 313 | 74 | 325 | 119 |
| Embedding (crops, image tower, classifier) | 225 | 137 | 232 | 145 |
| Association | 44 | 56 | 47 | 51 |
| **Whole frame** | 789 | 430 | 992 | 691 |

On 0.3.1 the mask statistics, CPU work around the models, were the largest stage on this board. Segmentation and association move as well, by 4 to 28 ms and in opposite directions (segmentation lower, association higher), without a change in their code that would explain it. The board's CPU clocks are left to the governor in these runs and follow the load; that is the likely reason and was not isolated.

### At the Recording's Own Rate

Frames processed, with the frames the ingest lane dropped in parentheses, and the time the first frame took:

| Recording | Lossless | `fastsam`, 0.3.1 | `fastsam`, after 0.3.1 | `dual`, 0.3.1 | `dual`, after 0.3.1 |
|---|---|---|---|---|---|
| fr1 | 45 | 6 (53), 17.5 s | 38 (11), 0.5 s | 6 (53), 19.8 s | 29 (29), 1.0 s |
| fr3 | 194 | 56 (198), 38.8 s | 149 (73), 9.4 s | 45 (209), 39.3 s | 108 (141), 3.2 s |
| pioneer | 237 | 210 (65), 32.8 s | 236 (7), 2.8 s | 178 (130), 33.4 s | 228 (37), 3.1 s |
| office | 121 | 58 (83), 15.1 s | 121 (1), 0.7 s | 46 (95), 16.9 s | 89 (52), 1.0 s |
| session1 | 53 | 45 (26), 25.0 s | 53 (0), 1.9 s | 41 (33), 29.9 s | 53 (3), 2.0 s |
| **All five** | **650** | 375 | **597** | 316 | **507** |

Two things changed here. The frame is shorter, and the first frame is no longer the first call of every model: 0.3.1 ran each model's first inference on the first frame of the session, 15 to 39 s on this board, while the recording kept playing. The engine now does that at start-up, on synthetic input ([`segmentation.warmup_inference`](getting-started/configuration.md#segmentation-backend)), which adds 14 to 18 s to the engine's start-up on this board.

The first frame then takes 0.5 to 3.2 s in nine of these ten runs. In the tenth, `fastsam` on `fr3`, it took 9.4 s, and the frames that arrived meanwhile were dropped. On the two multi-gigabyte recordings (`fr3`, `pioneer`) the first frame is erratic: 2 to 14 s over the eight runs made on them, lossless ones included, most of it in the segmenter's first call. The recording is being read into the board's free memory at that moment, which is the likely cause; it was not isolated, and a live camera fills no page cache.

Of the confirmed objects a lossless run stores, the run at rate has this share within 0.3 m:

| | fr1 | fr3 | pioneer | office | session1 |
|---|---|---|---|---|---|
| `fastsam`, 0.3.1 | 69 % | 42 % | 85 % | 83 % | 91 % |
| `fastsam`, after 0.3.1 | 98 % | 87 % | 99 % | 100 % | 100 % |
| `dual`, 0.3.1 | 70 % | 35 % | 68 % | 85 % | 84 % |
| `dual`, after 0.3.1 | 100 % | 75 % | 89 % | 95 % | 100 % |

Under `fastsam` the runs on `office` and `session1`, and under `dual` the run on `session1`, store exactly the memory of the lossless run, the same fingerprint. The two handheld recordings (`fr1`, `fr3`) admit a frame every 0.34 s and the board needs about 0.46 s per frame at rate under `fastsam`, so they still lose frames.

### Memory on the Board

RAM in use on the whole board while frames are processed (the median of one sample per second; on a Jetson the GPU's memory is this RAM):

| | 0.3.1 | After 0.3.1 |
|---|---|---|
| `fastsam`, mean of the five recordings | 5.0 GB | 4.3 GB |
| `dual`, mean of the five recordings | 5.1 GB | 4.3 GB |
| Highest sample of a run, loading included | 4.9 to 5.4 GB | 4.2 to 4.7 GB |

The board holds 1.0 to 1.2 GB before the engine starts (1.5 and 1.8 GB in two of the 0.3.1 runs), so the engine's own share falls from about 3.7 to 3.2 GB (`fastsam`) and from 3.9 to 3.2 GB (`dual`). 0.3.1 kept the working memory of the vocabulary's text encoding cached for the life of the process; it is now released at start-up.

### What These Figures Do Not Cover

- One board, in its 25 W power mode with the CPU and GPU clocks left to the governors; the MAXN SUPER mode and `jetson_clocks` were not used.
- One run per row (the `dual` run on `session1` at rate is a second run: the first replay ended early). The lossless figures repeat within a few percent; at the recording's own rate the count of processed frames moves by a few frames between runs, and by more when the first frame is slow.
- A replay from disk, not a live camera: capture and transport cost nothing here.
- TensorRT. The package runs its models in PyTorch and builds no engines.
- The time per frame grows with the memory: association and the view update look at the objects stored so far, so a longer session than these recordings (20 to 87 s) ends with longer frames.

### Test Configuration (Edge)

| Parameter | Value |
|-----------|-------|
| Device | NVIDIA Jetson Orin Nano Super developer kit, 8 GB |
| System | Jetson Linux R36.4.7 (JetPack 6), CUDA 12.6 |
| Power mode | 25 W, governors as shipped (CPU up to 1.34 GHz) |
| Python, torch | 3.10, torch 2.8.0 from the Jetson wheel index |
| Models | FastSAM-x, YOLOE-26s-seg (prompt-free), SigLIP ViT-B-16, in PyTorch |
| 0.3.1 | the commit before this change (release 0.3.1 plus two changes outside the pipeline) |
| Measured | 2026-11-09 and 2026-11-10, one run per row |

### Reproducing (Edge)

```bash
# lossless: the time per frame and the memory fingerprint
rtsm eval <recording> --repeats 1 --out runs/lossless --set segmentation.backend=fastsam

# at the recording's own rate: what the board keeps
rtsm eval <recording> --repeats 1 --out runs/at-rate --set segmentation.backend=fastsam --speed 1.0 --set ingest.policy=latest
```

`<recording>` is a ROS 1 `.bag`, a rosbag2 directory, an `.mcap` file or a Lens recording ([Eval guide](guides/eval.md)). The per-frame times are the `timing_ms` of the frame lines in the run's `events.jsonl`; the memory fingerprint is in `repeats.json`.

---

## Accuracy Evaluation

!!! info "Coming soon"
    Precision, recall, and spatial position accuracy metrics will be added when the ArUco marker ground-truth evaluation framework is complete. See [Roadmap](index.md).
