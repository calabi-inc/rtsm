# Perception Pipeline

The perception pipeline extracts object instances from RGB-D frames and encodes them for matching and search. The segmentation stage is swappable — the rest of the pipeline is shared across all backends.

---

## Pipeline Stages

```
RGB Frame → Segmentation → Heuristics → Top-K Scoring → CLIP Encode → Vocab Classify
              (swappable)   (filtering)   (priority)     (embedding)    (labeling)
```

---

## 1. Segmentation

The first stage produces instance masks from the RGB image. RTSM supports multiple backends:

### grounded_sam2 (default, Apache-2.0)

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) detects objects via text prompt, then [SAM2](https://github.com/facebookresearch/sam2) generates high-quality masks from the detected bounding boxes.

- **Architecture**: Transformer (Swin backbone + Hiera ViT)
- **Input**: the RGB frame + a text prompt (the packaged indoor vocabulary, or `segmentation.grounded_sam2.vocab`)
- **Output**: ~29 masks/frame on the benchmark recording (text-prompted detection), at the frame's resolution
- **Seg time**: 236 ms mean

### dual (FastSAM + YOLOE, AGPL-3.0)

Two CNN models run independently, then masks are cross-validated via IoU:

- **FastSAM**: Class-agnostic segmentation (~23 masks/frame)
- **YOLOE**: Open-vocabulary detection + segmentation (~11 masks/frame, 4 585 built-in categories)
- **Merge**: IoU > 0.40 = dual-confirmed, remainder kept as single-source
- **Output**: ~26 masks/frame (merged)
- **Seg time**: 61 ms mean

Dual-confirmed masks receive a priority boost; on the benchmark recording they are selected into top-K slightly more often than they occur (1.1x).

### Other backends

| Backend | Description | Seg time |
|---------|-------------|----------|
| `sam2` | SAM2 auto-mask (segment everything, no labels) | 1 215 ms |
| `fastsam` | FastSAM only (class-agnostic, fast) | 26 ms |
| `yoloe` | YOLOE only (open-vocab, prompt-free) | 26 ms |

*Mean on an RTX 5090, a 1920x1440 iPhone recording ([Benchmarks](../benchmarks.md)).*

### Frame-quality gate

Before any segmentation pass, the frame-quality gate (`gates.*`) measures grey
level, contrast, and the valid-depth fraction on a strided subsample of the
frame and skips black, blank, or depth-less frames. Skips are counted as
`frame_rejections` in the latency analytics. See
[Configuration](../getting-started/configuration.md#frame-quality-gate).

---

## 2. Mask Heuristics

Heuristic filters remove unsuitable masks using depth and geometric information:

| Filter | Purpose | Config key |
|--------|---------|------------|
| Min area | Remove noise/tiny fragments (hard reject) | `filters.min_area_px` (500) |
| Depth range | Depth outside the range counts as invalid | `filters.depth.z_min_m` / `z_max_m` |
| Depth validity | Reject masks with too little valid depth after edge erosion (hard reject) | `staging.depth_valid_min` (0.02), `staging.depth_erode_px` (1) |
| Depth spread | Reject noisy depth regions (hard reject) | `filters.depth.sigma_max_m` (0.50) |
| 3D centroid | Minimum valid depth before a 3D centroid is computed | `staging.centroid_min_valid` (0.05) |
| Coverage / border | Rank down wall-sized or edge-touching masks (soft, see Top-K) | `staging.w_coverage`, `coverage_soft_cap`, `w_coverage_oversize`, `w_border_fraction` |
| Planarity | Detect and score planar surfaces | `planarity.*` |

!!! note "Heuristics cost varies by backend"
    The statistics of a mask are computed on its bounding box, so their cost follows the pixels the masks cover. SAM2 returns masks at the frame's resolution, FastSAM and YOLOE at a 640-px side unless `retina_masks` is on: on the 1920x1440 benchmark recording the stage takes 91 ms under `grounded_sam2` and 26 ms under `dual`. Up to release 0.3.1 it passed over the whole frame for every mask (553 and 55 ms on the same recording).

---

## 3. Top-K Selection

After filtering, masks are scored and the top K (default: 15) are kept to bound CLIP compute:

**Priority scoring** considers:

- Coverage (mask area relative to frame)
- Depth validity (% of mask with valid depth)
- Border fraction (penalty for edge-touching)
- Depth spread (penalty for noisy depth)
- Structure score (planarity + geometry)
- Dual-confirmation boost (if applicable)

Same-frame deduplication removes overlapping candidates before CLIP.

---

## 4. CLIP Encoding

Each selected mask is:

1. Cropped from the RGB image (with 6px padding)
2. Resized to 224x224
3. Encoded via **SigLIP ViT-B-16** (webli) by default (`clip.model`, `clip.pretrained`)

**Output**: 768-dimensional L2-normalized embedding vector (with the default model)

These embeddings enable:

- Matching observations across frames (cosine similarity)
- Semantic search via text queries
- Object identity tracking over time

**Speed**: ~24 ms for up to 15 candidates on an RTX 5090, crops and classification included (one batch on the GPU).

---

## 5. Vocabulary Classification

Object labels are assigned by comparing the CLIP embedding to pre-computed text embeddings from a configurable vocabulary (`config/clip/vocab.yaml`):

```python
similarities = cosine_similarity(image_embedding, text_embeddings)
label = vocab[argmax(similarities)]
confidence = max(similarities)
```

Labels are tracked as EWMA (exponentially weighted moving average) scores across observations, so transient misclassifications don't persist.

---

## Performance Summary

*Mean per processed frame on an RTX 5090, a 1920x1440 iPhone recording. See [Benchmarks](../benchmarks.md) for full data, and for a Jetson Orin Nano Super.*

| Stage | dual (ms) | grounded_sam2 (ms) |
|-------|-----------|-------------------|
| Segmentation | 61 | 236 |
| Heuristics | 26 | 91 |
| Scoring | 0.3 | 0.4 |
| CLIP encode | 24 | 21 |
| Association | 5 | 6 |
| **Total** | **126** | **365** |

---

## Next Steps

- [Memory Model](memory-model.md) — How observations become persistent objects
- [Architecture](architecture.md) — Full system overview
- [Benchmarks](../benchmarks.md) — Detailed performance data
