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
- **Input**: 640x480 RGB + text prompt (30-class indoor vocabulary)
- **Output**: ~13 masks/frame (text-prompted detection)
- **Seg time**: 222 ms mean

### dual (FastSAM + YOLOE, AGPL-3.0)

Two CNN models run independently, then masks are cross-validated via IoU:

- **FastSAM**: Class-agnostic segmentation (~24 masks/frame)
- **YOLOE**: Open-vocabulary detection + segmentation (~11 masks/frame, 1200+ LVIS categories)
- **Merge**: IoU > 0.40 = dual-confirmed, remainder kept as single-source
- **Output**: ~29 masks/frame (merged)
- **Seg time**: 116 ms mean

Dual-confirmed masks receive a priority boost and are 1.5x more likely to be selected into top-K.

### Other backends

| Backend | Description | Seg time |
|---------|-------------|----------|
| `sam2` | SAM2 auto-mask (segment everything, no labels) | ~860 ms |
| `fastsam` | FastSAM only (class-agnostic, fast) | ~50 ms |
| `yoloe` | YOLOE only (open-vocab, prompt-free) | ~60 ms |

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
    SAM2-based masks take **4x longer** to filter than FastSAM masks (239 ms vs 60 ms), likely due to higher-fidelity mask boundaries requiring more compute in depth validation and planarity checks.

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
3. Encoded via **CLIP ViT-B/32** (OpenAI)

**Output**: 512-dimensional L2-normalized embedding vector

These embeddings enable:

- Matching observations across frames (cosine similarity)
- Semantic search via text queries
- Object identity tracking over time

**Speed**: ~23 ms for 15 candidates (batch encode on GPU).

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

*Measured on RTX 5090, 640x480 input. See [Benchmarks](../benchmarks.md) for full data.*

| Stage | dual (ms) | grounded_sam2 (ms) |
|-------|-----------|-------------------|
| Segmentation | 116 | 222 |
| Heuristics | 60 | 239 |
| Scoring | 0.2 | 0.2 |
| CLIP encode | 23 | 24 |
| Association | 6 | 4 |
| **Total** | **210** | **510** |

---

## Next Steps

- [Memory Model](memory-model.md) — How observations become persistent objects
- [Architecture](architecture.md) — Full system overview
- [Benchmarks](../benchmarks.md) — Detailed performance data
