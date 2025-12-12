# Perception Pipeline

The perception pipeline extracts object instances from RGB-D frames and encodes them for matching and search.

---

## Pipeline Stages

```
RGB Frame → FastSAM → Mask Filter → CLIP Encode → Vocab Classify
```

---

## 1. Segmentation (FastSAM)

[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) generates instance masks from the RGB image.

- **Input**: 640×480 RGB image
- **Output**: Variable number of binary masks
- **Speed**: ~10-15ms per frame

FastSAM is a CNN-based approximation of SAM (Segment Anything), trading some accuracy for 50× faster inference.

---

## 2. Mask Filtering

Heuristic filters remove unsuitable masks:

| Filter | Purpose |
|--------|---------|
| Min area (0.1%) | Remove noise/tiny fragments |
| Max area (50%) | Remove walls/floors/background |
| Aspect ratio | Remove extreme shapes |
| Edge touching | Optionally filter partial objects |

This typically rejects 10-15% of masks as insignificant.

---

## 3. Top-K Selection

After filtering, we keep only the top K masks (default: 20) per frame to bound compute cost. Selection prioritizes:

1. Mask confidence score
2. Area (medium-sized preferred)
3. Distance from frame center

---

## 4. CLIP Encoding

Each mask is:

1. Cropped from the RGB image (with padding)
2. Resized to 224×224
3. Encoded via CLIP ViT-B/32

**Output**: 512-dimensional embedding vector

These embeddings enable:

- Matching observations across frames
- Semantic search via text queries

---

## 5. Vocabulary Classification

Object labels are assigned by comparing the CLIP embedding to pre-computed text embeddings:

```python
text_embeddings = clip.encode_text([
    "a photo of a mug",
    "a photo of a backpack",
    "a photo of a chair",
    ...
])

similarities = cosine_similarity(image_embedding, text_embeddings)
label = vocab[argmax(similarities)]
confidence = max(similarities)
```

The vocabulary is configurable — add domain-specific objects for your use case.

---

## Performance

| Stage | Time (RTX 5090) |
|-------|-----------------|
| FastSAM | ~12ms |
| Mask filtering | <1ms |
| CLIP encode (20 masks) | ~15ms |
| Vocab classify | <1ms |
| **Total** | **<30ms** |

---

## Next Steps

- [Memory Model](memory-model.md) — How observations become persistent objects
- [Architecture](architecture.md) — Full system overview
