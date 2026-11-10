"""The embedding stage's shortcuts give the values of the code they replace, exactly: crops as one tensor instead
of one PIL image and one transform call each, the vocabulary classifier's scores in one transfer, the crop's mean
colour and fill through OpenCV; plus the start-up warm-up, the vocabulary helper and the row count a segmenter
may hand to the mask statistics."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from rtsm import engine
from rtsm.core.pipeline import Candidate, Pipeline
from rtsm.io.ingest_queue import IngestQueue
from rtsm.models.clip import inference as inf
from rtsm.models.clip.vocab_classifier import ClipVocabClassifier, LabelParams
from rtsm.models.segmentation import segmentation_vocab
from rtsm.utils.mask_staging import MaskStats

T = pytest.importorskip("torchvision.transforms")


# ───────────────────────── crops as one tensor ─────────────────────────

class _Tower(torch.nn.Module):
    """Stands in for an image tower: an output that depends on every input value."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 3 * 224 * 224).view(1, -1))

    def encode_image(self, x):
        flat = x.flatten(1) * self.w
        return torch.stack([flat[:, i::7].sum(dim=1) for i in range(7)], dim=1)


def _rgb(img):
    return img.convert("RGB")


def _clip_style(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    return T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), _rgb, T.ToTensor(),
                      T.Normalize(mean, std)])


def _siglip_style():
    return T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), _rgb, T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _crops(n, seed=0, size=224):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]


@pytest.mark.parametrize("make", [_clip_style, _siglip_style])
def test_tensor_preprocessing_gives_the_transform_values(make):
    pre = make()
    params = inf.tensor_preprocess_params(pre, "cpu")
    assert params is not None and params[0] == 224
    crops = _crops(5)
    ref = torch.stack([pre(Image.fromarray(c)) for c in crops])
    assert torch.equal(ref, inf._tensor_batch(crops, params, "cpu"))


@pytest.mark.parametrize("make", [_clip_style, _siglip_style])
def test_embeddings_are_identical_with_and_without_the_tensor_path(make):
    pre, model = make(), _Tower()
    params = inf.tensor_preprocess_params(pre, "cpu")
    crops = [c[..., ::-1] for c in _crops(19, seed=1)]      # views with a negative stride, as the pipeline passes them
    ref = inf.encode_images_batch([Image.fromarray(c) for c in crops], model, pre, device="cpu", batch_size=8, keep_on_device=True)
    new = inf.encode_images_batch(crops, model, pre, device="cpu", batch_size=8, keep_on_device=True, tensor_pre=params)
    old_arrays = inf.encode_images_batch(crops, model, pre, device="cpu", batch_size=8, keep_on_device=True)   # arrays, no params
    assert ref.shape == (19, 7) and torch.equal(ref, new) and torch.equal(ref, old_arrays)


def test_a_transform_that_is_not_the_identity_at_the_input_size_keeps_the_per_image_path():
    resize_then_crop = T.Compose([T.Resize(256), T.CenterCrop(224), _rgb, T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    assert inf.tensor_preprocess_params(resize_then_crop, "cpu") is None
    no_normalise = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    assert inf.tensor_preprocess_params(no_normalise, "cpu") is None
    assert inf.tensor_preprocess_params(lambda im: torch.zeros(3, 224, 224), "cpu") is None


def test_inputs_the_tensor_path_does_not_cover_fall_back():
    pre, model = _siglip_style(), _Tower()
    params = inf.tensor_preprocess_params(pre, "cpu")
    a, b = _crops(2, seed=2)
    for chunk in ([a, Image.fromarray(b)],                                  # one PIL image in the chunk
                  [a, _crops(1, size=100)[0]],                               # one array of another size
                  [a.astype(np.float32) / 255.0, b.astype(np.float32) / 255.0]):   # float arrays
        assert inf._tensor_batch(chunk, params, "cpu") is None
        ref = inf.encode_images_batch(chunk, model, pre, device="cpu", batch_size=4, keep_on_device=True)
        new = inf.encode_images_batch(chunk, model, pre, device="cpu", batch_size=4, keep_on_device=True, tensor_pre=params)
        assert torch.equal(ref, new)


def test_the_real_model_transform_passes_the_probe():
    open_clip = pytest.importorskip("open_clip")
    pre = open_clip.image_transform(224, is_train=False)
    params = inf.tensor_preprocess_params(pre, "cpu")
    assert params is not None and params[0] == 224
    crops = _crops(3, seed=5)
    assert torch.equal(torch.stack([pre(Image.fromarray(c)) for c in crops]), inf._tensor_batch(crops, params, "cpu"))


def test_the_probe_covers_every_input_value():
    """Scaling and normalising act on one byte of one channel at a time: a probe that holds all 256 values in each
    channel decides the equality for every image, not for a sample."""
    for size in (224, 256, 16):
        p = inf._probe_image(size)
        assert p.shape == (size, size, 3) and p.dtype == np.uint8
        assert all(len(np.unique(p[..., c])) == 256 for c in range(3))
    assert np.array_equal(inf._probe_image(224), inf._probe_image(224))


@pytest.mark.gpu
@pytest.mark.parametrize("make", [_clip_style, _siglip_style])
def test_the_tensor_path_is_bit_identical_on_the_gpu(make):
    """On CUDA a division by a Python number is a multiplication by its reciprocal, one bit off the CPU's division
    for about half of the byte values; the tensor path divides by a tensor on the device and matches the transform."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    pre = make()
    params = inf.tensor_preprocess_params(pre, "cuda")
    assert params is not None
    crops = _crops(6, seed=9)
    ref = torch.stack([pre(Image.fromarray(c)) for c in crops])
    assert torch.equal(ref, inf._tensor_batch(crops, params, "cuda").cpu())
    by_scalar = torch.from_numpy(np.stack(crops)).cuda().permute(0, 3, 1, 2).contiguous().to(torch.float32).div(255)
    by_scalar = by_scalar.sub_(params[2]).div_(params[3]).cpu()
    assert not torch.equal(ref, by_scalar)                # the shortcut this file exists to keep out


# ───────────────────────── the classifier's scores in one transfer ─────────────────────────

def _per_scalar_classify(clf, img_feats, params=None, return_topk=None):
    """classify_feats as it was: every score and index read from its tensor on its own."""
    if img_feats.numel() == 0:
        return []
    if img_feats.dim() == 1:
        img_feats = img_feats.unsqueeze(0)
    text_feats = clf.text_feats.to(device=img_feats.device, dtype=img_feats.dtype)
    k = int(min(return_topk or 5, text_feats.shape[0]))
    logits = img_feats @ text_feats.T
    topv, topi = logits.max(dim=1)
    masked = logits.clone()
    masked.scatter_(1, topi.unsqueeze(1), float("-inf"))
    secondv, _ = masked.max(dim=1)
    tkv, tki = logits.topk(k=k, dim=1)
    if params is None:
        params = LabelParams(min_top=float(clf.params_yaml["min_top"]), min_margin=float(clf.params_yaml["min_margin"]), topk=k)
    out = []
    for i in range(logits.size(0)):
        topk_list = [(clf.class_ids[int(tki[i, j])], float(tkv[i, j]), int(tki[i, j])) for j in range(k)]
        tv = float(topv[i].item())
        margin = float((topv[i] - secondv[i]).item())
        if tv >= params.min_top and margin >= params.min_margin:
            out.append((clf.class_ids[int(topi[i].item())], tv, int(topi[i].item()), topk_list))
        else:
            out.append(("unknown", tv, -1, topk_list))
    return out


def _classifier(n_classes=23, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    clf = ClipVocabClassifier.__new__(ClipVocabClassifier)
    clf.text_feats = torch.nn.functional.normalize(torch.randn(n_classes, dim, generator=g), dim=1)
    clf.class_ids = [f"class_{i}" for i in range(n_classes)]
    clf.params_yaml = {"min_top": 0.15, "min_margin": 0.01}
    return clf, g


@pytest.mark.parametrize("n_classes,topk", [(23, 5), (3, 5), (23, 1), (2, None)])
def test_classifier_output_is_what_the_per_scalar_reads_gave(n_classes, topk):
    clf, g = _classifier(n_classes)
    feats = torch.nn.functional.normalize(torch.randn(40, 32, generator=g), dim=1)
    ref = _per_scalar_classify(clf, feats, return_topk=topk)
    new = clf.classify_feats(feats, return_topk=topk)
    assert new == ref and {r[0] for r in ref} != {"unknown"} and any(r[0] == "unknown" for r in ref)
    assert all(type(r[1]) is float and type(r[2]) is int for r in new)
    one = clf.classify_feats(feats[0], return_topk=topk)
    assert one == _per_scalar_classify(clf, feats[0], return_topk=topk) and len(one) == 1
    assert clf.classify_feats(torch.empty(0, 32)) == []
    strict = LabelParams(min_top=0.99, min_margin=0.5, topk=3)
    assert clf.classify_feats(feats, params=strict, return_topk=3) == _per_scalar_classify(clf, feats, params=strict, return_topk=3)


# ───────────────────────── the crop's mean colour and fill ─────────────────────────

def _pipeline(tmp_path, staging=None):
    from rtsm.evaluation.event_log import EventLogWriter
    w = EventLogWriter(enabled=False, configured_path=str(tmp_path / "e.jsonl"))
    return Pipeline(cfg={"staging": staging or {}}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                    associator=None, ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w)


def _numpy_crop(rgb, mask, bbox, sx, sy, pad=6, size=224):
    """The embedding crop as it was computed: a float mean over the crop and a boolean-indexed fill."""
    H, W, _ = rgb.shape
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None
    region = mask[int(y0 / sy): int(y1 / sy), int(x0 / sx): int(x1 / sx)]
    if region.numel() > 0:
        m = cv2.resize(region.cpu().numpy().astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        crop[m == 0] = crop.mean(axis=(0, 1)).astype(np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def _candidate(i, mask, bbox):
    x0, y0, x1, y1 = bbox
    st = MaskStats(idx=i, area_px=int(mask.sum()), bbox=bbox, coverage=0.1, border_fraction=0.0, depth_valid=1.0,
                   depth_p50=1.0, depth_spread=0.0)
    return Candidate(idx=i, mask=mask, stats=st, priority=1.0)


@pytest.mark.parametrize("rgb_hw,mask_hw", [((120, 160), (120, 160)), ((240, 320), (120, 160)), ((90, 130), (60, 100))])
def test_mean_fill_crops_are_bit_identical(tmp_path, rgb_hw, mask_hw):
    rng = np.random.default_rng(11)
    pipe = _pipeline(tmp_path)
    rgb = rng.integers(0, 256, (*rgb_hw, 3), dtype=np.uint8)
    sx, sy = rgb_hw[1] / mask_hw[1], rgb_hw[0] / mask_hw[0]
    cands, boxes = [], []
    for i in range(30):
        y0, x0 = int(rng.integers(0, mask_hw[0] - 2)), int(rng.integers(0, mask_hw[1] - 2))
        y1, x1 = int(rng.integers(y0 + 1, mask_hw[0] + 1)), int(rng.integers(x0 + 1, mask_hw[1] + 1))
        mask = torch.zeros(mask_hw, dtype=torch.bool)
        mask[y0:y1, x0:x1] = torch.from_numpy(rng.random((y1 - y0, x1 - x0)) < float(rng.uniform(0.0, 1.0)))
        cands.append(_candidate(i, mask, (x0, y0, x1, y1)))
        boxes.append((x0, y0, x1, y1))
    pipe._make_crops_inplace(cands, rgb, sx, sy)
    n = 0
    for c, bbox in zip(cands, boxes):
        ref = _numpy_crop(rgb, c.mask, bbox, sx, sy)
        assert (c.crop is None) == (ref is None)
        if ref is not None:
            assert c.crop.dtype == np.uint8 and np.array_equal(c.crop, ref)
            n += 1
    assert n >= 25


def test_other_fill_modes_and_frame_types_take_the_numpy_path(tmp_path):
    rng = np.random.default_rng(12)
    mask = torch.zeros((40, 40), dtype=torch.bool)
    mask[10:30, 12:28] = True
    rgb = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    for fill, value in (("white", 255), ("black", 0)):
        c = _candidate(0, mask, (12, 10, 28, 30))
        _pipeline(tmp_path, {"bg_fill": fill, "crop_pad_px": 4, "clip_input": 28})._make_crops_inplace([c], rgb)
        assert c.crop.shape == (28, 28, 3) and (c.crop[0, 0] == value).all()
    c = _candidate(0, mask, (12, 10, 28, 30))
    _pipeline(tmp_path, {"bg_fill": "blur"})._make_crops_inplace([c], rgb)
    assert c.crop is not None
    wide = rng.integers(0, 256, (40, 40, 4), dtype=np.uint8)                 # not a 3-channel frame: the numpy lines
    c = _candidate(0, mask, (12, 10, 28, 30))
    _pipeline(tmp_path, {"crop_pad_px": 0, "clip_input": 16})._make_crops_inplace([c], wide)
    assert c.crop.shape == (16, 16, 4) and np.array_equal(c.crop, _numpy_crop(wide, mask, (12, 10, 28, 30), 1.0, 1.0, pad=0, size=16))


def test_the_pipeline_hands_arrays_to_the_image_tower(tmp_path):
    seen = {}

    class Clip:
        def encode_images_batch(self, images, batch_size=16):
            seen["images"] = images
            seen["batch_size"] = batch_size
            return torch.arange(len(images) * 4, dtype=torch.float32).view(len(images), 4)

    pipe = _pipeline(tmp_path)
    pipe.clip, pipe.vocab_clf = Clip(), None
    rng = np.random.default_rng(3)
    cands = [_candidate(i, torch.ones((8, 8), dtype=torch.bool), (0, 0, 8, 8)) for i in range(3)]
    crops = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    cands[0].crop, cands[2].crop = crops[0], crops[1]                        # the middle candidate has no crop
    pipe._clip_batch_inplace(cands)
    assert len(seen["images"]) == 2 and all(isinstance(im, np.ndarray) for im in seen["images"])
    assert np.array_equal(seen["images"][0], crops[0][..., ::-1]) and np.array_equal(seen["images"][1], crops[1][..., ::-1])
    assert np.array_equal(cands[0].emb_vis, [0, 1, 2, 3]) and cands[1].emb_vis is None and np.array_equal(cands[2].emb_vis, [4, 5, 6, 7])


# ───────────────────────── the warm-up at start-up ─────────────────────────

class _Segmenter:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def segment(self, image, vocab=None):
        self.calls.append((image.size, image.mode, vocab))
        if self.fail:
            raise RuntimeError("no engine")
        return SimpleNamespace(count=7)


def test_warm_segmenter_runs_two_frames_of_the_configured_size_with_the_backend_vocabulary():
    seg = _Segmenter()
    cfg = {"camera": {"width": 320, "height": 200},
           "segmentation": {"backend": "grounded_sam2", "grounded_sam2": {"vocab": ["chair", "table"]}}}
    info = engine.warm_segmenter(seg, cfg)
    assert seg.calls == [((320, 200), "RGB", ["chair", "table"])] * 2 and info["masks"] == 7 and info["ms"] >= 0
    seg = _Segmenter()
    engine.warm_segmenter(seg, {})                                           # no camera section: 640x480, no vocabulary
    assert seg.calls == [((640, 480), "RGB", None)] * 2


def test_warm_segmenter_skips_a_backend_without_a_model_and_survives_a_failure():
    seg = _Segmenter()
    seg.consumes_detections = True
    assert engine.warm_segmenter(seg, {}) is None and seg.calls == []
    seg = _Segmenter(fail=True)
    assert engine.warm_segmenter(seg, {}) is None and len(seg.calls) == 1     # logged, not raised


def test_the_warm_up_frame_is_reproducible_and_not_flat():
    a, b = np.asarray(engine._warmup_frame(640, 480)), np.asarray(engine._warmup_frame(640, 480))
    assert a.shape == (480, 640, 3) and a.dtype == np.uint8 and np.array_equal(a, b)
    assert len(np.unique(a.reshape(-1, 3), axis=0)) > 10
    assert np.asarray(engine._warmup_frame(32, 24)).shape == (24, 32, 3)     # a frame smaller than the rectangles' range


def test_warm_embedder_runs_the_per_frame_batch_and_half_a_batch():
    sizes, classified = [], []

    class Clip:
        def encode_images_batch(self, images, batch_size=16):
            assert all(isinstance(im, np.ndarray) and im.shape == (224, 224, 3) and im.dtype == np.uint8 for im in images)
            sizes.append((len(images), batch_size))
            return torch.ones(len(images), 4)

    class Clf:
        def ensure_text_on_device(self):
            classified.append("device")

        def classify_feats(self, feats, return_topk=None):
            classified.append((tuple(feats.shape), return_topk))
            return []

    assert engine.warm_embedder(Clip(), Clf(), {})["ms"] >= 0
    assert sizes == [(15, 16), (8, 16)] and classified == ["device", ((8, 4), 5)]
    sizes.clear()
    engine.warm_embedder(Clip(), None, {"clip": {"batch_size": 4}, "staging": {"topk_preclip": 10}})
    assert sizes == [(4, 4), (2, 4)]

    class Broken:
        def encode_images_batch(self, images, batch_size=16):
            raise RuntimeError("no device")

    assert engine.warm_embedder(Broken(), Clf(), {}) is None


def test_segmentation_vocab_per_backend():
    seg = {"yoloe": {"vocab": ["cup"]}, "grounded_sam2": {"vocab": ["chair"]}}
    for backend, want in (("dual", ["cup"]), ("yoloe", ["cup"]), ("grounded_sam2", ["chair"]), ("fastsam", None),
                          ("sam2", None), ("external", None)):
        assert segmentation_vocab({"segmentation": {"backend": backend, **seg}}) == want
    assert segmentation_vocab({}) is None


# ───────────────────────── the row count a segmenter hands over ─────────────────────────

def _fastsam_with(mask_data):
    from rtsm.models.segmentation.fastsam_segmenter import FastSAMSegmenter
    seg = FastSAMSegmenter(model_path="unused.pt", device="cpu", retina_masks=False)
    result = SimpleNamespace(masks=SimpleNamespace(data=mask_data), boxes=None)
    seg._model = lambda image, **kw: [result]
    return seg.segment(Image.new("RGB", (16, 12)))


def test_fastsam_makes_no_row_count_for_masks_on_the_cpu():
    data = (torch.rand(4, 12, 16, generator=torch.Generator().manual_seed(0)) > 0.5).float()
    out = _fastsam_with(data)
    assert out.mask_row_counts is None and out.masks.dtype is torch.bool and out.masks.shape == (4, 12, 16)


@pytest.mark.gpu
def test_fastsam_counts_rows_on_the_gpu_for_the_stack_it_returns():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    data = (torch.rand(5, 12, 16, generator=torch.Generator().manual_seed(1)) > 0.4).float().cuda()
    out = _fastsam_with(data)
    assert out.masks.device.type == "cpu" and out.mask_row_counts.dtype == np.int64
    assert np.array_equal(out.mask_row_counts, out.masks.sum(dim=2).numpy())
