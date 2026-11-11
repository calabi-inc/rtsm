"""run_heuristics computes every per-mask statistic on the mask's bounding box. This file keeps the version it
replaced, which passed over the full frame for every mask, and requires the two to agree field by field, exactly
(values, dtypes, drop details), on randomised stacks: masks on the borders, empty and one-pixel masks, a depth map
with holes or at another resolution, every erosion radius, with and without intrinsics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rtsm.utils import mask_staging as ms

STAT_FIELDS = ("idx", "area_px", "bbox", "coverage", "border_fraction", "depth_valid", "depth_p50", "depth_spread",
               "planar_inlier_pct", "planar_rms_m", "plane_normal_cam", "plane_offset_cam", "centroid_px", "centroid_cam")


def full_frame_heuristics(ann_bool, depth_m, cfg):
    """The per-mask loop as it was: every statistic from a pass over the whole frame, through the module's helpers."""
    N, H, W = ann_bool.shape
    total_px = float(H * W)
    area_min = int(cfg["filters"]["min_area_px"])
    fx, fy, cx, cy = cfg["camera"].get("fx"), cfg["camera"].get("fy"), cfg["camera"].get("cx"), cfg["camera"].get("cy")
    have_intr = all(v is not None for v in (fx, fy, cx, cy))
    min_valid_for_centroid = float(cfg.get("staging", {}).get("centroid_min_valid", 0.25))
    erode_px = int(cfg.get("staging", {}).get("depth_erode_px", 2))
    depth_valid_min = float(cfg.get("staging", {}).get("depth_valid_min", 0.15))
    depth_spread_max = float(cfg.get("filters", {}).get("depth", {}).get("sigma_max_m", 0.35))
    plan_stride = int(cfg.get("planarity", {}).get("sample_stride", 3))
    plan_rms_thr = float(cfg.get("planarity", {}).get("rms_residual_max_m", 0.02))
    track_drops = bool(cfg.get("diagnostics", {}).get("track_drops", False))
    kept, infos, drop_details = [], [], []
    rej_area = rej_valid = rej_spread = 0
    for i in range(N):
        m = ann_bool[i]
        area = int(m.sum().item())
        if area < area_min:
            rej_area += 1
            if track_drops:
                drop_details.append({"idx": i, "reason": "area", "bbox": list(ms._compute_mask_bbox(m)),
                                     "area_px": area, "area_min": area_min})
            continue
        bbox = ms._compute_mask_bbox(m)
        coverage = area / total_px
        border_fraction = ms._border_fraction(m, area)
        if depth_m is not None:
            depth_valid, depth_p50, depth_spread = ms._depth_quantiles(m, depth_m, erode_px=erode_px)
        else:
            depth_valid, depth_p50, depth_spread = 0.0, None, None
        if depth_m is not None:
            if depth_valid < depth_valid_min:
                rej_valid += 1
                if track_drops:
                    drop_details.append({"idx": i, "reason": "depth_valid", "bbox": list(bbox), "area_px": area,
                                         "coverage": float(coverage), "depth_valid": float(depth_valid),
                                         "depth_valid_min": float(depth_valid_min)})
                continue
            if depth_spread is not None and depth_spread > depth_spread_max:
                rej_spread += 1
                if track_drops:
                    drop_details.append({"idx": i, "reason": "depth_spread", "bbox": list(bbox), "area_px": area,
                                         "coverage": float(coverage), "depth_valid": float(depth_valid),
                                         "depth_spread": float(depth_spread),
                                         "depth_spread_max": float(depth_spread_max)})
                continue
        cpx = ms._centroid_px(m)
        ccam = None
        if depth_m is not None and have_intr and depth_valid >= min_valid_for_centroid:
            ccam = ms._centroid_cam(m, depth_m, fx, fy, cx, cy, stride=2, erode_px=erode_px)
        planar_inlier_pct = planar_rms_m = plane_normal_cam = plane_offset_cam = None
        if depth_m is not None and have_intr and area >= area_min:
            pts = ms._sample_cam_points_from_mask(m, depth_m, fx, fy, cx, cy, stride=max(1, plan_stride))
            if pts is not None and pts.shape[0] >= 16:
                n_cam, d_cam = ms._fit_plane_pca(pts)
                inlier_pct, rms = ms._planarity_metrics(pts, n_cam, d_cam, plan_rms_thr)
                planar_inlier_pct, planar_rms_m = float(inlier_pct), float(rms)
                plane_normal_cam, plane_offset_cam = n_cam, float(d_cam)
        kept.append(m)
        infos.append(ms.MaskStats(idx=i, area_px=area, bbox=bbox, border_fraction=border_fraction, coverage=coverage,
                                  depth_valid=depth_valid, depth_p50=depth_p50, depth_spread=depth_spread,
                                  planar_inlier_pct=planar_inlier_pct, planar_rms_m=planar_rms_m,
                                  plane_normal_cam=plane_normal_cam, plane_offset_cam=plane_offset_cam,
                                  centroid_px=cpx, centroid_cam=ccam))
    diag = ms.FilterDiagnostics(total_input=N, dropped_area=rej_area, dropped_depth_valid=rej_valid,
                                dropped_depth_spread=rej_spread, passed=len(kept), drop_details=drop_details)
    return kept, infos, diag


def _same(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
    return type(x) is type(y) and x == y


def differences(ref, new):
    """Every field in which two run_heuristics results differ (empty = identical)."""
    (k0, s0, d0), (k1, s1, d1) = ref, new
    out = []
    if len(s0) != len(s1) or len(k0) != len(k1):
        return [f"kept {len(s0)} != {len(s1)}"]
    for a, b, ma, mb in zip(s0, s1, k0, k1):
        if not torch.equal(ma, mb):
            out.append(f"mask {a.idx}: kept mask differs")
        for f in STAT_FIELDS:
            if not _same(getattr(a, f), getattr(b, f)):
                out.append(f"mask {a.idx} {f}: {getattr(a, f)!r} != {getattr(b, f)!r}")
    for f in ("total_input", "dropped_area", "dropped_depth_valid", "dropped_depth_spread", "passed", "drop_details"):
        if getattr(d0, f) != getattr(d1, f):
            out.append(f"diagnostics {f}: {getattr(d0, f)!r} != {getattr(d1, f)!r}")
    return out


def _random_mask(rng, H, W):
    m = np.zeros((H, W), dtype=bool)
    kind = int(rng.integers(0, 9))
    if kind == 0:                                           # empty
        return m
    if kind == 1:                                           # one pixel, anywhere (corners included)
        m[int(rng.integers(0, H)), int(rng.integers(0, W))] = True
        return m
    if kind == 2:                                           # the whole frame
        m[:] = True
        return m
    y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
    y1, x1 = int(rng.integers(y0 + 1, H + 1)), int(rng.integers(x0 + 1, W + 1))
    if kind == 3:                                           # a row or a column on a border
        if rng.random() < 0.5:
            m[int(rng.choice([0, H - 1])), x0:x1] = True
        else:
            m[y0:y1, int(rng.choice([0, W - 1]))] = True
        return m
    if kind in (4, 5):                                      # a rectangle, often against a border
        if rng.random() < 0.4:
            y0, x0 = 0, 0
        if rng.random() < 0.4:
            y1, x1 = H, W
        m[y0:y1, x0:x1] = True
        return m
    if kind == 6:                                           # two separate pieces
        m[y0:y1, x0:x1] = True
        yy, xx = int(rng.integers(0, H)), int(rng.integers(0, W))
        m[yy:yy + 3, xx:xx + 3] = True
        return m
    block = rng.random((y1 - y0, x1 - x0)) < float(rng.uniform(0.2, 0.95))   # a ragged blob inside a rectangle
    m[y0:y1, x0:x1] = block
    return m


def _random_depth(rng, H, W):
    choice = int(rng.integers(0, 6))
    if choice == 0:
        return None
    dh, dw = (H, W)
    if choice == 1:                                         # a depth map at another resolution (the Lens case)
        dh, dw = max(1, H // 2), max(1, W // 3)
    elif choice == 2:
        dh, dw = H + 5, W + 7
    yy, xx = np.mgrid[0:dh, 0:dw]
    d = (1.0 + 0.01 * xx + 0.02 * yy + rng.normal(0.0, float(rng.choice([0.0, 0.01, 0.3])), (dh, dw))).astype(np.float32)
    holes = rng.random((dh, dw)) < float(rng.choice([0.0, 0.1, 0.6, 1.0]))
    d[holes] = rng.choice(np.array([np.nan, 0.0, -1.0, np.inf], dtype=np.float32), size=int(holes.sum()))
    return d


def _random_cfg(rng, H, W):
    cam = {"fx": 30.0 + float(rng.random()), "fy": 31.0, "cx": W / 2.0, "cy": H / 2.0}
    if rng.random() < 0.15:
        cam = {}
    return {"filters": {"min_area_px": int(rng.choice([0, 1, 4, 25])),
                        "depth": {"sigma_max_m": float(rng.choice([0.02, 0.35, 5.0]))}},
            "camera": cam,
            "staging": {"depth_erode_px": int(rng.choice([0, 1, 2, 3])),
                        "depth_valid_min": float(rng.choice([0.0, 0.02, 0.15, 0.5])),
                        "centroid_min_valid": float(rng.choice([0.0, 0.05, 0.25]))},
            "planarity": {"sample_stride": int(rng.choice([1, 3])), "rms_residual_max_m": 0.02},
            "diagnostics": {"track_drops": bool(rng.random() < 0.5)}}


SHAPES = [(48, 64), (37, 53), (1, 40), (40, 1), (2, 2), (64, 64), (30, 96)]


@pytest.mark.parametrize("seed", range(40))
def test_bounding_box_statistics_equal_the_full_frame_ones(seed):
    rng = np.random.default_rng(seed)
    for _ in range(8):
        H, W = SHAPES[int(rng.integers(0, len(SHAPES)))]
        n = int(rng.integers(0, 9))
        ann = torch.from_numpy(np.stack([_random_mask(rng, H, W) for _ in range(n)]) if n else np.zeros((0, H, W), dtype=bool))
        depth = _random_depth(rng, H, W)
        cfg = _random_cfg(rng, H, W)
        ref = full_frame_heuristics(ann, depth, cfg)
        new = ms.run_heuristics(ann, depth, cfg)
        assert differences(ref, new) == []


def test_every_kind_of_mask_is_reached():
    """The generator above must produce kept masks with every optional field filled, or the comparison proves little."""
    rng = np.random.default_rng(7)
    seen = {"kept": 0, "centroid_cam": 0, "planar": 0, "area": 0, "depth_valid": 0, "depth_spread": 0}
    for _ in range(400):
        H, W = SHAPES[int(rng.integers(0, len(SHAPES)))]
        n = int(rng.integers(1, 9))
        ann = torch.from_numpy(np.stack([_random_mask(rng, H, W) for _ in range(n)]))
        _, stats, diag = ms.run_heuristics(ann, _random_depth(rng, H, W), _random_cfg(rng, H, W))
        seen["kept"] += len(stats)
        seen["centroid_cam"] += sum(s.centroid_cam is not None for s in stats)
        seen["planar"] += sum(s.plane_normal_cam is not None for s in stats)
        seen["area"] += diag.dropped_area
        seen["depth_valid"] += diag.dropped_depth_valid
        seen["depth_spread"] += diag.dropped_depth_spread
    assert all(v > 20 for v in seen.values()), seen


def test_no_masks():
    cfg = {"filters": {"min_area_px": 10}, "camera": {}}
    for ann in (torch.zeros((0, 1, 1), dtype=torch.bool), torch.zeros((0, 12, 16), dtype=torch.bool)):
        kept, stats, diag = ms.run_heuristics(ann, np.ones((12, 16), np.float32), cfg)
        assert kept == [] and stats == [] and diag.total_input == 0 and diag.passed == 0


def test_an_empty_mask_passes_an_area_floor_of_zero_like_before():
    ann = torch.zeros((1, 8, 8), dtype=torch.bool)
    cfg = {"filters": {"min_area_px": 0}, "camera": {"fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 4.0},
           "staging": {"depth_valid_min": 0.0}}
    for depth in (None, np.ones((8, 8), np.float32)):
        ref, new = full_frame_heuristics(ann, depth, cfg), ms.run_heuristics(ann, depth, cfg)
        assert differences(ref, new) == [] and len(new[1]) == 1 and new[1][0].border_fraction == 0.0


def _stack(seed=3, n=6, H=40, W=56):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([_random_mask(rng, H, W) for _ in range(n)])), _random_depth(np.random.default_rng(1), H, W)


def test_row_counts_from_the_segmenter_give_the_same_statistics():
    ann, depth = _stack()
    cfg = {"filters": {"min_area_px": 4}, "camera": {"fx": 30.0, "fy": 30.0, "cx": 28.0, "cy": 20.0},
           "diagnostics": {"track_drops": True}}
    ref = ms.run_heuristics(ann, depth, cfg)
    counts = ann.sum(dim=2).numpy()
    assert differences(ref, ms.run_heuristics(ann, depth, cfg, row_counts=counts)) == []
    assert differences(ref, ms.run_heuristics(ann, depth, cfg, row_counts=counts.astype(np.int32))) == []


def test_row_counts_of_another_stack_are_not_used():
    """A count whose shape is not (N, H) cannot belong to this stack: the function counts for itself."""
    ann, depth = _stack()
    cfg = {"filters": {"min_area_px": 4}, "camera": {}}
    ref = ms.run_heuristics(ann, depth, cfg)
    for wrong in (np.zeros((ann.shape[0] + 1, ann.shape[1]), np.int64), np.zeros((ann.shape[0], ann.shape[1] - 1), np.int64),
                  np.zeros((ann.shape[0],), np.int64)):
        assert differences(ref, ms.run_heuristics(ann, depth, cfg, row_counts=wrong)) == []


def test_row_count_helper():
    rng = np.random.default_rng(0)
    a = rng.random((5, 17, 23)) < 0.3
    assert np.array_equal(ms._row_counts(a), a.sum(axis=2)) and ms._row_counts(a).dtype == np.int64
    assert np.array_equal(ms._row_counts(a[:, ::2, ::-1]), a[:, ::2, ::-1].sum(axis=2))      # a view that is not contiguous
    assert ms._row_counts(np.zeros((0, 4, 4), bool)).shape == (0, 4) and ms._row_counts(np.zeros((3, 4, 0), bool)).shape == (3, 4)
