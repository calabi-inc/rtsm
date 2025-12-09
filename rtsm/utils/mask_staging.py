"""
Mask Staging: filtering, scoring, and annotating instance segmentation masks
before they are used for downstream tasks such as tracking, association, or 3D reconstruction.
used for downstream tasks such as tracking, association, or 3D reconstruction.

This module provides utilities to:
  - Compute per-mask statistics (area, coverage, border fraction, etc.)
  - Apply heuristic filters (e.g., min area, border contact, planarity)
  - Annotate masks with geometric features (centroid, planarity, depth stats)
  - Select and prioritize masks for further processing

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import cv2

# ---------- erosion helper ----------

def _erode_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Erode mask by kernel_size to remove edge pixels before depth sampling."""
    m_np = mask.numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(m_np, kernel, iterations=1)
    return torch.from_numpy(eroded.astype(bool))

# ---------- mask stats ----------

@dataclass
class MaskStats:
    idx: int
    area_px: int
    bbox: Tuple[int,int,int,int]
    # --- SOFT FEATURES (drive priority + acceptance) ---
    coverage: float
    border_fraction: float              # fraction of mask pixels on image border
    depth_valid: float
    depth_p50: Optional[float]
    depth_spread: Optional[float]
    # --- GEOMETRY (fast planarity estimate) ---
    planar_inlier_pct: Optional[float] = None   # fraction of samples within rms threshold
    planar_rms_m: Optional[float] = None        # RMS point-to-plane distance (m)
    plane_normal_cam: Optional[np.ndarray] = None  # (3,) unit normal in camera frame
    plane_offset_cam: Optional[float] = None    # signed offset d in n·x + d = 0 (>=0)
    # --- UTILITIES (for merge/association; not soft features) ---
    centroid_px: Optional[Tuple[float,float]] = None
    centroid_cam: Optional[np.ndarray] = None

    # TODO: Add planarity for potential structure estimation

def _compute_mask_bbox(mask: torch.Tensor) -> Tuple[int,int,int,int]:
    """Tight bbox (x0,y0,x1,y1) for a binary mask [H,W] (x1/y1 exclusive)."""
    yx = mask.nonzero()
    if yx.numel() == 0:
        return (0,0,0,0)
    y0 = int(yx[:,0].min()); y1 = int(yx[:,0].max()) + 1
    x0 = int(yx[:,1].min()); x1 = int(yx[:,1].max()) + 1
    return (x0,y0,x1,y1)

def _depth_quantiles(mask: torch.Tensor, depth_m: np.ndarray, erode_px: int = 0):
    # Optionally erode mask to avoid edge depth artifacts
    if erode_px > 0:
        mask = _erode_mask(mask, kernel_size=erode_px * 2 + 1)
    m = mask.contiguous().numpy()
    z = depth_m[m]
    good = np.isfinite(z) & (z > 0.0)
    if not good.any(): return 0.0, None, None  # valid_pct, p50, spread
    zg = z[good]
    p10, p50, p90 = np.percentile(zg, [10, 50, 90]).astype(np.float32)
    valid_pct = float(good.mean())
    return valid_pct, float(p50), float(p90 - p10)

def _centroid_px(mask: torch.Tensor) -> Optional[Tuple[float,float]]:
    yx = mask.nonzero()
    if yx.numel() == 0: return None
    y_mean = float(yx[:,0].float().mean().item())
    x_mean = float(yx[:,1].float().mean().item())
    return (x_mean, y_mean)

def _centroid_cam(mask: torch.Tensor, depth_m: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float,
                  stride: int = 2, erode_px: int = 0) -> Optional[np.ndarray]:
    """Compute 3D centroid in camera frame from mask and depth.

    Uses fast vectorized single-pixel depth sampling.
    """
    # Optionally erode mask to avoid edge depth artifacts
    if erode_px > 0:
        mask = _erode_mask(mask, kernel_size=erode_px * 2 + 1)
    m = mask.contiguous().numpy()
    ys, xs = np.where(m)
    if ys.size == 0: return None
    ys = ys[::stride]; xs = xs[::stride]
    if ys.size == 0: return None

    # Fast vectorized single-pixel depth sampling
    z = depth_m[ys, xs].astype(np.float32)

    good = np.isfinite(z) & (z > 0.0)
    if not good.any(): return None
    ys = ys[good].astype(np.float32)
    xs = xs[good].astype(np.float32)
    z  = z[good]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    Z = z
    c = np.array([X.mean(), Y.mean(), Z.mean()], dtype=np.float32)
    return c

def _border_fraction(mask: torch.Tensor, total: int) -> float:
    if total == 0: return 0.0
    edge = (mask[0,:].sum() + mask[-1,:].sum() + mask[:,0].sum() + mask[:,-1].sum())
    return float(edge.item()) / float(total)


# ---------- fast planarity (subsample + PCA/SVD) ----------
def _sample_cam_points_from_mask(
    mask: torch.Tensor,
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    stride: int,
) -> Optional[np.ndarray]:
    m = mask.contiguous().numpy()
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    ys = ys[::stride]
    xs = xs[::stride]
    if ys.size == 0:
        return None
    z = depth_m[ys, xs].astype(np.float32)
    good = np.isfinite(z) & (z > 0.0)
    if not good.any():
        return None
    ys = ys[good].astype(np.float32)
    xs = xs[good].astype(np.float32)
    z  = z[good]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    pts = np.stack([X, Y, z], axis=1).astype(np.float32)
    if pts.shape[0] < 3:
        return None
    return pts


def _fit_plane_pca(points_cam: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit plane to points (N,3) in camera frame using PCA.
    Returns (unit_normal, signed_offset_d) for plane n·x + d = 0, with d>=0.
    """
    p0 = points_cam.mean(axis=0)
    Q = points_cam - p0
    # covariance SVD; smallest singular vector is normal
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1, :].astype(np.float32)
    n_norm = float(np.linalg.norm(n) + 1e-12)
    n = n / n_norm
    d = float(-np.dot(n, p0.astype(np.float32)))
    # ensure non-negative offset for consistency
    if d < 0.0:
        n = -n
        d = -d
    return n.astype(np.float32), float(d)


def _planarity_metrics(
    points_cam: np.ndarray,
    n: np.ndarray,
    d: float,
    inlier_rms_thresh_m: float,
) -> Tuple[float, float]:
    """
    Compute planar_inlier_pct and planar_rms_m given plane parameters.
    Distance is point-to-plane |n·x + d| (since n normalized).
    planar_rms_m is RMS over all samples; inlier pct uses threshold.
    """
    resid = np.abs(points_cam @ n.reshape(3,) + d).astype(np.float32)
    if resid.size == 0:
        return 0.0, None  # type: ignore
    rms = float(np.sqrt(np.mean(resid * resid)))
    inlier = (resid <= float(inlier_rms_thresh_m))
    inlier_pct = float(np.count_nonzero(inlier)) / float(resid.size)
    return inlier_pct, rms


# ---------- predicate checks ----------
def min_area_ok(area: int, min_area_px: int) -> bool:
    return area >= min_area_px

def calculate_mask_stats(mask: torch.Tensor, depth_m: Optional[np.ndarray], intrinsics: Optional[np.ndarray], H: int, W: int, cfg: Dict[str, Any]):
    area = int(mask.sum().item())
    border_fraction = _border_fraction(mask, area)
    bbox = _compute_mask_bbox(mask)
    coverage = float(mask.sum().item()) / (H * W)
    border_fraction = float(mask[0,:].sum() + mask[-1,:].sum() + mask[:,0].sum() + mask[:,-1].sum()) / (H * W)

    return area, coverage, bbox, border_fraction
# ---------- main: per-mask heuristic stack ----------

def run_heuristics(      
    ann_bool: torch.Tensor,                # [N,H,W] torch.bool CPU (from prepare_ann)
    depth_m: Optional[np.ndarray],         # [H,W] float32 meters or None   
    cfg,
) -> Tuple[List[torch.Tensor], List[MaskStats]]:
    """
    Returns:
        kept_masks: list of torch.bool views (aligned to image grid)
        stats:      list of MaskStats (same order)
    """
    assert ann_bool.dtype is torch.bool and ann_bool.device.type == "cpu" and ann_bool.ndim == 3
    kept: List[torch.Tensor] = []
    infos: List[MaskStats] = []

    N, H, W = ann_bool.shape
    total_px = float(H * W)
    area_min = int(cfg["filters"]["min_area_px"])

    # Depth range gates (treat outside as invalid)
    zmin = cfg.get("filters", {}).get("depth", {}).get("z_min_m", 0.20)   # D435i safe default
    zmax = cfg.get("filters", {}).get("depth", {}).get("z_max_m", 5.00)   # indoors default

    # Camera intrinsics for centroid_cam/planarity
    fx, fy, cx, cy = cfg["camera"].get("fx"), cfg["camera"].get("fy"), cfg["camera"].get("cx"), cfg["camera"].get("cy")
    have_intr = all(v is not None for v in (fx, fy, cx, cy))
    min_valid_for_centroid = float(cfg.get("staging", {}).get("centroid_min_valid", 0.25))

    # Depth edge filtering config
    erode_px = int(cfg.get("staging", {}).get("depth_erode_px", 2))
    depth_valid_min = float(cfg.get("staging", {}).get("depth_valid_min", 0.15))
    depth_spread_max = float(cfg.get("filters", {}).get("depth", {}).get("sigma_max_m", 0.35))

    # Planarity cfg
    plan_stride = int(cfg.get("planarity", {}).get("sample_stride", 3))
    plan_rms_thr = float(cfg.get("planarity", {}).get("rms_residual_max_m", 0.02))

    for i in range(N):
        m = ann_bool[i]
        area = int(m.sum().item())
        if area < area_min:
            continue

        bbox = _compute_mask_bbox(m)
        coverage = area / total_px
        border_fraction = _border_fraction(m, area)

        # depth features (use eroded mask to avoid edge artifacts)
        if depth_m is not None:
            depth_valid, depth_p50, depth_spread = _depth_quantiles(m, depth_m, erode_px=erode_px)
        else:
            depth_valid, depth_p50, depth_spread = 0.0, None, None

        # Hard rejection: skip masks with poor depth quality
        if depth_m is not None:
            if depth_valid < depth_valid_min:
                continue
            if depth_spread is not None and depth_spread > depth_spread_max:
                continue

        # centroids (use eroded mask for 3D centroid)
        cpx = _centroid_px(m)
        ccam: Optional[np.ndarray] = None
        if depth_m is not None and have_intr and depth_valid >= min_valid_for_centroid:
            ccam = _centroid_cam(m, depth_m, fx, fy, cx, cy, stride=2, erode_px=erode_px)

        # planarity (cheap; annotate only)
        planar_inlier_pct: Optional[float] = None
        planar_rms_m: Optional[float] = None
        plane_normal_cam: Optional[np.ndarray] = None
        plane_offset_cam: Optional[float] = None
        if depth_m is not None and have_intr and area >= area_min:
            pts = _sample_cam_points_from_mask(m, depth_m, fx, fy, cx, cy, stride=max(1, plan_stride))
            if pts is not None and pts.shape[0] >= 16:  # need some support
                n_cam, d_cam = _fit_plane_pca(pts)
                inlier_pct, rms = _planarity_metrics(pts, n_cam, d_cam, plan_rms_thr)
                planar_inlier_pct = float(inlier_pct)
                planar_rms_m = float(rms)
                plane_normal_cam = n_cam
                plane_offset_cam = float(d_cam)

        stats = MaskStats(
            idx=i,
            area_px=area,
            bbox=bbox,
            border_fraction=border_fraction,
            coverage=coverage,
            depth_valid=depth_valid,
            depth_p50=depth_p50,
            depth_spread=depth_spread,
            planar_inlier_pct=planar_inlier_pct,
            planar_rms_m=planar_rms_m,
            plane_normal_cam=plane_normal_cam,
            plane_offset_cam=plane_offset_cam,
            centroid_px=cpx,
            centroid_cam=ccam,
        )
        kept.append(m)
        infos.append(stats)

    return kept, infos