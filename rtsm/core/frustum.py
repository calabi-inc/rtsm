"""
Frustum projection for the P2 ``view`` ledger line (Gate 4.5 plan, P2 stage C).

Pure NumPy, no state. The projection is the associator's
(``rtsm/core/association._project_px``) vectorised: world -> camera with
``T_cam_world``, pinhole with the RGB-space intrinsics; a point is IN FRUSTUM
when ``z > z_min`` and its pixel lies inside the RGB image. The observed depth
is sampled from the frame's depth map at the projected pixel with the SAME
RGB -> depth coordinate rule the mask stage uses
(``rtsm/utils/mask_staging._depth_coords``; session1's depth is 256x192 under a
1920x1440 RGB), as the NaN-aware median of a small window.

Model ``v1_occlusion_agnostic``: an object behind a wall but inside the image is
listed with ``expected_depth < observed_depth``; the consumer (P5's decay, P3's
metrics) decides visibility from the two numbers. Nothing here judges.

Pose math: tested against ``_project_px`` point by point on random poses
(``tests/evaluation/test_frustum.py``) -- the CLAUDE.md rule for anything that
touches poses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FRUSTUM_MODEL_V1 = "v1_occlusion_agnostic"
Z_MIN_M = 0.05           # closer than this is "at the lens", not in view
BEHIND_EPS = 1e-6        # the associator's own behind-camera threshold


def project_points(P: Any, T_cam_world: Any, intr: Dict[str, float], hw: Sequence[int],
                   z_min: float = Z_MIN_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project N world points. Returns ``(u, v, z, in_frustum)``: ``u``/``v`` in
    RGB pixels (NaN where ``z <= 1e-6``, where the associator returns None),
    ``z`` the camera-frame depth, ``in_frustum`` = ``z > z_min`` and
    ``0 <= u < W`` and ``0 <= v < H`` with ``hw = (H, W)``."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T_cam_world, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    pc = P @ R.T + t
    z = pc[:, 2]
    fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    ok = z > BEHIND_EPS
    u = np.full(P.shape[0], np.nan)
    v = np.full(P.shape[0], np.nan)
    u[ok] = fx * pc[ok, 0] / z[ok] + cx
    v[ok] = fy * pc[ok, 1] / z[ok] + cy
    H, W = int(hw[0]), int(hw[1])
    inside = ok & (z > float(z_min)) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, inside


def sample_depth(depth_m: Optional[np.ndarray], us: Any, vs: Any, rgb_hw: Sequence[int],
                 window: int = 3) -> np.ndarray:
    """NaN-aware median of a ``window x window`` neighbourhood of the depth map
    around each projected RGB pixel, after mapping the pixel to depth
    resolution with the mask stage's rule. NaN where no finite, positive depth
    is found (or no depth map)."""
    us = np.asarray(us, dtype=np.float64).reshape(-1)
    vs = np.asarray(vs, dtype=np.float64).reshape(-1)
    out = np.full(us.shape[0], np.nan)
    if depth_m is None or us.shape[0] == 0:
        return out
    from rtsm.utils.mask_staging import _depth_coords
    d = np.asarray(depth_m)
    dh, dw = int(d.shape[0]), int(d.shape[1])
    H, W = int(rgb_hw[0]), int(rgb_hw[1])
    fin = np.isfinite(us) & np.isfinite(vs)
    ys = np.clip(np.floor(np.where(fin, vs, 0.0)), 0, H - 1).astype(int)
    xs = np.clip(np.floor(np.where(fin, us, 0.0)), 0, W - 1).astype(int)
    dy, dx = _depth_coords(ys, xs, (dh, dw), (H, W))
    r = max(0, int(window) // 2)
    for i in range(us.shape[0]):
        if not fin[i]:
            continue
        y0, y1 = max(0, int(dy[i]) - r), min(dh, int(dy[i]) + r + 1)
        x0, x1 = max(0, int(dx[i]) - r), min(dw, int(dx[i]) + r + 1)
        win = d[y0:y1, x0:x1]
        good = win[np.isfinite(win) & (win > 0)]
        if good.size:
            out[i] = float(np.median(good))
    return out


def frustum_view(objects: Sequence[Any], T_cam_world: Any, intr: Dict[str, float], rgb_hw: Sequence[int],
                 depth_m: Optional[np.ndarray] = None, *, z_min: float = Z_MIN_M,
                 window: int = 3) -> Tuple[List[Dict[str, Any]], int]:
    """The in-frustum subset of ``objects`` (duck-typed: ``id``, ``xyz_world``,
    ``confirmed``, ``hits``, ``stability``, ``label_primary``) as ledger entries
    with ``expected_depth`` (camera z of the stored position) and
    ``observed_depth`` (the depth map at that pixel), plus the number of
    objects considered."""
    n = len(objects)
    if n == 0:
        return [], 0
    P = np.stack([np.asarray(o.xyz_world, dtype=np.float64).reshape(3) for o in objects])
    u, v, z, inside = project_points(P, T_cam_world, intr, rgb_hw, z_min)
    idx = np.flatnonzero(inside)
    observed = sample_depth(depth_m, u[idx], v[idx], rgb_hw, window) if idx.size else np.zeros(0)
    entries: List[Dict[str, Any]] = []
    for j, i in enumerate(idx):
        o = objects[i]
        od = float(observed[j]) if j < observed.shape[0] else float("nan")
        entries.append({
            "id": getattr(o, "id", None),
            "confirmed": bool(getattr(o, "confirmed", False)),
            "hits": int(getattr(o, "hits", 0) or 0),
            "stability": round(float(getattr(o, "stability", 0.0) or 0.0), 3),
            "label_primary": getattr(o, "label_primary", None),
            "u": round(float(u[i]), 1),
            "v": round(float(v[i]), 1),
            "expected_depth": round(float(z[i]), 3),
            "observed_depth": (round(od, 3) if np.isfinite(od) else None),
        })
    return entries, n
