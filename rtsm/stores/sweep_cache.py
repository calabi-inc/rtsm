"""
SweepCache — SE(3)-aware sweeps (cell, vbin) state and helpers.

This is a drop-in, testable sweep cache with:
  - (cell, vbin) sweep state: last_ts, last_cam_pos, tiny look-cell LRU
  - Helpers to quantize cells and view bins (yaw/pitch bins)

"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int, int]
VBin = Tuple[int, int]  # (yaw_bin, pitch_bin)

class SweepCache:
    def __init__(
        self,
        grid_size_m: float = 0.25,
        per_cell_cap: int = 64,
        neighbors_max: int = 128,
        two_d: bool = False,
        yaw_bins: int = 12,
        pitch_bins: int = 5,
        pitch_deg: float = 60.0,
        look_lru_keep: int = 8,
    ) -> None:
        # --- spatial grid ---
        self.grid = float(grid_size_m)
        self.cap = int(per_cell_cap)
        self.neighbors_max = int(neighbors_max)
        self.two_d = bool(two_d)

        # --- membership ---
        self.h: Dict[Cell, Set[str]] = defaultdict(set)   # cell -> {oid}
        self.oid_cell: Dict[str, Cell] = {}               # reverse map

        # --- sweep state keyed by (cell, vbin) ---
        self.yaw_bins = int(yaw_bins)
        self.pitch_bins = int(pitch_bins)
        self.pitch_max = math.radians(float(pitch_deg))
        self.look_keep = int(look_lru_keep)

        self.swept_view_ts: Dict[Tuple[Cell, VBin], float] = {}
        self.last_cam_pos: Dict[Tuple[Cell, VBin], np.ndarray] = {}
        self.recent_look_cells: Dict[Tuple[Cell, VBin], Deque[Cell]] = {}

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------
    def cell_from_xyz(self, xyz: np.ndarray) -> Cell:
        gx = int(math.floor(float(xyz[0]) / self.grid))
        gy = int(math.floor(float(xyz[1]) / self.grid))
        if self.two_d:
            gz = 0
        else:
            gz = int(math.floor(float(xyz[2]) / self.grid))
        return (gx, gy, gz)

    def vbin_from_forward(self, fwd: np.ndarray) -> VBin:
        # expects a (3,) vector; will normalize
        f = np.asarray(fwd, dtype=float)
        n = float(np.linalg.norm(f))
        if n == 0.0:
            # degenerate; put into bin 0, mid pitch
            return (0, self.pitch_bins // 2)
        f = f / n
        # yaw in [-pi, pi]
        yaw = math.atan2(float(f[1]), float(f[0]))
        # pitch in [-pi/2, pi/2]
        pitch = math.asin(max(-1.0, min(1.0, float(f[2]))))
        # clamp pitch to useful range ±pitch_max
        pitch = max(-self.pitch_max, min(self.pitch_max, pitch))
        # normalize to [0,1]
        yaw_u = (yaw + math.pi) / (2.0 * math.pi)
        pitch_u = (pitch + self.pitch_max) / (2.0 * self.pitch_max if self.pitch_max > 0 else 1.0)
        yb = int(self.yaw_bins * yaw_u) % self.yaw_bins
        pb = int(self.pitch_bins * pitch_u)
        # guard upper edge
        pb = min(pb, self.pitch_bins - 1)
        return (yb, pb)

    # Convenience: forward vector from R_wc (3x3) or quaternion (xyzw)
    @staticmethod
    def forward_from_Rwc(Rwc: np.ndarray) -> np.ndarray:
        # optical +Z forward convention
        return np.asarray(Rwc, dtype=float) @ np.array([0.0, 0.0, 1.0])

    @staticmethod
    def R_from_quat_xyzw(q: Iterable[float]) -> np.ndarray:
        x, y, z, w = map(float, q)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Sweep state (cell, vbin)
    # ------------------------------------------------------------------
    def cell_view_age(self, cell: Cell, vbin: VBin, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        key = (cell, vbin)
        t = self.swept_view_ts.get(key)
        if t is None:
            return float("inf")
        return now - t

    def mark_cell_view_swept(self, cell: Cell, vbin: VBin, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self.swept_view_ts[(cell, vbin)] = now

    # last camera position per (cell, vbin)
    def get_last_cam_pos(self, cell: Cell, vbin: VBin) -> Optional[np.ndarray]:
        return self.last_cam_pos.get((cell, vbin))

    def set_last_cam_pos(self, cell: Cell, vbin: VBin, cam_pos: np.ndarray) -> None:
        self.last_cam_pos[(cell, vbin)] = np.asarray(cam_pos, dtype=float).copy()

    # look-cell LRU per (cell, vbin)
    def look_cell_recent(self, cell: Cell, vbin: VBin, look_cell: Cell) -> bool:
        dq = self.recent_look_cells.get((cell, vbin))
        if dq is None:
            return False
        return look_cell in dq

    def remember_look_cell(self, cell: Cell, vbin: VBin, look_cell: Cell, keep: Optional[int] = None) -> None:
        key = (cell, vbin)
        if key not in self.recent_look_cells:
            self.recent_look_cells[key] = deque(maxlen=int(keep or self.look_keep))
        dq = self.recent_look_cells[key]
        if look_cell in dq:
            return
        dq.append(look_cell)

    # Convenience: compute perpendicular (parallax) baseline given stored last cam pos
    def baseline_perp(self, cell: Cell, vbin: VBin, cam_pos: np.ndarray, fwd_unit: np.ndarray) -> float:
        last = self.get_last_cam_pos(cell, vbin)
        if last is None:
            return float("inf")
        f = np.asarray(fwd_unit, dtype=float)
        n = float(np.linalg.norm(f))
        if n == 0.0:
            return 0.0
        f = f / n
        delta = np.asarray(cam_pos, dtype=float) - last
        proj = float(delta @ f)
        perp = delta - proj * f
        return float(np.linalg.norm(perp))

    # ------------------------------------------------------------------
    # Tiny helpers for common pose → (cell, vbin)
    # ------------------------------------------------------------------
    def cell_and_vbin_from_pose(
        self,
        twc_xyz: np.ndarray,
        q_wc_xyzw: Optional[Iterable[float]] = None,
        Rwc: Optional[np.ndarray] = None,
    ) -> Tuple[Cell, VBin, np.ndarray]:
        """Return (cell, vbin, forward_world) from translation + rotation.
        Provide either quaternion (xyzw) or Rwc.
        """
        cell = self.cell_from_xyz(twc_xyz)
        if Rwc is None:
            assert q_wc_xyzw is not None, "Provide q_wc_xyzw or Rwc"
            Rwc = self.R_from_quat_xyzw(q_wc_xyzw)
        fwd = self.forward_from_Rwc(Rwc)
        vbin = self.vbin_from_forward(fwd)
        return cell, vbin, fwd

    # ------------------------------------------------------------------
    # Clear / Reset
    # ------------------------------------------------------------------
    def clear(self) -> Dict[str, int]:
        """
        Clear all sweep cache state.

        Clears object membership, sweep timestamps, camera positions, and look-cell LRUs.
        Forces full re-sweep on next pipeline run.

        Returns dict with counts of what was cleared.
        """
        cells_count = len(self.h)
        oids_count = len(self.oid_cell)
        view_states = len(self.swept_view_ts)
        cam_snapshots = len(self.last_cam_pos)
        look_lrus = len(self.recent_look_cells)

        # Clear object membership
        self.h.clear()
        self.oid_cell.clear()

        # Clear sweep state
        self.swept_view_ts.clear()
        self.last_cam_pos.clear()
        self.recent_look_cells.clear()

        return {
            "cells_cleared": cells_count,
            "objects_cleared": oids_count,
            "view_states_cleared": view_states,
            "cam_snapshots_cleared": cam_snapshots,
            "look_lrus_cleared": look_lrus,
        }

    def stats(self) -> Dict[str, int]:
        """Return current sweep cache statistics."""
        return {
            "cells": len(self.h),
            "objects": len(self.oid_cell),
            "view_states": len(self.swept_view_ts),
            "cam_snapshots": len(self.last_cam_pos),
            "look_lrus": len(self.recent_look_cells),
        }


