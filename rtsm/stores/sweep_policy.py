"""
SweepPolicy: logic for deciding when to perform
a "heavy" sweep of a spatial cell and view bin, based on time-to-live (TTL),
parallax/baseline, and optional novelty triggers. The policy is configurable
and expects a SweepCache-like object for state tracking.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

# ------------------------- Bucket interface expected by SweepPolicy -------------------------

class SweepCacheLike(Protocol):
    """Protocol for the minimal methods SweepPolicy needs from the SweepGrid.

    API:
      - cell_view_age: seconds since last sweep for (cell, vbin)
      - get_last_cam_pos / set_last_cam_pos: track last camera position for (cell, vbin)
      - look_cell_recent / remember_look_cell: tiny LRU of recently hit target cells
      - mark_cell_view_swept: record the last sweep timestamp for (cell, vbin)
    """

    def cell_view_age(
        self, cell: Tuple[int, int, int], vbin: Tuple[int, int], now: Optional[float] = None
    ) -> float: ...

    def get_last_cam_pos(
        self, cell: Tuple[int, int, int], vbin: Tuple[int, int]
    ) -> Optional[np.ndarray]: ...

    def set_last_cam_pos(
        self, cell: Tuple[int, int, int], vbin: Tuple[int, int], cam_pos: np.ndarray
    ) -> None: ...

    def look_cell_recent(
        self, cell: Tuple[int, int, int], vbin: Tuple[int, int], look_cell: Tuple[int, int, int]
    ) -> bool: ...

    def remember_look_cell(
        self,
        cell: Tuple[int, int, int],
        vbin: Tuple[int, int],
        look_cell: Tuple[int, int, int],
        keep: int = 8,
    ) -> None: ...

    def mark_cell_view_swept(
        self, cell: Tuple[int, int, int], vbin: Tuple[int, int], now: Optional[float] = None
    ) -> None: ...


# ------------------------- Policy configuration and decision -------------------------

@dataclass
class SweepPolicyConfig:
    """Knobs to tune when a (cell, vbin) should be (re)processed.

    ttl_s: soft cadence; consider sweeping when age >= ttl_s *and* novelty exists.
    hard_max_s: absolute max; sweep even without novelty when age >= hard_max_s.
    min_baseline_m: flat parallax baseline in meters (sideways motion threshold).
    k_depth: scales baseline with range Z; effective threshold is max(min_baseline_m, k_depth * Z).
    keep_lookcells: tiny LRU size for recently hit look-cells per (cell, vbin).
    require_novelty_on_ttl: if True, age>=ttl_s also needs parallax or new look-cell.
    """

    ttl_s: float = 4.0
    hard_max_s: float = 20.0
    min_baseline_m: float = 0.20
    k_depth: float = 0.02
    keep_lookcells: int = 8
    require_novelty_on_ttl: bool = True


@dataclass
class SweepDecision:
    do_sweep: bool
    reason: str
    age_s: float
    baseline_perp_m: float
    min_required_baseline_m: float
    look_is_new: bool

# ------------------------- Sweep policy -------------------------

class SweepPolicy:

    def __init__(self, cfg: Optional[SweepPolicyConfig] = None) -> None:
        self.cfg = cfg or SweepPolicyConfig()

    #  core math helpers
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v
        return v / n

    def baseline_perp(self, last_pos: np.ndarray, cam_pos: np.ndarray, fwd_unit: np.ndarray) -> float:
        """Sideways motion magnitude relative to forward direction.

        baseline_perp = || (cam_pos - last_pos) - ((cam_pos - last_pos)·f) f ||
        """
        f = self._normalize(fwd_unit)
        delta = cam_pos - last_pos
        proj = float(delta @ f)
        perp = delta - proj * f
        return float(np.linalg.norm(perp))

    def min_baseline(self, Z: Optional[float]) -> float:
        if Z is None or not np.isfinite(Z):
            return self.cfg.min_baseline_m
        return max(self.cfg.min_baseline_m, self.cfg.k_depth * float(Z))

    # ------------------------- decision -------------------------

    def decide(
        self,
        sweep_cache: SweepCacheLike,
        *,
        cell: Tuple[int, int, int],
        vbin: Tuple[int, int],
        cam_pos: np.ndarray,
        fwd_unit: np.ndarray,
        Z: Optional[float],
        look_cell: Optional[Tuple[int, int, int]] = None,
        now: Optional[float] = None,
    ) -> SweepDecision:
        """Return a rich decision for whether to sweep.

        Parameters
        ----------
        sweep_cache : SweepCacheLike
            SweepCache-like state store.
        cell, vbin : tuple
            Keys for staleness state.
        cam_pos : np.ndarray shape (3,)
            Current camera position in world.
        fwd_unit : np.ndarray shape (3,)
            Forward direction (need not be normalized).
        Z : float or None
            Working range (e.g., center-pixel depth in meters). If None, only flat baseline applies.
        look_cell : tuple or None
            Target cell hit by center ray (optional novelty signal).
        now : float or None
            Timestamp for age computation. Defaults to time.monotonic().
        """
        now = time.monotonic() if now is None else now
        age = sweep_cache.cell_view_age(cell, vbin, now)
        # hard backstop regardless of novelty
        if age >= self.cfg.hard_max_s:
            return SweepDecision(True, "hard_max_s", age, 0.0, self.min_baseline(Z), False)

        # parallax baseline
        last_pos = sweep_cache.get_last_cam_pos(cell, vbin)
        if last_pos is None:
            # Never seen this (cell,vbin) before → force sweep
            baseline = float("inf")
        else:
            baseline = self.baseline_perp(last_pos, cam_pos, fwd_unit)

        min_base = self.min_baseline(Z)

        # look-cell novelty gate
        look_is_new = False
        if look_cell is not None:
            look_is_new = not sweep_cache.look_cell_recent(cell, vbin, look_cell)

        # Primary triggers
        if baseline >= min_base:
            return SweepDecision(True, "parallax", age, baseline, min_base, look_is_new)

        if age >= self.cfg.ttl_s:
            if not self.cfg.require_novelty_on_ttl or baseline >= self.cfg.min_baseline_m or look_is_new:
                return SweepDecision(True, "ttl+novelty", age, baseline, min_base, look_is_new)

        return SweepDecision(False, "skip", age, baseline, min_base, look_is_new)

    # ------------------------- bookkeeping after a sweep -------------------------

    def record_after_sweep(
        self,
        sweep_cache: SweepCacheLike,
        *,
        cell: Tuple[int, int, int],
        vbin: Tuple[int, int],
        cam_pos: np.ndarray,
        look_cell: Optional[Tuple[int, int, int]] = None,
        now: Optional[float] = None,
    ) -> None:
        """Update bucket state after a successful heavy pass."""
        sweep_cache.mark_cell_view_swept(cell, vbin, now)
        sweep_cache.set_last_cam_pos(cell, vbin, cam_pos)
        if look_cell is not None:
            sweep_cache.remember_look_cell(cell, vbin, look_cell, keep=self.cfg.keep_lookcells)


def should_sweep(
    sweep_cache: SweepCacheLike,
    *,
    cell: Tuple[int, int, int],
    vbin: Tuple[int, int],
    cam_pos: np.ndarray,
    fwd_unit: np.ndarray,
    Z: Optional[float],
    look_cell: Optional[Tuple[int, int, int]] = None,
    now: Optional[float] = None,
    cfg: Optional[SweepPolicyConfig] = None,
) -> bool:
    policy = SweepPolicy(cfg)
    decision = policy.decide(
        sweep_cache,
        cell=cell,
        vbin=vbin,
        cam_pos=cam_pos,
        fwd_unit=fwd_unit,
        Z=Z,
        look_cell=look_cell,
        now=now,
    )
    return decision.do_sweep
