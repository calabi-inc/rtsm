"""
ProximityIndex: A spatial index for fast neighbor queries in world coordinates.

Responsibility:
- Maintains a mapping from object IDs to spatial grid cells (2D or 3D).
- Supports efficient insertion, removal, and neighbor queries for objects based on their world position.
- Handles per-cell capacity limits and recency-based pruning.
- Optionally prunes dead object IDs using a user-provided callable.
- Provides statistics on cell/object counts and per-cell occupancy.
- Thread-safe for concurrent access.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Dict, Set, List, Union
from collections import defaultdict
import numpy as np
import math
import threading
import time
import logging

logger = logging.getLogger(__name__)

Cell2D = Tuple[int, int]
Cell3D = Tuple[int, int, int]
Cell = Union[Cell2D, Cell3D]

# --------- shared grid spec (quantization + neighbors) ---------

@dataclass(frozen=True, slots=True)
class GridSpec:
    """
    World-space uniform grid.
    - cell_m: cell size in meters
    - use_3d: True => (ix,iy,iz) keys; False => (ix,iy) keys
    """
    cell_m: float = 0.25
    use_3d: bool = True

    def cell(self, xyz_world: np.ndarray) -> Cell:
        x, y, z = float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])
        ix = int(math.floor(x / self.cell_m))
        iy = int(math.floor(y / self.cell_m))
        if self.use_3d:
            iz = int(math.floor(z / self.cell_m))
            return (ix, iy, iz)
        return (ix, iy)

    def neighbors(self, c: Cell, rings: int = 1) -> Iterable[Cell]:
        if rings <= 0:
            yield c
            return
        if self.use_3d:
            ix, iy, iz = c  # type: ignore[misc]
            for dx in range(-rings, rings + 1):
                for dy in range(-rings, rings + 1):
                    for dz in range(-rings, rings + 1):
                        yield (ix + dx, iy + dy, iz + dz)
        else:
            ix, iy = c  # type: ignore[misc]
            for dx in range(-rings, rings + 1):
                for dy in range(-rings, rings + 1):
                    yield (ix + dx, iy + dy)

# --------- proximity index (object membership by cell) ---------

class ProximityIndex:
    """
    Fast proximity lookup for live WM objects (proto + confirmed).

    Design:
      - WM owns lifecycles (create/move/expire) and calls insert/update/remove.
      - Per-cell cap with immediate eviction (keeps memory & query sets bounded).
      - Lazy prune on reads (drops IDs no longer in WM).
      - Query-time clamps: neighbors_max & recency-based trimming.

    Optional WM lookup:
      Pass wm_lookup(oid) -> (confirmed: bool, stability: float, last_seen_mono: float)
      so eviction can prefer dropping protos, then low-stability, then oldest.

    Thread-safety:
      A light RLock guards internal maps; reads still copy small sets.
    """

    def __init__(
        self,
        grid: GridSpec,
        per_cell_cap: int = 32,
        neighbors_max: int = 128,
    ) -> None:
        self.grid = grid
        self.per_cell_cap = int(per_cell_cap)
        self.neighbors_max = int(neighbors_max)

        self._members: Dict[Cell, Set[str]] = defaultdict(set)  # cell -> {oid}
        self._oid_cell: Dict[str, Cell] = {}                   # oid -> cell
        self._touch: Dict[str, float] = {}                     # oid -> last access tick/time
        self._tick: float = 0.0                                # monotonically increasing
        self._lock = threading.RLock()

    # ----- internal helpers -----

    def _now_tick(self) -> float:
        # monotonic tick; using perf_counter to avoid wall-time jumps
        self._tick = max(self._tick + 1.0, time.perf_counter())
        return self._tick

    def _touch_oid(self, oid: str) -> None:
        self._touch[oid] = self._now_tick()

    def _evict_overflow(
        self,
        cell: Cell,
        wm_lookup: Optional[Callable[[str], Optional[Tuple[bool, float, float]]]] = None,
    ) -> None:
        """Evict immediately if cell exceeds cap. Uses WM info if available; else LRU fallback."""
        s = self._members.get(cell)
        if not s:
            return
        overflow = len(s) - self.per_cell_cap
        if overflow <= 0:
            return

        def rank(oid: str) -> Tuple[int, float, float]:
            # Lower is evicted first
            if wm_lookup is not None:
                info = wm_lookup(oid)
                if info is not None:
                    confirmed, stability, last_seen = info
                    # Prefer to evict protos (confirmed=False => 0), then low stability, then oldest last_seen
                    return (0 if not confirmed else 1, stability, last_seen)
            # Fallback: LRU by internal touch (older first)
            touch = self._touch.get(oid, 0.0)
            return (1, 0.0, touch)

        victims = sorted(s, key=rank)[:overflow]
        for v in victims:
            s.discard(v)
            self._oid_cell.pop(v, None)
            # leave WM to expire it; index only mirrors membership

    # ----- public API -----

    def clear(self) -> None:
        with self._lock:
            self._members.clear()
            self._oid_cell.clear()
            self._touch.clear()
            self._tick = 0.0

    def insert(
        self,
        oid: str,
        xyz_world: np.ndarray,
        *,
        wm_lookup: Optional[Callable[[str], Optional[Tuple[bool, float, float]]]] = None,
    ) -> None:
        """Add an object to its current cell (called by WM on create)."""
        c = self.grid.cell(xyz_world)
        with self._lock:
            self._members[c].add(oid)
            self._oid_cell[oid] = c
            self._touch_oid(oid)
            self._evict_overflow(c, wm_lookup)
        try:
            logger.info(
                "[PI] insert oid=%s cell=%s size_cell=%d total=%d",
                oid,
                c,
                len(self._members.get(c, set())),
                len(self._oid_cell),
            )
        except Exception:
            pass

    def update(
        self,
        oid: str,
        old_xyz_world: np.ndarray,
        new_xyz_world: np.ndarray,
        *,
        wm_lookup: Optional[Callable[[str], Optional[Tuple[bool, float, float]]]] = None,
    ) -> None:
        """
        Move object if it crossed a cell boundary (called by WM after EMA/KF pose update).
        """
        c_old = self.grid.cell(old_xyz_world)
        c_new = self.grid.cell(new_xyz_world)
        if c_old == c_new:
            self._touch_oid(oid)
            return
        with self._lock:
            # remove from old
            s_old = self._members.get(c_old)
            if s_old is not None:
                s_old.discard(oid)
            # add to new
            self._members[c_new].add(oid)
            self._oid_cell[oid] = c_new
            self._touch_oid(oid)
            self._evict_overflow(c_new, wm_lookup)
        try:
            logger.info(
                "[PI] update oid=%s %s->%s size_new=%d",
                oid,
                c_old,
                c_new,
                len(self._members.get(c_new, set())),
            )
        except Exception:
            pass

    def remove(self, oid: str, last_xyz_world: Optional[np.ndarray] = None) -> None:
        """Remove object from its recorded cell (called by WM on expire/merge)."""
        with self._lock:
            cell = self._oid_cell.pop(oid, None)
            if cell is None and last_xyz_world is not None:
                cell = self.grid.cell(last_xyz_world)
            if cell is not None:
                s = self._members.get(cell)
                if s is not None:
                    s.discard(oid)
            self._touch.pop(oid, None)
        try:
            logger.info("[PI] remove oid=%s cell=%s", oid, cell)
        except Exception:
            pass

    def nearby_ids(
        self,
        xyz_world: np.ndarray,
        rings: int = 1,
        *,
        prune_with: Optional[Callable[[str], bool]] = None,
    ) -> List[str]:
        """
        Get object IDs in the ±rings neighborhood of xyz_world's cell.
        - prune_with(oid) should return True if the ID is still live in WM.
          Stale IDs are lazily removed from the index.
        - The returned list is clamped to neighbors_max by recency (most-recent first).
        """
        c = self.grid.cell(xyz_world)
        with self._lock:
            ids: Set[str] = set()
            # Debug: capture neighbor occupancy
            neighbor_counts: List[Tuple[Cell, int]] = []
            for n in self.grid.neighbors(c, rings=rings):
                s = self._members.get(n)
                if s:
                    ids |= s
                    neighbor_counts.append((n, len(s)))
            try:
                logger.info(
                    "[PI] query cell=%s rings=%d neighbors_hit=%d ids_before_prune=%d",
                    c,
                    rings,
                    len(neighbor_counts),
                    len(ids),
                )
                if neighbor_counts:
                    # Log at most first 16 neighbor occupancies to avoid spam
                    logger.info(
                        "[PI] neighbor occupancy sample: %s",
                        neighbor_counts[:16],
                    )
            except Exception:
                pass
            # Lazy prune dead IDs (if WM callable provided)
            if prune_with is not None:
                dead: List[str] = [oid for oid in ids if not prune_with(oid)]
                if dead:
                    for oid in dead:
                        # quick remove from recorded cell, if known
                        cell = self._oid_cell.pop(oid, None)
                        if cell is not None:
                            self._members.get(cell, set()).discard(oid)
                        self._touch.pop(oid, None)
                    ids.difference_update(dead)

            # Clamp by recency (keep most-recent touched)
            if len(ids) > self.neighbors_max:
                ids_sorted = sorted(ids, key=lambda k: self._touch.get(k, 0.0), reverse=True)
                ids = set(ids_sorted[: self.neighbors_max])

            # Touch survivors (they’re active)
            now = self._now_tick()
            for oid in ids:
                self._touch[oid] = now

            return list(ids)

    # ----- utilities -----

    def cell_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._members.values() if s)

    def object_count(self) -> int:
        with self._lock:
            return len(self._oid_cell)

    def stats(self) -> dict:
        with self._lock:
            per_cell_sizes = [len(s) for s in self._members.values() if s]
            return {
                "cells": len(per_cell_sizes),
                "objects": len(self._oid_cell),
                "avg_per_cell": (sum(per_cell_sizes) / len(per_cell_sizes)) if per_cell_sizes else 0.0,
                "max_per_cell": max(per_cell_sizes) if per_cell_sizes else 0,
                "per_cell_cap": self.per_cell_cap,
                "neighbors_max": self.neighbors_max,
            }

    def cell_size(self, xyz_world: np.ndarray) -> int:
        """Return the current membership size for the cell containing xyz_world."""
        c = self.grid.cell(xyz_world)
        with self._lock:
            s = self._members.get(c)
            return len(s) if s else 0