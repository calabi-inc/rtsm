"""
Ingress gate for heavy processing

Responsibilities
- Always accept keyframes
- Prevent non-keyframe sweeps that collide with a recently processed keyframe (dup window)
- Apply view-based TTL/parallax gating via SweepPolicy for non-keyframes

Notes
- Monotonic clock is used for wall-time gates; sensor `ts_ns` is used for dup-window proximity.
- This class is intentionally stateless regarding sweep_grid/policy beyond small caches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from rtsm.stores.sweep_cache import SweepCache, Cell, VBin
from rtsm.stores.sweep_policy import SweepPolicy, SweepPolicyConfig


@dataclass
class IngestDecision:
    accept: bool
    reason: str


class IngestGate:

    def __init__(self, cfg: Optional[dict[str, Any]] = None, policy: Optional[SweepPolicy] = None) -> None:
        self.cfg = cfg or {}
        self.policy = policy or SweepPolicy(self._policy_cfg_from(self.cfg))

        ingest_cfg = self.cfg.get("ingest", {})
        self.dup_window_ns: int = int(ingest_cfg.get("dup_window_ns", int(0.2 * 1e9)))
        # Non-keyframe grace delay after a keyframe arrives (monotonic seconds)
        self.non_kf_grace_s: float = float(ingest_cfg.get("non_kf_grace_s", 0.03))

        self._last_keyframe_ts_ns: Optional[int] = None
        self._last_kf_arrival_mono: float = 0.0

    # ----------------------------- public API -----------------------------
    def should_accept(
        self,
        *,
        is_keyframe: bool,
        ts_ns: int,
        sweep_cache: SweepCache,
        cell: Cell,
        vbin: VBin,
        cam_pos: np.ndarray,
        fwd_unit: np.ndarray,
        Z: Optional[float],
        look_cell: Optional[Cell] = None,
        now_mono: Optional[float] = None,
    ) -> IngestDecision:
        """Return whether to allow heavy processing for this frame/view.

        For keyframes, apply a simple time-based minimum period. For non-keyframes,
        block frames that are too close (by sensor ts) to the last accepted keyframe
        and then consult the SweepPolicy for TTL/parallax gating.
        """
        now_mono = time.monotonic() if now_mono is None else now_mono

        if is_keyframe:
            # Remember arrival time to protect a brief window for non-KF frames
            self._last_kf_arrival_mono = now_mono
            return IngestDecision(True, "keyframe")

        # Non-keyframe grace: if a keyframe just arrived, delay non-KF acceptance
        if (now_mono - self._last_kf_arrival_mono) < self.non_kf_grace_s:
            return IngestDecision(False, "non_kf_grace")

        # Non-keyframe: check for proximity to last accepted keyframe by sensor time
        if self._last_keyframe_ts_ns is not None and self.dup_window_ns > 0:
            if abs(int(ts_ns) - int(self._last_keyframe_ts_ns)) <= self.dup_window_ns:
                return IngestDecision(False, "near_recent_keyframe")

        # Ask the sweep policy
        decision = self.policy.decide(
            sweep_cache,
            cell=cell,
            vbin=vbin,
            cam_pos=np.asarray(cam_pos, dtype=float),
            fwd_unit=np.asarray(fwd_unit, dtype=float),
            Z=None if Z is None else float(Z),
            look_cell=look_cell,
            now=now_mono,
        )
        return IngestDecision(decision.do_sweep, decision.reason)

    def record_processed(
        self,
        *,
        is_keyframe: bool,
        ts_ns: int,
        sweep_cache: SweepCache,
        cell: Cell,
        vbin: VBin,
        cam_pos: np.ndarray,
        look_cell: Optional[Cell] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Book-keeping to be called after heavy processing has been accepted and run."""
        now_mono = time.monotonic() if now_mono is None else now_mono
        if is_keyframe:
            self._last_keyframe_ts_ns = int(ts_ns)

        self.policy.record_after_sweep(
            sweep_cache,
            cell=cell,
            vbin=vbin,
            cam_pos=np.asarray(cam_pos, dtype=float),
            look_cell=look_cell,
            now=now_mono,
        )

    # --------------------------- helpers ---------------------------
    @staticmethod
    def _policy_cfg_from(cfg: dict[str, Any]) -> SweepPolicyConfig:
        # Support both legacy key 'sweep' and current 'sweep_policy'
        sweep = cfg.get("sweep", cfg.get("sweep_policy", {}))
        return SweepPolicyConfig(
            ttl_s=float(sweep.get("ttl_s", 4.0)),
            hard_max_s=float(sweep.get("hard_max_s", 20.0)),
            min_baseline_m=float(sweep.get("min_baseline_m", 0.20)),
            k_depth=float(sweep.get("k_depth", 0.02)),
            keep_lookcells=int(sweep.get("keep_lookcells", 8)),
            require_novelty_on_ttl=bool(sweep.get("require_novelty_on_ttl", True)),
        )


