"""Frame-quality gate: skip frames that cannot yield usable masks.

Runs before segmentation on a strided subsample of the frame, so it costs well
under a millisecond and saves a full segmentation pass on black, blank, or
depth-less frames. The defaults are deliberately conservative; a frame-level
gate that is too strict silently starves the map, so raise thresholds only with
replay evidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("rtsm.frame_gate")

REASON_DARK = "dark"
REASON_FLAT = "flat"
REASON_DEPTH = "depth"
REASONS = (REASON_DARK, REASON_FLAT, REASON_DEPTH)


@dataclass(frozen=True)
class FrameGateDecision:
    accept: bool
    reason: str            # "" when accepted, otherwise one of REASONS
    brightness: float      # mean grey level, 0-255
    contrast: float        # grey standard deviation
    depth_valid: float     # fraction of finite, positive depth pixels; 1.0 when no depth given


class FrameQualityGate:
    """Cheap per-frame usability check driven by the ``gates`` config section."""

    def __init__(self, cfg: Dict[str, Any], log_interval_s: float = 10.0):
        g = cfg.get("gates") or {}
        self.enabled = bool(g.get("enable", True))
        self.min_brightness = float(g.get("min_brightness", 5.0))
        self.min_std = float(g.get("min_std", 5.0))
        self.min_depth_valid = float(g.get("min_depth_valid", 0.02))
        self.stride = max(1, int(g.get("sample_stride", 4)))
        self.checked = 0
        self.rejections: Dict[str, int] = {reason: 0 for reason in REASONS}
        self._log_interval_s = float(log_interval_s)
        self._last_log_mono = float("-inf")
        self._unlogged = 0

    def check(self, rgb: Any, depth_m: Optional[np.ndarray]) -> FrameGateDecision:
        """Measure the frame and decide. Statistics are computed even when disabled."""
        img = np.asarray(rgb)
        sub = img[:: self.stride, :: self.stride]
        if sub.ndim == 3 and sub.shape[-1] >= 3:
            s = sub[..., :3].astype(np.float32)
            grey = s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114
        else:
            grey = sub.astype(np.float32)
        brightness = float(grey.mean()) if grey.size else 0.0
        contrast = float(grey.std()) if grey.size else 0.0

        depth_valid = 1.0
        if depth_m is not None:
            d = np.asarray(depth_m)[:: self.stride, :: self.stride]
            if d.size:
                with np.errstate(invalid="ignore"):
                    depth_valid = float(np.mean(np.isfinite(d) & (d > 0)))
            else:
                depth_valid = 0.0

        reason = ""
        if self.enabled:
            if brightness < self.min_brightness:
                reason = REASON_DARK
            elif contrast < self.min_std:
                reason = REASON_FLAT
            elif depth_valid < self.min_depth_valid:
                reason = REASON_DEPTH

        self.checked += 1
        if reason:
            self.rejections[reason] += 1
            self._unlogged += 1
        return FrameGateDecision(not reason, reason, brightness, contrast, depth_valid)

    def maybe_log(self, decision: FrameGateDecision, now_mono: Optional[float] = None) -> bool:
        """Log the first rejection immediately, then at most one summary per interval."""
        now = time.monotonic() if now_mono is None else now_mono
        if now - self._last_log_mono < self._log_interval_s:
            return False
        logger.warning(
            "frame-quality gate skipped %d frame(s) since last report (%s); latest reason=%s "
            "brightness=%.1f std=%.1f depth_valid=%.2f",
            self._unlogged,
            ", ".join(f"{k}={v}" for k, v in self.rejections.items()),
            decision.reason, decision.brightness, decision.contrast, decision.depth_valid,
        )
        self._last_log_mono = now
        self._unlogged = 0
        return True
