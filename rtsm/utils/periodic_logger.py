"""
Periodic summary logger for RTSM pipeline statistics.

Provides low-overhead periodic logging of system health metrics:
- Frame throughput (FPS)
- Association statistics (matched/created)
- Working memory object counts
- Queue depth

Enabled by default, configurable via config/rtsm.yaml.
"""

from __future__ import annotations
import time
import logging

logger = logging.getLogger(__name__)


class PeriodicLogger:
    """Logs system summary at configurable intervals."""

    def __init__(self, interval_s: float = 5.0, enabled: bool = True):
        """
        Initialize the periodic logger.

        Args:
            interval_s: Seconds between summary logs (default 5.0)
            enabled: Whether periodic logging is enabled (default True)
        """
        self.interval_s = interval_s
        self.enabled = enabled
        self._last_log_time = time.monotonic()
        self._frame_count = 0
        self._total_matched = 0
        self._total_created = 0
        self._total_masks = 0

    def tick(self, matched: int = 0, created: int = 0, masks: int = 0) -> None:
        """
        Called after each frame to accumulate stats.

        Args:
            matched: Number of objects matched this frame
            created: Number of objects created this frame
            masks: Number of masks kept after heuristics
        """
        self._frame_count += 1
        self._total_matched += matched
        self._total_created += created
        self._total_masks += masks

    def maybe_log(self, wm_stats: dict, queue_size: int = 0) -> bool:
        """
        Log summary if interval elapsed.

        Args:
            wm_stats: Dict from working_memory.stats() with 'objects', 'confirmed' keys
            queue_size: Current ingest queue size

        Returns:
            True if summary was logged, False otherwise
        """
        if not self.enabled:
            return False

        now = time.monotonic()
        elapsed = now - self._last_log_time
        if elapsed < self.interval_s:
            return False

        # Calculate FPS
        fps = self._frame_count / max(0.1, elapsed)

        # Extract WM stats with defaults
        n_objects = wm_stats.get("objects", 0)
        n_confirmed = wm_stats.get("confirmed", 0)

        logger.info(
            f"[summary] frames={self._frame_count} fps={fps:.1f} | "
            f"matched={self._total_matched} created={self._total_created} | "
            f"wm: {n_objects} objs ({n_confirmed} confirmed) | "
            f"queue={queue_size}"
        )

        # Reset counters
        self._frame_count = 0
        self._total_matched = 0
        self._total_created = 0
        self._total_masks = 0
        self._last_log_time = now

        return True

    def reset(self) -> None:
        """Reset all counters and timestamp."""
        self._frame_count = 0
        self._total_matched = 0
        self._total_created = 0
        self._total_masks = 0
        self._last_log_time = time.monotonic()
