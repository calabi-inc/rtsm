"""
FrameWindow: Efficient, TTL-based sliding window for RGB and depth frames.

- Maintains sorted timestamp lists and dicts for fast lookup and eviction.
- Accepts integer nanosecond timestamps or ROS2 Time-like objects.
- Used for buffering recent frames for association, matching, or visualization.
- Supports per-frame intrinsics for dynamic camera calibration.
"""

from __future__ import annotations
import bisect, time
from typing import List, Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rtsm.core.datamodel import PinholeIntrinsics

try:
    from builtin_interfaces.msg import Time  # type: ignore
except Exception:
    Time = None  # ROS not available; we will accept integers for timestamps

NS = 10**9
def to_ns(t: Any) -> int:
    if isinstance(t, int):
        return t
    # Duck-type ROS2 Time-like
    try:
        return int(t.sec) * NS + int(t.nanosec)
    except Exception:
        # Fallback for tuple-like (sec, nsec)
        try:
            sec, nsec = t
            return int(sec) * NS + int(nsec)
        except Exception:
            raise TypeError("Unsupported timestamp type; expected int ns or ROS Time-like")

class FrameWindow:
    def __init__(self, ttl_sec=30.0, max_items=2000, slop_sec=0.03):
        self.rgb = {}              # ts_ns -> img
        self.depth = {}            # ts_ns -> depth
        self.intrinsics = {}       # ts_ns -> PinholeIntrinsics (per-frame)
        self.rgb_ts: List[int] = []    # sorted!
        self.depth_ts: List[int] = []  # sorted!
        self.ttl_ns = int(ttl_sec * NS)
        self.max = max_items
        self.slop_ns = int(slop_sec * NS)
        self.watermark = 0         # latest stamp seen across both streams

    def _insert_sorted(self, arr: List[int], ts: int):
        i = bisect.bisect_left(arr, ts)
        if i == len(arr) or arr[i] != ts:
            arr.insert(i, ts)

    def _evict_by_stamp(self, arr: List[int], store: dict):
        cutoff = self.watermark - self.ttl_ns
        # drop old by stamp
        while arr and (arr[0] < cutoff or len(arr) > self.max):
            old = arr.pop(0)
            store.pop(old, None)

    def add_rgb(self, stamp: Any, img):
        ts = to_ns(stamp)
        self.rgb[ts] = img
        self._insert_sorted(self.rgb_ts, ts)
        if ts > self.watermark: self.watermark = ts
        self._evict_by_stamp(self.rgb_ts, self.rgb)

    def add_depth(self, stamp: Any, d):
        ts = to_ns(stamp)
        self.depth[ts] = d
        self._insert_sorted(self.depth_ts, ts)
        if ts > self.watermark: self.watermark = ts
        self._evict_by_stamp(self.depth_ts, self.depth)

    def add_rgbd(self, stamp: Any, rgb, depth, intrinsics: Optional["PinholeIntrinsics"] = None):
        """
        Atomically add bundled RGB-D frame with optional per-frame intrinsics.

        This is the preferred method for camera.rgbd topic where RGB and depth
        arrive together in a single message.

        Args:
            stamp: Timestamp (int nanoseconds or ROS Time-like)
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            intrinsics: Optional per-frame camera intrinsics
        """
        ts = to_ns(stamp)
        self.rgb[ts] = rgb
        self.depth[ts] = depth
        if intrinsics is not None:
            self.intrinsics[ts] = intrinsics
        self._insert_sorted(self.rgb_ts, ts)
        # Depth uses same timestamp, ensure it's in the sorted list too
        self._insert_sorted(self.depth_ts, ts)
        if ts > self.watermark:
            self.watermark = ts
        self._evict_by_stamp(self.rgb_ts, self.rgb)
        self._evict_by_stamp(self.depth_ts, self.depth)
        self._evict_intrinsics()

    def _evict_intrinsics(self):
        """Evict intrinsics entries whose timestamps are no longer in rgb_ts."""
        valid_ts = set(self.rgb_ts)
        stale = [ts for ts in self.intrinsics if ts not in valid_ts]
        for ts in stale:
            del self.intrinsics[ts]

    def _nearest(self, ts: int, arr: List[int]) -> Optional[int]:
        i = bisect.bisect_left(arr, ts)
        cand = []
        if i < len(arr): cand.append(arr[i])
        if i > 0: cand.append(arr[i-1])
        if not cand: return None
        best = min(cand, key=lambda k: abs(k - ts))
        return best if abs(best - ts) <= self.slop_ns else None

    def assemble_pair(self, stamp: Any) -> Tuple[Any, Any, Optional["PinholeIntrinsics"]]:
        """
        Assemble RGB, depth, and intrinsics for a given timestamp.

        Args:
            stamp: Target timestamp (int nanoseconds or ROS Time-like)

        Returns:
            Tuple of (rgb, depth, intrinsics). Any may be None if not found within slop.
        """
        ts = to_ns(stamp)
        rgb_ts = ts if ts in self.rgb else self._nearest(ts, self.rgb_ts)
        depth_ts = self._nearest(ts, self.depth_ts)
        rgb = self.rgb.get(rgb_ts) if rgb_ts is not None else None
        depth = self.depth.get(depth_ts) if depth_ts is not None else None
        # Get intrinsics from the RGB timestamp (they arrive together in camera.rgbd)
        intr = self.intrinsics.get(rgb_ts) if rgb_ts is not None else None
        return rgb, depth, intr

    def clear(self) -> dict:
        """
        Clear all buffered frames.

        Returns dict with counts of what was cleared.
        """
        rgb_count = len(self.rgb)
        depth_count = len(self.depth)
        intr_count = len(self.intrinsics)

        self.rgb.clear()
        self.depth.clear()
        self.intrinsics.clear()
        self.rgb_ts.clear()
        self.depth_ts.clear()
        self.watermark = 0

        return {
            "rgb_frames_cleared": rgb_count,
            "depth_frames_cleared": depth_count,
            "intrinsics_cleared": intr_count,
        }

    def stats(self) -> dict:
        """Return current frame buffer statistics."""
        return {
            "rgb_frames": len(self.rgb),
            "depth_frames": len(self.depth),
            "intrinsics": len(self.intrinsics),
            "watermark_ns": self.watermark,
        }
