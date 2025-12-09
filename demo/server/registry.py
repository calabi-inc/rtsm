"""
Keyframe Registry for RTSM Demo

Stores all processed keyframes with their point clouds and poses.
Backend is the source of truth for mesh management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import threading


@dataclass
class KeyframeRecord:
    """A single keyframe with its point cloud and pose."""
    mesh_id: str                    # Unique ID (from SLAM kf_id)
    timestamp_ns: int               # Capture timestamp
    positions: np.ndarray           # (N, 3) float32 camera-frame XYZ
    colors: np.ndarray              # (N, 3) uint8 RGB
    pose: Optional[np.ndarray] = None  # (4, 4) float32 Twc row-major, or None if not yet received
    map_id: str = "0"               # SLAM map ID


class KeyframeRegistry:
    """
    Thread-safe registry for all keyframes.

    Backend assigns mesh_id and controls lifecycle.
    Frontend only renders based on backend commands.
    """

    def __init__(self):
        self._keyframes: Dict[str, KeyframeRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        mesh_id: str,
        timestamp_ns: int,
        positions: np.ndarray,
        colors: np.ndarray,
        pose: Optional[np.ndarray] = None,
        map_id: str = "0"
    ) -> KeyframeRecord:
        """
        Register a new keyframe or update existing.

        Returns the KeyframeRecord.
        """
        with self._lock:
            if mesh_id in self._keyframes:
                # Update existing
                rec = self._keyframes[mesh_id]
                rec.positions = positions
                rec.colors = colors
                rec.timestamp_ns = timestamp_ns
                if pose is not None:
                    rec.pose = pose
                rec.map_id = map_id
            else:
                # Create new
                rec = KeyframeRecord(
                    mesh_id=mesh_id,
                    timestamp_ns=timestamp_ns,
                    positions=positions,
                    colors=colors,
                    pose=pose,
                    map_id=map_id
                )
                self._keyframes[mesh_id] = rec
            return rec

    def update_pose(self, mesh_id: str, pose: np.ndarray) -> bool:
        """
        Update pose for an existing keyframe.

        Returns True if keyframe exists and was updated.
        """
        with self._lock:
            if mesh_id not in self._keyframes:
                return False
            self._keyframes[mesh_id].pose = pose
            return True

    def get(self, mesh_id: str) -> Optional[KeyframeRecord]:
        """Get a keyframe by mesh_id."""
        with self._lock:
            return self._keyframes.get(mesh_id)

    def delete(self, mesh_id: str) -> bool:
        """Delete a keyframe. Returns True if it existed."""
        with self._lock:
            if mesh_id in self._keyframes:
                del self._keyframes[mesh_id]
                return True
            return False

    def get_all(self) -> List[KeyframeRecord]:
        """Get all keyframes (for syncing new clients)."""
        with self._lock:
            return list(self._keyframes.values())

    def get_all_ids(self) -> List[str]:
        """Get all mesh IDs."""
        with self._lock:
            return list(self._keyframes.keys())

    def exists(self, mesh_id: str) -> bool:
        """Check if a keyframe exists."""
        with self._lock:
            return mesh_id in self._keyframes

    def count(self) -> int:
        """Get total number of keyframes."""
        with self._lock:
            return len(self._keyframes)

    def clear(self) -> int:
        """Clear all keyframes. Returns count of deleted."""
        with self._lock:
            count = len(self._keyframes)
            self._keyframes.clear()
            return count

    def stats(self) -> dict:
        """Get registry statistics."""
        with self._lock:
            total_points = sum(rec.positions.shape[0] for rec in self._keyframes.values())
            poses_set = sum(1 for rec in self._keyframes.values() if rec.pose is not None)
            return {
                "keyframes": len(self._keyframes),
                "total_points": total_points,
                "poses_set": poses_set
            }
