"""
Point Cloud Processor for RTSM Demo

Handles:
- JPEG decoding (OpenCV)
- PNG 16-bit depth decoding (OpenCV)
- Depth filtering (5x5 median + jump rejection)
- Back-projection to 3D camera coordinates
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2


@dataclass
class ProcessorConfig:
    """Configuration for point cloud processing."""
    subsample_step: int = 2
    depth_min_m: float = 0.30
    depth_max_m: float = 4.0
    filter_window: int = 2      # 5x5 window (2*2+1)
    filter_min_valid: int = 5
    jump_abs_m: float = 0.05
    jump_rel: float = 0.10


class PointCloudProcessor:
    """Processes RGB-D frames into point clouds."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.cfg = config or ProcessorConfig()

    def decode_jpeg(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to BGR uint8 array."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG")
        return img

    def decode_depth_png(self, png_bytes: bytes, depth_scale: float = 0.001) -> np.ndarray:
        """
        Decode PNG 16-bit depth to float32 meters.

        Args:
            png_bytes: PNG-encoded 16-bit depth image
            depth_scale: Scale factor to convert to meters (default 0.001 for mm)

        Returns:
            (H, W) float32 depth in meters
        """
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        # Decode as unchanged to preserve 16-bit depth
        depth_u16 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            raise ValueError("Failed to decode depth PNG")

        # Convert to float32 meters
        depth_m = depth_u16.astype(np.float32) * depth_scale
        return depth_m

    def unpack_depth_z16(self, depth_bytes: bytes, width: int, height: int, depth_scale: float = 0.001) -> np.ndarray:
        """
        Unpack raw 16-bit little-endian depth to float32 meters.

        This is a fallback for raw Z16 format (not PNG).

        Args:
            depth_bytes: Raw 16-bit LE depth data
            width: Image width
            height: Image height
            depth_scale: Scale factor to convert to meters

        Returns:
            (H, W) float32 depth in meters
        """
        expected = width * height * 2
        if len(depth_bytes) != expected:
            raise ValueError(f"Depth size mismatch: got {len(depth_bytes)}, expected {expected}")
        arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((height, width))
        depth_m = arr.astype(np.float32) * depth_scale
        return depth_m

    def filter_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Apply depth filtering:
        1. Range clamp (min/max depth)
        2. Median filter for noise reduction
        3. Jump rejection using neighbor comparison

        Returns filtered depth with invalid pixels set to NaN.
        """
        filtered = depth_m.copy()

        # Range filter
        invalid = (filtered < self.cfg.depth_min_m) | (filtered > self.cfg.depth_max_m) | (filtered == 0)
        filtered[invalid] = np.nan

        # Median filter (5x5)
        kernel_size = self.cfg.filter_window * 2 + 1
        # cv2.medianBlur doesn't handle NaN, so we work around it
        valid_mask = np.isfinite(filtered)
        temp = filtered.copy()
        temp[~valid_mask] = 0

        # Apply median filter
        median = cv2.medianBlur(temp.astype(np.float32), kernel_size)

        # Jump rejection: compare each pixel to its median neighborhood
        jump_threshold = np.maximum(self.cfg.jump_abs_m, self.cfg.jump_rel * np.abs(filtered))
        jump_mask = np.abs(filtered - median) > jump_threshold
        filtered[jump_mask] = np.nan

        # Also invalidate pixels with too few valid neighbors
        valid_count = cv2.boxFilter(valid_mask.astype(np.float32), -1, (kernel_size, kernel_size), normalize=False)
        filtered[valid_count < self.cfg.filter_min_valid] = np.nan

        return filtered

    def backproject(
        self,
        depth_m: np.ndarray,
        rgb: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project depth to 3D points in camera frame.

        Args:
            depth_m: (H, W) float32 depth in meters (NaN for invalid)
            rgb: (H, W, 3) uint8 BGR image
            K: (3, 3) or (9,) camera intrinsics matrix

        Returns:
            positions: (N, 3) float32 XYZ in camera frame
            colors: (N, 3) uint8 RGB colors
        """
        K = np.asarray(K).reshape(3, 3) if K.size == 9 else K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        h, w = depth_m.shape
        step = self.cfg.subsample_step

        # Create coordinate grids with subsampling
        u_coords = np.arange(0, w, step, dtype=np.float32)
        v_coords = np.arange(0, h, step, dtype=np.float32)
        uu, vv = np.meshgrid(u_coords, v_coords)

        # Sample depth and RGB at subsampled locations
        z_sampled = depth_m[::step, ::step]
        rgb_sampled = rgb[::step, ::step]

        # Find valid pixels
        valid = np.isfinite(z_sampled) & (z_sampled > 0)

        # Extract valid values
        z = z_sampled[valid]
        u = uu[valid]
        v = vv[valid]

        # Back-project to camera coordinates
        # z = depth_meters[v, u]
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack positions
        positions = np.stack([x, y, z], axis=1).astype(np.float32)

        # Extract colors (BGR -> RGB)
        colors_bgr = rgb_sampled[valid]
        colors = colors_bgr[:, ::-1].astype(np.uint8)  # BGR to RGB

        return positions, colors

    def process(
        self,
        jpeg_bytes: bytes,
        depth_bytes: bytes,
        K: np.ndarray,
        width: int = 640,
        height: int = 480,
        depth_scale: float = 0.001,
        depth_encoding: str = 'png'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full processing pipeline: decode, filter, backproject.

        Args:
            jpeg_bytes: JPEG-encoded RGB image
            depth_bytes: Encoded depth (PNG or raw Z16)
            K: Camera intrinsics (3x3 or flat 9-element)
            width: Image width
            height: Image height
            depth_scale: Scale factor for depth (multiply to get meters)
            depth_encoding: 'png' for PNG uint16, 'z16' for raw bytes

        Returns:
            positions: (N, 3) float32 XYZ in camera frame
            colors: (N, 3) uint8 RGB colors
        """
        # Decode JPEG
        bgr = self.decode_jpeg(jpeg_bytes)

        # Validate dimensions
        if bgr.shape[0] != height or bgr.shape[1] != width:
            # Resize if needed
            bgr = cv2.resize(bgr, (width, height))

        # Decode depth based on encoding
        if depth_encoding == 'png':
            depth_m = self.decode_depth_png(depth_bytes, depth_scale)
        else:
            depth_m = self.unpack_depth_z16(depth_bytes, width, height, depth_scale)

        # Ensure depth matches expected dimensions
        if depth_m.shape[0] != height or depth_m.shape[1] != width:
            depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)

        # Filter depth
        depth_filtered = self.filter_depth(depth_m)

        # Back-project
        positions, colors = self.backproject(depth_filtered, bgr, K)

        return positions, colors
