from __future__ import annotations
import cv2
import numpy as np

# ---------- for testing ----------
def load_depth_png_as_meters(png_path: str, *, depth_scale: float = 0.001, invalid_value: int = 0) -> np.ndarray:
    """
    Load a 16-bit single-channel depth PNG and return float32 meters.
    - depth_scale: meters per raw unit (0.001 for millimeters).
    - invalid_value: raw pixel value that means 'no depth' (usually 0) → set to NaN.
    """
    z = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if z is None:
        raise FileNotFoundError(png_path)
    if z.ndim == 3:
        # some tools write depth into the first channel; keep channel 0
        z = z[..., 0]
    if z.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG; got dtype={z.dtype}, shape={z.shape}")

    z16 = z.astype(np.uint16)
    depth_m = z16.astype(np.float32) * depth_scale
    if invalid_value is not None:
        depth_m[z16 == invalid_value] = np.nan
    return depth_m  # shape (H, W), dtype float32, meters with NaNs for invalids