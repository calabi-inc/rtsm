import argparse
import os
import math
import glob
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import yaml


def load_camera_intrinsics(config_path: str) -> Tuple[float, float, float, float, int, int]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cam = cfg.get('camera', {})
    fx = float(cam['fx'])
    fy = float(cam['fy'])
    cx = float(cam['cx'])
    cy = float(cam['cy'])
    width = int(cam['width'])
    height = int(cam['height'])
    return fx, fy, cx, cy, width, height


def parse_frame_trajectory(traj_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Parse TUM-style FrameTrajectory.txt lines:
      t x y z qx qy qz qw
    Returns mapping from integer nanoseconds timestamp to (R, t) in meters.
    """
    ts_to_pose: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with open(traj_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            t_sec = float(parts[0])
            x, y, z = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            R = quat_to_rotmat(qx, qy, qz, qw)
            t = np.array([x, y, z], dtype=np.float64)
            t_ns = int(round(t_sec * 1e9))
            ts_to_pose[t_ns] = (R, t)
    return ts_to_pose


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Normalize quaternion
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    wx = qw*qx
    wy = qw*qy
    wz = qw*qz
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R


def build_timestamp_index(dir_path: str, ext: str) -> List[int]:
    """Return sorted list of integer ns timestamps parsed from filenames like 123.456.png/.npy"""
    paths = glob.glob(os.path.join(dir_path, f"*.{ext}"))
    ts_list: List[int] = []
    for p in paths:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        try:
            ts_sec = float(name)
            ts_list.append(int(round(ts_sec * 1e9)))
        except Exception:
            continue
    ts_list.sort()
    return ts_list


def find_nearest(ts_target: int, ts_sorted: List[int]) -> int:
    """Binary-search nearest timestamp in a sorted ns list."""
    if not ts_sorted:
        raise ValueError("Empty timestamp list")
    lo, hi = 0, len(ts_sorted) - 1
    if ts_target <= ts_sorted[lo]:
        return ts_sorted[lo]
    if ts_target >= ts_sorted[hi]:
        return ts_sorted[hi]
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_sorted[mid] == ts_target:
            return ts_sorted[mid]
        if ts_sorted[mid] < ts_target:
            lo = mid + 1
        else:
            hi = mid - 1
    # lo is the insertion point
    cand_hi = ts_sorted[lo] if lo < len(ts_sorted) else ts_sorted[-1]
    cand_lo = ts_sorted[lo-1] if lo-1 >= 0 else ts_sorted[0]
    return cand_lo if abs(ts_target - cand_lo) <= abs(ts_target - cand_hi) else cand_hi


def depth_npy_to_meters(depth_path: str, depth_scale: float = 0.0, *, _debug_decision: Dict[str, bool] | None = None) -> np.ndarray:
    """
    Load depth NPY and return float32 meters with NaN for invalid.

    Rules:
    - If depth_scale > 0, multiply raw values by depth_scale.
    - Else, if dtype is integer (uint16/uint32/int32), assume millimeters → scale by 0.001.
    - Else, if dtype is float, auto-detect by statistics: if median(valid) > 50, assume mm; else assume meters.
    """
    arr = np.load(depth_path)
    arr_np = np.asarray(arr)

    # Identify invalids (zero for ints; non-finite for floats)
    if np.issubdtype(arr_np.dtype, np.integer):
        invalid_mask = (arr_np == 0)
    else:
        invalid_mask = ~np.isfinite(arr_np)

    # Choose scale
    scale = float(depth_scale) if depth_scale and depth_scale > 0 else None
    if scale is None:
        if np.issubdtype(arr_np.dtype, np.integer):
            scale = 0.001  # mm → m
            if _debug_decision is not None and not _debug_decision.get('reported', False):
                print(f"[depth] dtype={arr_np.dtype} → assuming millimeters (scale 0.001)")
                _debug_decision['reported'] = True
        else:
            # Float: auto detect via median of positive finite values
            finite = np.isfinite(arr_np)
            positives = (arr_np > 0)
            vals = arr_np[finite & positives]
            med = float(np.median(vals)) if vals.size > 0 else 0.0
            if med > 50.0:
                scale = 0.001
                if _debug_decision is not None and not _debug_decision.get('reported', False):
                    print(f"[depth] float median={med:.3f} → assuming millimeters (scale 0.001)")
                    _debug_decision['reported'] = True
            else:
                scale = 1.0
                if _debug_decision is not None and not _debug_decision.get('reported', False):
                    print(f"[depth] float median={med:.3f} → assuming meters (scale 1.0)")
                    _debug_decision['reported'] = True

    depth_m = arr_np.astype(np.float32) * scale
    # Apply invalid mask
    depth_m = depth_m.astype(np.float32)
    depth_m[invalid_mask] = np.nan
    return depth_m


def load_rgb(rgb_path: str) -> np.ndarray:
    img = Image.open(rgb_path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def backproject_depth_to_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backproject to camera coordinates with optional pixel downsampling.
    Returns (x, y, z, u_valid, v_valid) where (u_valid, v_valid) are pixel indices
    corresponding to the returned 3D points.
    """
    h, w = depth_m.shape
    if stride > 1:
        z_grid = depth_m[::stride, ::stride]
        u_coords, v_coords = np.meshgrid(
            np.arange(0, w, stride, dtype=np.float32),
            np.arange(0, h, stride, dtype=np.float32),
        )
    else:
        z_grid = depth_m
        u_coords, v_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

    valid = ~np.isnan(z_grid) & (z_grid > 0)
    z = z_grid[valid]
    u = u_coords[valid]
    v = v_coords[valid]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return x, y, z, u.astype(np.int32), v.astype(np.int32)


def transform_points(R: np.ndarray, t: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
    return (R @ xyz_cam.T).T + t[None, :]


def write_ply(ply_path: str, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
    n = points_xyz.shape[0]
    with open(ply_path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points_xyz[i]
            r, g, b = colors_rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='Fuse test_dataset RGB-D into colored PLY using FrameTrajectory poses.')
    parser.add_argument('--dataset', type=str, default='test_dataset', help='Path to dataset root containing rgb/, depth/, FrameTrajectory.txt')
    parser.add_argument('--config', type=str, default='config/rtsm.yaml', help='Config YAML with camera intrinsics (D435i)')
    parser.add_argument('--traj', type=str, default=None, help='FrameTrajectory.txt path (defaults to <dataset>/FrameTrajectory.txt)')
    parser.add_argument('--out', type=str, default='fused_world.ply', help='Output PLY path')
    parser.add_argument('--max_frames', type=int, default=0, help='Optional cap on number of frames (0 = all)')
    parser.add_argument('--pixel_stride', type=int, default=1, help='Backprojection pixel stride (>=1). 2 keeps 1/4 pixels, etc.')
    parser.add_argument('--voxel_size', type=float, default=0.0, help='Optional voxel grid downsample size in meters (0 = disabled)')
    parser.add_argument('--depth_scale', type=float, default=0.0, help='Override raw depth scale (e.g., 0.001 for mm→m). 0 = auto')
    args = parser.parse_args()

    fx, fy, cx, cy, width, height = load_camera_intrinsics(args.config)
    dataset = args.dataset
    rgb_dir = os.path.join(dataset, 'rgb')
    depth_dir = os.path.join(dataset, 'depth')
    traj_path = args.traj or os.path.join(dataset, 'FrameTrajectory.txt')

    # Build timestamp indices (ns)
    rgb_ts = build_timestamp_index(rgb_dir, 'png')
    depth_ts = build_timestamp_index(depth_dir, 'npy')
    if not rgb_ts:
        raise RuntimeError(f"No RGB PNGs found in {rgb_dir}")
    if not depth_ts:
        raise RuntimeError(f"No depth NPYs found in {depth_dir}")

    ts_to_pose = parse_frame_trajectory(traj_path)
    if not ts_to_pose:
        raise RuntimeError(f"No poses parsed from {traj_path}")

    fused_xyz: List[np.ndarray] = []
    fused_rgb: List[np.ndarray] = []

    # Iterate over pose timestamps to avoid bias; pair nearest RGB+Depth for each pose
    pose_ts_sorted = sorted(ts_to_pose.keys())
    if args.max_frames > 0:
        pose_ts_sorted = pose_ts_sorted[:args.max_frames]

    for t_pose_ns in pose_ts_sorted:
        t_rgb_ns = find_nearest(t_pose_ns, rgb_ts)
        t_depth_ns = find_nearest(t_pose_ns, depth_ts)

        rgb_path = os.path.join(rgb_dir, f"{t_rgb_ns/1e9:.6f}.png")
        # Ensure string retains full precision used in filenames
        rgb_path = os.path.join(rgb_dir, f"{format(t_rgb_ns/1e9, '.6f')}.png")
        depth_path = os.path.join(depth_dir, f"{format(t_depth_ns/1e9, '.6f')}.npy")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            # Try with higher precision if needed
            rgb_path = os.path.join(rgb_dir, f"{format(t_rgb_ns/1e9, '.6f')}.png")
            depth_path = os.path.join(depth_dir, f"{format(t_depth_ns/1e9, '.6f')}.npy")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue

        rgb = load_rgb(rgb_path)
        # On first use, carry a debug dict to print one-time detection
        if 'depth_debug' not in globals():
            globals()['depth_debug'] = {'reported': False}
        depth_m = depth_npy_to_meters(depth_path, depth_scale=float(args.depth_scale), _debug_decision=globals()['depth_debug'])

        if rgb.shape[1] != depth_m.shape[1] or rgb.shape[0] != depth_m.shape[0]:
            # If needed, crop/resize depth to match RGB (aligned pipeline expected for D435i)
            h = min(rgb.shape[0], depth_m.shape[0])
            w = min(rgb.shape[1], depth_m.shape[1])
            rgb = rgb[:h, :w]
            depth_m = depth_m[:h, :w]

        x, y, z, u_idx, v_idx = backproject_depth_to_points(depth_m, fx, fy, cx, cy, stride=max(1, args.pixel_stride))
        if x.size == 0:
            continue
        xyz_cam = np.stack([x, y, z], axis=1)
        (R, t) = ts_to_pose[t_pose_ns]
        xyz_world = transform_points(R, t, xyz_cam)

        # Gather colors corresponding to selected (subsampled) valid pixels
        colors = rgb[v_idx, u_idx, :]

        fused_xyz.append(xyz_world.astype(np.float32))
        fused_rgb.append(colors.astype(np.uint8))

    if not fused_xyz:
        raise RuntimeError("No valid points fused. Check input paths and alignment.")

    all_xyz = np.concatenate(fused_xyz, axis=0)
    all_rgb = np.concatenate(fused_rgb, axis=0)

    # Optional voxel grid downsampling in world space
    if args.voxel_size and args.voxel_size > 0.0:
        vox = float(args.voxel_size)
        coords = np.floor(all_xyz / vox).astype(np.int64)
        # Unique voxel coords, keep first index
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        all_xyz = all_xyz[unique_idx]
        all_rgb = all_rgb[unique_idx]

    write_ply(args.out, all_xyz, all_rgb)
    print(f"Wrote {all_xyz.shape[0]} points to {args.out}")


if __name__ == '__main__':
    main()


