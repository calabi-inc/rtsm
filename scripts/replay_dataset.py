import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import io
import zmq
import struct



def _is_number(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _parse_timestamp_from_stem(stem: str) -> Optional[float]:
    # Expect stems like "1754989062.627478"
    if _is_number(stem):
        try:
            return float(stem)
        except Exception:
            return None
    return None


def _list_rgb_frames(rgb_dir: str) -> List[Tuple[float, str]]:
    if not os.path.isdir(rgb_dir):
        return []
    entries = []
    for fn in os.listdir(rgb_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem, _ = os.path.splitext(fn)
        ts = _parse_timestamp_from_stem(stem)
        if ts is None:
            continue
        entries.append((ts, os.path.join(rgb_dir, fn)))
    entries.sort(key=lambda x: x[0])
    return entries


def _depth_path_for_timestamp(depth_npy_dir: str, ts: float) -> Optional[str]:
    # The dataset uses timestamp filenames; format back to the original string
    # Keep full precision if present in rgb filenames; here we probe several representations
    candidates = [
        f"{ts}",
        f"{ts:.6f}",
        f"{ts:.9f}",
        ("%.6f" % ts),
        ("%.9f" % ts),
    ]
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        p = os.path.join(depth_npy_dir, c + ".npy")
        if os.path.exists(p):
            return p
    # Fallback: search by prefix match if necessary (slow for large dirs)
    try:
        for fn in os.listdir(depth_npy_dir):
            if not fn.endswith(".npy"):
                continue
            stem, _ = os.path.splitext(fn)
            try:
                if abs(float(stem) - ts) < 1e-6:
                    return os.path.join(depth_npy_dir, fn)
            except Exception:
                continue
    except Exception:
        pass
    return None


@dataclass
class ReplayItem:
    timestamp: float
    rgb_path: str
    depth_path: Optional[str]


@dataclass
class PoseItem:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class KeyframeItem(PoseItem):
    kf_id: int


def build_replay_index(dataset_dir: str) -> List[ReplayItem]:
    rgb_dir = os.path.join(dataset_dir, "rgb")
    depth_npy_dir = os.path.join(dataset_dir, "depth_npy")

    rgb_entries = _list_rgb_frames(rgb_dir)
    items: List[ReplayItem] = []
    for ts, rgb_path in rgb_entries:
        depth_path = _depth_path_for_timestamp(depth_npy_dir, ts)
        items.append(ReplayItem(timestamp=ts, rgb_path=rgb_path, depth_path=depth_path))
    return items


def _sleep_until(target_monotonic: float):
    now = time.monotonic()
    remaining = target_monotonic - now
    if remaining > 0:
        time.sleep(remaining)



def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB array must be HxWx3")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _encode_depth_npy(depth_m: np.ndarray) -> bytes:
    if depth_m.dtype != np.float32:
        depth_m = depth_m.astype(np.float32)
    if depth_m.ndim != 2:
        raise ValueError("Depth array must be HxW float32 meters")
    buf = io.BytesIO()
    # Use compressed container to reduce size
    np.savez_compressed(buf, depth=depth_m)
    return buf.getvalue()


def _load_pose_file(path: str) -> List[PoseItem]:
    poses: List[PoseItem] = []
    if not path or not os.path.exists(path):
        return poses
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                ts = float(parts[0])
                tx = float(parts[1]); ty = float(parts[2]); tz = float(parts[3])
                qx = float(parts[4]); qy = float(parts[5]); qz = float(parts[6]); qw = float(parts[7])
                poses.append(PoseItem(ts, tx, ty, tz, qx, qy, qz, qw))
            except Exception:
                continue
    poses.sort(key=lambda p: p.timestamp)
    return poses


def _load_keyframe_file(path: str) -> List[KeyframeItem]:
    kfs: List[KeyframeItem] = []
    base: List[PoseItem] = _load_pose_file(path)
    for idx, p in enumerate(base):
        kfs.append(KeyframeItem(p.timestamp, p.tx, p.ty, p.tz, p.qx, p.qy, p.qz, p.qw, idx))
    return kfs


def _find_nearest_pose(poses: List[PoseItem], ts: float) -> Optional[PoseItem]:
    if not poses:
        return None
    # Binary search for nearest by timestamp
    lo, hi = 0, len(poses) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if poses[mid].timestamp < ts:
            lo = mid + 1
        else:
            hi = mid
    cand_idx = lo
    cand = poses[cand_idx]
    if cand_idx > 0 and abs(poses[cand_idx - 1].timestamp - ts) < abs(cand.timestamp - ts):
        cand = poses[cand_idx - 1]
    return cand


def _pack_u64(v: int) -> bytes:
    return struct.pack("<Q", int(v))


def _pack_u32(v: int) -> bytes:
    return struct.pack("<I", int(v))


def _pack_f32(v: float) -> bytes:
    return struct.pack("<f", float(v))


def replay(
    items: List[ReplayItem],
    speed: float = 1.0,
    max_delay_s: Optional[float] = None,
    loop: bool = False,
    show: bool = False,
    zmq_enable: bool = False,
    zmq_endpoint: str = "tcp://127.0.0.1:6001",
    zmq_rgb_topic: str = "camera/front/image_raw",
    zmq_depth_topic: str = "camera/front/depth/image_raw",
    traj_frame_path: Optional[str] = None,
    traj_keyframe_path: Optional[str] = None,
    zmq_pose_topic: str = "slam/pose",
    zmq_kf_topic: str = "slam/keyframe",
):
    try:
        import cv2  # Optional visualization
    except Exception:
        cv2 = None
        show = False

    if not items:
        print("No frames to replay.")
        return

    # Setup ZeroMQ if requested
    zmq_ctx = None
    zmq_sock = None
    seq_rgb = 0
    seq_depth = 0
    poses: List[PoseItem] = []
    kfs: List[KeyframeItem] = []
    # Precomputed mappings for O(1) lookups per frame
    pose_idx_for_frame: List[int] = []
    kf_indices_per_frame: List[List[int]] = []
    if traj_frame_path:
        poses = _load_pose_file(traj_frame_path)
    if traj_keyframe_path:
        kfs = _load_keyframe_file(traj_keyframe_path)
    # Build precomputed maps if we have trajectories
    frame_ts: List[float] = [it.timestamp for it in items]
    if frame_ts:
        first_frame_ts = frame_ts[0]
        last_frame_ts = frame_ts[-1]
    else:
        first_frame_ts = last_frame_ts = 0.0
    if kfs:
        kf_min = kfs[0].timestamp
        kf_max = kfs[-1].timestamp
        kf_in_window = sum(1 for k in kfs if first_frame_ts <= k.timestamp <= last_frame_ts)
    else:
        kf_min = kf_max = 0.0
        kf_in_window = 0
    if zmq_enable:
        print(
            f"[replay] frames={len(frame_ts)} poses={len(poses)} kfs={len(kfs)} "
            f"frame_ts=[{first_frame_ts:.6f},{last_frame_ts:.6f}] "
            f"kf_ts=[{kf_min:.6f},{kf_max:.6f}] expected_kfs_in_window={kf_in_window}"
        )
    if poses:
        # Two-pointer sweep to find nearest pose for each frame timestamp (O(N+M))
        pose_idx_for_frame = [-1] * len(frame_ts)
        pi = 0
        for fi, ts in enumerate(frame_ts):
            # advance pose index while it improves closeness
            while pi + 1 < len(poses) and abs(poses[pi + 1].timestamp - ts) <= abs(poses[pi].timestamp - ts):
                pi += 1
            pose_idx_for_frame[fi] = pi
    else:
        pose_idx_for_frame = [-1] * len(frame_ts)
    if kfs:
        # Assign keyframes to the first frame whose timestamp >= kf timestamp
        kf_indices_per_frame = [[] for _ in range(len(frame_ts))]
        ki = 0
        fi = 0
        while ki < len(kfs) and fi < len(frame_ts):
            if kfs[ki].timestamp <= frame_ts[fi]:
                kf_indices_per_frame[fi].append(ki)
                ki += 1
            else:
                fi += 1
        # Any remaining KFs after last frame are dropped (no later frame to attach)
    else:
        kf_indices_per_frame = [[] for _ in range(len(frame_ts))]
    if zmq_enable:
        print(f"[replay] Initializing ZeroMQ PUB at '{zmq_endpoint}' ...")
        zmq_ctx = zmq.Context()
        zmq_sock = zmq_ctx.socket(zmq.PUB)
        zmq_sock.bind(zmq_endpoint)
        # Allow a brief moment for subscribers to connect (avoid slow joiner drop)
        time.sleep(0.1)

    while True:
        t0_ros = items[0].timestamp
        t0_mon = time.monotonic()
        for i, it in enumerate(items):
            # Schedule by ROS-like time derived from filename
            dt_ros = (it.timestamp - t0_ros) / max(1e-9, speed)
            target = t0_mon + dt_ros
            if max_delay_s is not None:
                target = min(target, time.monotonic() + max(0.0, max_delay_s))
            _sleep_until(target)

            # Load RGB
            rgb_img = np.array(Image.open(it.rgb_path).convert("RGB"))

            # Load Depth (meters) if available
            depth_m = None
            if it.depth_path is not None and os.path.exists(it.depth_path):
                try:
                    depth_m = np.load(it.depth_path).astype(np.float32)
                except Exception as e:
                    print(f"Failed to load depth npy '{it.depth_path}': {e}")
                    depth_m = None

            print(f"[{i+1}/{len(items)}] t={it.timestamp:.6f} rgb='{os.path.basename(it.rgb_path)}' depth={'yes' if depth_m is not None else 'no'}")

            # Publish to ZeroMQ topics
            if zmq_sock is not None:
                ts_ns = int(round(it.timestamp * 1e9))
                # RGB as raw bytes (rgb8)
                try:
                    if rgb_img.dtype != np.uint8:
                        rgb_img = rgb_img.astype(np.uint8)
                    H, W, _ = rgb_img.shape
                    rgb_bytes = rgb_img.tobytes(order="C")
                    seq_rgb += 1
                    zmq_sock.send_multipart([
                        zmq_rgb_topic.encode(),
                        _pack_u64(ts_ns),
                        _pack_u32(seq_rgb),
                        b"rgb8",
                        _pack_u32(W),
                        _pack_u32(H),
                        rgb_bytes,
                    ])
                except Exception as e:
                    print(f"[replay] Failed to encode/send RGB: {e}")
                # Depth as raw float32 meters
                if depth_m is not None:
                    try:
                        if depth_m.dtype != np.float32:
                            depth_m = depth_m.astype(np.float32)
                        H, W = depth_m.shape
                        depth_bytes = depth_m.tobytes(order="C")
                        seq_depth += 1
                        zmq_sock.send_multipart([
                            zmq_depth_topic.encode(),
                            _pack_u64(ts_ns),
                            _pack_u32(seq_depth),
                            b"f32",
                            _pack_u32(W),
                            _pack_u32(H),
                            depth_bytes,
                        ])
                    except Exception as e:
                        print(f"[replay] Failed to encode/send depth: {e}")

                # Continuous pose (precomputed index)
                if pose_idx_for_frame:
                    pi = pose_idx_for_frame[i]
                    if 0 <= pi < len(poses):
                        p = poses[pi]
                        zmq_sock.send_multipart([
                            zmq_pose_topic.encode(),
                            _pack_u64(ts_ns),
                            _pack_f32(p.tx), _pack_f32(p.ty), _pack_f32(p.tz),
                            _pack_f32(p.qx), _pack_f32(p.qy), _pack_f32(p.qz), _pack_f32(p.qw),
                        ])

                # Keyframe pose(s) assigned to this frame
                if kf_indices_per_frame:
                    for ki in kf_indices_per_frame[i]:
                        if 0 <= ki < len(kfs):
                            kf = kfs[ki]
                            ts_kf_ns = int(round(kf.timestamp * 1e9))
                            zmq_sock.send_multipart([
                                zmq_kf_topic.encode(),
                                _pack_u64(ts_kf_ns),
                                _pack_u32(kf.kf_id),
                                _pack_f32(kf.tx), _pack_f32(kf.ty), _pack_f32(kf.tz),
                                _pack_f32(kf.qx), _pack_f32(kf.qy), _pack_f32(kf.qz), _pack_f32(kf.qw),
                            ])

            if show and cv2 is not None:
                vis = rgb_img.copy()
                if depth_m is not None:
                    dm = depth_m.copy()
                    # Simple visualization: normalize ignoring NaNs/zeros
                    finite = np.isfinite(dm)
                    if finite.any():
                        vmin = float(np.nanpercentile(dm[finite], 5))
                        vmax = float(np.nanpercentile(dm[finite], 95))
                        vmax = max(vmax, vmin + 1e-6)
                        dm = np.clip((dm - vmin) / (vmax - vmin), 0.0, 1.0)
                        dm = (dm * 255.0).astype(np.uint8)
                        dm = np.stack([dm, dm, dm], axis=-1)
                        vis = np.hstack([vis, dm])
                cv2.imshow("replay (rgb | depth)", vis[:, :, ::-1])
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    return

        if not loop:
            break

    # Teardown ZMQ
    if zmq_sock is not None:
        try:
            zmq_sock.close(0)
        except Exception:
            pass
    if zmq_ctx is not None:
        try:
            zmq_ctx.term()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Replay test_dataset with original timing from filenames.")
    p.add_argument("--dataset", default="test_dataset", help="Dataset root directory (contains rgb/ and depth_npy/)")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (1.0 = real-time)")
    p.add_argument("--max-delay", type=float, default=None, help="Optional cap on per-frame wait seconds")
    p.add_argument("--loop", action="store_true", help="Loop playback")
    p.add_argument("--show", action="store_true", help="Show RGB/Depth preview window")
    # ZeroMQ options
    p.add_argument("--zmq", action="store_true", help="Publish frames via ZeroMQ PUB")
    p.add_argument("--zmq-endpoint", default="tcp://127.0.0.1:6001", help="ZeroMQ bind endpoint for PUB socket")
    p.add_argument("--zmq-rgb-topic", default="camera/front/image_raw", help="ZeroMQ topic for RGB frames")
    p.add_argument("--zmq-depth-topic", default="camera/front/depth/image_raw", help="ZeroMQ topic for depth frames")
    p.add_argument("--traj-frame", default=None, help="Path to FrameTrajectory.txt (TUM format)")
    p.add_argument("--traj-keyframe", default=None, help="Path to KeyFrameTrajectory.txt (TUM format)")
    p.add_argument("--zmq-pose-topic", default="slam/pose", help="ZeroMQ topic for continuous pose")
    p.add_argument("--zmq-kf-topic", default="slam/keyframe", help="ZeroMQ topic for keyframe pose")
    args = p.parse_args()

    items = build_replay_index(args.dataset)
    if not items:
        print(f"No frames found in '{args.dataset}/rgb'. Ensure filenames are numeric timestamps.")
        return
    missing_depth = sum(1 for it in items if it.depth_path is None)
    if missing_depth > 0:
        print(f"Warning: {missing_depth} frames missing matching depth npy in '{args.dataset}/depth_npy'.")

    # Default trajectory paths if not provided
    traj_frame = args.traj_frame or os.path.join(args.dataset, "FrameTrajectory.txt")
    traj_kf = args.traj_keyframe or os.path.join(args.dataset, "KeyFrameTrajectory.txt")

    replay(
        items,
        speed=args.speed,
        max_delay_s=args.max_delay,
        loop=args.loop,
        show=args.show,
        zmq_enable=args.zmq,
        zmq_endpoint=args.zmq_endpoint,
        zmq_rgb_topic=args.zmq_rgb_topic,
        zmq_depth_topic=args.zmq_depth_topic,
        traj_frame_path=traj_frame,
        traj_keyframe_path=traj_kf,
        zmq_pose_topic=args.zmq_pose_topic,
        zmq_kf_topic=args.zmq_kf_topic,
    )


if __name__ == "__main__":
    main()




