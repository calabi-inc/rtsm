import argparse
import sys
import time
from typing import Optional, Tuple

import numpy as np
import zmq

from rtsm.stores.sweep_cache import SweepCache
from rtsm.core.ingest_gate import IngestGate


def _u64(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def _u32(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def _f32(b: bytes) -> float:
    return float(np.frombuffer(b, dtype=np.float32, count=1)[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Test IngestGate decisions over ZeroMQ streams (prints accepts).")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:6001")
    p.add_argument("--two_d", action="store_true", help="Use 2D bucket (Z collapsed)")
    p.add_argument("--grid", type=float, default=0.25, help="Bucket grid size in meters")
    p.add_argument("--dup_window_ns", type=int, default=int(0.2 * 1e9), help="Non-KF dup window in ns around last KF")
    args = p.parse_args()

    # State: latest pose (ts, xyz, q_xyzw)
    pose_ts_ns: Optional[int] = None
    pose_xyz: Optional[np.ndarray] = None
    pose_q_xyzw: Optional[Tuple[float, float, float, float]] = None

    # Bucket and gate
    sweep_cache = SweepCache(two_d=args.two_d, grid_size_m=args.grid)
    gate = IngestGate(cfg={"ingest": {"dup_window_ns": int(args.dup_window_ns)}})

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.endpoint)
    for t in ("camera/front/image_raw", "camera/front/depth/image_raw", "slam/pose", "slam/keyframe"):
        sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    print(f"[test_ingest_gate] SUB {args.endpoint}")
    try:
        while True:
            parts = sock.recv_multipart()
            topic = parts[0].decode(errors="ignore")

            if topic == "slam/pose":
                # [topic, u64 ts_ns, 7 x f32] => x y z qx qy qz qw
                if len(parts) != 9:
                    print(f"[test_ingest_gate] bad pose parts={len(parts)}", file=sys.stderr)
                    continue
                pose_ts_ns = _u64(parts[1])
                x, y, z = _f32(parts[2]), _f32(parts[3]), _f32(parts[4])
                qx, qy, qz, qw = _f32(parts[5]), _f32(parts[6]), _f32(parts[7]), _f32(parts[8])
                pose_xyz = np.array([x, y, z], dtype=float)
                pose_q_xyzw = (qx, qy, qz, qw)

            elif topic == "slam/keyframe":
                # [topic, u64 ts_ns, u32 id, 7 x f32]
                if len(parts) != 10:
                    print(f"[test_ingest_gate] bad keyframe parts={len(parts)}", file=sys.stderr)
                    continue
                ts_ns = _u64(parts[1])
                # kf_id = _u32(parts[2])  # unused here
                x, y, z = _f32(parts[3]), _f32(parts[4]), _f32(parts[5])
                qx, qy, qz, qw = _f32(parts[6]), _f32(parts[7]), _f32(parts[8]), _f32(parts[9])
                xyz = np.array([x, y, z], dtype=float)
                q_xyzw = (qx, qy, qz, qw)

                # Compute view keys
                cell, vbin, fwd = sweep_cache.cell_and_vbin_from_pose(twc_xyz=xyz, q_wc_xyzw=q_xyzw)
                dec = gate.should_accept(
                    is_keyframe=True,
                    ts_ns=ts_ns,
                    sweep_cache=sweep_cache,
                    cell=cell,
                    vbin=vbin,
                    cam_pos=xyz,
                    fwd_unit=fwd,
                    Z=None,
                    look_cell=None,
                    now_mono=time.monotonic(),
                )
                if dec.accept:
                    print(f"[test_ingest_gate] ACCEPT keyframe ts={ts_ns} reason={dec.reason} cell={cell} vbin={vbin}")
                    gate.record_processed(
                        is_keyframe=True,
                        ts_ns=ts_ns,
                        sweep_cache=sweep_cache,
                        cell=cell,
                        vbin=vbin,
                        cam_pos=xyz,
                        look_cell=None,
                        now_mono=time.monotonic(),
                    )

            elif topic.startswith("camera/front/"):
                # [topic, u64 ts_ns, u32 seq, enc, u32 W, u32 H, bytes]
                if len(parts) < 7:
                    print(f"[test_ingest_gate] bad image parts={len(parts)}", file=sys.stderr)
                    continue
                ts_ns = _u64(parts[1])
                if pose_xyz is None or pose_q_xyzw is None:
                    # no pose yet
                    continue
                # Compute view keys from latest pose
                cell, vbin, fwd = sweep_cache.cell_and_vbin_from_pose(twc_xyz=pose_xyz, q_wc_xyzw=pose_q_xyzw)
                dec = gate.should_accept(
                    is_keyframe=False,
                    ts_ns=ts_ns,
                    sweep_cache=sweep_cache,
                    cell=cell,
                    vbin=vbin,
                    cam_pos=pose_xyz,
                    fwd_unit=fwd,
                    Z=None,
                    look_cell=None,
                    now_mono=time.monotonic(),
                )
                if dec.accept:
                    print(f"[test_ingest_gate] ACCEPT image ts={ts_ns} reason={dec.reason} cell={cell} vbin={vbin}")
                    gate.record_processed(
                        is_keyframe=False,
                        ts_ns=ts_ns,
                        sweep_cache=sweep_cache,
                        cell=cell,
                        vbin=vbin,
                        cam_pos=pose_xyz,
                        look_cell=None,
                        now_mono=time.monotonic(),
                    )
            else:
                # ignore other topics
                pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()


