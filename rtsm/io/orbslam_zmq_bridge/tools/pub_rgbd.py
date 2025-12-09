#!/usr/bin/env python3
import argparse
import time
import os
import glob
import struct
import numpy as np
import cv2
import zmq


def le_u64(v: int) -> bytes:
    return struct.pack('<Q', int(v))


def send_frame(sock, t_ns: int, rgb_bgr: np.ndarray, depth_u16: np.ndarray):
    ok, jpg = cv2.imencode('.jpg', rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return
    sock.send_multipart([
        b'rgbd/input',
        le_u64(t_ns),
        jpg.tobytes(),
        depth_u16.tobytes(),
    ])


def synthetic_sequence(w: int, h: int, depth_val: int, n: int):
    for i in range(n):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(rgb, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        depth = np.full((h, w), depth_val, dtype=np.uint16)
        yield rgb, depth


def folder_sequence(rgb_glob: str, depth_glob: str):
    rgb_files = sorted(glob.glob(rgb_glob))
    depth_files = sorted(glob.glob(depth_glob))
    for rf, df in zip(rgb_files, depth_files):
        rgb = cv2.imread(rf, cv2.IMREAD_COLOR)
        depth = cv2.imread(df, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        if depth.dtype != np.uint16:
            raise RuntimeError(f"Depth image must be 16UC1: {df}")
        yield rgb, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True, help='tcp://host:port to connect (SUB of bridge)')
    ap.add_argument('--hz', type=float, default=10.0)
    ap.add_argument('--rgb-glob', default='')
    ap.add_argument('--depth-glob', default='')
    ap.add_argument('--w', type=int, default=640)
    ap.add_argument('--h', type=int, default=480)
    ap.add_argument('--depth-val', type=int, default=1000, help='synthetic depth in raw units (e.g., mm if factor=1000)')
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(args.endpoint)

    if args.rgb_glob and args.depth_glob:
        frames = folder_sequence(args.rgb_glob, args.depth_glob)
    else:
        frames = synthetic_sequence(args.w, args.h, args.depth_val, 1_000_000_000)

    period = 1.0 / args.hz if args.hz > 0 else 0.0
    t0 = time.time_ns()
    for rgb, depth in frames:
        t_ns = time.time_ns()
        send_frame(pub, t_ns, rgb, depth)
        if period > 0:
            to_sleep = period - (time.time_ns() - t_ns) * 1e-9
            if to_sleep > 0:
                time.sleep(to_sleep)


if __name__ == '__main__':
    main()


