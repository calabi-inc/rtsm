#!/usr/bin/env python3
import argparse
import struct
import numpy as np
import zmq


def parse_u64_le(b: bytes) -> int:
    return struct.unpack('<Q', b)[0]


def parse_mat4f_le(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype='<f4')
    if arr.size != 16:
        raise ValueError('Expected 16 float32s')
    return arr.reshape(4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True, help='tcp://host:port to connect')
    ap.add_argument('--topic', default='slam/tracking_pose')
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, args.topic.encode('ascii'))

    print(f"Subscribed to {args.topic} at {args.endpoint}")
    while True:
        frames = sub.recv_multipart()
        topic = frames[0].decode('ascii')
        if topic == 'slam/tracking_pose':
            t_ns = parse_u64_le(frames[1])
            Twc = parse_mat4f_le(frames[2])
            print(f"tracking t={t_ns} ns\n{Twc}")
        elif topic == 'slam/kf_pose':
            kf_id = parse_u64_le(frames[1])
            t_ns = parse_u64_le(frames[2])
            Twc = parse_mat4f_le(frames[3])
            print(f"kf {kf_id} t={t_ns} ns\n{Twc}")
        else:
            print(f"unknown topic {topic} with {len(frames)-1} payload frames")


if __name__ == '__main__':
    main()


