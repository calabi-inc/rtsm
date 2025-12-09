import argparse
import sys

import zmq


def main():
    p = argparse.ArgumentParser(description="Simple ZeroMQ subscriber that prints timestamps from replay.")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:6001", help="ZeroMQ endpoint to connect to")
    p.add_argument(
        "--topic",
        action="append",
        default=None,
        help="Topic to subscribe to (repeat for multiple). Default: camera/front/image_raw, camera/front/depth/image_raw, slam/pose, slam/keyframe",
    )
    args = p.parse_args()

    topics = (
        args.topic
        if args.topic is not None
        else [
            "camera/front/image_raw",
            "camera/front/depth/image_raw",
            "slam/pose",
            "slam/keyframe",
        ]
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.endpoint)
    for t in topics:
        sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    print(f"[test_zmq] Connected to {args.endpoint}, subscribed to {topics}")
    try:
        while True:
            parts = sock.recv_multipart()
            topic_b = parts[0]
            topic = topic_b.decode(errors="ignore")
            if topic.startswith("camera/front/"):
                # [topic, u64 ts_ns, u32 seq, enc, u32 W, u32 H, bytes]
                if len(parts) < 7:
                    print(f"[test_zmq] Unexpected image parts: {len(parts)}", file=sys.stderr)
                    continue
                ts_ns = int.from_bytes(parts[1], byteorder="little", signed=False)
                ts_s = ts_ns / 1e9
                print(f"topic={topic} ts_s={ts_s:.6f}")
            elif topic == "slam/pose":
                # [topic, u64 ts_ns, 7 x f32]
                if len(parts) != 9:
                    print(f"[test_zmq] Unexpected pose parts: {len(parts)}", file=sys.stderr)
                    continue
                ts_ns = int.from_bytes(parts[1], byteorder="little", signed=False)
                ts_s = ts_ns / 1e9
                print(f"topic={topic} ts_s={ts_s:.6f}")
            elif topic == "slam/keyframe":
                # [topic, u64 ts_ns, u32 id, 7 x f32]
                if len(parts) != 10:
                    print(f"[test_zmq] Unexpected keyframe parts: {len(parts)}", file=sys.stderr)
                    continue
                ts_ns = int.from_bytes(parts[1], byteorder="little", signed=False)
                ts_s = ts_ns / 1e9
                print(f"topic={topic} ts_s={ts_s:.6f}")
            else:
                print(f"[test_zmq] Unknown topic: {topic}")
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


