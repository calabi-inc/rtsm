import argparse
import sys
import time

import roslibpy

def _stamp_to_float(stamp):
    if not isinstance(stamp, dict):
        return None
    # ROS1: {secs, nsecs}; ROS2: {sec, nanosec}
    if 'secs' in stamp:
        sec = stamp.get('secs', 0)
        nsec = stamp.get('nsecs', 0)
    else:
        sec = stamp.get('sec', 0)
        nsec = stamp.get('nanosec', stamp.get('nsec', 0))
    try:
        return float(sec) + float(nsec) * 1e-9
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='Subscribe to /rgb and /depth via rosbridge and print compact events.')
    ap.add_argument('--host', default='localhost', help='rosbridge host (default: localhost)')
    ap.add_argument('--port', type=int, default=9090, help='rosbridge port (default: 9090)')
    ap.add_argument('--rgb-topic', default='/rgb', help='RGB topic name (default: /rgb)')
    ap.add_argument('--depth-topic', default='/depth', help='Depth topic name (default: /depth)')
    args = ap.parse_args()

    ros = roslibpy.Ros(host=args.host, port=args.port)
    try:
        ros.run()
    except Exception as e:
        print(f"[test_ros] failed to connect to rosbridge at {args.host}:{args.port}: {e}")
        sys.exit(1)
    if not ros.is_connected:
        print(f"[test_ros] not connected to rosbridge at {args.host}:{args.port}")
        sys.exit(2)
    print('[test_ros] connected to rosbridge')

    rgb = roslibpy.Topic(ros, args.rgb_topic, 'sensor_msgs/Image')
    depth = roslibpy.Topic(ros, args.depth_topic, 'sensor_msgs/Image')

    def on_rgb(msg):
        h = msg.get('header', {})
        t = _stamp_to_float(h.get('stamp'))
        w = msg.get('width', '?')
        hgt = msg.get('height', '?')
        print(f"[rgb] t={t if t is not None else 'na'} size={w}x{hgt}")

    def on_depth(msg):
        h = msg.get('header', {})
        t = _stamp_to_float(h.get('stamp'))
        w = msg.get('width', '?')
        hgt = msg.get('height', '?')
        print(f"[depth] t={t if t is not None else 'na'} size={w}x{hgt}")

    rgb.subscribe(on_rgb)
    depth.subscribe(on_depth)

    print('[test_ros] subscriptions active. Press Ctrl+C to exit.')
    try:
        while ros.is_connected:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rgb.unsubscribe()
        except Exception:
            pass
        try:
            depth.unsubscribe()
        except Exception:
            pass
        ros.terminate()


if __name__ == '__main__':
    main()


