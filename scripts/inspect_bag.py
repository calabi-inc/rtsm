#!/usr/bin/env python
"""Print what a bag actually contains, the way the eval reader will see it
(P3 task 1 input survey): storage / compression, topics with types, counts and
rates, image encodings and sizes, CameraInfo K, the TF frames and which of them
are dynamic, stamp ranges and the log-time vs header-stamp offset.

    python scripts/inspect_bag.py recordings/external/rgbd_dataset_freiburg3_long_office_household.bag
    python scripts/inspect_bag.py recordings/external/r2b_cafe
    python scripts/inspect_bag.py recordings/session1_bag --frames 3

Handles ROS 1 .bag files and rosbag2 directories (sqlite3 or mcap storage)
through rosbags' AnyReader; needs the eval extra.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--frames", type=int, default=2, help="how many image messages per topic to decode for the shape report")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print('rosbags is not installed: pip install "rtsm[eval]"', file=sys.stderr)
        return 2
    p = Path(args.path)
    report = {"path": str(p), "topics": {}, "tf": {}, "images": {}, "camera_info": {}, "stamps": {}}
    # rosbag2 bags written before message definitions were embedded (e.g. NVIDIA's r2b, rosbag2 v5) carry no
    # type definitions: a default typestore is required, and harmless for bags that do embed them.
    from rosbags.typesys import Stores, get_typestore
    with AnyReader([p], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        report["message_count"] = r.message_count
        report["duration_s"] = round(r.duration / 1e9, 3)
        report["start_ns"], report["end_ns"] = r.start_time, r.end_time
        conns = list(r.connections)
        for c in conns:
            rate = c.msgcount / (r.duration / 1e9) if r.duration else None
            report["topics"][c.topic] = {"type": c.msgtype, "count": c.msgcount, "hz": (round(rate, 2) if rate else None)}
        img_seen: Counter = Counter()
        tf_frames: dict = defaultdict(set)
        tf_static_frames: dict = defaultdict(set)
        hdr_offsets: dict = defaultdict(list)
        stamp_min: dict = {}
        stamp_max: dict = {}
        image_conns = [c for c in conns if c.msgtype in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/CameraInfo")]
        tf_conns = [c for c in conns if c.msgtype in ("tf2_msgs/msg/TFMessage", "tf/msg/tfMessage", "tf/tfMessage")]
        for conn, log_ns, raw in r.messages(connections=image_conns + tf_conns):
            msg = r.deserialize(raw, conn.msgtype)
            if conn.msgtype in ("tf2_msgs/msg/TFMessage", "tf/msg/tfMessage", "tf/tfMessage"):
                for tr in msg.transforms:
                    key = (tr.header.frame_id, tr.child_frame_id)
                    (tf_static_frames if conn.topic.endswith("tf_static") else tf_frames)[key].add(conn.topic)
                    hdr_ns = int(tr.header.stamp.sec) * 1_000_000_000 + int(tr.header.stamp.nanosec)
                    stamp_min[conn.topic] = min(stamp_min.get(conn.topic, hdr_ns), hdr_ns)
                    stamp_max[conn.topic] = max(stamp_max.get(conn.topic, hdr_ns), hdr_ns)
                continue
            hdr_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            stamp_min[conn.topic] = min(stamp_min.get(conn.topic, hdr_ns), hdr_ns)
            stamp_max[conn.topic] = max(stamp_max.get(conn.topic, hdr_ns), hdr_ns)
            if len(hdr_offsets[conn.topic]) < 200:
                hdr_offsets[conn.topic].append((log_ns - hdr_ns) / 1e6)
            if conn.msgtype == "sensor_msgs/msg/CameraInfo":
                if conn.topic not in report["camera_info"]:
                    k = [round(float(v), 3) for v in msg.k] if hasattr(msg, "k") else [round(float(v), 3) for v in msg.K]
                    d = list(msg.d) if hasattr(msg, "d") else list(msg.D)
                    report["camera_info"][conn.topic] = {"width": int(msg.width), "height": int(msg.height), "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                                                         "distortion_model": msg.distortion_model, "n_d": len(d), "frame_id": msg.header.frame_id}
                continue
            if img_seen[conn.topic] >= args.frames:
                continue
            img_seen[conn.topic] += 1
            entry = report["images"].setdefault(conn.topic, {"type": conn.msgtype, "frame_id": msg.header.frame_id, "samples": []})
            if conn.msgtype == "sensor_msgs/msg/Image":
                import numpy as np
                data = np.asarray(msg.data, dtype=np.uint8)
                s = {"width": int(msg.width), "height": int(msg.height), "encoding": msg.encoding, "step": int(msg.step), "is_bigendian": int(msg.is_bigendian), "bytes": int(data.size)}
                if msg.encoding in ("16UC1", "mono16"):
                    a = data.view(np.uint16).reshape(int(msg.height), int(msg.width))
                    s.update({"zeros_frac": round(float((a == 0).mean()), 4), "p50": int(np.median(a[a > 0])) if (a > 0).any() else None, "max": int(a.max())})
                elif msg.encoding == "32FC1":
                    a = data.view(np.float32).reshape(int(msg.height), int(msg.width))
                    finite = np.isfinite(a) & (a > 0)
                    s.update({"invalid_frac": round(float(1 - finite.mean()), 4), "p50_m": round(float(np.median(a[finite])), 3) if finite.any() else None, "nan_frac": round(float(np.isnan(a).mean()), 4)})
                entry["samples"].append(s)
            else:
                entry["samples"].append({"format": msg.format, "bytes": len(msg.data)})
        report["tf"] = {"dynamic": sorted(f"{a} -> {b}" for (a, b) in tf_frames), "static": sorted(f"{a} -> {b}" for (a, b) in tf_static_frames)}
        report["stamps"] = {t: {"header_min_ns": stamp_min[t], "header_max_ns": stamp_max[t], "span_s": round((stamp_max[t] - stamp_min[t]) / 1e9, 3),
                                "log_minus_header_ms_median": (round(sorted(hdr_offsets[t])[len(hdr_offsets[t]) // 2], 3) if hdr_offsets.get(t) else None)}
                            for t in stamp_min}
    if args.json:
        print(json.dumps(report, indent=1, default=str))
        return 0
    print(f"{report['path']}: {report['message_count']} messages over {report['duration_s']} s")
    print("topics:")
    for t, v in sorted(report["topics"].items()):
        print(f"  {t:<50} {v['type']:<34} n={v['count']:<7} {v['hz'] or ''} Hz")
    print("images:")
    for t, v in report["images"].items():
        print(f"  {t} ({v['type'].split('/')[-1]}, frame_id={v['frame_id']!r}): {v['samples'][0] if v['samples'] else '-'}")
    print("camera_info:")
    for t, v in report["camera_info"].items():
        print(f"  {t}: {v}")
    print(f"tf dynamic: {report['tf']['dynamic']}")
    print(f"tf static:  {report['tf']['static']}")
    print("stamps:")
    for t, v in report["stamps"].items():
        print(f"  {t}: span {v['span_s']} s, log-header median {v['log_minus_header_ms_median']} ms, header_min {v['header_min_ns']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
