#!/usr/bin/env python3
"""
Run a configured segmenter over a bag and write a sibling bag that carries its
detections as a ``vision_msgs/msg/Detection2DArray`` topic (P3, the detections
adapter). The result is what a customer's own detector would have recorded:
the gate feeds it back through ``rtsm eval --set segmentation.backend=external``
and compares against the native run; it is also a demonstration of the format.

    python scripts/export_detections.py recordings/session1_bag recordings/session1_bag_det
    python scripts/export_detections.py in.bag out_bag --set segmentation.backend=dual --no-scores
    python scripts/export_detections.py in_bag out_bag --max-frames 300 --topic /my/detector

Every message of the input is copied (deserialised and re-serialised as CDR,
so ROS 1 bags come out as rosbag2); one Detection2DArray per paired RGB frame
is added on ``--topic`` with the frame's stamp and frame id. Boxes come from
the segmenter's boxes (or the bounding boxes of its masks), labels from its
detection labels, scores from its confidences; ``--no-scores`` writes NaN
scores, which the adapter reads as "the detector reports none". Needs the
GPU extras: the segmenter is the real one.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

if importlib.util.find_spec("rtsm") is None:  # running from a checkout without an install: the repo root is one level up
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("export_detections")


def _boxes_from_result(seg, H: int, W: int) -> np.ndarray:
    boxes = getattr(seg, "boxes", None)
    if boxes is not None and len(boxes):
        b = np.asarray(boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else boxes, dtype=np.float32).reshape(-1, 4)
        return b
    masks = getattr(seg, "masks", None)
    if masks is None or not len(masks):
        return np.zeros((0, 4), dtype=np.float32)
    m = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    mh, mw = m.shape[1:]
    sy, sx = H / float(mh), W / float(mw)
    out = []
    for k in range(m.shape[0]):
        ys, xs = np.where(m[k])
        if ys.size == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        out.append([xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy])
    return np.asarray(out, dtype=np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    from rtsm.cfg.cli import add_config_arguments, config_from_args
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="a ROS 1 .bag, a rosbag2 directory or a bare .mcap the bag reader accepts")
    ap.add_argument("output", help="the rosbag2 directory to write (MCAP storage, chunk zstd)")
    ap.add_argument("--topic", default="/rtsm/detections")
    ap.add_argument("--no-scores", action="store_true", help="write NaN scores: the detector reports no confidence")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-id", default=None, help="frame id for the detections headers (default: the camera frame)")
    add_config_arguments(ap)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)

    from PIL import Image
    from rosbags.rosbag2 import CompressionFormat, CompressionMode, StoragePlugin, Writer
    from rtsm.engine import load_models
    from rtsm.io import codecs
    from rtsm.io.bag_reader import BagStats, _open_stream, iter_bag_frames, probe_bag
    from rtsm.io.msgdefs import DETECTION_2D, register_vision_msgs

    inp, out = Path(args.input), Path(args.output)
    if out.exists():
        ap.error(f"output exists: {out}")
    probe = probe_bag(inp, typestore=str((cfg.get("io") or {}).get("bag", {}).get("typestore") or "humble"))
    if probe.refusal:
        ap.error("bag refused: " + "; ".join(f"{c}: {d}" for c, d in probe.refusal))
    cam_frame = args.frame_id or probe.camera_frame or "camera"

    seg_cfg = cfg.get("segmentation") or {}
    backend = str(seg_cfg.get("backend"))
    if backend == "external":
        ap.error("segmentation.backend=external has no detector to export; choose a model backend")
    vocab = None
    if backend in ("dual", "yoloe"):
        vocab = (seg_cfg.get("yoloe") or {}).get("vocab")
    elif backend == "grounded_sam2":
        vocab = (seg_cfg.get("grounded_sam2") or {}).get("vocab")
    logger.info("loading the %s backend", backend)
    models = load_models(cfg)
    segmenter = models.segmenter

    # 1. copy every message, re-serialised as CDR
    stream = _open_stream(inp, probe.bag.get("typestore", "humble"))
    ts = stream._reader.typestore if hasattr(stream, "_reader") else stream._ts
    register_vision_msgs(ts, "ros2")
    T = ts.types
    w = Writer(out, version=9, storage_plugin=StoragePlugin.MCAP)
    w.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
    n_copied = 0
    t0 = time.monotonic()
    with w:
        conns = {}
        for topic, msgtype, log_ns, msg in stream.messages([t.name for t in stream.info.topics.values()]):
            if topic not in conns:
                conns[topic] = w.add_connection(topic, msgtype, typestore=ts)
            w.write(conns[topic], int(log_ns), ts.serialize_cdr(msg, msgtype))
            n_copied += 1
        stream.close()
        logger.info("copied %d messages in %.1f s", n_copied, time.monotonic() - t0)

        # 2. the detections
        det_conn = w.add_connection(args.topic, DETECTION_2D, typestore=ts)
        Hdr, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
        D2A, D2, OHP, OH, BB, P2, PT = (T[DETECTION_2D], T["vision_msgs/msg/Detection2D"], T["vision_msgs/msg/ObjectHypothesisWithPose"],
                                        T["vision_msgs/msg/ObjectHypothesis"], T["vision_msgs/msg/BoundingBox2D"], T["vision_msgs/msg/Pose2D"], T["vision_msgs/msg/Point2D"])
        PWC, Pose, Pt, Q = T["geometry_msgs/msg/PoseWithCovariance"], T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/Point"], T["geometry_msgs/msg/Quaternion"]
        pose0 = PWC(pose=Pose(position=Pt(x=0.0, y=0.0, z=0.0), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)), covariance=np.zeros(36))
        st = BagStats()
        n_frames = n_det = 0
        t1 = time.monotonic()
        for fr in iter_bag_frames(inp, stats=st, max_frames=args.max_frames, typestore=probe.bag.get("typestore", "humble")):
            h = fr.header
            bgr = codecs.decode_rgb(h.rgb.data, h.rgb.encoding, h.rgb.width, h.rgb.height)
            H, W = bgr.shape[:2]
            pil = Image.fromarray(bgr[..., ::-1])
            seg = segmenter.segment(pil, vocab=vocab)
            boxes = _boxes_from_result(seg, H, W)
            labels = list(getattr(seg, "detection_labels", None) or getattr(seg, "labels", None) or [])
            confs = getattr(seg, "label_confidence", None)
            if confs is None and getattr(seg, "scores", None) is not None:
                confs = [float(s) for s in seg.scores]
            dets = []
            for k in range(boxes.shape[0]):
                x0, y0, x1, y1 = (float(v) for v in boxes[k])
                if x1 <= x0 or y1 <= y0:
                    continue
                label = (str(labels[k]) if k < len(labels) and labels[k] else "object")
                score = float("nan") if args.no_scores or confs is None or k >= len(confs) or confs[k] is None else float(confs[k])
                stamp = Time(sec=int(h.t_sensor_ns) // 1_000_000_000, nanosec=int(h.t_sensor_ns) % 1_000_000_000)
                dets.append(D2(header=Hdr(stamp=stamp, frame_id=cam_frame), results=[OHP(hypothesis=OH(class_id=label, score=score), pose=pose0)],
                               bbox=BB(center=P2(position=PT(x=(x0 + x1) / 2, y=(y0 + y1) / 2), theta=0.0), size_x=x1 - x0, size_y=y1 - y0), id=""))
            stamp = Time(sec=int(h.t_sensor_ns) // 1_000_000_000, nanosec=int(h.t_sensor_ns) % 1_000_000_000)
            msg = D2A(header=Hdr(stamp=stamp, frame_id=cam_frame), detections=dets)
            log_ns = int(round((h.t_wall_utc_s or 0.0) * 1e9)) + 50_000_000 if h.t_wall_utc_s else int(h.t_sensor_ns) + 50_000_000
            w.write(det_conn, log_ns, ts.serialize_cdr(msg, DETECTION_2D))
            n_frames += 1
            n_det += len(dets)
            if n_frames % 20 == 0:
                logger.info("%d frames, %d detections, %.1f s", n_frames, n_det, time.monotonic() - t1)
    for m in (getattr(models, "segmenter", None), getattr(models, "clip", None)):
        closer = getattr(m, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
    print(f"wrote {out}: {n_copied} messages copied + {n_frames} Detection2DArray on {args.topic} ({n_det} detections, scores {'absent' if args.no_scores else 'present'}) in {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
