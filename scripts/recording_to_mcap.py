#!/usr/bin/env python
"""Convert a Lens recording (messages.bin + index.jsonl + meta.json) into a
rosbag2 directory with MCAP storage (P3 task 0).

    python scripts/recording_to_mcap.py recordings/session1 recordings/session1_bag
    python scripts/recording_to_mcap.py recordings/session1 /tmp/s1_20 --max-frames 20 --no-compression

Needs the eval extra (pip install "rtsm[eval]"). The ARKit->OpenCV camera flip
is baked into /tf by default (the receiver applies it once at ingest; session1
was anchored with it on); --no-flip keeps the raw ARKit pose and records
pose_convention=arkit in the bag's custom data. Layout and parity test:
rtsm/evaluation/recording_mcap.py, tests/evaluation/test_recording_mcap.py.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording_dir")
    ap.add_argument("out_dir", help="rosbag2 directory to create (metadata.yaml + <name>.mcap)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-flip", action="store_true", help="do NOT bake the ARKit->OpenCV flip into /tf")
    ap.add_argument("--no-compression", action="store_true", help="write uncompressed MCAP chunks (default: zstd per message)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    try:
        from rtsm.evaluation.recording_mcap import convert_recording_to_mcap
    except ModuleNotFoundError:                      # run from a checkout without `pip install -e .`
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rtsm.evaluation.recording_mcap import convert_recording_to_mcap
    summary = convert_recording_to_mcap(
        args.recording_dir, args.out_dir, apply_camera_flip=not args.no_flip, max_frames=args.max_frames,
        compression=(None if args.no_compression else "zstd"), overwrite=args.overwrite,
    )
    print(json.dumps(summary.as_dict(), indent=1))
    return 0 if summary.frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
