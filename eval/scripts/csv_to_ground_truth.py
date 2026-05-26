#!/usr/bin/env python3
"""
Convert marker_assignments.csv into per-scene ground_truth.yaml files.

For each unique scene_id in the CSV, writes:
  eval/datasets/<scene_id>/ground_truth.yaml

The 3D positions are NOT filled in here — they get computed later by the
offline ArUco detector (rtsm/evaluation/scenes/aruco.py) from the recorded
RGB frames + ARKit pose. This script only emits the marker→object mapping.

Usage:
  python eval/scripts/csv_to_ground_truth.py
  python eval/scripts/csv_to_ground_truth.py --csv eval/marker_assignments.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "eval" / "marker_assignments.csv"
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"

REQUIRED_COLS = ("scene_id", "marker_id", "object_id", "label")
ARUCO_ACCURACY_M = 0.01


def yaml_escape(s: str) -> str:
    if any(c in s for c in ':#[]{}\n"\''):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: CSV is missing required column(s): {missing}\n       Expected at least: {list(REQUIRED_COLS)}")
        rows = []
        for i, row in enumerate(reader, start=2):
            scene = (row.get("scene_id") or "").strip()
            mid_raw = (row.get("marker_id") or "").strip()
            obj = (row.get("object_id") or "").strip()
            label = (row.get("label") or "").strip()
            if not scene and not mid_raw and not obj:
                continue  # blank line
            if not (scene and mid_raw and obj and label):
                sys.exit(f"ERROR: row {i} is missing required field(s): {row!r}")
            try:
                mid = int(mid_raw)
            except ValueError:
                sys.exit(f"ERROR: row {i} marker_id is not an integer: {mid_raw!r}")
            if not (0 <= mid <= 49):
                sys.exit(f"ERROR: row {i} marker_id {mid} is outside DICT_4X4_50 range 0..49")
            rows.append(
                dict(
                    scene_id=scene,
                    marker_id=mid,
                    object_id=obj,
                    label=label,
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return rows


def validate_uniqueness(rows: list[dict]) -> None:
    seen_marker = defaultdict(list)
    seen_object = defaultdict(list)
    for r in rows:
        seen_marker[(r["scene_id"], r["marker_id"])].append(r)
        seen_object[(r["scene_id"], r["object_id"])].append(r)
    errors = []
    for key, rs in seen_marker.items():
        if len(rs) > 1:
            errors.append(f"  scene={key[0]!r} marker_id={key[1]} appears {len(rs)} times")
    for key, rs in seen_object.items():
        if len(rs) > 1:
            errors.append(f"  scene={key[0]!r} object_id={key[1]!r} appears {len(rs)} times")
    if errors:
        sys.exit("ERROR: duplicate (scene_id, marker_id) or (scene_id, object_id):\n" + "\n".join(errors))


def write_scene_yaml(scene_id: str, scene_rows: list[dict]) -> Path:
    scene_dir = DATASETS_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_path = scene_dir / "ground_truth.yaml"

    lines: list[str] = []
    lines.append(f"# Auto-generated from eval/marker_assignments.csv on {date.today().isoformat()}")
    lines.append("# Edit the CSV and re-run csv_to_ground_truth.py instead of editing this file.")
    lines.append(f"scene_id: {yaml_escape(scene_id)}")
    lines.append('marker_dictionary: "DICT_4X4_50"')
    lines.append("marker_size_m: 0.05")
    lines.append("objects:")
    for r in sorted(scene_rows, key=lambda x: x["marker_id"]):
        lines.append(f"  - id: {yaml_escape(r['object_id'])}")
        lines.append(f"    label: {yaml_escape(r['label'])}")
        lines.append(f"    position_source: aruco_marker_{r['marker_id']}")
        lines.append(f"    position_accuracy_m: {ARUCO_ACCURACY_M}")
        lines.append("    position_world: null  # filled by aruco.py from recorded frames")
        if r["notes"]:
            lines.append(f"    notes: {yaml_escape(r['notes'])}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to marker_assignments.csv")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    rows = load_rows(args.csv)
    if not rows:
        sys.exit("ERROR: CSV has no data rows.")

    validate_uniqueness(rows)

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["scene_id"]].append(r)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(rows)} rows across {len(by_scene)} scene(s) from {args.csv.relative_to(REPO_ROOT)}\n")
    for scene_id, scene_rows in sorted(by_scene.items()):
        out = write_scene_yaml(scene_id, scene_rows)
        print(f"  {scene_id:30s}  {len(scene_rows):2d} objects  ->  {out.relative_to(REPO_ROOT)}")

    print(f"\nDone. Positions are null until you run the offline ArUco detector on the recorded frames.")


if __name__ == "__main__":
    main()
