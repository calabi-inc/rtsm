#!/usr/bin/env python3
"""
RTSM Edge Datasheet Benchmark — all backends, single machine.
=============================================================
Sweeps every segmentation backend through the SAME replay session on the
current machine and emits per-backend raw JSON + a flat datasheet table.

Unlike scripts/benchmark_backends.py (which compares exactly dual vs
grounded_sam2), this runs the full set so we get FastSAM-only edge numbers
and an internally-consistent comparison on the current code (SigLIP era).

Usage:
    python scripts/benchmark_datasheet.py                 # all backends
    python scripts/benchmark_datasheet.py fastsam         # subset
    python scripts/benchmark_datasheet.py fastsam yoloe dual grounded_sam2

Runs are SEQUENTIAL by necessity: concurrent GPU jobs would contend and
corrupt per-frame latency. Each backend's raw JSON is written immediately
so a mid-sweep failure never loses completed runs.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Reuse the proven single-backend runner + helpers from the committed harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_backends as bb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
CONFIG_PATH = bb.CONFIG_PATH

# Order: most important (edge default) first so the key number lands early,
# slowest/OOM-risk (sam2 auto-mask) last.
DEFAULT_BACKENDS = [
    {"name": "fastsam",       "label": "FastSAM-only (class-agnostic)"},
    {"name": "yoloe",         "label": "YOLOE-only (open-vocab pf)"},
    {"name": "dual",          "label": "FastSAM + YOLOE (dual)"},
    {"name": "grounded_sam2", "label": "Grounding DINO + SAM2"},
    {"name": "sam2",          "label": "SAM2 auto-mask"},
]


def _f(d, *keys, default=0.0):
    """Nested float fetch."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _ms(d, stage, stat="mean"):
    return _f(d, stage, stat) * 1000.0


def compile_datasheet(results: list[dict], gpu: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stages = [
        ("t_segmentation", "Segmentation"),
        ("t_heuristics",   "Heuristics"),
        ("t_scoring",      "Scoring"),
        ("t_clip",         "Embed (SigLIP)"),
        ("t_association",  "Association"),
    ]

    # End-to-end latency table
    lat_rows = []
    for r in results:
        if "error" in r:
            lat_rows.append(f"| {r['label']} | ERROR: {r['error']} | | | |")
            continue
        la = r["latency"]
        lat_rows.append(
            f"| {r['label']} | {_ms(la,'t_total','mean'):.1f} | "
            f"{_ms(la,'t_total','p50'):.1f} | {_ms(la,'t_total','p95'):.1f} | "
            f"{_ms(la,'t_total','max'):.1f} |"
        )

    # Per-stage mean breakdown table
    stage_header = "| Backend | " + " | ".join(s[1] for s in stages) + " | **Total** |"
    stage_sep = "|" + "---|" * (len(stages) + 2)
    stage_rows = []
    for r in results:
        if "error" in r:
            continue
        la = r["latency"]
        cells = " | ".join(f"{_ms(la, s[0], 'mean'):.1f}" for s in stages)
        stage_rows.append(
            f"| {r['label']} | {cells} | **{_ms(la,'t_total','mean'):.1f}** |"
        )

    # Object / throughput table
    obj_rows = []
    for r in results:
        if "error" in r:
            continue
        la = r["latency"]
        wm = r["working_memory"]
        seg = r["segmentation"]
        total_obj = wm.get("objects", 0)
        confirmed = wm.get("confirmed", 0)
        hz = la.get("processing_hz", 0)
        masks = seg.get("mean_total", "N/A")
        obj_rows.append(
            f"| {r['label']} | {la.get('frame_count','N/A')} | "
            f"{hz if isinstance(hz,str) else f'{hz:.2f}'} | {masks} | "
            f"{total_obj} | {confirmed} |"
        )

    license_map = {
        "FastSAM-only (class-agnostic)": "AGPL-3.0 (Ultralytics)",
        "YOLOE-only (open-vocab pf)": "AGPL-3.0 (Ultralytics)",
        "FastSAM + YOLOE (dual)": "AGPL-3.0 (Ultralytics)",
        "Grounding DINO + SAM2": "Apache-2.0",
        "SAM2 auto-mask": "Apache-2.0",
    }
    lic_rows = [f"| {r['label']} | {license_map.get(r['label'],'?')} |"
                for r in results if "error" not in r]

    return f"""# RTSM Backend Datasheet — RTX 5090 (desktop reference)

**Generated:** {now}
**GPU:** {gpu}
**Recording:** `recordings/session1` (240 msgs, 75.8 s, iPhone ARKit RGB-D + pose)
**Embedder:** SigLIP ViT-B-16 (webli), 768-dim
**Mask resolution:** 640×640 (`retina_masks: false`)
**Replay:** real-time (original recording cadence)

> **Latency is per *processed* frame.** In deployment RTSM gates the camera
> stream internally (~20 Hz in → ~5 Hz processed), so memory state refreshes
> at ~4-5 Hz while the heavy perception cost stays off the agent's hot loop.
> Queries against current memory are sub-50 ms warm regardless of backend.

---

## 1. End-to-end pipeline latency (ms/frame)

| Backend | Mean | P50 | P95 | Max |
|---|---|---|---|---|
{chr(10).join(lat_rows)}

## 2. Per-stage mean breakdown (ms) — where the cost concentrates

{stage_header}
{stage_sep}
{chr(10).join(stage_rows)}

## 3. Throughput & object discovery (same 75.8 s session)

| Backend | Frames | Proc Hz | Masks/frame | Objects | Confirmed |
|---|---|---|---|---|---|
{chr(10).join(obj_rows)}

## 4. License

| Backend | License |
|---|---|
{chr(10).join(lic_rows)}

---

## Edge-deployment read

- **`fastsam`** is the proposed edge default: lightest segmentation stage,
  class-agnostic masks, SigLIP supplies semantics. Trade-off vs `dual`:
  fewer dual-confirmed masks, but the latency headroom matters on Jetson.
- **`dual`** is the desktop default (best object discovery) but carries two
  model forward passes — the wrong trade on Orin-class compute.
- **`grounded_sam2`** is the Apache-2.0 path for users avoiding AGPL.
- Jetson Orin numbers to follow (entry-tier floor on Orin Nano Super; tiers
  matching reflex's auto-probe to be filled as hardware is available).

*Generated by `scripts/benchmark_datasheet.py` — per-frame latency from the
RTSM analytics API, same deterministic replay input across all backends.*
"""


def main():
    requested = sys.argv[1:]
    if requested:
        backends = [b for b in DEFAULT_BACKENDS if b["name"] in requested]
        # allow names not in default list too
        known = {b["name"] for b in DEFAULT_BACKENDS}
        for name in requested:
            if name not in known:
                backends.append({"name": name, "label": name})
    else:
        backends = DEFAULT_BACKENDS

    REPORT_DIR.mkdir(exist_ok=True)
    gpu = bb.get_gpu_info()
    print(f"GPU: {gpu}")
    print(f"Backends: {[b['name'] for b in backends]}\n")

    backup = str(CONFIG_PATH) + ".datasheet_bak"
    shutil.copy2(CONFIG_PATH, backup)

    results = []
    try:
        for b in backends:
            res = bb.run_one_backend(b)
            res["_gpu_info"] = gpu
            results.append(res)
            raw_path = REPORT_DIR / f"datasheet_raw_{b['name']}.json"
            with open(raw_path, "w") as f:
                json.dump(res, f, indent=2, default=str)
            print(f"  saved {raw_path.name}"
                  f"  ({'ERROR' if 'error' in res else 'ok'})")
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        Path(backup).unlink(missing_ok=True)

    sheet = compile_datasheet(results, gpu)
    out = REPORT_DIR / "edge_datasheet_5090.md"
    out.write_text(sheet, encoding="utf-8")
    print(f"\nDatasheet: {out}")
    # quick console echo of the key numbers
    for r in results:
        if "error" in r:
            print(f"  {r['label']:38s} ERROR {r['error']}")
        else:
            la = r["latency"]
            print(f"  {r['label']:38s} mean={_ms(la,'t_total','mean'):6.1f}ms "
                  f"p95={_ms(la,'t_total','p95'):6.1f}ms "
                  f"obj={r['working_memory'].get('objects','?')}")


if __name__ == "__main__":
    main()
