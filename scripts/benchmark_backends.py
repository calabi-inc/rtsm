#!/usr/bin/env python3
"""
Backend Comparison Benchmark — RTSM
====================================
Runs the same replay session through two segmentation backends and
generates a comprehensive comparison report suitable for publication.

Usage:
    python scripts/benchmark_backends.py

Backends compared:
    A) dual        — FastSAM + YOLOE (AGPL, IoU-merge)
    B) grounded_sam2 — Grounding DINO + SAM2 (Apache 2.0, prompted)

The script:
    1. Launches RTSM with backend A, replays session1, collects per-frame stats
    2. Repeats for backend B
    3. Generates a Markdown report with latency, throughput, object, and
       segmentation breakdowns — all from the same deterministic input.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "recordings" / "session1"
CONFIG_PATH = ROOT / "rtsm" / "cfg" / "rtsm.yaml"
REPORT_DIR = ROOT / "reports"

# Extra CLI arguments appended to every `python -m rtsm --replay ...` launch,
# e.g. ["--profile", "examples/rc_car_agent/e1-demo2.profile.yaml"]. Set by
# main() / benchmark_datasheet.py from --profile; empty = packaged yaml only.
# Profiles are layered AFTER the base file, so patch_config()'s backend patch
# still wins as long as the profile does not pin segmentation.backend.
RTSM_EXTRA_ARGS: List[str] = []

API_PORT = 8002
POLL_INTERVAL = 5          # seconds between API polls
DRAIN_WAIT = 25            # extra seconds after replay finishes for pipeline to drain
MAX_WAIT = 900             # 15 min safety cap per run

BACKENDS = [
    {"name": "dual", "label": "FastSAM + YOLOE (dual)"},
    {"name": "grounded_sam2", "label": "Grounding DINO + SAM2"},
]


# ─────────────────────── API helpers ───────────────────────

def api_get(path: str) -> Optional[Dict[str, Any]]:
    """GET JSON from the RTSM REST API."""
    try:
        url = f"http://127.0.0.1:{API_PORT}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def wait_for_api(timeout: float = 180) -> bool:
    """Block until the API responds."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if api_get("/healthz") is not None:
            return True
        time.sleep(2)
    return False


# ─────────────────────── Config patching ───────────────────────

def patch_config(backend: str) -> None:
    """Patch config/rtsm.yaml in-place (backup is created)."""
    import yaml
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["segmentation"]["backend"] = backend
    # Keep visualization enabled so vis server does Tier 2 rollups,
    # but we don't connect any clients
    cfg["visualization"]["enable"] = True
    # Increase analytics buffer to capture all frames (session1 has 162)
    cfg["analytics"]["buffer_frames"] = 500
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ─────────────────────── Run one backend ───────────────────────

def run_one_backend(backend_info: Dict[str, str]) -> Dict[str, Any]:
    """Run replay with one backend, collect results, kill process."""
    name = backend_info["name"]
    label = backend_info["label"]
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Backend: {name}")
    print(f"{'='*60}\n")

    # Patch config
    patch_config(name)

    # Write process output to a log file (don't capture stdout/stderr
    # to avoid pipe deadlocks on Windows).
    log_path = REPORT_DIR / f"run_{name}.log"
    REPORT_DIR.mkdir(exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "rtsm", "--replay", str(RECORDING), *RTSM_EXTRA_ARGS],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(f"  PID: {proc.pid}")
    if RTSM_EXTRA_ARGS:
        print(f"  RTSM args: {' '.join(RTSM_EXTRA_ARGS)}")
    print(f"  Waiting for API to come up...")

    try:
        if not wait_for_api(timeout=300):
            print("  ERROR: API did not start in time")
            # Show last lines from log
            log_f.flush()
            try:
                with open(log_path, "r") as lf:
                    lines = lf.readlines()
                    for line in lines[-10:]:
                        print(f"    LOG: {line.rstrip()}")
            except Exception:
                pass
            proc.kill()
            return {"label": label, "error": "API timeout"}

        print(f"  API ready. Waiting for replay to complete...")

        # Wait for replay to finish and pipeline to drain.
        # Strategy: poll /stats/analytics and watch frame_count stabilize.
        # The recording is 458s with 162 frames (~2.8s avg gap), so natural
        # inter-frame pauses can be long. We require stability for longer
        # to avoid false positives. Additionally, check process logs for
        # the "[replay] Complete" marker.
        t0 = time.monotonic()
        last_frame_count = 0
        stable_polls = 0
        replay_done_detected = False

        while time.monotonic() - t0 < MAX_WAIT:
            time.sleep(POLL_INTERVAL)

            # Check process log for replay completion marker
            if not replay_done_detected:
                try:
                    log_f.flush()
                    with open(log_path, "r") as lf:
                        log_content = lf.read()
                        if "[replay] Complete" in log_content:
                            replay_done_detected = True
                except Exception:
                    pass

            analytics = api_get("/stats/analytics")
            if analytics:
                lat = analytics.get("latency", {}).get("aggregate", {})
                fc = lat.get("frame_count", 0)
                ph = lat.get("processing_hz", -1)
                marker = " [REPLAY DONE]" if replay_done_detected else ""
                print(f"    frames={fc}  processing_hz={ph:.2f}  "
                      f"elapsed={time.monotonic()-t0:.0f}s{marker}     ", end="\r")

                if fc == last_frame_count and fc > 5:
                    stable_polls += 1
                else:
                    stable_polls = 0
                last_frame_count = fc

                # If replay thread finished AND pipeline drained (stable 3 polls)
                if replay_done_detected and stable_polls >= 3:
                    print(f"\n  Replay complete. {fc} frames processed.")
                    break
                # Fallback: very long stability without replay marker (>60s idle)
                if stable_polls >= 12:
                    print(f"\n  Pipeline idle for 60s. {fc} frames processed.")
                    break
            else:
                if proc.poll() is not None:
                    print("\n  Process exited unexpectedly!")
                    return {"label": label, "error": "Process crashed"}
        else:
            print(f"\n  WARNING: Timeout reached after {MAX_WAIT}s")

        # Extra drain wait for association/WM to settle
        print(f"  Draining pipeline ({DRAIN_WAIT}s)...")
        time.sleep(DRAIN_WAIT)

        # ── Collect final results from all API endpoints ──
        print(f"  Collecting results...")
        analytics = api_get("/stats/analytics") or {}
        detailed = api_get("/stats/detailed") or {}
        basic_stats = api_get("/stats") or {}
        objects_all = api_get("/objects") or {}
        # /objects defaults to the first 100 objects; the multiset anchors
        # recorded in eval/baselines/ were computed over that page, so it is
        # kept as-is. objects_full is the whole map (server max 500) for the
        # post-reconcile anchors -- do not compare it against the old shas.
        objects_full = api_get("/objects?limit=500") or {}

        result = {
            "label": label,
            "backend": name,
            "timestamp": datetime.now().isoformat(),
            "latency": analytics.get("latency", {}).get("aggregate", {}),
            "latency_hourly": analytics.get("latency", {}).get("hourly", []),
            "segmentation": analytics.get("segmentation", {}).get("aggregate", {}),
            "segmentation_hourly": analytics.get("segmentation", {}).get("hourly", []),
            "working_memory": basic_stats,
            "detailed": detailed,
            "objects": objects_all,
            "objects_full": objects_full,
            "rtsm_extra_args": list(RTSM_EXTRA_ARGS),
        }

        print(f"  Done. Shutting down...")
        return result

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_f.close()
        # Wait for port to be released
        time.sleep(5)


# ─────────────────────── Formatting helpers ───────────────────────

def _ms(val: Any, key: str = "mean") -> str:
    """Format a timing dict value (in seconds) to ms string."""
    if isinstance(val, dict):
        v = val.get(key, 0)
    else:
        v = val or 0
    try:
        return f"{float(v)*1000:.1f}"
    except (TypeError, ValueError):
        return "N/A"


def _delta(a: dict, b: dict, key: str) -> str:
    """Compute delta string (ms) between two timing dicts."""
    va = a.get(key, 0) if isinstance(a, dict) else 0
    vb = b.get(key, 0) if isinstance(b, dict) else 0
    try:
        va, vb = float(va), float(vb)
    except (TypeError, ValueError):
        return "N/A"
    if va == 0 and vb == 0:
        return "N/A"
    diff = (vb - va) * 1000
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}"


def _pct(val: Any) -> str:
    if val is None or val == 0:
        return "N/A"
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _v(d: dict, *keys: str) -> str:
    """Extract value by trying multiple keys, including nested dicts."""
    for k in keys:
        if k in d:
            return str(d[k])
    for v in d.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return str(v[k])
    return "N/A"


def _speedup(a_sec: float, b_sec: float) -> str:
    """Compute relative speedup: how much faster A is than B (or vice versa)."""
    if a_sec == 0 or b_sec == 0:
        return "N/A"
    ratio = b_sec / a_sec
    if ratio > 1:
        return f"{ratio:.2f}x slower"
    else:
        return f"{1/ratio:.2f}x faster"


# ─────────────────────── GPU info ───────────────────────

def get_gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem = props.total_mem / (1024**3)
            return f"{name} ({mem:.1f} GB VRAM)"
    except Exception:
        pass
    # Fallback: try nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
        return out
    except Exception:
        pass
    return "N/A"


# ─────────────────────── Report generation ───────────────────────

def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive Markdown comparison report."""
    r_a, r_b = results[0], results[1]
    la, lb = r_a["latency"], r_b["latency"]
    sa, sb = r_a["segmentation"], r_b["segmentation"]
    wm_a_raw, wm_b_raw = r_a["working_memory"], r_b["working_memory"]
    # Normalize WM: compute proto = objects - confirmed
    wm_a = {
        "total": wm_a_raw.get("objects", 0),
        "confirmed": wm_a_raw.get("confirmed", 0),
        "proto": wm_a_raw.get("objects", 0) - wm_a_raw.get("confirmed", 0),
        "avg_hits": wm_a_raw.get("avg_hits", 0),
    }
    wm_b = {
        "total": wm_b_raw.get("objects", 0),
        "confirmed": wm_b_raw.get("confirmed", 0),
        "proto": wm_b_raw.get("objects", 0) - wm_b_raw.get("confirmed", 0),
        "avg_hits": wm_b_raw.get("avg_hits", 0),
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = r_a.get("_gpu_info", get_gpu_info())

    # Recording metadata
    meta = {}
    meta_path = RECORDING / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    total_frames = meta.get("total_binary_messages", "N/A")
    duration = meta.get("duration_s", 0)
    device = meta.get("device_name", "N/A")
    rec_cfg = meta.get("rtsm_config_snapshot", {})
    cam = rec_cfg.get("camera", {})

    # CLIP model from config
    import yaml
    with open(CONFIG_PATH, "r") as f:
        _cfg = yaml.safe_load(f)
    clip_cfg = _cfg.get("clip", {})
    clip_model_name = clip_cfg.get("model", "unknown")
    clip_pretrained = clip_cfg.get("pretrained", "")
    clip_model = f"{clip_model_name} ({clip_pretrained})" if clip_pretrained else clip_model_name

    # Hourly time-series summary (for variance / jitter analysis)
    h_a = r_a.get("latency_hourly", [])
    h_b = r_b.get("latency_hourly", [])
    n_buckets_a = len([b for b in h_a if b.get("frames_in_bucket", 0) > 0])
    n_buckets_b = len([b for b in h_b if b.get("frames_in_bucket", 0) > 0])

    # Total drops from hourly
    drops_a = sum(b.get("queue_drops", 0) + b.get("gate_rejections", 0) +
                  b.get("throttle_skips", 0) + b.get("tracking_drops", 0)
                  for b in h_a)
    drops_b = sum(b.get("queue_drops", 0) + b.get("gate_rejections", 0) +
                  b.get("throttle_skips", 0) + b.get("tracking_drops", 0)
                  for b in h_b)

    # Total association events from hourly
    total_matched_a = sum(b.get("assoc_matched", 0) for b in h_a)
    total_created_a = sum(b.get("assoc_created", 0) for b in h_a)
    total_matched_b = sum(b.get("assoc_matched", 0) for b in h_b)
    total_created_b = sum(b.get("assoc_created", 0) for b in h_b)

    # Peak WM from hourly
    peak_wm_a = max((b.get("wm_total", 0) for b in h_a), default=0)
    peak_wm_b = max((b.get("wm_total", 0) for b in h_b), default=0)
    peak_confirmed_a = max((b.get("wm_confirmed", 0) for b in h_a), default=0)
    peak_confirmed_b = max((b.get("wm_confirmed", 0) for b in h_b), default=0)

    # Compute segmentation stage % of total
    def _seg_pct(lat: dict) -> str:
        t_seg = lat.get("t_segmentation", {})
        t_tot = lat.get("t_total", {})
        s = t_seg.get("mean", 0) if isinstance(t_seg, dict) else 0
        t = t_tot.get("mean", 0) if isinstance(t_tot, dict) else 0
        if t == 0:
            return "N/A"
        return f"{s/t*100:.1f}%"

    # Pre-compute all delta values to avoid dict literals in f-strings
    empty = {}
    stages = ['t_total', 't_segmentation', 't_heuristics', 't_scoring', 't_clip', 't_association']
    d_mean = {s: _delta(la.get(s, empty), lb.get(s, empty), 'mean') for s in stages}
    d_p50 = {s: _delta(la.get(s, empty), lb.get(s, empty), 'p50') for s in stages}
    d_p95 = {s: _delta(la.get(s, empty), lb.get(s, empty), 'p95') for s in stages}
    d_max = {s: _delta(la.get(s, empty), lb.get(s, empty), 'max') for s in stages}

    cam_res = f"{cam.get('width', 'N/A')}x{cam.get('height', 'N/A')}"

    report = f"""# RTSM Backend Comparison Report

**Generated:** {now}
**System:** {platform.node()} / {platform.system()} {platform.release()}
**GPU:** {gpu}
**Python:** {platform.python_version()}

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Recording | `recordings/session1` |
| Device | {device} |
| Total binary frames | {total_frames} |
| Recording duration | {duration:.1f} s |
| RGB resolution | {cam_res} |
| Replay rate | Real-time (original recording rate) |
| CLIP model | {clip_model} |
| Mask resolution | 640x640 (`retina_masks: false`) |
| Top-K pre-CLIP | 15 |
| Association `cos_min` | 0.90 |
| Promote hits | 2 |
| Proto TTL | 10.0 s |
| Analytics buffer | 500 frames (captures full session) |

---

## 1. Backends Under Test

| Property | A: {r_a['label']} | B: {r_b['label']} |
|---|---|---|
| Config key | `dual` | `grounded_sam2` |
| License | AGPL-3.0 (Ultralytics) | Apache-2.0 |
| Detection model | FastSAM-x + YOLOE-v8s-seg-pf | Grounding DINO Tiny |
| Segmentation model | FastSAM-x + YOLOE-v8s-seg-pf | SAM2.1 Hiera Small |
| Vocabulary | YOLOE prompt-free (1200+ LVIS) | 30-class indoor text prompt |
| Merge strategy | IoU cross-validation (threshold 0.40) | Box-prompted mask generation |

---

## 2. Latency Analysis

### 2.1 End-to-End Pipeline Latency (ms)

| Metric | A: dual | B: grounded_sam2 | Delta (B-A) |
|---|---|---|---|
| **Mean** | {_ms(la.get('t_total'), 'mean')} | {_ms(lb.get('t_total'), 'mean')} | {d_mean['t_total']} |
| **P50 (median)** | {_ms(la.get('t_total'), 'p50')} | {_ms(lb.get('t_total'), 'p50')} | {d_p50['t_total']} |
| **P95** | {_ms(la.get('t_total'), 'p95')} | {_ms(lb.get('t_total'), 'p95')} | {d_p95['t_total']} |
| **Max** | {_ms(la.get('t_total'), 'max')} | {_ms(lb.get('t_total'), 'max')} | {d_max['t_total']} |

### 2.2 Per-Stage Breakdown — Mean (ms)

| Stage | A: dual | B: grounded_sam2 | Delta (B-A) | % of A total | % of B total |
|---|---|---|---|---|---|
| Segmentation | {_ms(la.get('t_segmentation'), 'mean')} | {_ms(lb.get('t_segmentation'), 'mean')} | {d_mean['t_segmentation']} | {_seg_pct_stage(la, 't_segmentation')} | {_seg_pct_stage(lb, 't_segmentation')} |
| Heuristics | {_ms(la.get('t_heuristics'), 'mean')} | {_ms(lb.get('t_heuristics'), 'mean')} | {d_mean['t_heuristics']} | {_seg_pct_stage(la, 't_heuristics')} | {_seg_pct_stage(lb, 't_heuristics')} |
| Scoring | {_ms(la.get('t_scoring'), 'mean')} | {_ms(lb.get('t_scoring'), 'mean')} | {d_mean['t_scoring']} | {_seg_pct_stage(la, 't_scoring')} | {_seg_pct_stage(lb, 't_scoring')} |
| CLIP | {_ms(la.get('t_clip'), 'mean')} | {_ms(lb.get('t_clip'), 'mean')} | {d_mean['t_clip']} | {_seg_pct_stage(la, 't_clip')} | {_seg_pct_stage(lb, 't_clip')} |
| Association | {_ms(la.get('t_association'), 'mean')} | {_ms(lb.get('t_association'), 'mean')} | {d_mean['t_association']} | {_seg_pct_stage(la, 't_association')} | {_seg_pct_stage(lb, 't_association')} |
| **Total** | **{_ms(la.get('t_total'), 'mean')}** | **{_ms(lb.get('t_total'), 'mean')}** | **{d_mean['t_total']}** | 100% | 100% |

### 2.3 Per-Stage Breakdown — P95 (ms)

| Stage | A: dual | B: grounded_sam2 |
|---|---|---|
| Segmentation | {_ms(la.get('t_segmentation'), 'p95')} | {_ms(lb.get('t_segmentation'), 'p95')} |
| Heuristics | {_ms(la.get('t_heuristics'), 'p95')} | {_ms(lb.get('t_heuristics'), 'p95')} |
| Scoring | {_ms(la.get('t_scoring'), 'p95')} | {_ms(lb.get('t_scoring'), 'p95')} |
| CLIP | {_ms(la.get('t_clip'), 'p95')} | {_ms(lb.get('t_clip'), 'p95')} |
| Association | {_ms(la.get('t_association'), 'p95')} | {_ms(lb.get('t_association'), 'p95')} |
| **Total** | **{_ms(la.get('t_total'), 'p95')}** | **{_ms(lb.get('t_total'), 'p95')}** |

### 2.4 Segmentation as % of Total Pipeline

| Backend | Seg % of total (mean) |
|---|---|
| A: dual | {_seg_pct(la)} |
| B: grounded_sam2 | {_seg_pct(lb)} |

---

## 3. Throughput & Real-Time Performance

| Metric | A: dual | B: grounded_sam2 |
|---|---|---|
| Frames processed | {la.get('frame_count', 'N/A')} | {lb.get('frame_count', 'N/A')} |
| Window duration (s) | {la.get('window_duration_s', 'N/A')} | {lb.get('window_duration_s', 'N/A')} |
| Processing Hz | {la.get('processing_hz', 'N/A')} | {lb.get('processing_hz', 'N/A')} |
| Mean queue depth | {la.get('mean_queue_depth', 'N/A')} | {lb.get('mean_queue_depth', 'N/A')} |
| Warmup frames skipped | {la.get('warmup_skipped', 'N/A')} | {lb.get('warmup_skipped', 'N/A')} |

---

## 4. Segmentation Output Analysis

| Metric | A: dual | B: grounded_sam2 |
|---|---|---|
| Backend | {sa.get('backend', 'N/A')} | {sb.get('backend', 'N/A')} |
| Mean masks/frame (post-merge) | {sa.get('mean_total', 'N/A')} | {sb.get('mean_total', 'N/A')} |
| Mean raw FastSAM masks/frame | {sa.get('mean_fastsam_raw', 'N/A')} | {sb.get('mean_fastsam_raw', 'N/A')} |
| Mean raw YOLOE masks/frame | {sa.get('mean_yoloe_raw', 'N/A')} | {sb.get('mean_yoloe_raw', 'N/A')} |
| Staged survival rate | {_pct(sa.get('staged_survival_rate'))} | {_pct(sb.get('staged_survival_rate'))} |

### 4.1 Confirmation Source Breakdown (dual backend)

| Source | Rate |
|---|---|
| Dual-confirmed (FastSAM + YOLOE IoU match) | {_pct(sa.get('dual_rate'))} |
| FastSAM-only | {_pct(sa.get('fastsam_only_rate'))} |
| YOLOE-only | {_pct(sa.get('yoloe_only_rate'))} |

### 4.2 Selected Masks by Source (dual backend, top-K)

| Source | Selection rate |
|---|---|
| Dual-confirmed | {_pct(_nested_float(sa, 'selected_rate_by_source', 'dual'))} |
| FastSAM-only | {_pct(_nested_float(sa, 'selected_rate_by_source', 'fastsam_only'))} |
| YOLOE-only | {_pct(_nested_float(sa, 'selected_rate_by_source', 'yoloe_only'))} |

---

## 5. Mask Pipeline Funnel

| Stage | A: dual | B: grounded_sam2 |
|---|---|---|
| Raw masks/frame (model output) | {sa.get('mean_total', 'N/A')} | {sb.get('mean_total', 'N/A')} |
| Into heuristics | {la.get('mean_masks_in', 'N/A')} | {lb.get('mean_masks_in', 'N/A')} |
| Staged survival | {_pct(sa.get('staged_survival_rate'))} | {_pct(sb.get('staged_survival_rate'))} |
| Top-K candidates (CLIP input) | {la.get('mean_candidates', 'N/A')} | {lb.get('mean_candidates', 'N/A')} |
| Overall mask survival (cand/raw) | {_pct(la.get('mask_survival_rate'))} | {_pct(lb.get('mask_survival_rate'))} |

---

## 6. Object Discovery & Working Memory

### 6.1 Final Working Memory State

| Metric | A: dual | B: grounded_sam2 |
|---|---|---|
| Total objects | {wm_a['total']} | {wm_b['total']} |
| Confirmed | {wm_a['confirmed']} | {wm_b['confirmed']} |
| Proto (unconfirmed) | {wm_a['proto']} | {wm_b['proto']} |
| Avg hits/object | {wm_a['avg_hits']:.1f} | {wm_b['avg_hits']:.1f} |

### 6.2 Object Confirmation Rate

| Metric | A: dual | B: grounded_sam2 |
|---|---|---|
| Confirmation rate | {_ratio(wm_a['confirmed'], wm_a['total'])} | {_ratio(wm_b['confirmed'], wm_b['total'])} |
| Avg observations/object | {wm_a['avg_hits']:.1f} | {wm_b['avg_hits']:.1f} |

---

## 7. Comparative Summary

| Dimension | A: dual (FastSAM+YOLOE) | B: grounded_sam2 (GDINO+SAM2) |
|---|---|---|
| **License** | AGPL-3.0 | Apache-2.0 |
| **Mean latency** | {_ms(la.get('t_total'), 'mean')} ms | {_ms(lb.get('t_total'), 'mean')} ms |
| **P95 latency** | {_ms(la.get('t_total'), 'p95')} ms | {_ms(lb.get('t_total'), 'p95')} ms |
| **Masks/frame** | {sa.get('mean_total', 'N/A')} | {sb.get('mean_total', 'N/A')} |
| **Objects discovered** | {wm_a['total']} | {wm_b['total']} |
| **Objects confirmed** | {wm_a['confirmed']} | {wm_b['confirmed']} |
| **Vocabulary** | 1200+ LVIS (prompt-free) | 30-class indoor (text-prompted) |
| **Mask consensus** | Dual-model IoU validation | Single-model (SAM2 quality) |

---

*Report generated by `scripts/benchmark_backends.py`*
*RTSM — Real-Time Spatio-Semantic Memory*
*Recording: session1 ({total_frames} frames, {duration:.1f}s)*
"""
    return report


def _seg_pct_stage(lat: dict, stage_key: str) -> str:
    """Compute stage % of total pipeline time."""
    t_stage = lat.get(stage_key, {})
    t_total = lat.get("t_total", {})
    s = t_stage.get("mean", 0) if isinstance(t_stage, dict) else 0
    t = t_total.get("mean", 0) if isinstance(t_total, dict) else 0
    if t == 0:
        return "N/A"
    return f"{s/t*100:.1f}%"


def _nested_float(d: dict, *keys: str) -> Optional[float]:
    """Walk nested keys to get a float value."""
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _ratio(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num/denom*100:.1f}%"


# ─────────────────────── Main ───────────────────────

def parse_common_args(argv: Optional[List[str]] = None):
    """--profile PATH (repeatable) -> RTSM_EXTRA_ARGS; returns (args, rest).

    Shared with benchmark_datasheet.py so both harnesses accept the same
    pass-through. Unknown arguments are returned untouched (the datasheet
    treats them as backend names).
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", action="append", default=[],
                        help="sparse config profile passed to `python -m rtsm --profile` "
                             "(repeatable; resolved relative to the repo root)")
    args, rest = parser.parse_known_args(argv)
    RTSM_EXTRA_ARGS.clear()
    for p in args.profile:
        RTSM_EXTRA_ARGS.extend(["--profile", p])
    return args, rest


def main():
    parse_common_args()
    print("=" * 60)
    print("  RTSM Backend Comparison Benchmark")
    print("=" * 60)
    print(f"  Recording: {RECORDING}")
    print(f"  Backends:  {', '.join(b['name'] for b in BACKENDS)}")
    if RTSM_EXTRA_ARGS:
        print(f"  RTSM args: {' '.join(RTSM_EXTRA_ARGS)}")
    print()

    if not RECORDING.exists():
        print(f"ERROR: Recording not found at {RECORDING}")
        sys.exit(1)

    # Backup original config
    backup = str(CONFIG_PATH) + ".bench_bak"
    shutil.copy2(CONFIG_PATH, backup)

    # Detect GPU early (before torch is unloaded by subprocess)
    gpu_info = get_gpu_info()
    print(f"  GPU: {gpu_info}\n")

    all_results = []
    try:
        for backend in BACKENDS:
            result = run_one_backend(backend)
            all_results.append(result)

            # Save raw JSON per backend
            REPORT_DIR.mkdir(exist_ok=True)
            raw_path = REPORT_DIR / f"raw_{backend['name']}.json"
            with open(raw_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Raw data saved: {raw_path}")
    finally:
        # Restore original config
        shutil.copy2(backup, CONFIG_PATH)
        if os.path.exists(backup):
            os.remove(backup)

    # Generate comparison report
    if len(all_results) == 2 and all("error" not in r for r in all_results):
        # Inject GPU info into results for report
        for r in all_results:
            r["_gpu_info"] = gpu_info
        report = generate_report(all_results)
        report_path = REPORT_DIR / "backend_comparison.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n{'='*60}")
        print(f"  Report saved: {report_path}")
        print(f"{'='*60}")

        # Print summary to console
        la = all_results[0]["latency"]
        lb = all_results[1]["latency"]
        print(f"\n  Quick summary:")
        print(f"    dual mean:           {_ms(la.get('t_total'), 'mean')} ms")
        print(f"    grounded_sam2 mean:  {_ms(lb.get('t_total'), 'mean')} ms")
    else:
        print("\nWARNING: Some runs failed. Check raw JSON files for details.")
        for r in all_results:
            if r.get("error"):
                print(f"  {r['label']}: {r['error']}")


if __name__ == "__main__":
    main()
