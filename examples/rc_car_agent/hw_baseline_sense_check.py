"""
Hand-held sense check for the two baseline-search inputs (2026-08-17).

Run with RTSM up and the phone streaming, then walk around holding the
phone (or the whole car) and watch the prints:

    .venv/Scripts/python.exe hw_baseline_sense_check.py [--query "tissue box"]

  * WALL IS CLOSE  — forward depth clearance dropped below the walk
                     threshold (the exact signal the searcher's relocate
                     walk checks). Point at a wall and approach it.
  * TISSUE BOX DETECTED — the freshness-gated semantic query returned a
                     fresh hit: the SAME gate the baseline's acquisition
                     poll applies. Point at the object and hold it in
                     view; on a fresh map the seconds until this prints
                     ARE the proto->searchable maturation latency.

Status line updates in place; state TRANSITIONS print on their own lines
with timestamps. Ctrl-C to exit. Read-only — no motion commands.
"""

from __future__ import annotations

import argparse
import sys
import time

from baseline_search import fresh_hits
from config import load_config
from rtsm_client import RtsmClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="tissue box")
    args = ap.parse_args()

    cfg = load_config()
    rtsm = RtsmClient(cfg.rtsm.url, timeout_s=2.0)
    gate_s = cfg.baseline.freshness_gate_s
    wall_m = cfg.baseline.min_walk_clearance_m

    print(__doc__)
    print(f"query={args.query!r}  freshness_gate={gate_s}s  "
          f"wall_threshold={wall_m}m\n")

    wall_close = None          # tri-state so the first reading prints
    detected = None
    first_detect_mono = None
    t_start = time.monotonic()

    while True:
        try:
            pose, clearance = rtsm.get_pose_and_clearance()
        except Exception:  # noqa: BLE001
            pose, clearance = None, None

        c_m = None
        if clearance and time.time() - clearance.get("timestamp", 0) <= 2.0:
            c_m = float(clearance.get("clearance_m", 0.0))

        # ── wall rule (same threshold the relocate walk uses) ────────────
        now_wall_close = (c_m is not None and c_m < wall_m)
        if now_wall_close != wall_close:
            stamp = time.monotonic() - t_start
            if now_wall_close:
                print(f"\n[{stamp:6.1f}s] WALL IS CLOSE  ({c_m:.2f} m ahead)")
            elif wall_close is not None:
                print(f"\n[{stamp:6.1f}s] wall clear     "
                      f"({c_m if c_m is not None else float('nan'):.2f} m ahead)")
            wall_close = now_wall_close

        # ── freshness-gated query (the baseline's acquisition rule) ─────
        hit = None
        try:
            res = rtsm.semantic_query(args.query,
                                      top_k=cfg.baseline.gate_fetch_k)
            fresh = fresh_hits(res.results, time.time(), gate_s,
                               cfg.baseline.clock_skew_tol_s)
            if fresh:
                hit = fresh[0]
        except Exception:  # noqa: BLE001
            pass

        now_detected = hit is not None
        if now_detected != detected:
            stamp = time.monotonic() - t_start
            if now_detected:
                if first_detect_mono is None:
                    first_detect_mono = time.monotonic()
                    print(f"\n[{stamp:6.1f}s] {args.query.upper()} DETECTED  "
                          f"(FIRST detection {stamp:.1f}s after start)")
                else:
                    print(f"\n[{stamp:6.1f}s] {args.query.upper()} DETECTED")
                age = time.time() - hit.last_seen_wall_utc
                print(f"           id={hit.id}  score={hit.score:.4f}  "
                      f"confirmed={hit.confirmed}  seen {age:.1f}s ago  "
                      f"xyz=[{hit.xyz_world[0]:.2f}, {hit.xyz_world[2]:.2f}]")
            elif detected is not None:
                print(f"\n[{stamp:6.1f}s] {args.query} lost "
                      f"(no fresh hit in {gate_s}s window)")
            detected = now_detected

        # ── status line ──────────────────────────────────────────────────
        pose_age = ("n/a" if pose is None
                    else f"{time.time() - pose.timestamp:4.1f}s")
        c_txt = "none" if c_m is None else f"{c_m:4.2f}m"
        print(f"\rpose_age={pose_age}  clearance={c_txt}  "
              f"wall_close={str(bool(wall_close)):5}  "
              f"detected={str(bool(detected)):5}   ",
              end="", flush=True)
        time.sleep(0.35)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ndone.")
        sys.exit(0)
