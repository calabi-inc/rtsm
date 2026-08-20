"""
Hand-held sense check for the two baseline-search inputs (2026-08-17).

Run with RTSM up and the phone streaming, then walk around holding the
phone (or the whole car):

    .venv/Scripts/python.exe hw_baseline_sense_check.py [--query "tissue box"]

Prints, with timestamps:

    [  12.3s] tissue box 1 found
    [  18.9s] tissue box 1 lost
    [  25.0s] tissue box 2 found        <- a second registration (new id)
    [  31.2s] wall close (0.44 m)
    [  40.8s] wall clear (1.62 m)

"tissue box N" numbers distinct memory ids in order of first appearance
(same physical box re-registered under a new id = next number). found /
lost use the SAME freshness gate the baseline's acquisition poll applies;
wall close/clear uses the SAME clearance threshold the relocate walk
checks. Transitions are debounced (2 consecutive readings) and fetch
failures freeze state rather than printing false losses. Read-only.
Ctrl-C to exit.
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

    DEBOUNCE = 2
    wall_state = None                 # True/False/None(never committed)
    wall_pend, wall_pend_n = None, 0
    ordinal: dict = {}                # id -> "tissue box N" number
    live: dict = {}                   # id -> True(live)/False(lost)
    absent: dict = {}                 # id -> consecutive absent count
    fetch_fails = 0
    t_start = time.monotonic()

    while True:
        stamp = time.monotonic() - t_start

        try:
            pose, clearance = rtsm.get_pose_and_clearance()
            stats_ok = True
        except Exception:  # noqa: BLE001
            pose, clearance, stats_ok = None, None, False

        c_m = None
        if clearance and time.time() - clearance.get("timestamp", 0) <= 2.0:
            c_m = float(clearance.get("clearance_m", 0.0))

        fresh, query_ok = [], True
        try:
            res = rtsm.semantic_query(args.query,
                                      top_k=cfg.baseline.gate_fetch_k)
            fresh = fresh_hits(res.results, time.time(), gate_s,
                               cfg.baseline.clock_skew_tol_s)
        except Exception:  # noqa: BLE001
            query_ok = False

        if stats_ok and query_ok:
            fetch_fails = 0
        else:
            fetch_fails += 1
            if fetch_fails == 4:
                print(f"\n[{stamp:6.1f}s] warning: RTSM not answering — "
                      f"holding state until it recovers")

        # ── wall (debounced; unknown never flips state) ──────────────────
        reading = None if c_m is None else (c_m < wall_m)
        if reading is None or reading == wall_state:
            wall_pend, wall_pend_n = None, 0
        else:
            if reading == wall_pend:
                wall_pend_n += 1
            else:
                wall_pend, wall_pend_n = reading, 1
            if wall_pend_n >= DEBOUNCE:
                wall_state = reading
                wall_pend, wall_pend_n = None, 0
                word = "wall close" if wall_state else "wall clear"
                print(f"\n[{stamp:6.1f}s] {word} ({c_m:.2f} m)")

        # ── per-object found/lost (debounced via absent counter) ────────
        if query_ok:
            fresh_ids = {h.id for h in fresh}
            for h in fresh:
                if h.id not in ordinal:
                    ordinal[h.id] = len(ordinal) + 1
                    live[h.id] = True
                    absent[h.id] = 0
                    print(f"\n[{stamp:6.1f}s] {args.query} "
                          f"{ordinal[h.id]} found")
                elif not live[h.id]:
                    live[h.id] = True
                    absent[h.id] = 0
                    print(f"\n[{stamp:6.1f}s] {args.query} "
                          f"{ordinal[h.id]} found again")
                else:
                    absent[h.id] = 0
            for oid, is_live in live.items():
                if not is_live or oid in fresh_ids:
                    continue
                absent[oid] += 1
                if absent[oid] >= DEBOUNCE:
                    live[oid] = False
                    print(f"\n[{stamp:6.1f}s] {args.query} "
                          f"{ordinal[oid]} lost")

        # ── status line ──────────────────────────────────────────────────
        pose_age = ("n/a " if pose is None
                    else f"{time.time() - pose.timestamp:4.1f}s")
        c_txt = " ?  " if c_m is None else f"{c_m:4.2f}m"
        wall_txt = {True: "YES", False: "no ", None: " ? "}[wall_state]
        n_live = sum(1 for v in live.values() if v)
        health = "" if fetch_fails == 0 else f"  fetch_fails={fetch_fails}"
        print(f"\rpose_age={pose_age}  clearance={c_txt}  "
              f"wall_close={wall_txt}  {args.query}s_live={n_live}"
              f"{health}   ",
              end="", flush=True)
        time.sleep(0.35)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ndone.")
        sys.exit(0)
