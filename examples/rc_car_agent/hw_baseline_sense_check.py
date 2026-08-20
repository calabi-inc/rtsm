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

    # Debounced two-state machines with an explicit UNKNOWN third value:
    # a failed fetch is "don't know", never "gone" — one RTSM hiccup used
    # to flip BOTH printers in the same tick, which read as the wall and
    # detection triggers being tangled together (they share the server,
    # not any logic). State changes need DEBOUNCE consecutive definite
    # readings.
    DEBOUNCE = 2
    wall_state = None            # committed state: True/False/None(=never)
    wall_pend, wall_pend_n = None, 0
    det_state = None
    det_pend, det_pend_n = None, 0
    first_detect_stamp = None
    fetch_fails = 0
    t_start = time.monotonic()

    def commit(pend_val, pend_n, reading, state):
        """Debounce helper: returns (new_pend, new_pend_n, fire) where
        fire=True when `reading` has been seen DEBOUNCE times in a row and
        differs from the committed state. reading=None never fires."""
        if reading is None or reading == state:
            return None, 0, False
        if reading == pend_val:
            pend_n += 1
        else:
            pend_val, pend_n = reading, 1
        return pend_val, pend_n, pend_n >= DEBOUNCE

    while True:
        stamp = time.monotonic() - t_start

        # ── one /stats fetch: pose + clearance ───────────────────────────
        try:
            pose, clearance = rtsm.get_pose_and_clearance()
            stats_ok = True
        except Exception:  # noqa: BLE001
            pose, clearance, stats_ok = None, None, False

        c_m = None
        if clearance and time.time() - clearance.get("timestamp", 0) <= 2.0:
            c_m = float(clearance.get("clearance_m", 0.0))

        # ── freshness-gated query (the baseline's acquisition rule) ─────
        hit, query_ok = None, True
        try:
            res = rtsm.semantic_query(args.query,
                                      top_k=cfg.baseline.gate_fetch_k)
            fresh = fresh_hits(res.results, time.time(), gate_s,
                               cfg.baseline.clock_skew_tol_s)
            if fresh:
                hit = fresh[0]
        except Exception:  # noqa: BLE001
            query_ok = False

        # ── fetch-health accounting (visible, never a state flip) ───────
        if stats_ok and query_ok:
            fetch_fails = 0
        else:
            fetch_fails += 1
            if fetch_fails == 4:
                print(f"\n[{stamp:6.1f}s] WARNING: RTSM not answering "
                      f"(4 consecutive fetch failures) — states frozen "
                      f"until it recovers")

        # ── wall machine (reading is None when clearance unknown) ───────
        wall_reading = None if c_m is None else (c_m < wall_m)
        wall_pend, wall_pend_n, fire = commit(wall_pend, wall_pend_n,
                                              wall_reading, wall_state)
        if fire:
            wall_state = wall_reading
            if wall_state:
                print(f"\n[{stamp:6.1f}s] WALL IS CLOSE  ({c_m:.2f} m ahead)")
            else:
                print(f"\n[{stamp:6.1f}s] wall clear     ({c_m:.2f} m ahead)")

        # ── detection machine (reading None when the QUERY failed;
        #    a successful query with no fresh hit is a definite False) ───
        det_reading = None if not query_ok else (hit is not None)
        det_pend, det_pend_n, fire = commit(det_pend, det_pend_n,
                                            det_reading, det_state)
        if fire:
            det_prev = det_state
            det_state = det_reading
            if det_state:
                extra = ""
                if first_detect_stamp is None:
                    first_detect_stamp = stamp
                    extra = f"  (FIRST detection {stamp:.1f}s after start)"
                print(f"\n[{stamp:6.1f}s] {args.query.upper()} DETECTED{extra}")
                if hit is not None:
                    age = time.time() - hit.last_seen_wall_utc
                    print(f"           id={hit.id}  score={hit.score:.4f}  "
                          f"confirmed={hit.confirmed}  seen {age:.1f}s ago  "
                          f"xyz=[{hit.xyz_world[0]:.2f}, "
                          f"{hit.xyz_world[2]:.2f}]")
            elif det_prev is True:       # never print "lost" before a find
                print(f"\n[{stamp:6.1f}s] {args.query} lost "
                      f"(no fresh hit in {gate_s}s window)")

        # ── status line ──────────────────────────────────────────────────
        pose_age = ("n/a " if pose is None
                    else f"{time.time() - pose.timestamp:4.1f}s")
        c_txt = " ?  " if c_m is None else f"{c_m:4.2f}m"
        det_txt = {True: "YES", False: "no ", None: " ? "}[det_state]
        wall_txt = {True: "YES", False: "no ", None: " ? "}[wall_state]
        health = "" if fetch_fails == 0 else f"  fetch_fails={fetch_fails}"
        print(f"\rpose_age={pose_age}  clearance={c_txt}  "
              f"wall_close={wall_txt}  detected={det_txt}{health}   ",
              end="", flush=True)
        time.sleep(0.35)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ndone.")
        sys.exit(0)
