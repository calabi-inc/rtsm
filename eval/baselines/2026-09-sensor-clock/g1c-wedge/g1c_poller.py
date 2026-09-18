#!/usr/bin/env python3
"""G1-C poller: a separate process standing in for the RC-car agent's nav loop and planner.

  /stats            at 10 Hz, timeout 0.6 s   (examples/rc_car_agent/config.yaml: nav.poll_hz 10, http_timeout_s 0.6)
  /search/semantic  at 0.5 Hz, timeout 3.0 s  (rtsm_client.RTSMClient default timeout_s 3.0 — the planner's search)
  /healthz          at 1 Hz, timeout 2.0 s    (watchdog frame_flow state + ingest lane counters)
  RSS of --pid      at 1 Hz via psutil

Like the agent's rtsm_client it uses bare requests.get per call with an explicit ``Connection: close``
header (a new TCP connection each time — no Session keep-alive) against the 127.0.0.1 literal. Every call, including a timeout or error, is one JSON line in <out-dir>/poller.jsonl
(flushed per line, so a kill loses nothing). Stops when --stop-file appears or after --duration.
Timing: time.perf_counter() throughout.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import time

import psutil
import requests

QUERIES = ["red cup", "plush teddy bear", "tissue box", "backpack", "water bottle", "book"]
HDR = {"Connection": "close"}


class Writer:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, rec: dict):
        line = json.dumps(rec) + "\n"
        with self.lock:
            self.f.write(line)
            self.f.flush()

    def close(self):
        with self.lock:
            self.f.close()


def loop(kind, hz, fn, stop: threading.Event, writer: Writer, t0: float):
    period = 1.0 / hz
    next_t = time.perf_counter()
    while not stop.is_set():
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(next_t - now, 0.05))
            continue
        sched = next_t
        next_t += period
        if next_t < now - period:      # fell behind (e.g. a 0.6 s timeout at 10 Hz): re-anchor, do not burst
            next_t = now + period
        t_call = time.perf_counter()
        rec = {"kind": kind, "t": round(t_call - t0, 4), "wall": round(time.time(), 4),
               "sched_late": round(t_call - sched, 4)}
        try:
            rec.update(fn())
            rec["ok"] = rec.get("ok", True)
        except requests.Timeout as e:
            rec.update({"ok": False, "error": f"timeout: {e}"})
        except Exception as e:
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        rec["latency"] = round(time.perf_counter() - t_call, 5)
        writer.write(rec)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8002")
    p.add_argument("--pid", type=int, required=True, help="rtsm process id (RSS sampling)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stop-file", required=True)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--stats-hz", type=float, default=10.0)
    p.add_argument("--stats-timeout", type=float, default=0.6)
    p.add_argument("--search-hz", type=float, default=0.5)
    p.add_argument("--search-timeout", type=float, default=3.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    writer = Writer(os.path.join(args.out_dir, "poller.jsonl"))
    proc = psutil.Process(args.pid)
    stop = threading.Event()
    t0 = time.perf_counter()
    qcycle = itertools.cycle(QUERIES)
    writer.write({"kind": "meta", "t": 0.0, "wall": round(time.time(), 4), "pid": args.pid, "api": args.api,
                  "stats_hz": args.stats_hz, "stats_timeout": args.stats_timeout,
                  "search_hz": args.search_hz, "search_timeout": args.search_timeout})

    def get_stats():
        r = requests.get(f"{args.api}/stats", timeout=args.stats_timeout, headers=HDR)
        r.raise_for_status()
        d = r.json()
        rp = d.get("robot_pose") or {}
        lanes = d.get("ingest_lanes") or {}
        return {
            "status": r.status_code,
            "objects": d.get("objects"), "confirmed": d.get("confirmed"),
            "ingest_q": d.get("ingest_q"),
            "pose": None if not rp else {k: rp.get(k) for k in (
                "timestamp", "sensor_ts_ns", "frame_epoch", "age_s", "stale", "writes_accepted",
                "sensor_ts_regressions", "rejected_writes", "pose_clock")},
            "lanes": {k: lanes.get(k) for k in (
                "policy", "depth", "lane_full", "max_depth_seen", "age_dropped", "nonkf_superseded",
                "kf_dropped", "kf_lane_full", "admitted_kf", "admitted_nonkf", "blocked_s", "closed")},
        }

    def get_search():
        q = next(qcycle)
        r = requests.get(f"{args.api}/search/semantic", params={"query": q, "top_k": 5}, timeout=args.search_timeout, headers=HDR)
        body = None
        try:
            body = r.json()
        except Exception:
            pass
        hits = None
        if isinstance(body, dict):
            res = body.get("results")
            hits = len(res) if isinstance(res, list) else None
        return {"status": r.status_code, "ok": r.status_code == 200, "query": q, "hits": hits,
                "detail": (body or {}).get("detail") if isinstance(body, dict) and r.status_code != 200 else None}

    def get_health():
        r = requests.get(f"{args.api}/healthz", timeout=2.0, headers=HDR)
        d = r.json()
        ff = d.get("frame_flow") or {}
        ing = d.get("ingest") or {}
        try:
            rss = proc.memory_info().rss
        except Exception:
            rss = None
        return {"status": r.status_code, "health": d.get("status"), "reasons": d.get("reasons"),
                "ff_state": ff.get("state"), "ff_reasons": ff.get("reasons"), "ff_backlog": ff.get("backlog"),
                "ingest": {k: ing.get(k) for k in ("policy", "depth", "lane_full", "max_depth_seen", "age_dropped",
                                                    "nonkf_superseded", "kf_dropped", "admitted_kf", "admitted_nonkf")},
                "semantic_index": d.get("semantic_index"), "rss": rss}

    threads = [
        threading.Thread(target=loop, args=("stats", args.stats_hz, get_stats, stop, writer, t0), daemon=True),
        threading.Thread(target=loop, args=("search", args.search_hz, get_search, stop, writer, t0), daemon=True),
        threading.Thread(target=loop, args=("health", 1.0, get_health, stop, writer, t0), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while not stop.is_set():
            if os.path.exists(args.stop_file) or (time.perf_counter() - t0) >= args.duration:
                stop.set()
                break
            time.sleep(0.2)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        writer.write({"kind": "end", "t": round(time.perf_counter() - t0, 4), "wall": round(time.time(), 4)})
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
