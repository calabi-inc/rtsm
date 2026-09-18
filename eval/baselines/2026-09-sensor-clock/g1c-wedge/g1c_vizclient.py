#!/usr/bin/env python3
"""G1-C headless dashboard client: attaches to the visualization websocket exactly where the browser
does (the single-port ``/ws`` on the API port; the broadcaster is shared with the viz server's own
port) and reads + discards every message, so the receiver JPEG-encodes every admitted frame for the
PiP feed, keyframes are TSDF-integrated, and objects/analytics are pushed — the CPU work E1 had on
the same interpreter while the wedge formed.

Writes one JSON line per second to <out-dir>/vizclient.jsonl: messages, bytes, CAMF frames, and
whether the connection is still up. Exits on --duration or when the socket closes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import websockets

MAGIC_CAMF = b"CAMF"


async def run(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    jl = open(os.path.join(args.out_dir, "vizclient.jsonl"), "w", encoding="utf-8")
    t_start = time.perf_counter()
    sec = {"n": 0, "bytes": 0, "camf": 0, "json": 0}
    cur = 0
    closed_reason = None
    total = {"n": 0, "bytes": 0, "camf": 0, "json": 0}

    def flush_sec(s):
        jl.write(json.dumps({"t": s, "wall": round(time.time(), 3), **sec}) + "\n")
        jl.flush()
        for k in sec:
            sec[k] = 0

    try:
        async with websockets.connect(args.url, max_size=None, ping_interval=20, ping_timeout=20,
                                      open_timeout=10, close_timeout=2) as ws:
            jl.write(json.dumps({"connected": True, "wall": round(time.time(), 3), "url": args.url}) + "\n")
            jl.flush()
            while True:
                now = time.perf_counter() - t_start
                if now >= args.duration:
                    break
                s = int(now)
                while cur < s:
                    flush_sec(cur)
                    cur += 1
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                sec["n"] += 1
                total["n"] += 1
                if isinstance(msg, (bytes, bytearray)):
                    sec["bytes"] += len(msg)
                    total["bytes"] += len(msg)
                    if msg[:4] == MAGIC_CAMF:
                        sec["camf"] += 1
                        total["camf"] += 1
                else:
                    sec["bytes"] += len(msg)
                    total["bytes"] += len(msg)
                    sec["json"] += 1
                    total["json"] += 1
    except Exception as e:
        closed_reason = f"{type(e).__name__}: {e}"
    flush_sec(cur)
    jl.write(json.dumps({"closed": True, "reason": closed_reason, "elapsed_s": round(time.perf_counter() - t_start, 3),
                         "total": total}) + "\n")
    jl.close()
    print(json.dumps({"closed_reason": closed_reason, "total": total}))
    return 0 if closed_reason is None else 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws")
    p.add_argument("--duration", type=float, default=400.0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
