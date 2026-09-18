#!/usr/bin/env python3
"""G1-C sender: a separate process that plays `recordings/session1/messages.bin` into a LIVE
`python -m rtsm` websocket receiver the way Calabi Lens does — hello/hello_ack handshake, then the
raw binary messages at the recorded pacing divided by SPEED — looped for DURATION seconds with a
FRESH session_id per loop (the receiver bumps its frame epoch on a new id, so the pose mailbox sees
a restart, never a regression).

Transport is UNCOMPRESSED (`compression=None`): the phone sends raw NV12; a permessage-deflate
negotiation would put zlib on both ends and turn `send()` time into compression time. The negotiated
extensions are recorded and the gate refuses a run that negotiated any.

Measures what the phone would feel: the time spent inside ``await ws.send(raw)`` (in steady state this
is the receiver's per-frame read time once uvicorn's 32-message read-ahead queue and the socket buffers
are full — transport backpressure), the achieved input rate, connection failures, close() duration per
loop, and it records each frame's header `timestamp_ns` + send wall time so the poller's observed
`(frame_epoch, sensor_ts_ns)` can be joined into an end-to-end pose lag.

Between loops the sender waits (<= --drain-wait s) until `/stats.robot_pose.writes_accepted` has caught
up with the frames sent, so the next loop's epoch can never interleave with the previous loop's tail.

Output: <out-dir>/sender.jsonl (one line per frame / loop event) + <out-dir>/sender_summary.json.
All timing uses time.perf_counter() (QueryPerformanceCounter): time.monotonic() ticks at 15.6 ms on
Python 3.12 / Windows. Cross-process alignment uses time.time().
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import struct
import sys
import time
import uuid

import websockets

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


def load_frames(recording_dir: str):
    idx_path = os.path.join(recording_dir, "index.jsonl")
    bin_path = os.path.join(recording_dir, "messages.bin")
    entries = [json.loads(l) for l in open(idx_path, encoding="utf-8") if l.strip()]
    frames = []
    with open(bin_path, "rb") as f:
        for e in entries:
            f.seek(e["offset"])
            raw = f.read(e["length"])
            if len(raw) != e["length"]:
                raise RuntimeError(f"short read at seq {e['seq']}")
            (jl,) = struct.unpack_from("<I", raw, 0)
            hdr = json.loads(raw[4:4 + jl].decode("utf-8"))
            frames.append((e["seq"], float(e["t_mono_s"]), raw, int(hdr.get("timestamp_ns") or 0),
                           str(hdr.get("tracking_state", ""))))
    return frames


async def pace_until(deadline: float) -> None:
    """Sleep until perf_counter() >= deadline: asyncio.sleep to ~2 ms before, then yield-spin."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            await asyncio.sleep(remaining - 0.002)
        else:
            await asyncio.sleep(0)


def writes_accepted(api: str):
    if not api or requests is None:
        return None
    try:
        r = requests.get(f"{api}/stats", timeout=1.0, headers={"Connection": "close"})
        return int(((r.json().get("robot_pose") or {}).get("writes_accepted")) or 0)
    except Exception:
        return None


async def run(args) -> int:
    frames = load_frames(args.recording)
    t_rec0 = frames[0][1]
    gaps = [b[1] - a[1] for a, b in zip(frames, frames[1:])]
    target_interframe = statistics.median(gaps) / args.speed
    target_hz = len(frames) / ((frames[-1][1] - frames[0][1] + statistics.median(gaps)) / args.speed)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    jl = open(os.path.join(out_dir, "sender.jsonl"), "w", encoding="utf-8")

    stalls = []
    sent_total = 0
    loops_done = 0
    loops_started = 0
    failures = []
    per_sec = {}
    close_s = []
    drain_waits = []
    extensions_seen = []
    t_stream0 = None
    wall_stream0 = None
    stop = False
    speed = float(args.speed)
    duration = float(args.duration)

    while not stop:
        session_id = str(uuid.uuid4()).upper()
        loops_started += 1
        loop_no = loops_started
        t_loop_connect = time.perf_counter()
        try:
            async with websockets.connect(args.url, compression=None, max_size=None, ping_interval=20,
                                          ping_timeout=20, close_timeout=5, open_timeout=10) as ws:
                ext_hdr = None
                try:
                    ext_hdr = ws.response.headers.get("Sec-WebSocket-Extensions")
                except Exception:
                    pass
                ext_names = []
                try:
                    ext_names = [e.name for e in ws.protocol.extensions]
                except Exception:
                    pass
                extensions_seen.append({"header": ext_hdr, "negotiated": ext_names})
                hello = {"type": "hello", "protocol_version": 1, "session_id": session_id,
                         "device_name": "g1c-sender"}
                await ws.send(json.dumps(hello))
                ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if ack.get("type") != "hello_ack" or ack.get("status") != "ok":
                    failures.append({"loop": loop_no, "phase": "handshake", "error": f"bad ack {ack!r}"})
                    await asyncio.sleep(1.0)
                    continue
                t_handshake = time.perf_counter() - t_loop_connect
                base = time.perf_counter()
                loop_sent = 0
                for seq, t_rec, raw, ts_ns, tracking in frames:
                    deadline = base + (t_rec - t_rec0) / speed
                    await pace_until(deadline)
                    t0 = time.perf_counter()
                    wall0 = time.time()
                    await ws.send(raw)
                    t1 = time.perf_counter()
                    if t_stream0 is None:
                        t_stream0 = t0
                        wall_stream0 = wall0
                    stall = t1 - t0
                    stalls.append(stall)
                    sent_total += 1
                    loop_sent += 1
                    rel = t0 - t_stream0
                    per_sec[int(rel)] = per_sec.get(int(rel), 0) + 1
                    jl.write(json.dumps({"loop": loop_no, "seq": seq, "ts": ts_ns, "t": round(rel, 6),
                                         "wall": round(wall0, 6), "stall": round(stall, 6),
                                         "late": round(t0 - deadline, 6), "kf": (loop_sent == 1 or loop_sent % 30 == 0)}) + "\n")
                    if rel >= duration:
                        stop = True
                        break
                jl.flush()
                loops_done += 1
                tc0 = time.perf_counter()
            # `async with` exit = close handshake; the server reads its parked backlog first
            close_s.append(time.perf_counter() - tc0)
            # Drain gate: the receiver must have written every pose of this loop before the next epoch begins.
            dw0 = time.perf_counter()
            waited = None
            while args.drain_wait > 0 and (time.perf_counter() - dw0) < args.drain_wait:
                wa = writes_accepted(args.api)
                if wa is None or wa >= sent_total:
                    waited = wa
                    break
                await asyncio.sleep(0.05)
            drain_waits.append(time.perf_counter() - dw0)
            jl.write(json.dumps({"loop_end": loop_no, "session_id": session_id, "handshake_s": round(t_handshake, 4),
                                 "close_s": round(close_s[-1], 4), "drain_wait_s": round(drain_waits[-1], 4),
                                 "writes_accepted_seen": waited, "sent_total": sent_total,
                                 "t": round(time.perf_counter() - (t_stream0 or base), 6), "wall": round(time.time(), 6),
                                 "complete": not stop}) + "\n")
            jl.flush()
        except Exception as e:  # connection refused/closed, ping timeout, send error
            failures.append({"loop": loop_no, "phase": "stream", "error": f"{type(e).__name__}: {e}",
                             "t": round((time.perf_counter() - t_stream0) if t_stream0 else -1.0, 3)})
            jl.write(json.dumps({"failure": failures[-1]}) + "\n")
            jl.flush()
            if t_stream0 is not None and time.perf_counter() - t_stream0 >= duration:
                break
            await asyncio.sleep(1.0)
            if len(failures) >= args.max_failures:
                break
    jl.close()

    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]

    streamed_s = (time.perf_counter() - t_stream0) if t_stream0 else 0.0
    full_secs = [k for k in per_sec if k < int(streamed_s)]
    summary = {
        "recording": os.path.abspath(args.recording),
        "url": args.url,
        "compression": None,
        "extensions": extensions_seen[:3] + (["..."] if len(extensions_seen) > 3 else []),
        "extensions_negotiated_any": any(e["negotiated"] for e in extensions_seen),
        "speed": speed,
        "duration_target_s": duration,
        "streamed_s": round(streamed_s, 3),
        "wall_stream0": wall_stream0,
        "frames_sent": sent_total,
        "frames_per_loop": len(frames),
        "loops_started": loops_started,
        "loops_completed": loops_done,
        "achieved_hz": round(sent_total / streamed_s, 3) if streamed_s > 0 else None,
        "target_hz": round(target_hz, 3),
        "recorded_median_gap_s": round(statistics.median(gaps), 4),
        "target_interframe_s": round(target_interframe, 4),
        "send_stall_s": {"p50": pct(stalls, 50), "p90": pct(stalls, 90), "p99": pct(stalls, 99),
                         "max": max(stalls) if stalls else None, "n": len(stalls)},
        "close_s": {"p50": pct(close_s, 50), "max": max(close_s) if close_s else None, "n": len(close_s)},
        "drain_wait_s": {"p50": pct(drain_waits, 50), "max": max(drain_waits) if drain_waits else None},
        "per_second_min_full": min((per_sec[k] for k in full_secs), default=None),
        "per_second_below_90pct_target": sum(1 for k in full_secs if per_sec[k] < 0.9 * target_hz),
        "full_seconds": len(full_secs),
        "failures": failures,
        "per_second_sent": {str(k): v for k, v in sorted(per_sec.items())},
    }
    with open(os.path.join(out_dir, "sender_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_second_sent",)}, indent=1))
    return 0 if not failures else 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", default="recordings/session1")
    p.add_argument("--url", default="ws://127.0.0.1:8765/stream")
    p.add_argument("--api", default="http://127.0.0.1:8002", help="for the between-loop drain gate ('' disables)")
    p.add_argument("--speed", type=float, default=1.0, help="pacing divisor over the recorded t_mono gaps")
    p.add_argument("--duration", type=float, default=150.0, help="seconds of streaming (from the first frame sent)")
    p.add_argument("--drain-wait", type=float, default=5.0, help="max seconds to wait between loops for writes_accepted to catch up")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-failures", type=int, default=5)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
