#!/usr/bin/env python3
"""G1-C wedge proof — driver (execution plan v2 line 145; design, review outcomes and revisions in the G1-C
section of eval/baselines/2026-09-sensor-clock/README.md).

Runs the WHOLE `python -m rtsm` process (packaged defaults since P1 task 7: grounded_sam2, visualization OFF,
TSDF OFF, analytics ON, watchdog ON, diagnostics OFF) with a separate sender process streaming
recordings/session1 raw bytes through the live websocket receiver (uncompressed, like the phone), a separate
poller process standing in for the agent (/stats 10 Hz, /search/semantic 0.5 Hz, /healthz 1 Hz, RSS 1 Hz), and
a headless dashboard client on /ws whenever the runner log reports a viz server (the --viz runs).
Evaluates the predicates that correspond to the E1 wedge symptoms (2026-08-10) and prints a verdict per run.

    python eval/baselines/2026-09-sensor-clock/g1c-wedge/g1c_gate.py [--runs C,R1,RB,RS,F,R1Z,R1T] [--out DIR]

Runs: C = calibration (the driver suspends the rtsm process for 3 s mid-stream; every symptom predicate must trip,
else the harness is INVALID and nothing else runs); R1 = the E1 condition (phone cadence, packaged throttle, 300 s);
RB = pipeline overload with bounded lanes (throttle 0.2 s); RS = receiver saturated (3x pacing; the achieved ingest
Hz is the measurement); F = the pre-P1 legacy queue under the RB load (must show queue growth, else the harness
cannot see a wedge); R1Z / R1T = dashboard runs (--viz; --viz + TSDF on), run by default, non-gating. Since P1 task 7 the
packaged config is headless, so R1/RB/RS/F run without a viz server; the viz/TSDF state of every run is read from
the runner's own log lines and recorded in finals.json.

Raw artifacts (poller.jsonl, sender.jsonl, vizclient.jsonl, rtsm_<run>.log, finals.json) go to --out (default:
this directory; gitignored except the scripts, gate.out and README; summary.json is force-added). Harness timing is
perf_counter; cross-process alignment uses time.time() (wall) stamped by each process.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                      # <repo>/eval/baselines/2026-09-sensor-clock/g1c-wedge -> <repo>
API = "http://127.0.0.1:8002"
STREAM_URL = "ws://127.0.0.1:8765/stream"
VIZ_WS_URL = "ws://127.0.0.1:8002/ws"
RECORDING = ROOT / "recordings" / "session1"
FAISS_DIR = ROOT / "model_store" / "faiss"
NOBROWSER = HERE / "nobrowser"
HDR = {"Connection": "close"}

# ---------------------------------------------------------------------------------------------------------------
# Run matrix. speed = pacing divisor over the recorded gaps (1.0 = the phone's own cadence, median 186 ms, 5.4 Hz).
# A speed the receiver cannot sustain backpressures the sender (send() stalls), like a phone that keeps pushing —
# it is measured, never hidden. nonkf = ingest.nonkf_min_interval_s (packaged 0.5 s).
# ---------------------------------------------------------------------------------------------------------------
RUNS = {
    "C":   dict(policy="latest", speed=1.0, duration=60.0, nonkf=None, role="calibrate", suspend_at=25.0, suspend_s=3.0,
                what="CALIBRATION: rtsm suspended 3 s at +25 s; every symptom predicate must trip"),
    "R1":  dict(policy="latest", speed=1.0, duration=300.0, nonkf=None, role="hard",
                what="the E1 condition: phone cadence (1x), packaged throttle, 300 s"),
    "RB":  dict(policy="latest", speed=1.0, duration=150.0, nonkf=0.2, role="hard",
                what="phone cadence, throttle 0.2 s (pipeline overload, bounded lanes), 150 s"),
    "RS":  dict(policy="latest", speed=3.0, duration=150.0, nonkf=None, role="hard-saturated",
                what="receiver saturated (3x pacing), packaged throttle, 150 s"),
    "F":   dict(policy="legacy", speed=1.0, duration=150.0, nonkf=0.2, role="falsify",
                what="LEGACY 512-deep queue under the RB load, 150 s (must show queue growth)"),
    # Since P1 task 7 the packaged config is headless (visualization.enable false, tsdf false): R1/RB/RS/F above run
    # exactly that. The dashboard states are non-gating attribution runs (run by default), launched with the
    # runner's own flags:
    "R1Z": dict(policy="latest", speed=1.0, duration=150.0, nonkf=None, role="attribution", extra_args=["--viz"],
                what="dashboard on (--viz), TSDF off = per-keyframe clouds, client attached, 150 s (non-gating)"),
    "R1T": dict(policy="latest", speed=1.0, duration=150.0, nonkf=None, role="attribution", extra_args=["--viz"],
                extra_sets=["visualization.tsdf.enable=true"],
                what="dashboard on + TSDF fusion on (the pre-task-7 packaged config), client attached, 150 s (non-gating)"),
    # Pre-task-7 attribution runs, kept only so --reeval of the 2026-09-18 artifacts still resolves their specs:
    "R1V": dict(policy="latest", speed=1.0, duration=150.0, nonkf=None, role="attribution", legacy=True,
                extra_sets=["visualization.tsdf.enable=false"],
                what="(2026-09-18) attribution: R1 load with TSDF integration OFF (viz client still attached), 150 s"),
    "R1N": dict(policy="latest", speed=1.0, duration=150.0, nonkf=None, role="attribution", no_viz=True, legacy=True,
                what="(2026-09-18) attribution: R1 load with NO viz server (--no-viz), 150 s"),
}
DEFAULT_ORDER = "C,R1,RB,RS,F,R1Z,R1T"

# Predicate constants (plan line 145 + task-5 addendum + design review 2026-09-18)
HTTP_TIMEOUT_S = 0.6        # the plan's bound for /stats (stricter than the agent's RTSMClient 3.0 s default)
HTTP_P99_S = 0.25
SEARCH_TIMEOUT_S = 3.0      # RTSMClient default timeout — the agent's real bound for every RTSM call
POSE_MIN_PER_10S = 30       # protocol 4b floor: >= 3 fresh poses/s over 10 s
POSE_FRESH_FRACTION = 0.85  # of min(poll rate, achieved input rate) x 10 s
POSE_MAX_GAP_S = 0.5        # inside stale_abort_s 2.5; equals robot_pose.stale_after_s
LAG_P99_S = 0.5             # end-to-end sender->observed pose lag (+<=0.1 s poll resolution)
LAG_MAX_S = 1.0
LOAD_SLIP_MAX_S = 1.0       # the sender never fell more than this behind the phone's cadence (paced runs)
LOAD_MIN_RATIO = 0.95       # achieved input Hz / target Hz over the whole run
RSS_GROWTH_MAX = 300 * 1024 * 1024
RSS_BASELINE_S = 60.0       # process footprint (TSDF volume, CUDA host allocations) settles by ~60 s at 1x; 20 s for short runs
TSDF_FRAME_BYTES = 640 * 480 * (3 + 4)     # visualization.tsdf.integration_max_width 640 x 480: uint8 RGB + float32 depth
TSDF_RING = 200                            # visualization.tsdf.frame_buffer_size
LANE_MAX_DEPTH = 4          # keyframe lane 3 + one non-KF slot (structural invariant of policy latest)
LEGACY_WEDGE_QUEUE = 20
WARMUP_S = 2.0
FF_BAD = {"backlogged", "hung", "receiver_dead", "starved"}
LOG_BAD_PATTERNS = ["frame parse error", "connection error", "callback error", "ingest queue refused frame",
                    "write rejected", "pose key went backwards"]


def log(msg: str) -> None:
    print(msg, flush=True)


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ports_free(ports=(8002, 8765), timeout=60.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if all(port_free(p) for p in ports):
            return True
        time.sleep(0.5)
    return False


def api_get(path: str, timeout: float = 5.0, **params):
    r = requests.get(f"{API}{path}", params=params or None, timeout=timeout, headers=HDR)
    r.raise_for_status()
    return r.json()


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


def r3(x):
    return None if x is None else round(float(x), 3)


def mb(x):
    return None if x is None else round(x / 2**20)


def launch_rtsm(run: str, spec: dict, out: Path):
    cmd = [sys.executable, "-X", "utf8", "-u", "-m", "rtsm", "--set", f"ingest.policy={spec['policy']}"]
    if spec.get("nonkf") is not None:
        cmd += ["--set", f"ingest.nonkf_min_interval_s={spec['nonkf']}"]
    for kv in spec.get("extra_sets", []):
        cmd += ["--set", kv]
    cmd += list(spec.get("extra_args", []))
    if spec.get("no_viz"):
        cmd += ["--no-viz"]
    log_path = out / f"rtsm_{run}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(NOBROWSER)}
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=open(log_path, "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT, env=env)
    return proc, log_path, cmd


def wait_ready(proc, log_path: Path, timeout=300.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            txt = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            txt = ""
        if "RTSM is running" in txt:
            try:
                api_get("/healthz", timeout=2.0)
                if not port_free(8765):
                    return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


def terminate(proc, name: str, grace=15.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    log(f"  [{name}] stopped (rc={proc.poll()})")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for l in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------------------------------------------
def execute(run: str, spec: dict, out: Path) -> dict:
    log(f"\n=== {run}: {spec['what']} (policy {spec['policy']}, speed {spec['speed']}x, {spec['duration']} s) ===")
    if not wait_ports_free():
        raise RuntimeError("ports 8002/8765 not free")
    rtsm = viz = poller = sender = None
    stop_file = out / f"{run}.stop"
    stop_file.unlink(missing_ok=True)
    rdir = out / run
    rdir.mkdir(parents=True, exist_ok=True)
    for f in ("sender.jsonl", "sender_summary.json", "poller.jsonl", "vizclient.jsonl", "finals.json"):
        (rdir / f).unlink(missing_ok=True)
    result = {"run": run, "spec": spec}
    calib = {}
    try:
        rtsm, log_path, cmd = launch_rtsm(run, spec, out)
        log(f"  rtsm pid {rtsm.pid}: {' '.join(cmd[4:])}; waiting for models + API...")
        if not wait_ready(rtsm, log_path):
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
            raise RuntimeError(f"rtsm not ready in time / exited rc={rtsm.poll()}\n{tail}")
        t_ready = time.time()
        rss_ready = psutil.Process(rtsm.pid).memory_info().rss
        base_stats = api_get("/stats")
        base_analytics = api_get("/stats/analytics")
        boot = log_path.read_text(encoding="utf-8", errors="replace")
        viz_on = "Visualization server initialized" in boot
        tsdf_on = "TSDF fusion enabled" in boot
        log(f"  ready: RSS {rss_ready/2**20:.0f} MB, objects {base_stats.get('objects')}, "
            f"ingest policy {(base_stats.get('ingest_lanes') or {}).get('policy')}, viz={'on' if viz_on else 'off'}, tsdf={'on' if tsdf_on else 'off'}")

        viz_attached = None
        if viz_on:
            viz = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(HERE / "g1c_vizclient.py"),
                                    "--url", VIZ_WS_URL, "--duration", str(spec["duration"] + 400),
                                    "--out-dir", str(rdir)], cwd=str(ROOT))
            # The viz server runs in integrated mode (no own port) and no API route reports the broadcaster's
            # client count -> registration is confirmed by both sides' records.
            for _ in range(40):
                vz_ok = any(r.get("connected") for r in read_jsonl(rdir / "vizclient.jsonl"))
                srv_ok = "[api/ws] Client connected" in log_path.read_text(encoding="utf-8", errors="replace")
                if vz_ok and srv_ok:
                    viz_attached = "log-confirmed"
                    break
                if viz.poll() is not None:
                    raise RuntimeError(f"viz client exited rc={viz.returncode}")
                time.sleep(0.5)
            else:
                raise RuntimeError("viz client did not register on the broadcaster")
            log(f"  viz client attached ({viz_attached})")

        poller = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(HERE / "g1c_poller.py"),
                                   "--api", API, "--pid", str(rtsm.pid), "--out-dir", str(rdir),
                                   "--stop-file", str(stop_file), "--duration", str(spec["duration"] + 600)],
                                  cwd=str(ROOT))
        time.sleep(5.0)                                   # idle baseline for HTTP latency
        sender = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(HERE / "g1c_sender.py"),
                                   "--recording", str(RECORDING), "--url", STREAM_URL, "--api", API,
                                   "--speed", str(spec["speed"]), "--duration", str(spec["duration"]),
                                   "--out-dir", str(rdir)],
                                  cwd=str(ROOT), stdout=open(rdir / "sender.stdout", "w"), stderr=subprocess.STDOUT)
        t_sender0 = time.perf_counter()
        log(f"  sender started (pid {sender.pid}); streaming {spec['duration']} s at {spec['speed']}x ...")
        suspended = False
        while sender.poll() is None:
            el = time.perf_counter() - t_sender0
            if el > spec["duration"] + 240:
                raise RuntimeError("sender did not finish")
            if rtsm.poll() is not None:
                raise RuntimeError(f"rtsm exited during streaming rc={rtsm.poll()}")
            if spec.get("suspend_at") is not None and not suspended and el >= spec["suspend_at"]:
                suspended = True
                p = psutil.Process(rtsm.pid)
                calib["suspend_wall"] = time.time()
                p.suspend()
                time.sleep(spec["suspend_s"])
                p.resume()
                calib["resume_wall"] = time.time()
                log(f"  [calibration] rtsm suspended {spec['suspend_s']} s at +{el:.1f} s")
            time.sleep(0.5)
        log(f"  sender finished rc={sender.returncode}; draining 10 s")
        time.sleep(10.0)
        viz_alive_at_end = (viz is not None and viz.poll() is None)
        finals = {}
        for path in ("/stats", "/healthz", "/stats/analytics"):
            try:
                finals[path] = api_get(path, timeout=10.0)
            except Exception as e:
                finals[path] = {"error": repr(e)}
        stop_file.touch()
        try:
            poller.wait(timeout=15)
        except Exception:
            terminate(poller, "poller")
        (rdir / "finals.json").write_text(json.dumps({
            "cmd": cmd, "t_ready": t_ready, "rss_ready": rss_ready, "base_stats": base_stats,
            "base_analytics_rollup": base_analytics.get("rollup"), "viz_attached": viz_attached,
            "viz_alive_at_end": viz_alive_at_end, "calib": calib, "viz_on": viz_on, "tsdf_on": tsdf_on,
            "finals": finals}, indent=1, default=str), encoding="utf-8")
        result.update(evaluate(run, spec, rdir, log_path, finals, rss_ready, viz_alive_at_end, calib,
                               viz_on=viz_on, tsdf_on=tsdf_on))
    finally:
        terminate(sender, "sender", 5)
        terminate(poller, "poller", 5)
        terminate(viz, "vizclient", 5)
        terminate(rtsm, "rtsm", 20)
        stop_file.unlink(missing_ok=True)
        wait_ports_free(timeout=30)
    return result


# ---------------------------------------------------------------------------------------------------------------
def evaluate(run: str, spec: dict, d: Path, log_path: Path, finals: dict, rss_ready: int, viz_alive_at_end, calib: dict,
             viz_on=None, tsdf_on=None) -> dict:
    ss = json.loads((d / "sender_summary.json").read_text(encoding="utf-8"))
    sj = read_jsonl(d / "sender.jsonl")
    pol = read_jsonl(d / "poller.jsonl")
    vz = read_jsonl(d / "vizclient.jsonl")
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    # Viz / TSDF state: from the runner's own log (task 7+); pre-task-7 artifacts fall back to the spec.
    if viz_on is None:
        viz_on = ("Visualization server initialized" in txt) or (not spec.get("no_viz") and "Visualization" in txt)
    if tsdf_on is None:
        tsdf_on = ("TSDF fusion enabled" in txt)
    w0 = float(ss["wall_stream0"]) if ss.get("wall_stream0") else None
    if w0 is None:
        return {"verdict": "ERROR", "error": "sender never sent a frame", "sender": ss}
    w1 = w0 + float(ss["streamed_s"])
    inwin = lambda r: (w0 + WARMUP_S) <= r["wall"] <= w1
    stats = [r for r in pol if r["kind"] == "stats"]
    search = [r for r in pol if r["kind"] == "search"]
    health = [r for r in pol if r["kind"] == "health"]
    S = [r for r in stats if inwin(r)]
    Q = [r for r in search if inwin(r)]
    H = [r for r in health if inwin(r)]
    frames = [r for r in sj if "seq" in r]
    P = {}
    info = {}
    saturated = spec["role"] == "hard-saturated"
    hz_achieved = float(ss.get("achieved_hz") or 0.0)

    # P1 HTTP responsiveness (/stats)
    lat = [r["latency"] for r in S]
    slow = [r for r in S if (not r.get("ok")) or r["latency"] > HTTP_TIMEOUT_S]
    p99 = pct(lat, 99)
    sched_late_p99 = pct([r.get("sched_late", 0.0) for r in S], 99)
    P["P1_stats_http"] = (len(slow) == 0 and p99 is not None and p99 < HTTP_P99_S,
                          f"n={len(S)} errors/timeouts+slow(>0.6s)={len(slow)} p50={r3(pct(lat,50))} p99={r3(p99)} max={r3(max(lat) if lat else None)} "
                          f"(idle p99={r3(pct([r['latency'] for r in stats if r['wall'] < w0], 99))}; poll schedule slip p99={r3(sched_late_p99)})")
    # P1b /search/semantic
    qlat = [r["latency"] for r in Q]
    qbad = [r for r in Q if (not r.get("ok")) or r["latency"] > SEARCH_TIMEOUT_S]
    q503 = [r for r in Q if r.get("status") == 503]
    P["P1b_search_http"] = (len(qbad) - len(q503) == 0 and len(q503) <= 1,
                            f"n={len(Q)} bad={len(qbad)} (503={len(q503)}) p50={r3(pct(qlat,50))} p99={r3(pct(qlat,99))} max={r3(max(qlat) if qlat else None)} hits_last={[r.get('hits') for r in Q[-3:]]}")
    # P2 / P3 pose freshness from the 10 Hz samples
    keyed = [(r["wall"], (r["pose"] or {}).get("frame_epoch"), (r["pose"] or {}).get("sensor_ts_ns"))
             for r in S if r.get("ok") and r.get("pose")]
    poll_hz = 10.0
    expected10 = max(POSE_MIN_PER_10S, POSE_FRESH_FRACTION * min(poll_hz, hz_achieved) * 10.0)
    windows = []
    t = w0 + WARMUP_S
    while t + 10.0 <= w1:
        windows.append(len({(e, s) for (w, e, s) in keyed if t <= w < t + 10.0}))
        t += 10.0
    P["P2_pose_rate"] = (len(windows) > 0 and min(windows) >= expected10,
                         f"distinct poses per 10 s window: min={min(windows) if windows else None} required>={r3(expected10)} (floor {POSE_MIN_PER_10S}; 0.85 x min(poll {poll_hz:.0f} Hz, input {hz_achieved:.2f} Hz) x 10) windows={windows}")
    gaps = []
    last_key = None
    last_change = None
    for (w, e, s) in keyed:
        k = (e, s)
        if k != last_key:
            if last_change is not None:
                gaps.append(w - last_change)
            last_change = w
            last_key = k
    stale_n = sum(1 for r in S if r.get("ok") and (r.get("pose") or {}).get("stale"))
    P["P3_pose_gaps"] = (bool(gaps) and max(gaps) < POSE_MAX_GAP_S and stale_n <= 0.01 * max(1, len(S)),
                         f"max gap between pose changes={r3(max(gaps) if gaps else None)} s (p99={r3(pct(gaps,99))}), stale samples={stale_n}/{len(S)}")
    # P4 single writer
    rp = ((finals.get("/stats") or {}).get("robot_pose")) or {}
    epochs = []
    for (_, e, _) in keyed:
        if e not in epochs:
            epochs.append(e)
    P["P4_pose_writer"] = (rp.get("writes_accepted") == ss["frames_sent"] and rp.get("sensor_ts_regressions") == 0 and rp.get("rejected_writes") == 0,
                           f"writes_accepted={rp.get('writes_accepted')} frames_sent={ss['frames_sent']} regressions={rp.get('sensor_ts_regressions')} rejected={rp.get('rejected_writes')} epochs_seen={len(epochs)} loops={ss['loops_started']}")
    # P5 transport
    counters = (((finals.get("/stats/analytics") or {}).get("latency") or {}).get("aggregate") or {}).get("counters") or {}
    received = counters.get("received")
    st = ss["send_stall_s"]
    interframe = ss["target_interframe_s"]
    kf_stalls = [r["stall"] for r in frames if r.get("kf")]
    nk_stalls = [r["stall"] for r in frames if not r.get("kf")]
    no_ext = not ss.get("extensions_negotiated_any")
    base_ok = no_ext and not ss["failures"] and received == ss["frames_sent"]
    ext0 = (ss.get("extensions") or [None])[0]
    cls = (f"stall p99 post-KF={r3(pct(kf_stalls,99))} non-KF={r3(pct(nk_stalls,99))}; close_s p50={r3(ss['close_s']['p50'])} "
           f"max={r3(ss['close_s']['max'])}; drain_wait max={r3(ss['drain_wait_s']['max'])}")
    # Delivered load = the sender kept the phone's schedule. `late` = send start minus the pacing deadline (resets per
    # loop, so the between-loop drain wait is reported separately, not counted). Integer per-second buckets of a
    # 5.4 Hz stream alternate 5/6 and the recording's own 0.35 s gaps make single seconds of 4 frames normal.
    slips = [r["late"] for r in frames if r.get("late") is not None]
    slip_p99 = pct(slips, 99)
    slip_max = max(slips) if slips else None
    ratio = (hz_achieved / float(ss["target_hz"])) if ss.get("target_hz") else None
    per_sec = {int(k): v for k, v in (ss.get("per_second_sent") or {}).items()}
    full = int(ss["streamed_s"])
    secs = [per_sec.get(k, 0) for k in range(full)]
    w5 = [sum(secs[i:i + 5]) for i in range(0, max(0, full - 4))]
    load_txt = (f"delivered load: achieved {hz_achieved} Hz = {r3(ratio)} x target {ss['target_hz']} Hz; schedule slip p99={r3(slip_p99)} "
                f"max={r3(slip_max)} s (<{LOAD_SLIP_MAX_S}); info: min frames per 5 s window={min(w5) if w5 else None}, "
                f"single seconds below 90%={ss.get('per_second_below_90pct_target')}/{ss.get('full_seconds')}")
    if saturated:
        P["P5_transport"] = (base_ok,
                             f"SATURATED by design: achieved {hz_achieved} Hz = the receiver's ceiling with the full process (target {ss['target_hz']}); "
                             f"stall p50={r3(st['p50'])} p99={r3(st['p99'])} max={r3(st['max'])}; received={received} sent={ss['frames_sent']} "
                             f"failures={len(ss['failures'])} extensions={ext0}; {cls}")
    else:
        delivered_ok = (st["max"] is not None and st["max"] < 1.0 and slip_max is not None and slip_max < LOAD_SLIP_MAX_S
                        and ratio is not None and ratio >= LOAD_MIN_RATIO)
        P["P5_transport"] = (base_ok and delivered_ok,
                             f"{load_txt}; max stall={r3(st['max'])} (<1 s), "
                             f"received={received} sent={ss['frames_sent']} failures={len(ss['failures'])} extensions={ext0}; {cls}")
        P["P5b_plan_stall"] = (st["p99"] is not None and st["p99"] < interframe,
                               f"plan text: send() stall p99={r3(st['p99'])} < one inter-frame interval {r3(interframe)} s (p50={r3(st['p50'])} p90={r3(st['p90'])})")
    # P6 RSS, TSDF-ring-aware
    rss = [(r["wall"], r["rss"]) for r in health if r.get("rss")]
    base20 = next(((w, v) for (w, v) in rss if w >= w0 + 20.0), None)
    after20 = [v for (w, v) in rss if w >= w0 + 20.0]
    base_at = RSS_BASELINE_S if (w1 - w0) >= RSS_BASELINE_S + 30.0 else 20.0
    base_pt = next(((w, v) for (w, v) in rss if w >= w0 + base_at), None)
    after = [v for (w, v) in rss if w >= w0 + base_at]
    kf_walls = [r["wall"] for r in frames if r.get("kf")]
    if base_pt and tsdf_on:
        kf_before = sum(1 for w in kf_walls if w < base_pt[0])
        tsdf_bytes = (min(TSDF_RING, len(kf_walls)) - min(TSDF_RING, kf_before)) * TSDF_FRAME_BYTES
    else:
        tsdf_bytes = 0
    last60 = [(w, v) for (w, v) in rss if w1 - 60.0 <= w <= w1]
    slope = None
    if len(last60) >= 2 and last60[-1][0] > last60[0][0]:
        slope = (last60[-1][1] - last60[0][1]) / (last60[-1][0] - last60[0][0]) * 60 / 2**20   # MB/min
    raw_growth = (max(after) - base_pt[1]) if (base_pt and after) else None
    adj_growth = (raw_growth - tsdf_bytes) if raw_growth is not None else None
    raw20 = (max(after20) - base20[1]) if (base20 and after20) else None
    P["P6_rss"] = (adj_growth is not None and adj_growth < RSS_GROWTH_MAX,
                   f"ready={mb(rss_ready)} MB base(+{base_at:.0f}s)={mb(base_pt[1]) if base_pt else None} MB max={mb(max(after)) if after else None} MB "
                   f"raw growth={mb(raw_growth)} MB, TSDF ring credit={mb(tsdf_bytes)} MB ({(str(len(kf_walls)) + ' KFs x 2.05 MB, ring 200') if tsdf_on else 'TSDF off'}) -> "
                   f"adjusted growth={mb(adj_growth)} MB (<300); slope_last60={slope if slope is None else round(slope, 1)} MB/min; "
                   f"from +20 s: base={mb(base20[1]) if base20 else None} MB raw growth={mb(raw20)} MB (info)")
    # P7 lanes
    ing = (finals.get("/healthz") or {}).get("ingest") or {}
    qmax = max([r.get("ingest_q") or 0 for r in S] or [0])
    if spec["policy"] == "latest":
        P["P7_lanes"] = (ing.get("age_dropped") == 0 and qmax <= LANE_MAX_DEPTH and (ing.get("max_depth_seen") or 0) <= LANE_MAX_DEPTH,
                         f"age_dropped={ing.get('age_dropped')} ingest_q max={qmax} max_depth_seen={ing.get('max_depth_seen')} "
                         f"superseded={ing.get('nonkf_superseded')} kf_dropped={ing.get('kf_dropped')} admitted kf/nonkf={ing.get('admitted_kf')}/{ing.get('admitted_nonkf')} "
                         f"(depth bounds are structural under `latest`; age_dropped is the evidence)")
    else:
        P["P7_lanes"] = (qmax >= LEGACY_WEDGE_QUEUE,
                         f"LEGACY wedge signature: ingest_q max={qmax} (>= {LEGACY_WEDGE_QUEUE} required) final depth={ing.get('depth')} lane_full={ing.get('lane_full')}")
    # P8 watchdog: samples + the log's transition lines (timestamped, between the 1 Hz polls)
    states = [r.get("ff_state") for r in H]
    bad_states = [s for s in states if s in FF_BAD]
    any_backlogged = any(r.get("ff_state") == "backlogged" for r in health)
    trans = re.findall(r"^(\S+ \S+) .*\[watchdog\] frame flow: (\S+) -> (\S+)(.*)$", txt, flags=re.M)

    def _wall(ts: str):
        try:
            return time.mktime(time.strptime(ts.split(",")[0], "%Y-%m-%d %H:%M:%S")) + int(ts.split(",")[1]) / 1000.0
        except Exception:
            return None
    trans_in = [(ts, a, b, why.strip()) for (ts, a, b, why) in trans if w0 <= (_wall(ts) or 0) <= w1 + 1.0]
    bad_trans = [tr for tr in trans_in if tr[2] in {"backlogged", "hung", "receiver_dead"}]
    P["P8_frame_flow"] = (not bad_states and not any_backlogged and not bad_trans and not any(tr[2] == "backlogged" for tr in trans),
                          f"sampled states in window={dict((s, states.count(s)) for s in set(states))} backlogged_anywhere={any_backlogged}; "
                          f"log transitions in window={[(a, b, why[:60]) for (_, a, b, why) in trans_in][:6]}")
    # P9 rollup
    roll = (finals.get("/stats/analytics") or {}).get("rollup") or {}
    P["P9_rollup"] = (roll.get("alive") is True and roll.get("stalled") is False and roll.get("late_ticks") == 0 and roll.get("stale_rollups") == 0,
                      f"{roll}")
    # P10 receiver/process log
    counts = {pat: txt.count(pat) for pat in LOG_BAD_PATTERNS}
    n_hs = txt.count("Handshake OK")
    n_viz_clients = txt.count("[api/ws] Client connected")
    expect_clients = 1 if viz_on else 0
    P["P10_process_log"] = (all(v == 0 for v in counts.values()) and n_hs == ss["loops_started"] and n_viz_clients == expect_clients,
                            f"bad lines={counts} handshakes={n_hs}/loops {ss['loops_started']} (completed {ss['loops_completed']}) "
                            f"viz clients connected={n_viz_clients} (expected {expect_clients}: exactly one scripted client, no browser)")
    # P11 end-to-end pose lag: sender (loop, ts) send wall -> first poller observation of (epoch, ts)
    loop_epoch = {i + 1: e for i, e in enumerate(epochs)}
    sent_wall = {(loop_epoch.get(r["loop"]), r["ts"]): r["wall"] for r in frames if r.get("ts")}
    first_seen = {}
    for (w, e, s) in keyed:
        if (e, s) not in first_seen:
            first_seen[(e, s)] = w
    lags = [w - sent_wall[k] for k, w in first_seen.items() if k in sent_wall and w >= w0 + WARMUP_S]
    lag_p99 = pct(lags, 99)
    lag_max = max(lags) if lags else None
    if saturated:
        P["P11_pose_lag"] = (bool(lags), f"SATURATED (info): sender->observed lag p50={r3(pct(lags,50))} p99={r3(lag_p99)} max={r3(lag_max)} n={len(lags)} (matched {len(lags)}/{len(first_seen)} observed keys)")
    else:
        P["P11_pose_lag"] = (bool(lags) and lag_p99 < LAG_P99_S and lag_max < LAG_MAX_S,
                             f"sender->observed lag p50={r3(pct(lags,50))} p99={r3(lag_p99)} (<{LAG_P99_S}) max={r3(lag_max)} (<{LAG_MAX_S}) n={len(lags)} (matched {len(lags)}/{len(first_seen)} observed keys)")
    # viz attachment for the whole run
    if viz_on:
        closed_early = [r for r in vz if r.get("closed") and r.get("reason")]
        P["P12_viz_attached"] = (bool(viz_alive_at_end) and not closed_early,
                                 f"viz client alive at sender end={viz_alive_at_end}; closed early={[r.get('reason') for r in closed_early][:2]}")
    # info
    fs = finals.get("/stats") or {}
    vsec = [r for r in vz if "n" in r and r.get("t") is not None and w0 <= r.get("wall", 0) <= w1]
    info["objects"] = (fs.get("objects"), fs.get("confirmed"))
    info["viz_on"] = viz_on
    info["tsdf_on"] = tsdf_on
    info["viz"] = {"camf_per_s_median": statistics.median([r["camf"] for r in vsec]) if vsec else None,
                   "bytes_per_s_median": statistics.median([r["bytes"] for r in vsec]) if vsec else None}
    agg = (((finals.get("/stats/analytics") or {}).get("latency") or {}).get("aggregate") or {})
    info["latency_aggregate"] = {k: agg.get(k) for k in ("input_hz", "processing_hz", "effective_ratio", "frame_count")}
    info["counters"] = counters
    info["semantic_index"] = (finals.get("/healthz") or {}).get("semantic_index")
    info["health_final"] = {k: (finals.get("/healthz") or {}).get(k) for k in ("status", "reasons")}
    info["sender"] = {k: ss.get(k) for k in ("frames_sent", "loops_started", "loops_completed", "achieved_hz", "target_hz", "streamed_s")}
    info["log_warn_err"] = sorted({re.sub(r"^\S+ \S+ ", "", l)[:160] for l in txt.splitlines() if (" WARNING " in l or " ERROR " in l)})[:40]

    # verdict
    if spec["role"] == "calibrate":
        sw, rw = calib.get("suspend_wall"), calib.get("resume_wall")
        win = lambda r: sw is not None and sw <= r["wall"] <= rw + 4.0
        c = {}
        c["P1_trips"] = any((not r.get("ok")) or r["latency"] > HTTP_TIMEOUT_S for r in stats if win(r))
        c["P2_trips"] = bool(windows) and min(windows) < expected10
        # The gap is the symptom. A `stale` sample needs a poll to land while age_s > 0.5 s AFTER the resume; a
        # headless process refills the mailbox faster than the 10 Hz poller can catch that (0 stale in the task-7
        # calibration run despite a 2.68 s gap), so the stale count is reported, not required.
        c["P3_trips"] = bool(gaps) and max(gaps) >= 2.0
        c["P3_stale_samples"] = stale_n
        c["P5_trips"] = st["max"] is not None and st["max"] >= 2.0
        c["P9_trips"] = (roll.get("late_ticks") or 0) >= 1
        c["P11_trips"] = lag_max is not None and lag_max >= 2.0
        verdict = "CALIBRATED" if all(c.values()) else "NOT-CALIBRATED"
        info["calibration"] = {"suspend_wall": sw, "resume_wall": rw, **c}
    elif spec["role"] == "falsify":
        verdict = "WEDGE-VISIBLE" if P["P7_lanes"][0] else "WEDGE-NOT-VISIBLE"
    else:
        verdict = "PASS" if all(v[0] for v in P.values()) else "FAIL"
    log(f"  --- {run} predicates ---")
    for k, (ok, detail) in P.items():
        log(f"  {'ok  ' if ok else 'FAIL'} {k}: {detail}")
    log(f"  info: {json.dumps(info, default=str)}")
    log(f"  {run} VERDICT: {verdict}")
    return {"verdict": verdict, "predicates": {k: {"ok": v[0], "detail": v[1]} for k, v in P.items()}, "info": info}


# ---------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=DEFAULT_ORDER)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--smoke", action="store_true", help="driver mechanics check: C with 45 s of streaming")
    ap.add_argument("--reeval", action="store_true", help="no GPU: re-derive the predicates of every run under --out from its saved artifacts")
    args = ap.parse_args()
    if args.reeval:
        out = Path(args.out).resolve()
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        results = {}
        log(f"G1-C re-evaluation from saved artifacts — runs {runs} — {out}")
        for r in runs:
            rdir = out / r
            fin = rdir / "finals.json"
            if not fin.exists():
                log(f"  {r}: no finals.json, skipped")
                continue
            F = json.loads(fin.read_text(encoding="utf-8"))
            log(""); log(f"=== {r}: {RUNS[r]['what']} ===")
            results[r] = {"run": r, "spec": RUNS[r], **evaluate(r, RUNS[r], rdir, out / f"rtsm_{r}.log", F["finals"], int(F["rss_ready"]), F.get("viz_alive_at_end"), F.get("calib") or {}, viz_on=F.get("viz_on"), tsdf_on=F.get("tsdf_on"))}
        (out / "summary.json").write_text(json.dumps({"reeval": True, "runs": results}, indent=1, default=str), encoding="utf-8")
        verdicts = {r: results.get(r, {}).get("verdict") for r in runs if r in results}
        hard = [r for r in verdicts if RUNS[r]["role"] in ("hard", "hard-saturated")]
        hard_ok = bool(hard) and all(verdicts[r] == "PASS" for r in hard)
        f_ok = all(verdicts[r] == "WEDGE-VISIBLE" for r in verdicts if RUNS[r]["role"] == "falsify")
        c_ok = all(verdicts[r] == "CALIBRATED" for r in verdicts if RUNS[r]["role"] == "calibrate")
        log(""); log(f"HARD GATE (re-evaluated): {'PASS' if (hard_ok and f_ok and c_ok) else 'FAIL'} | harness valid: calibration {c_ok}, falsifier {f_ok} | "
            + " | ".join(f"{r} {verdicts[r]}" for r in verdicts))
        return 0 if (hard_ok and f_ok and c_ok) else 1
    if args.smoke:
        RUNS["C"] = {**RUNS["C"], "duration": 45.0, "what": "SMOKE (driver mechanics + calibration), 45 s"}
        args.runs = "C"
    out = Path(args.out).resolve()      # the child processes run with cwd=ROOT; a relative --out must not depend on it
    out.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(ROOT), text=True).strip()
    except Exception:
        commit = branch = "?"
    log(f"G1-C wedge proof — {branch} @ {commit} — runs {runs} — out {out}")
    moved = False
    if FAISS_DIR.is_dir() and not (ROOT / "model_store" / "faiss.pre-g1c").exists():
        shutil.move(str(FAISS_DIR), str(ROOT / "model_store" / "faiss.pre-g1c"))
        moved = True
    results = {}
    try:
        for r in runs:
            if RUNS[r].get("legacy"):
                log(f"  {r}: pre-task-7 attribution run, kept for --reeval only; not launched")
                continue
            try:
                results[r] = execute(r, RUNS[r], out)
            except Exception as e:
                log(f"  {r} ERROR: {e}")
                results[r] = {"verdict": "ERROR", "error": str(e)}
            finally:
                shutil.rmtree(FAISS_DIR, ignore_errors=True)     # empty store per run
            if RUNS[r]["role"] == "calibrate" and results[r].get("verdict") != "CALIBRATED":
                log("  calibration failed: the harness cannot see the symptoms it gates on; stopping here")
                break
    finally:
        shutil.rmtree(FAISS_DIR, ignore_errors=True)
        if moved:
            shutil.move(str(ROOT / "model_store" / "faiss.pre-g1c"), str(FAISS_DIR))
    (out / "summary.json").write_text(json.dumps({"branch": branch, "commit": commit, "runs": results}, indent=1, default=str), encoding="utf-8")
    verdicts = {r: results.get(r, {}).get("verdict") for r in runs}
    hard = [r for r in runs if RUNS[r]["role"] in ("hard", "hard-saturated")]
    hard_ok = bool(hard) and all(verdicts[r] == "PASS" for r in hard)
    f_ok = all(verdicts[r] == "WEDGE-VISIBLE" for r in runs if RUNS[r]["role"] == "falsify")
    c_ok = all(verdicts[r] == "CALIBRATED" for r in runs if RUNS[r]["role"] == "calibrate")
    log(f"\nHARD GATE: {'PASS' if (hard_ok and f_ok and c_ok) else 'FAIL'} | harness valid: calibration {c_ok}, falsifier {f_ok} | "
        + " | ".join(f"{r} {verdicts[r]}" for r in runs))
    log("G1C_GATE_DONE")
    return 0 if (hard_ok and f_ok and c_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
