"""
``rtsm eval <bag-or-recording>`` -- the headless eval runner (Gate 4.5 plan,
P3 task 2).

One process, no HTTP server: the models are loaded once, then for each of N
repeats a fresh runtime (memory, index, gate, sweep cache, queue, pipeline,
event log with the trace + ledgers) is built through ``rtsm.engine``, the
input is fed through the bag source (or the replay source for a Lens
recording) on the sensor clock with the lossless lane, the pipeline runs
until the source is done and the queue is drained, every object is force-
flushed to an isolated vector store, and the run directory receives
``events.jsonl`` (trace + pose / obs / view ledgers) and ``summary.json``.

Modes
- ``as_deployed`` (default): the deployed ingest settings -- keyframes every
  ``ingest.keyframe_every_n`` admitted frames (or the source's own keyframes
  when it flags them), the non-keyframe throttle ``ingest.nonkf_min_interval_s``,
  the sweep gate ENFORCED. This is exactly what ``python -m rtsm --replay`` /
  ``--bag`` does, so the G1-B anchor must reproduce (gate G3-2).
- ``dense``: keyframes every ``eval.keyframe_interval_s`` of sensor time, the
  throttle from ``eval.process_rate_hz``, the sweep gate in SHADOW mode (its
  decision is logged as ``gate_shadow`` on the dequeue line; the frame is
  processed anyway) -- the input for observation metrics that are then masked
  by what the deployed gate would have admitted.

Every run records the keyframe rule, the resolved flags, the config
fingerprint, the commit and the versions (``resolved`` in ``summary.json``).
The packaged YAML is never written; each run's config is a copy with the
vector-store path and the event-log path pointed into its run directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA = 1
MODES = ("as_deployed", "dense")


# ───────────────────────────── options + resolution ─────────────────────────────

@dataclass
class EvalOptions:
    input: str
    mode: str = "as_deployed"
    repeats: int = 1
    out: Optional[str] = None
    max_frames: Optional[int] = None
    max_wall_s: Optional[float] = None
    label: Optional[str] = None


@dataclass
class ResolvedEval:
    mode: str
    input: str
    input_kind: str                          # replay | bag
    clock: str                               # sensor
    policy: str                              # lossless
    keyframe_rule: Dict[str, Any]            # {"kind": "every_n", "n": 30} | {"kind": "interval", "interval_s": 1.0} | {"kind": "source"}
    nonkf_min_interval_s: float
    gate_mode: str                           # enforce | shadow
    require_tracking_normal: bool
    confidence_threshold: int
    apply_camera_flip: bool
    up_axis_default: str
    ledgers: bool
    ledger_format: str
    config_fingerprint: str
    git_commit: Optional[str]
    rtsm_version: Optional[str]
    python: str
    platform: str
    repeats: int
    max_frames: Optional[int]
    max_wall_s: Optional[float]


def input_kind(path: str | os.PathLike) -> str:
    p = Path(path)
    if p.is_dir() and (p / "messages.bin").is_file():
        return "replay"
    return "bag"


def _git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    try:
        root = repo_root or Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def _rtsm_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("rtsm")
    except Exception:  # noqa: BLE001
        return None


def resolve_eval(cfg: dict, opts: EvalOptions) -> ResolvedEval:
    """The run's settings from the config + the mode. Pure: reads ``cfg``,
    writes nothing (``configure_run`` derives the per-run config copy)."""
    from rtsm.cfg import config_fingerprint
    from rtsm.core.clock import resolve_clock_mode
    from rtsm.io.ingest_lanes import LaneConfig
    mode = str(opts.mode or "as_deployed").lower()
    if mode not in MODES:
        raise ValueError(f"eval mode {opts.mode!r} not in {MODES}")
    ev = dict(cfg.get("eval") or {})
    kind = input_kind(opts.input)
    clock = resolve_clock_mode((cfg.get("ingest") or {}).get("clock", "auto"), replay=True)
    lane_cfg = LaneConfig.from_cfg(cfg, replay=True)
    ws = (cfg.get("io") or {}).get("websocket") or {}
    vis = cfg.get("visualization") or {}
    if mode == "as_deployed":
        use_source = str(ev.get("use_source_keyframes", "auto")).lower()
        # No bag / recording flags its own keyframes in v1 (the front-end mints them); "auto" therefore resolves to the
        # receiver's every-N rule with the deployed N. An explicit true is honoured by the front-end when a source sets
        # keyframe_hint (ZeroMQ-style sources) and falls back to every-N otherwise.
        keyframe_rule = {"kind": "every_n", "n": int(lane_cfg.keyframe_every_n), "use_source_keyframes": use_source}
        throttle = float(lane_cfg.nonkf_min_interval_s)
        gate_default = "enforce"
    else:
        interval = float(ev.get("keyframe_interval_s", 1.0))
        rate = float(ev.get("process_rate_hz", 5.0))
        if interval <= 0 or rate <= 0:
            raise ValueError("eval.keyframe_interval_s and eval.process_rate_hz must be > 0")
        keyframe_rule = {"kind": "interval", "interval_s": interval}
        throttle = 1.0 / rate
        gate_default = "shadow"
    gate_mode = str(ev.get("gate_mode", "auto")).lower()
    if gate_mode == "auto":
        gate_mode = gate_default
    if gate_mode not in ("enforce", "shadow"):
        raise ValueError(f"eval.gate_mode {gate_mode!r} not in enforce | shadow | auto")
    up_axis_default = "y"
    if kind == "bag":
        try:
            from rtsm.evaluation.recording_mcap import read_bag_custom_data
            up_axis_default = "y" if read_bag_custom_data(opts.input).get("rtsm_source") == "lens_recording" else "z"
        except Exception:  # noqa: BLE001
            up_axis_default = "z"
    diag = cfg.get("diagnostics") or {}
    return ResolvedEval(
        mode=mode, input=str(opts.input), input_kind=kind, clock=clock, policy=lane_cfg.policy, keyframe_rule=keyframe_rule,
        nonkf_min_interval_s=throttle, gate_mode=gate_mode,
        require_tracking_normal=bool(ws.get("require_tracking_normal", True)), confidence_threshold=int(ws.get("confidence_threshold", 1)),
        apply_camera_flip=bool(vis.get("apply_camera_flip", False)), up_axis_default=up_axis_default,
        ledgers=True, ledger_format=str(diag.get("ledger_format", "jsonl")), config_fingerprint=config_fingerprint(cfg),
        git_commit=_git_commit(), rtsm_version=_rtsm_version(), python=platform.python_version(), platform=platform.platform(),
        repeats=int(opts.repeats), max_frames=opts.max_frames, max_wall_s=opts.max_wall_s,
    )


def configure_run(cfg: dict, resolved: ResolvedEval, run_dir: Path) -> dict:
    """The per-run config: a deep copy with the vector store and the event
    log inside ``run_dir``, diagnostics + ledgers on, viz / MCP off, the
    resolved gate mode. The input config object is never mutated."""
    c = copy.deepcopy(cfg)
    c.setdefault("vectors", {}).setdefault("faiss", {})["index_path"] = str(run_dir / "faiss" / "index.flatip")
    c["diagnostics"] = {**(c.get("diagnostics") or {}), "enabled": True, "event_log_path": str(run_dir / "events.jsonl"),
                        "ledgers": True, "ledger_format": resolved.ledger_format}
    c.setdefault("eval", {})["gate_mode"] = resolved.gate_mode
    c.setdefault("visualization", {})["enable"] = False
    c.setdefault("mcp", {})["enable"] = False
    c.setdefault("io", {})["receiver"] = "replay" if resolved.input_kind == "replay" else "bag"
    return c


# ───────────────────────────── memory summary + fingerprint ─────────────────────────────

def object_summary(o: Any) -> Dict[str, Any]:
    """The API's ``/objects`` summary shape (``rtsm/api/server.py::_obj_summary``),
    so ``memory.objects`` is comparable to every existing anchor record."""
    xyz = getattr(o, "xyz_world", None)
    return {
        "id": getattr(o, "id", None),
        "xyz_world": xyz.tolist() if xyz is not None else None,
        "created_wall_utc": float(getattr(o, "created_wall_utc", 0.0)),
        "created_mono": float(getattr(o, "created_mono", 0.0)),
        "stability": float(getattr(o, "stability", 0.0)),
        "hits": int(getattr(o, "hits", 0)),
        "confirmed": bool(getattr(o, "confirmed", False)),
        "label_primary": getattr(o, "label_primary", None),
        "view_bins": len(getattr(o, "view_bins", {}) or {}),
        "last_seen_mono": float(getattr(o, "last_seen_mono", 0.0)),
    }


def fingerprint(objects: List[Dict[str, Any]]) -> str:
    """The gate scripts' multiset fingerprint: sha256 over the sorted multiset
    of (label_primary, xyz rounded to 3 decimals, hits, confirmed), 16 hex chars.
    The G1-B anchor is ``ad6f71a5b89c8506``."""
    ms = Counter((o.get("label_primary"), tuple(round(float(v), 3) for v in (o.get("xyz_world") or [])),
                  int(o.get("hits") or 0), bool(o.get("confirmed"))) for o in objects)
    return hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]


def summarize_memory(wm: Any) -> Dict[str, Any]:
    objs = [object_summary(o) for o in wm.iter_objects()]
    stats = {}
    try:
        stats = dict(wm.stats())
    except Exception:  # noqa: BLE001
        stats = {}
    return {"objects": objs, "objects_count": len(objs), "confirmed_count": sum(1 for o in objs if o["confirmed"]),
            "fingerprint": fingerprint(objs), "working_memory": stats}


# ───────────────────────────── one run ─────────────────────────────

def _drain_and_stop(rt, src, *, max_wall_s: Optional[float]) -> Optional[str]:
    """Run the pipeline until the source is done AND the queue is empty.
    ``run_one_step`` is synchronous, so nothing is in flight once it returns
    with an empty, closed queue. Returns the abort reason, or None."""
    done = threading.Event()

    def waiter() -> None:
        try:
            src.wait()
        finally:
            close = getattr(rt.ingest_q, "close", None)
            if callable(close):
                close()
            done.set()

    threading.Thread(target=waiter, name="eval-source-waiter", daemon=True).start()
    t0 = time.monotonic()
    while True:
        rt.pipeline.run_one_step()
        if done.is_set() and rt.ingest_q.qsize() == 0:
            return None
        if max_wall_s is not None and (time.monotonic() - t0) > float(max_wall_s):
            try:
                src.stop()
            except Exception:  # noqa: BLE001
                pass
            return "max_wall_s"


def run_once(cfg: dict, models: Any, resolved: ResolvedEval, run_dir: Path, run_index: int, *,
             opts: EvalOptions, runtime_factory: Optional[Callable[..., Any]] = None,
             source_factory: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """One repeat into ``run_dir``. ``runtime_factory`` / ``source_factory``
    default to ``rtsm.engine.build_runtime`` / ``rtsm.io.sources.make_source``
    (tests inject stubs)."""
    from rtsm.core.clock import make_clock
    from rtsm.evaluation.event_log import EventLogWriter
    from rtsm.evaluation.ledger import by_kind, outcome_histogram, read_events
    from rtsm.io.contracts import SourceContext
    from rtsm.io.ingest_lanes import LaneConfig
    if runtime_factory is None:
        from rtsm.engine import build_runtime as runtime_factory
    if source_factory is None:
        from rtsm.io.sources import make_source as source_factory

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "faiss").mkdir(exist_ok=True)
    cfg_k = configure_run(cfg, resolved, run_dir)
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    lane_cfg = LaneConfig.from_cfg(cfg_k, replay=True)
    clock = make_clock(resolved.clock)
    event_log = EventLogWriter(
        enabled=True, configured_path=str(run_dir / "events.jsonl"),
        extra_meta={"ingest_clock": resolved.clock, "ingest_policy": lane_cfg.policy, "eval_mode": resolved.mode,
                    "keyframe_rule": resolved.keyframe_rule, "nonkf_min_interval_s": resolved.nonkf_min_interval_s,
                    "gate_mode": resolved.gate_mode, "run_index": run_index, "config_fingerprint": resolved.config_fingerprint},
        ledgers=True, ledger_format=resolved.ledger_format,
    )
    rt = runtime_factory(cfg_k, models, clock=clock, lane_cfg=lane_cfg, event_log=event_log, up_axis_default=resolved.up_axis_default)
    kf = resolved.keyframe_rule
    ctx = SourceContext(
        ingest_queue=rt.ingest_q, clock_mode=resolved.clock,
        keyframe_every_n=int(kf.get("n", lane_cfg.keyframe_every_n)),
        keyframe_interval_s=(float(kf["interval_s"]) if kf.get("kind") == "interval" else None),
        nonkf_min_interval_s=float(resolved.nonkf_min_interval_s),
        require_tracking_normal=resolved.require_tracking_normal, confidence_threshold=resolved.confidence_threshold,
        apply_camera_flip=resolved.apply_camera_flip,
        pose_sink=rt.wm.update_robot_pose, event_sink=event_log.sink(), ledger_sink=event_log.ledger_sink(),
        latency_analytics=getattr(rt.analytics, "latency", None),
    )
    if resolved.input_kind == "replay":
        src = source_factory("replay", cfg_k, ctx, recording_dir=resolved.input, replay_speed=1e9)   # as fast as accepted
        if opts.max_frames:
            logger.warning("[eval] --max-frames applies to bag inputs only; a recording replays whole")
    else:
        src = source_factory("bag", cfg_k, ctx, path=resolved.input, speed=None, max_frames=opts.max_frames)
    logger.info("[eval] run %d: %s (%s) -> %s", run_index, resolved.input, resolved.input_kind, run_dir)
    start_analytics = getattr(rt.analytics, "start", None)
    if callable(start_analytics):
        start_analytics()
    src.start()
    aborted = _drain_and_stop(rt, src, max_wall_s=resolved.max_wall_s)
    # force-flush every object to this run's vector store (the replay runner's rule)
    flushed = 0
    try:
        ready = rt.wm.collect_ready_for_upsert(force_all=True)
        if ready and rt.vectors is not None:
            rt.vectors.upsert_batch(ready)
            flushed = len(ready)
    except Exception as e:  # noqa: BLE001
        logger.warning("[eval] force flush failed: %s", e)
    memory = summarize_memory(rt.wm)
    stop_analytics = getattr(rt.analytics, "stop", None)
    if callable(stop_analytics):
        stop_analytics()
    for closer in (getattr(rt.vectors, "close", None), getattr(event_log, "close", None)):
        if callable(closer):
            try:
                closer()
            except Exception as e:  # noqa: BLE001
                logger.warning("[eval] close failed: %s", e)
    # the pipeline's own shutdown closes the MODELS, which the next repeat reuses: close the queue only
    close_q = getattr(rt.ingest_q, "close", None)
    if callable(close_q):
        close_q()
    wall_s = time.monotonic() - t_start

    rows = read_events(run_dir / "events.jsonl")
    kinds = by_kind(rows)
    receiver = Counter((r.get("decision"), r.get("reason")) for r in kinds.get("receiver", []))
    dequeue = Counter((r.get("outcome"), r.get("reason")) for r in kinds.get("dequeue", []))
    shadow = Counter(r.get("gate_shadow") for r in kinds.get("dequeue", []) if r.get("gate_shadow"))
    latency = seg = None
    try:
        latency = rt.analytics.latency.aggregate() if getattr(rt.analytics, "latency", None) is not None else None
        seg = rt.analytics.seg.aggregate() if getattr(rt.analytics, "seg", None) is not None else None
    except Exception:  # noqa: BLE001
        pass
    src_stats = None
    try:
        src_stats = src.stats() if hasattr(src, "stats") else src.liveness()
    except Exception:  # noqa: BLE001
        pass
    summary = {
        "schema": SUMMARY_SCHEMA, "input": resolved.input, "input_kind": resolved.input_kind, "mode": resolved.mode,
        "run_index": run_index, "run_dir": str(run_dir), "started_utc": started.isoformat(), "wall_s": round(wall_s, 3),
        "aborted": aborted, "resolved": asdict(resolved),
        "source": src_stats,
        "frames": {"receiver": {f"{d}:{r}" if r else str(d): n for (d, r), n in receiver.items()},
                   "dequeue": {f"{o}:{r}" if r else str(o): n for (o, r), n in dequeue.items()},
                   "outcomes": dict(outcome_histogram(rows)), "gate_shadow": dict(shadow),
                   "processed": sum(n for (o, _r), n in dequeue.items() if o == "processed"),
                   "kinds": {k: len(v) for k, v in kinds.items()}},
        "memory": memory, "flushed_to_vectors": flushed,
        "latency": latency, "segmentation": seg,
        "events_path": str(run_dir / "events.jsonl"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    logger.info("[eval] run %d done in %.1f s: objects %d / confirmed %d, fingerprint %s, processed %d%s", run_index, wall_s,
                memory["objects_count"], memory["confirmed_count"], memory["fingerprint"], summary["frames"]["processed"],
                f" ABORTED {aborted}" if aborted else "")
    return summary


# ───────────────────────────── N repeats ─────────────────────────────

@dataclass
class EvalResult:
    out_dir: Path
    resolved: ResolvedEval
    runs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def repeats(self) -> Dict[str, Any]:
        fps = [r["memory"]["fingerprint"] for r in self.runs]
        return {
            "schema": SUMMARY_SCHEMA, "runs": len(self.runs),
            "fingerprints": fps, "identical_fingerprints": (len(set(fps)) == 1) if fps else None,
            "objects": [r["memory"]["objects_count"] for r in self.runs], "confirmed": [r["memory"]["confirmed_count"] for r in self.runs],
            "processed": [r["frames"]["processed"] for r in self.runs], "wall_s": [r["wall_s"] for r in self.runs],
            "aborted": [r["aborted"] for r in self.runs], "run_dirs": [r["run_dir"] for r in self.runs],
        }


def run_eval(cfg: dict, opts: EvalOptions, *, models: Any = None, load_models: Optional[Callable[[dict], Any]] = None,
             runtime_factory: Optional[Callable[..., Any]] = None, source_factory: Optional[Callable[..., Any]] = None,
             close_models: bool = True) -> EvalResult:
    resolved = resolve_eval(cfg, opts)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = opts.label or Path(opts.input).name.replace(".", "_")
    out_dir = Path(opts.out) if opts.out else Path("eval_output") / f"{stem}-{resolved.mode}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved.json").write_text(json.dumps(asdict(resolved), indent=1, default=str), encoding="utf-8")
    if resolved.input_kind == "bag":
        from rtsm.io.bag_reader import probe_bag
        bag_cfg = dict((cfg.get("io") or {}).get("bag") or {})
        probe = probe_bag(opts.input, topics=dict(bag_cfg.get("topics") or {}),
                          tf_extrapolation_s=float(bag_cfg.get("tf_extrapolation_s", 0.05)),
                          world_frame=bag_cfg.get("world_frame") or None, camera_frame=bag_cfg.get("camera_frame") or None,
                          assume_aligned=bool(bag_cfg.get("assume_aligned", False)), typestore=str(bag_cfg.get("typestore") or "humble"))
        if probe.refusal:
            raise ValueError("bag refused: " + "; ".join(f"{c}: {d}" for c, d in probe.refusal))
        (out_dir / "bag_probe.json").write_text(json.dumps(probe.as_dict(), indent=1, default=str), encoding="utf-8")
    if models is None:
        if load_models is None:
            from rtsm.engine import load_models as load_models
        models = load_models(cfg)
    result = EvalResult(out_dir=out_dir, resolved=resolved)
    try:
        for k in range(1, int(opts.repeats) + 1):
            summary = run_once(cfg, models, resolved, out_dir / f"run_{k}", k, opts=opts,
                               runtime_factory=runtime_factory, source_factory=source_factory)
            result.runs.append(summary)
            (out_dir / "repeats.json").write_text(json.dumps(result.repeats, indent=1, default=str), encoding="utf-8")
    finally:
        if close_models:
            for m in (getattr(models, "segmenter", None), getattr(models, "clip", None)):
                closer = getattr(m, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # noqa: BLE001
                        pass
    return result


# ───────────────────────────── CLI ─────────────────────────────

def build_parser():
    import argparse
    from rtsm.cfg.cli import add_config_arguments
    ap = argparse.ArgumentParser(prog="rtsm eval", description="Headless evaluation run of a bag or a Lens recording (P3).")
    ap.add_argument("input", help="a ROS 1 .bag, a rosbag2 directory, a bare .mcap, or a Lens recording directory (messages.bin)")
    ap.add_argument("--mode", choices=MODES, default=None, help="as_deployed (default; the deployed ingest settings, gate enforced) | dense")
    ap.add_argument("--repeats", type=int, default=None, help="same-input repeats (default: eval.repeats or 1)")
    ap.add_argument("--out", type=str, default=None, help="output directory (default: eval_output/<input>-<mode>-<stamp>/)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many paired frames (bag inputs)")
    ap.add_argument("--max-wall-s", type=float, default=None, help="abort a run after this many seconds of wall time")
    ap.add_argument("--label", type=str, default=None, help="name for the output directory instead of the input's")
    add_config_arguments(ap)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    ap = build_parser()
    args = ap.parse_args(argv)
    from rtsm.cfg import ConfigError
    from rtsm.cfg.cli import config_from_args
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        ap.error(str(exc))
    ev = dict(cfg.get("eval") or {})
    opts = EvalOptions(
        input=args.input, mode=(args.mode or str(ev.get("mode", "as_deployed"))),
        repeats=(args.repeats if args.repeats is not None else int(ev.get("repeats", 1))),
        out=(args.out or ev.get("out") or None), max_frames=args.max_frames,
        max_wall_s=(args.max_wall_s if args.max_wall_s is not None else ev.get("max_wall_s")), label=args.label,
    )
    if not Path(opts.input).exists():
        ap.error(f"input not found: {opts.input}")
    if opts.repeats < 1:
        ap.error("--repeats must be >= 1")
    try:
        resolved = resolve_eval(cfg, opts)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"rtsm eval: {resolved.input} ({resolved.input_kind}) mode={resolved.mode} clock={resolved.clock} policy={resolved.policy} "
          f"keyframes={resolved.keyframe_rule} throttle={resolved.nonkf_min_interval_s:.3f}s gate={resolved.gate_mode} repeats={opts.repeats}")
    try:
        result = run_eval(cfg, opts)
    except ValueError as exc:
        ap.error(str(exc))
    rep = result.repeats
    print(json.dumps({"out_dir": str(result.out_dir), **rep}, indent=1, default=str))
    return 0 if not any(rep["aborted"]) else 1


if __name__ == "__main__":
    sys.exit(main())
