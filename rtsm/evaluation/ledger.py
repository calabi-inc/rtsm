"""
Ledger reader + rollups (Gate 4.5 plan, P2).

Pure functions over the diagnostic event log written by
``rtsm/evaluation/event_log.py``. The eval report (P3) reads THESE, never the
live analytics buffers (``rtsm/analytics``), which stay the operator's view.
NumPy only; pyarrow is imported lazily by ``to_parquet``.

  read_events(path)              -> list[dict]     every line, meta first
  by_kind(rows)                  -> {kind: [rows]}
  ledger_meta(rows)              -> dict           the meta line's ``ledgers`` block
  pose_health(rows, ...)         -> dict           rate / gaps / jitter / limited
                                                   episodes / discontinuities /
                                                   depth + confidence statistics,
                                                   per (source, epoch) and in total
  to_parquet(path, out_dir=None) -> {kind: Path}   one Parquet table per kind

CLI:
  python -m rtsm.evaluation.ledger summarize <events.jsonl>
  python -m rtsm.evaluation.ledger parquet   <events.jsonl> [--out DIR]

Conventions (pose ledger, schema 1): the pose STREAM of a (source, epoch)
group is the set of lines that carry a pose and a sensor stamp, ordered by
the stamp; rate, gaps, jitter and discontinuities are computed over it.
Tracking-limited episodes are maximal runs of lines whose tracking_state is
not "normal", in file order, for the sources that have a tracking state
(ARKit / websocket / replay); ZeroMQ lines all read "not_available" because
RTAB-Map publishes none, so no episode is ever counted there. The
discontinuity rule is the RC-car agent's (``examples/rc_car_agent/monitor.py``)
applied to the full 3-D translation, source-agnostic: a step larger than
``disc_base_m + disc_rate_mps * dt`` between consecutive stream poses. It is
a DETECTOR: nothing acts on it.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from rtsm.evaluation.event_log import KIND_POSE, LEDGER_KINDS, TS_NORMAL

logger = logging.getLogger(__name__)

TRACE_KINDS = ("receiver", "dequeue", "frame")
SOURCE_ZEROMQ = "zeromq"


# ───────────────────────────── reading ─────────────────────────────

def read_events(path: Any) -> List[dict]:
    """Every line of an event log as dicts; the first must be the meta line."""
    p = Path(path)
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{n}: not a JSON line: {e}") from None
    if not rows or rows[0].get("kind") != "meta":
        raise ValueError(f"{p}: the first line must be the meta line (kind: meta)")
    return rows


def by_kind(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        out[str(r.get("kind"))].append(r)
    return dict(out)


def ledger_meta(rows: Sequence[dict]) -> dict:
    """The meta line's ``ledgers`` block; ``{"enabled": False}`` for a file
    written before P2 (schema_version < 3)."""
    if not rows or rows[0].get("kind") != "meta":
        return {"enabled": False}
    block = rows[0].get("ledgers")
    return dict(block) if isinstance(block, dict) else {"enabled": False}


# ───────────────────────────── helpers ─────────────────────────────

def _finite(values: Iterable[Any]) -> np.ndarray:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return np.asarray(vals, dtype=float)


def _stats(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def _stamp(r: dict) -> Optional[int]:
    v = r.get("t_sensor_ns")
    if v is None:
        return None
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _close_episode(cur: dict) -> dict:
    s, e = cur.get("start_t_sensor_ns"), cur.get("end_t_sensor_ns")
    dur = (round((int(e) - int(s)) / 1e9, 4) if (s is not None and e is not None) else None)
    return {
        "start_t_sensor_ns": s,
        "end_t_sensor_ns": e,
        "n_frames": int(cur["n_frames"]),
        "duration_s": dur,
        "states": dict(cur["states"]),
    }


# ───────────────────────────── pose health ─────────────────────────────

def _group_health(g: List[dict], source: str, disc_base_m: float, disc_rate_mps: float,
                  gap_factor: float) -> dict:
    n = len(g)
    states = Counter(str(r.get("tracking_state")) for r in g)
    writes_expected = sum(1 for r in g if r.get("mailbox_write"))

    # The pose stream: lines with a pose AND a sensor stamp, ordered by stamp
    # (a stable sort keeps file order for equal stamps).
    stream = sorted((r for r in g if r.get("t_wc") is not None and _stamp(r) is not None),
                    key=lambda r: int(r["t_sensor_ns"]))
    stamps = np.asarray([int(r["t_sensor_ns"]) for r in stream], dtype=np.int64)
    dts = (np.diff(stamps) / 1e9) if stamps.size > 1 else np.zeros(0, dtype=float)
    span_s = float(stamps[-1] - stamps[0]) / 1e9 if stamps.size > 1 else 0.0
    sensor_hz = (round((stamps.size - 1) / span_s, 4) if span_s > 0 else None)
    med = float(np.median(dts)) if dts.size else None

    gaps: List[dict] = []
    if med is not None and med > 0:
        for i, dt in enumerate(dts):
            if dt > gap_factor * med:
                gaps.append({"after_t_sensor_ns": int(stamps[i]), "gap_s": round(float(dt), 4)})
    jitter_ms = (round(float(np.median(np.abs(dts - med))) * 1000.0, 3)
                 if (dts.size and med is not None) else None)

    discontinuities: List[dict] = []
    if len(stream) > 1:
        P = np.asarray([[float(v) for v in r["t_wc"]] for r in stream], dtype=float)
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        for i, d in enumerate(steps):
            dt = float(dts[i])
            if d > disc_base_m + disc_rate_mps * dt:
                discontinuities.append({"t_sensor_ns": int(stamps[i + 1]),
                                        "jump_m": round(float(d), 4), "dt_s": round(dt, 4)})

    # Tracking-limited episodes: file order, sources with a tracking state only.
    episodes: List[dict] = []
    if source != SOURCE_ZEROMQ:
        cur: Optional[dict] = None
        for r in g:
            state = str(r.get("tracking_state"))
            if state != TS_NORMAL:
                if cur is None:
                    cur = {"start_t_sensor_ns": _stamp(r), "end_t_sensor_ns": _stamp(r),
                           "n_frames": 0, "states": Counter()}
                cur["n_frames"] += 1
                st = _stamp(r)
                if st is not None:
                    cur["end_t_sensor_ns"] = st
                    if cur["start_t_sensor_ns"] is None:
                        cur["start_t_sensor_ns"] = st
                cur["states"][state] += 1
            elif cur is not None:
                episodes.append(_close_episode(cur))
                cur = None
        if cur is not None:
            episodes.append(_close_episode(cur))
    limited_frames = sum(e["n_frames"] for e in episodes)

    conf2 = []
    for r in g:
        h = r.get("conf_hist")
        if isinstance(h, (list, tuple)) and len(h) >= 3:
            tot = float(sum(h))
            if tot > 0:
                conf2.append(float(h[2]) / tot)

    return {
        "n_frames": n,
        "n_by_tracking_state": dict(states),
        "writes_expected": int(writes_expected),
        "n_stream": int(len(stream)),
        "span_s": round(span_s, 4),
        "sensor_hz": sensor_hz,
        "dt_ms": {k: (round(v * 1000.0, 3) if isinstance(v, float) else v) for k, v in _stats(dts).items()},
        "jitter_ms": jitter_ms,
        "n_gaps": len(gaps),
        "gaps": gaps,
        "n_limited_episodes": len(episodes),
        "limited_episodes": episodes,
        "limited_frames_frac": (round(limited_frames / n, 6) if n else None),
        "n_discontinuities": len(discontinuities),
        "discontinuities": discontinuities,
        "depth_valid_frac": _stats(r.get("depth_valid_frac") for r in g),
        "conf2_frac": _stats(conf2),
        "pose_errors": sum(1 for r in g if r.get("pose_error")),
    }


def pose_health(rows: Iterable[dict], *, disc_base_m: float = 0.5, disc_rate_mps: float = 1.0,
                gap_factor: float = 2.0) -> dict:
    """Pose-stream health from the ``pose`` ledger lines, per (source, epoch)
    group and in total. See the module docstring for the definitions.
    ``writes_expected`` counts the lines whose receiver called its pose sink:
    on a replay it must equal ``/stats.robot_pose.writes_accepted``."""
    pose = [r for r in rows if r.get("kind") == KIND_POSE]
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in pose:
        groups[(str(r.get("source")), r.get("epoch"))].append(r)
    out_groups: Dict[str, dict] = {}
    total = {"n_frames": 0, "writes_expected": 0, "n_gaps": 0, "n_limited_episodes": 0,
             "n_discontinuities": 0, "pose_errors": 0, "n_groups": 0}
    for (source, epoch), g in groups.items():
        res = _group_health(g, source, disc_base_m, disc_rate_mps, gap_factor)
        out_groups[f"{source}/{epoch}"] = res
        for k in ("n_frames", "writes_expected", "n_gaps", "n_limited_episodes", "n_discontinuities", "pose_errors"):
            total[k] += int(res[k])
        total["n_groups"] += 1
    return {
        "params": {"disc_base_m": disc_base_m, "disc_rate_mps": disc_rate_mps, "gap_factor": gap_factor},
        "groups": out_groups,
        "total": total,
    }


# ───────────────────────────── parquet ─────────────────────────────

def to_parquet(events_path: Any, out_dir: Any = None, kinds: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Convert an event log to one Parquet file per kind (``<stem>.<kind>.parquet``
    next to the JSONL, or under ``out_dir``). The JSONL stays the source of
    truth. Ledger kinds are converted first; a kind whose rows pyarrow cannot
    type is skipped with a warning (the trace kinds carry free-form dicts).
    Raises RuntimeError when pyarrow is missing."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("diagnostics.ledger_format=parquet needs pyarrow: pip install \"rtsm[eval]\"") from None
    p = Path(events_path)
    kinds_map = by_kind(read_events(p))
    dest_dir = Path(out_dir) if out_dir is not None else p.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = p.name[:-len(".jsonl")] if p.name.endswith(".jsonl") else p.stem
    wanted = list(kinds) if kinds else [k for k in (*LEDGER_KINDS, *TRACE_KINDS) if k in kinds_map]
    written: Dict[str, Path] = {}
    for kind in wanted:
        rows = kinds_map.get(kind) or []
        if not rows:
            continue
        dest = dest_dir / f"{stem}.{kind}.parquet"
        try:
            pq.write_table(pa.Table.from_pylist(rows), dest)
        except Exception:  # noqa: BLE001 -- one untypable kind must not lose the others
            logger.warning("ledger: parquet conversion skipped kind %r", kind, exc_info=True)
            continue
        written[kind] = dest
    return written


# ───────────────────────────── CLI ─────────────────────────────

def summarize(rows: Sequence[dict]) -> dict:
    kinds = by_kind(rows)
    return {
        "meta": rows[0] if rows else None,
        "counts": {k: len(v) for k, v in sorted(kinds.items())},
        "pose_health": pose_health(rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtsm.evaluation.ledger",
                                 description="Read an RTSM diagnostic event log (P2 ledgers).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="print counts per kind and the pose-health rollup as JSON")
    s.add_argument("events", help="path to events.jsonl")
    q = sub.add_parser("parquet", help="write one Parquet file per kind next to the JSONL (needs pyarrow)")
    q.add_argument("events", help="path to events.jsonl")
    q.add_argument("--out", default=None, help="output directory (default: next to the JSONL)")
    args = ap.parse_args(argv)
    if args.cmd == "summarize":
        print(json.dumps(summarize(read_events(args.events)), indent=2, default=str))
        return 0
    written = to_parquet(args.events, args.out)
    for kind, path in written.items():
        print(f"{kind}: {path}")
    return 0 if written else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
