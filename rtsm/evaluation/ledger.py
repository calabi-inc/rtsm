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
  observation_summary(rows)      -> dict           outcomes, per-frame and per-object
                                                   counts, match residuals, ranges,
                                                   view-bin coverage, the view/obs join
                                                   (in-frustum and matched / missed)
  frame_outcomes(rows)           -> {key: str}     one outcome per sensor frame joined
                                                   across receiver / lanes / dequeue lines
  outcome_histogram(rows)        -> Counter        the same, counted
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
a DETECTOR: nothing acts on it. ``delivery_lag`` is arrival time (the line's
monotonic stamp) minus sensor time, both relative to the first stream line:
its growth is the transport delivering frames slower than the sensor stamps
them (session1: +3.5 s over 40.5 s live; under replay the replayer's own
pacing drift adds ~9 ms per frame on top).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from rtsm.evaluation.event_log import KIND_OBS, KIND_POSE, KIND_VIEW, LEDGER_KINDS, OBS_CREATED, OBS_MATCHED, TS_NORMAL

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

    # Delivery lag: arrival (the write's monotonic stamp) minus sensor time,
    # both relative to the first stream line, in FILE order. Growth means the
    # transport delivers frames slower than the sensor stamps them (a queue
    # building on the sender or in the socket); a negative step is a catch-up
    # burst. Under replay this includes the replayer's own pacing drift
    # (~9 ms per frame on session1), so the live value is the meaningful one.
    delivery = {"end_s": None, "max_s": None, "slope_s_per_min": None, "n_catchups": 0}
    stream_file_order = [r for r in g if r.get("t_wc") is not None and _stamp(r) is not None
                         and isinstance(r.get("timestamp"), (int, float))]
    if len(stream_file_order) > 1:
        arr = np.asarray([float(r["timestamp"]) for r in stream_file_order]); arr -= arr[0]
        sen = np.asarray([int(r["t_sensor_ns"]) for r in stream_file_order], dtype=np.int64) / 1e9; sen -= sen[0]
        lag = arr - sen
        span_min = float(sen[-1]) / 60.0
        delivery = {
            "end_s": round(float(lag[-1]), 3),
            "max_s": round(float(lag.max()), 3),
            "slope_s_per_min": (round(float(lag[-1] / span_min), 4) if span_min > 0 else None),
            "n_catchups": int((np.diff(lag) < -0.05).sum()),
        }

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
        "delivery_lag": delivery,
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


# ───────────────────────────── frame outcomes ─────────────────────────────

_RX_KEEP_VERBATIM = ("tracking_state", "malformed", "parse_error", "duplicate_ts", "no_camera_frame")


def _frame_key(r: dict) -> Any:
    """Join key of a receiver / lanes / dequeue line: the sensor stamp when the
    header parsed (unique per frame on every source), else the receiver's
    rx_seq (malformed lines), else the file position."""
    ts = _stamp(r)
    if ts is not None:
        return ("ts", int(ts))
    if r.get("rx_seq") is not None:
        return ("rx", str(r.get("source")), int(r["rx_seq"]))
    return ("row", id(r))


def frame_outcomes(rows: Iterable[dict]) -> Dict[Any, str]:
    """One outcome string per sensor frame, precedence dequeue > lane drop >
    receiver decision (a frame is characterised by the furthest point it
    reached):
      processed | gate_rejected:<reason> | frame_rejected:<dark|flat|depth> |
      dropped:<reason> (pose conversion, queue_full, kf_lane_full, superseded,
      kf_dropped, age, closed) | throttled | tracking_state | malformed |
      parse_error | duplicate_ts | no_camera_frame | enqueued (still queued
      at shutdown)."""
    out: Dict[Any, str] = {}
    rank: Dict[Any, int] = {}

    def put(key: Any, value: str, level: int) -> None:
        if rank.get(key, -1) <= level:
            out[key] = value
            rank[key] = level

    for r in rows:
        kind = r.get("kind")
        if kind == "receiver":
            key = _frame_key(r)
            if r.get("source") == "lanes":
                put(key, f"dropped:{r.get('reason')}", 1)
            elif r.get("decision") == "enqueued":
                put(key, "enqueued", 0)
            else:
                reason = str(r.get("reason"))
                if reason == "throttle":
                    put(key, "throttled", 0)
                elif reason in _RX_KEEP_VERBATIM:
                    put(key, reason, 0)
                else:
                    put(key, f"dropped:{reason}", 0)
        elif kind == "dequeue":
            key = _frame_key(r)
            outcome = str(r.get("outcome"))
            if outcome == "processed":
                put(key, "processed", 2)
            else:
                put(key, f"{outcome}:{r.get('reason')}", 2)
    return out


def outcome_histogram(rows: Iterable[dict]) -> Counter:
    return Counter(frame_outcomes(rows).values())


# ───────────────────────────── observation summary ─────────────────────────────

def observation_summary(rows: Iterable[dict]) -> dict:
    """Inputs for the P3 metrics from the ``obs`` ledger lines: outcome counts,
    lines per frame, matched observations per object, the winning match's
    residuals, ranges, the view-bin coverage of matched + created
    observations, and the gate audit counters. No physical-object clustering
    here (that is P3)."""
    obs = [r for r in rows if r.get("kind") == KIND_OBS]
    outcomes = Counter(str(r.get("outcome")) for r in obs)
    matched = [r for r in obs if r.get("outcome") == OBS_MATCHED]
    created = [r for r in obs if r.get("outcome") == OBS_CREATED]
    per_object = Counter(r["object_id"] for r in matched if r.get("object_id"))
    objects_seen = set(per_object) | {r["object_id"] for r in created if r.get("object_id")}
    per_frame = Counter(r.get("t_sensor_ns") for r in obs)
    bins = Counter(r.get("view_bin") for r in matched + created if r.get("view_bin") is not None)
    # The view / obs join: for each frame with a `view` line, which in-frustum
    # objects were matched (seen again), which were missed, and whether any
    # object created on the frame was already listed (must never happen: the
    # view precedes association).
    views = [r for r in rows if r.get("kind") == KIND_VIEW]
    matched_by_ts: Dict[Any, set] = defaultdict(set)
    created_by_ts: Dict[Any, set] = defaultdict(set)
    for r in matched:
        if r.get("object_id"):
            matched_by_ts[r.get("t_sensor_ns")].add(r["object_id"])
    for r in created:
        if r.get("object_id"):
            created_by_ts[r.get("t_sensor_ns")].add(r["object_id"])
    in_and_matched = in_and_missed = matched_outside = created_in_view = 0
    depth_pairs = []
    for vw in views:
        ts = vw.get("t_sensor_ns")
        ids = {e.get("id") for e in (vw.get("objects") or [])}
        m = matched_by_ts.get(ts, set())
        in_and_matched += len(ids & m)
        in_and_missed += len(ids - m)
        matched_outside += len(m - ids)
        created_in_view += len(ids & created_by_ts.get(ts, set()))
        for e in (vw.get("objects") or []):
            if e.get("id") in m and e.get("observed_depth") is not None and e.get("expected_depth") is not None:
                depth_pairs.append(abs(float(e["expected_depth"]) - float(e["observed_depth"])))
    view_join = {
        "n_frames_with_view": len(views),
        "in_frustum_per_frame": _stats(len(vw.get("objects") or []) for vw in views),
        "n_live_per_frame": _stats(vw.get("n_live") for vw in views),
        "in_frustum_and_matched": in_and_matched,
        "in_frustum_and_missed": in_and_missed,
        "matched_outside_frustum": matched_outside,
        "created_already_in_view": created_in_view,
        "matched_depth_abs_err_m": _stats(depth_pairs),
        "view_ms": _stats(vw.get("view_ms") for vw in views),
    }
    return {
        "view": view_join,
        "n_obs": len(obs),
        "outcomes": dict(outcomes),
        "n_matched_without_scoring": sum(1 for r in matched if r.get("matched_without_scoring")),
        "n_frames_with_obs": len(per_frame),
        "obs_per_frame": _stats(per_frame.values()),
        "n_objects_seen": len(objects_seen),
        "n_objects_created": len({r["object_id"] for r in created if r.get("object_id")}),
        "matched_per_object": _stats(per_object.values()),
        "cos_sim": _stats(r.get("cos_sim") for r in matched if not r.get("matched_without_scoring")),
        "dist_m": _stats(r.get("dist_m") for r in matched if not r.get("matched_without_scoring")),
        "px_err": _stats(r.get("px_err") for r in matched if not r.get("matched_without_scoring")),
        "range_m": _stats(r.get("range_m") for r in obs),
        "n_nearby": _stats(r.get("n_nearby") for r in obs),
        "n_gate_survivors": _stats(r.get("n_gate_survivors") for r in obs),
        "view_bins": {str(k): int(v) for k, v in sorted(bins.items())},
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
        "observation_summary": observation_summary(rows),
        "frame_outcomes": dict(outcome_histogram(rows)),
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
