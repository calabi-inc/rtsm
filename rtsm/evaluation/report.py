"""
The free single-bag report (Gate 4.5 plan, P3 task 3): ``metrics.json`` +
``report.md`` over the run directories of one ``rtsm eval`` output.

  write_report(out_dir, cfg=None)  -> (metrics_path, report_path)
  rtsm report <out_dir> [--config/--profile/--set ...]

Every number is computed per run by ``rtsm.evaluation.metrics.compute_metrics``
and then aggregated over the N same-input repeats: the **floor** of a number
is its spread (max - min) across the runs -- what the system shows on
identical input, below which a difference between two bags or two versions
means nothing. Fewer than three runs cannot establish a floor; the report
says so at the top. The report is a pure function of the run directories:
no wall-clock stamps, so ``rtsm report`` reproduces the files byte for byte.

The free / paid line (master plan, 2026-09-18): everything here -- the
per-object, per-cluster and per-frame results as data, the same-bag repeat
comparison -- is core. Comparing against a stored baseline, attribution and
the CI gate live in ``rtsm-eval``.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rtsm.evaluation.metrics import METRICS_SCHEMA, MetricParams, compute_metrics

logger = logging.getLogger(__name__)

REPORT_SCHEMA = 1
MIN_RUNS_FOR_FLOOR = 3
CADENCE_LABEL = {"deployed": "deployed (as_deployed)", "representative": "representative (dense)", "exhaustive": "exhaustive (every_frame)"}


# ───────────────────────────── loading ─────────────────────────────

def run_dirs(out_dir: Path) -> List[Path]:
    """``run_<k>`` directories with both files, in numeric order."""
    dirs = []
    for d in out_dir.glob("run_*"):
        if d.is_dir() and (d / "summary.json").is_file() and (d / "events.jsonl").is_file():
            try:
                k = int(d.name.split("_", 1)[1])
            except ValueError:
                continue
            dirs.append((k, d))
    return [d for _, d in sorted(dirs)]


def run_metrics(run_dir: Path, params: MetricParams) -> dict:
    from rtsm.evaluation.ledger import read_events
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_events(run_dir / "events.jsonl")
    m = compute_metrics(rows, summary, params)
    m["run_dir"] = run_dir.name
    m["run_index"] = summary.get("run_index")
    m["detector"] = summary.get("detector")
    m["wall_s"] = summary.get("wall_s")
    m["aborted"] = summary.get("aborted")
    return m


# ───────────────────────────── aggregation ─────────────────────────────

def aggregate(per_run: Sequence[dict], radius_m: float) -> dict:
    """Floors on every scalar + the clusters matched across runs (run 1 is the reference)."""
    keys: List[str] = []
    seen = set()
    for m in per_run:
        for k in m.get("scalars", {}):
            if k not in seen:
                seen.add(k); keys.append(k)
    scalars: Dict[str, dict] = {}
    for k in keys:
        vals = [m.get("scalars", {}).get(k) for m in per_run]
        nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
        if nums:
            scalars[k] = {"value": _num(vals[0]), "mean": _round(sum(nums) / len(nums)), "min": _round(min(nums)), "max": _round(max(nums)),
                          "spread": _round(max(nums) - min(nums)), "n_runs": len(nums), "values": [_num(v) for v in vals]}
        else:
            scalars[k] = {"value": None, "mean": None, "min": None, "max": None, "spread": None, "n_runs": 0, "values": [None for _ in vals]}
    n_runs = len(per_run)
    ref = per_run[0].get("clusters", []) if per_run else []
    stable: List[dict] = []
    n_all = 0
    for c in ref:
        pos = np.asarray(c["position"], dtype=float)
        found = 1
        for m in per_run[1:]:
            others = m.get("clusters", [])
            if not others:
                continue
            P = np.asarray([o["position"] for o in others], dtype=float)
            d = np.linalg.norm(P - pos[None, :], axis=1)
            if d.size and float(d.min()) <= radius_m:
                found += 1
        if found == n_runs:
            n_all += 1
        stable.append({"cluster": c["cluster"], "in_runs": found, "stable": found == n_runs})
    counts = [len(m.get("clusters", [])) for m in per_run]
    fps = [(m.get("memory") or {}).get("fingerprint") for m in per_run]
    return {
        "n_runs": n_runs, "floor_established": n_runs >= MIN_RUNS_FOR_FLOOR, "scalars": scalars,
        "clusters_across_runs": {"reference_run": (per_run[0].get("run_dir") if per_run else None), "n_reference": len(ref), "n_in_all_runs": n_all,
                                 "n_only_in_some": len(ref) - n_all, "counts_per_run": counts, "per_cluster": stable},
        "fingerprints": fps, "identical_fingerprints": (len(set(fps)) == 1) if fps else None,
    }


def _num(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _round(float(v))
    return v


def _round(v: float) -> float:
    return round(float(v), 6)


# ───────────────────────────── markdown ─────────────────────────────

_PCT_HINTS = ("_rate", "_frac", "lower_bound", "disagreement", "rate_of_created", "frac_tracks")


def _fmt(key: str, v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if any(h in key for h in _PCT_HINTS) and 0.0 <= f <= 1.0:
        return f"{100.0 * f:.1f} %"
    if abs(f) >= 100:
        return f"{f:.1f}"
    return f"{f:.3f}"


def _detector_line(d: Optional[dict]) -> str:
    """The report's attribution of its numbers to a detector."""
    if not d:
        return "RTSM's own segmenter (backend not recorded)"
    if d.get("kind") != "external":
        return f"RTSM's own segmenter, backend `{d.get('backend')}`"
    parts = [f"**external** — `{d.get('topic')}` ({d.get('msgtype')}), scores **{d.get('scores')}**",
             f"{d.get('messages_paired', 0)} messages paired, {d.get('frames_without_detections', 0)} frames without detections, {d.get('messages_unpaired', 0)} messages unmatched"]
    if d.get("scores") in ("absent", "mixed"):
        parts.append(f"unscored labels stored with prior {d.get('unscored_label_prior')} (the ledger records `score: null`; label-confidence numbers are not measured)")
    if d.get("dropped"):
        parts.append("dropped: " + ", ".join(f"{k} {v}" for k, v in d["dropped"].items()))
    return "; ".join(parts)


def _cell(agg: dict, key: str, label: Optional[str] = None) -> str:
    s = agg["scalars"].get(key)
    if s is None or s["n_runs"] == 0:
        return f"| {label or key} | – | – |"
    floor = _fmt(key, s["spread"]) if s["n_runs"] >= MIN_RUNS_FOR_FLOOR else f"insufficient (n={s['n_runs']})"
    return f"| {label or key} | {_fmt(key, s['value'])} | {floor} |"


def _table(agg: dict, rows: Sequence[Tuple[str, str]]) -> List[str]:
    out = ["| metric | value (run 1) | floor (spread over runs) |", "|---|---|---|"]
    out += [_cell(agg, k, lbl) for k, lbl in rows]
    return out


def render_markdown(agg: dict, ref: dict, *, resolved: Optional[dict], input_name: str, params: MetricParams, repeats: Optional[dict]) -> str:
    n = agg["n_runs"]
    r = resolved or {}
    cadence = r.get("cadence") or ("deployed" if r.get("mode") == "as_deployed" else "representative" if r.get("mode") == "dense" else "exhaustive" if r.get("mode") == "every_frame" else "unknown")
    kf = r.get("keyframe_rule") or {}
    kf_text = (f"every {kf.get('n')} admitted frames" if kf.get("kind") == "every_n" else f"every {kf.get('interval_s')} s of sensor time" if kf.get("kind") == "interval"
               else f"the source's own keyframes" if kf.get("kind") == "source" else str(kf) if kf else "–")
    L: List[str] = []
    L.append(f"# rtsm eval report — `{input_name}`")
    L.append("")
    if not agg["floor_established"]:
        L.append(f"> **Floor not established: {n} run(s).** Every number below needs at least {MIN_RUNS_FOR_FLOOR} same-input runs to carry a floor "
                 f"(`rtsm eval … --repeats {MIN_RUNS_FOR_FLOOR}`). Read the values as one sample.")
        L.append("")
    L.append("## Run")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| input | `{r.get('input', input_name)}` ({r.get('input_kind', '–')}) |")
    L.append(f"| mode / cadence | `{r.get('mode', '–')}` — **{CADENCE_LABEL.get(cadence, cadence)}** |")
    L.append(f"| keyframe rule | {kf_text} |")
    L.append(f"| non-keyframe throttle | {r.get('nonkf_min_interval_s', '–')} s on the {r.get('clock', '–')} clock |")
    L.append(f"| sweep gate | {r.get('gate_mode', '–')}{' (shadow: every processed frame also reported restricted to the frames the deployed gate would have admitted — *masked*)' if ref.get('shadow_mode') else ''} |")
    L.append(f"| ingest policy | {r.get('policy', '–')} |")
    L.append(f"| repeats | {n}{' — fingerprints identical' if agg.get('identical_fingerprints') else ' — **fingerprints differ**' if agg.get('identical_fingerprints') is False else ''} |")
    L.append(f"| config fingerprint | `{r.get('config_fingerprint', '–')}` |")
    dirty = r.get("git_dirty")
    commit_txt = f"{r.get('git_commit', '–')}" + (f" (dirty, diff {r.get('tree_digest')})" if dirty else "")
    L.append(f"| commit / rtsm / python | `{commit_txt}` / {r.get('rtsm_version', '–')} / {r.get('python', '–')} |")
    L.append(f"| cluster radius | {params.cluster_radius_m} m (the associator's distance gate); sensitivity at {', '.join(f'{x:g}' for x in params.cluster_radii_m)} m |")
    L.append(f"| detector | {_detector_line(ref.get('detector'))} |")
    if repeats:
        L.append(f"| wall time per run | {', '.join(f'{w:.1f} s' for w in repeats.get('wall_s', []) if isinstance(w, (int, float)))} |")
    L.append("")
    L.append(f"The floor of a number is its spread over the {n} same-input runs: a difference between two bags or two versions smaller than the floor means nothing.")
    L.append("")

    L.append("## Frames and admission")
    L.append("")
    L += _table(agg, [("admission.receiver_lines", "sensor frames seen"), ("admission.enqueued", "admitted (enqueued)"), ("admission.throttled", "throttled"),
                      ("admission.throttled_frac", "throttled fraction"), ("admission.tracking_dropped", "dropped: tracking not normal"),
                      ("admission.processed", "processed"), ("admission.keyframes_processed", "keyframes processed"), ("admission.gate_rejected", "sweep gate rejected"),
                      ("admission.frame_rejected", "frame-quality rejected"), ("admission.processed_gap_p50_s", "gap between processed frames p50 (s)"),
                      ("admission.processed_gap_p95_s", "gap between processed frames p95 (s)"), ("admission.longest_unprocessed_s", "longest unprocessed stretch (s)"),
                      ("admission.processed_span_s", "processed span (s)")] +
               ([("admission.shadowed", "frames the deployed gate would have rejected"), ("admission.shadowed_frac", "… fraction of processed")] if ref.get("shadow_mode") else []))
    L.append("")
    adm = ref.get("admission") or {}
    if adm.get("dequeue"):
        L.append("Dequeue outcomes (run 1): " + ", ".join(f"`{k}` {v}" for k, v in adm["dequeue"].items()) + ".")
        L.append("")

    L.append("## Pose stream")
    L.append("")
    L += _table(agg, [("pose.n_frames", "pose lines"), ("pose.sensor_hz", "sensor rate (Hz)"), ("pose.span_s", "span (s)"), ("pose.jitter_ms", "stamp jitter (ms)"),
                      ("pose.n_gaps", "gaps"), ("pose.n_limited_episodes", "tracking-limited episodes"), ("pose.limited_frames_frac", "tracking-limited fraction"),
                      ("pose.n_discontinuities", "discontinuities (> 0.5 m + 1 m/s·dt)"), ("pose.pose_errors", "pose parse errors"),
                      ("pose.depth_valid_frac_p50", "depth valid fraction p50"), ("pose.conf2_frac_p50", "confidence-2 fraction p50")])
    L.append("")

    L.append("## Memory")
    L.append("")
    L += _table(agg, [("memory.objects", "objects at the end"), ("memory.confirmed", "confirmed"), ("memory.ids_created", "objects created over the run"),
                      ("memory.transient", "transient (created, not in the final memory)"), ("memory.survivors_single_hit", "survivors with a single hit"),
                      ("memory.matched_without_scoring", "matches made without scoring (associator fallback)")])
    mem = ref.get("memory") or {}
    L.append("")
    L.append(f"Fingerprint (run 1): `{mem.get('fingerprint')}`. Hits histogram: " + ", ".join(f"{k}: {v}" for k, v in (mem.get('hits_hist') or {}).items()) +
             ". View bins per object: " + ", ".join(f"{k}: {v}" for k, v in (mem.get('view_bins_hist') or {}).items()) + ".")
    L.append("")

    L.append("## Spatial clusters (proxy for physical objects)")
    L.append("")
    L += _table(agg, [("clusters.n", f"clusters at {params.cluster_radius_m:g} m"), ("clusters.n_with_survivor", "… with at least one surviving object"),
                      ("clusters.n_multi_member", "… with more than one object"), ("clusters.duplicates_all", "duplicate objects over the run (ids − clusters)"),
                      ("clusters.duplicates_survivors", "duplicate objects in the final memory")] +
               [(f"clusters.n_at_radius.{x:g}", f"clusters at {x:g} m (sensitivity)") for x in params.cluster_radii_m])
    L.append("")
    L.append("A cluster groups objects whose median raw positions lie within the radius (leader clustering in creation order, no chaining). "
             "Two real objects closer than the radius fall into one cluster; a duplicate spawn farther than the radius is not seen. The sensitivity rows bound that.")
    L.append("")
    cross = agg.get("clusters_across_runs") or {}
    if n > 1:
        L.append(f"Across the {n} runs: {cross.get('n_in_all_runs')} of run 1's {cross.get('n_reference')} clusters are present in every run, "
                 f"{cross.get('n_only_in_some')} only in some (cluster counts per run: {cross.get('counts_per_run')}).")
        L.append("")

    L.append("## Detection over in-frustum views")
    L.append("")
    L += _table(agg, [("detection.views", "views (cluster in the frustum on a processed frame)"), ("detection.reidentified", "re-identified (a member matched)"),
                      ("detection.duplicated", "duplicated (a member created instead)"), ("detection.missed", "missed"),
                      ("detection.detection_rate", "detection rate (re-identified + duplicated)"), ("detection.reid_rate", "re-identification rate")] +
               ([("detection.masked.views", "masked: views"), ("detection.masked.reid_rate", "masked: re-identification rate"), ("detection.masked.detection_rate", "masked: detection rate")] if ref.get("shadow_mode") else []))
    L.append("")
    det = ref.get("detection") or {}
    if det.get("by_range"):
        L.append("By expected range (object level, run 1):")
        L.append("")
        L.append("| range (m) | views | re-identified | rate |" + (" masked views | masked rate |" if ref.get("shadow_mode") else ""))
        L.append("|---|---|---|---|" + ("---|---|" if ref.get("shadow_mode") else ""))
        for b, row in det["by_range"].items():
            line = f"| {b} | {row['views']} | {row['reidentified']} | {_fmt('rate', row['reid_rate'])} |"
            if ref.get("shadow_mode"):
                line += f" {row.get('views_masked', 0)} | {_fmt('rate', row.get('reid_rate_masked'))} |"
            L.append(line)
        L.append("")
    L.append("The frustum model is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate here is a lower bound on what the perception could see.")
    L.append("")

    L.append("## Label disagreement")
    L.append("")
    L += _table(agg, [("labels.tracks_ge2", "objects with ≥ 2 labelled observations"), ("labels.disagreement_mean", "disagreement mean (1 − modal share)"),
                      ("labels.disagreement_p50", "disagreement p50"), ("labels.disagreement_p95", "disagreement p95"), ("labels.frac_tracks_disagreeing", "objects whose observations disagree"),
                      ("labels.clusters_with_survivor_label_conflict", "clusters whose surviving objects carry different primary labels")])
    L.append("")

    L.append("## Position scatter (raw observations vs the object median)")
    L.append("")
    L += _table(agg, [("scatter.n_tracks", f"objects with ≥ {params.min_obs_scatter} observations"), ("scatter.n_obs", "observations"),
                      ("scatter.along_rms_m", "along-ray RMS (m)"), ("scatter.lateral_rms_m", "lateral RMS (m)"), ("scatter.along_over_lateral", "along / lateral"),
                      ("scatter.along_slope_per_m", "|along| vs range: slope (m per m)"), ("scatter.along_intercept_m", "|along| vs range: intercept (m)"), ("scatter.along_r2", "|along| vs range: R²"),
                      ("scatter.lateral_slope_per_m", "lateral vs range: slope (m per m)")])
    L.append("")
    sc = ref.get("scatter") or {}
    if sc.get("by_range"):
        L.append("| range (m) | n | along RMS (m) | lateral RMS (m) |")
        L.append("|---|---|---|---|")
        for b, row in sc["by_range"].items():
            L.append(f"| {b} | {row['n']} | {_fmt('m', row['along_rms_m'])} | {_fmt('m', row['lateral_rms_m'])} |")
        L.append("")

    L.append("## Duplicate spawns")
    L.append("")
    L += _table(agg, [("duplicates.n", "objects created within the radius of a live object"), ("duplicates.rate_of_created", "… as a fraction of created objects"),
                      ("duplicates.original_in_view", "… while the original was in the frustum"), ("duplicates.reason.not_in_index", "reason: original not returned by the index"),
                      ("duplicates.reason.gated", "reason: nearby but failed the distance / z / reprojection gates"), ("duplicates.reason.low_similarity", f"reason: passed the gates, cosine below {params.cos_min:g}"),
                      ("duplicates.reason.other", "reason: other")])
    L.append("")
    L.append(f"Alive = a surviving object, or one observed less than {params.proto_ttl_s:g} s earlier (the proto TTL). The reason classes read the created line's audit fields (`n_nearby`, `n_gate_survivors`, `max_cos`).")
    L.append("")

    L.append("## Revisits")
    L.append("")
    L += _table(agg, [("revisits.clusters_with_revisit", "clusters seen in ≥ 2 visits"), ("revisits.n_revisits", "revisits"), ("revisits.reidentified", "re-identified on return"),
                      ("revisits.duplicated", "duplicated on return"), ("revisits.missed", "missed on return"), ("revisits.reid_lower_bound", "re-identification lower bound on revisits")])
    L.append("")
    L.append(f"A visit ends when the cluster leaves the frustum for ≥ {params.revisit_gap_s:g} s of sensor time. Missed and duplicated revisits count as failures, so the bound is a lower bound (occlusion is a miss here).")
    L.append("")

    L.append("## Worst moments (run 1)")
    L.append("")
    L.append("| t (s) | frame | kind | score | detail |")
    L.append("|---|---|---|---|---|")
    for m in ref.get("worst_moments") or []:
        detail = ", ".join(f"{k} {v}" for k, v in (m.get("detail") or {}).items() if v not in (None, 0, "", {}))
        L.append(f"| {m.get('t_rel_s')} | {m.get('frame_seq') if m.get('frame_seq') is not None else '–'} | {m.get('kind')} | {_fmt('score', m.get('score'))} | {detail} |")
    L.append("")
    L.append("Score = missed clusters + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame); pose gaps / discontinuities / tracking-limited episodes score by their size. `t` is sensor time since the first pose line.")
    L.append("")

    L.append("## Largest clusters (run 1)")
    L.append("")
    L.append("| cluster | objects (surviving) | observations | first–last seen (s) | re-id rate | visits | top labels | surviving labels |")
    L.append("|---|---|---|---|---|---|---|---|")
    clusters = sorted(ref.get("clusters") or [], key=lambda c: (-c["n_obs"], c["cluster"]))[:15]
    for c in clusters:
        labs = ", ".join(f"{t['label']} ({t['n']})" for t in (c.get("labels") or {}).get("top", [])[:3])
        surv = ", ".join((c.get("labels") or {}).get("survivor_label_primary") or []) or "–"
        L.append(f"| {c['cluster']} | {c['n_members']} ({c['n_survivors']}) | {c['n_obs']} | {c.get('first_seen_s')}–{c.get('last_seen_s')} | "
                 f"{_fmt('rate', c['detection']['reid_rate'])} | {c['visits']['n_visits']} | {labs} | {surv} |")
    L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Everything above is computed from the run's ledgers (`events.jsonl`: pose / obs / view lines, the frame-flow trace) and the final memory in `summary.json`. No ground truth, no labels required; label numbers use the detector's top-1 label per observation.")
    L.append("- Objects = every id the associator matched or created; a **transient** object was created and is not in the final memory (a proto that expired, or one the memory evicted).")
    L.append("- **Engine-confirmed** (\"confirmed\" above) means the memory promoted the object under its own rules (hits, stability, view bins); it is not ground-truth confirmation. **Re-identified** means the associator matched an existing object (a reassociation), not a verified identity.")
    L.append("- The fingerprint is the sha256 of the sorted multiset of (label_primary, xyz rounded to 3 decimals, hits, confirmed), 16 hex characters: identical fingerprints mean the same final memory at that resolution, not bit-identical runs.")
    L.append("- Raw observations (`p_world` as the associator computed it) are used everywhere; the memory's smoothed positions appear only in the fingerprint.")
    L.append("- Per-object, per-cluster, per-frame and per-duplicate records are in `metrics.json` (`objects`, `clusters`, `frames`, `duplicate_spawns`, `worst_moments`); each object and cluster carries the sensor stamps of the frames it was seen and missed on.")
    L.append(f"- Parameters: cluster radius {params.cluster_radius_m:g} m, revisit gap {params.revisit_gap_s:g} s, range bin {params.range_bin_m:g} m, scatter needs ≥ {params.min_obs_scatter} observations, proto TTL {params.proto_ttl_s:g} s, cosine threshold {params.cos_min:g}.")
    L.append("")
    return "\n".join(L)


# ───────────────────────────── writing ─────────────────────────────

def write_report(out_dir: Any, cfg: Optional[dict] = None, *, params: Optional[MetricParams] = None) -> Tuple[Path, Path]:
    """``metrics.json`` + ``report.md`` under ``out_dir`` from its ``run_*`` directories."""
    out = Path(out_dir)
    dirs = run_dirs(out)
    if not dirs:
        raise ValueError(f"{out}: no run_<k> directory with summary.json + events.jsonl")
    if params is None:
        if cfg is None:
            from rtsm.cfg import load_config
            cfg = load_config()
        params = MetricParams.from_cfg(cfg)
    resolved = None
    if (out / "resolved.json").is_file():
        try:
            resolved = json.loads((out / "resolved.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- provenance only
            resolved = None
    repeats = None
    if (out / "repeats.json").is_file():
        try:
            repeats = json.loads((out / "repeats.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            repeats = None
    per_run = [run_metrics(d, params) for d in dirs]
    agg = aggregate(per_run, params.cluster_radius_m)
    input_name = (resolved or {}).get("input") or (per_run[0].get("run_dir") if per_run else out.name)
    input_name = Path(str(input_name)).name
    doc = {"schema": REPORT_SCHEMA, "metrics_schema": METRICS_SCHEMA, "input": (resolved or {}).get("input"), "resolved": resolved,
           "params": per_run[0]["params"], "n_runs": len(per_run), "aggregate": agg, "runs": per_run}
    metrics_path = out / "metrics.json"
    report_path = out / "report.md"
    metrics_path.write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8", newline="\n")
    report_path.write_text(render_markdown(agg, per_run[0], resolved=resolved, input_name=input_name, params=params, repeats=repeats),
                           encoding="utf-8", newline="\n")
    return metrics_path, report_path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ───────────────────────────── CLI ─────────────────────────────

def build_parser():
    import argparse
    from rtsm.cfg.cli import add_config_arguments
    ap = argparse.ArgumentParser(prog="rtsm report", description="Write metrics.json + report.md from the run directories of an `rtsm eval` output (no models, no GPU).")
    ap.add_argument("out_dir", help="an `rtsm eval` output directory (holds run_1/, run_2/, …)")
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
    if not Path(args.out_dir).is_dir():
        ap.error(f"not a directory: {args.out_dir}")
    try:
        metrics_path, report_path = write_report(args.out_dir, cfg)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"metrics: {metrics_path}\nreport:  {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
