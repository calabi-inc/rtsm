"""
E1 aggregation — turns a directory of trial JSONLs into the paper's
numbers: TCR, TTA (with conservative censoring), path efficiency, and a
Mann-Whitney U comparison of the two conditions.

    .venv/Scripts/python.exe aggregate.py [--dir paper/demo2_data]
                                          [--tape-success-cm 50] [--json out.json]

Definitions (locked E1 protocol; censoring + exclusion semantics audited
2026-08-06, pre-registered before any campaign data):
  Exclusions (loudly counted, never silent):
        * trials whose trial_start `notes` begin with "INVALID"
          (operator-declared: bystander contact, wrong condition, ...)
        * results {cancelled, preempted, shutdown} — operator/system
          interventions with no protocol-legitimate mid-trial meaning
        * uncalibrated trials (is_calibrated false / rig_id null)
        * off-protocol budgets (recorded budget != locked cap)
  TCR   HEADLINE: verdict == "arrived" AND tape <= threshold (50 cm =
        40 cm controller stop radius + 10 cm centroid-vs-floor-cross
        projection & reading allowance, fixed pre-data). A safety-stop or
        timeout that happens to halt near the object is NOT a completion.
        Also reported: tape-only (any verdict) as a secondary, a 40/50/60
        cm sensitivity sweep, and the robot's believed-arrival rate
        (never the headline — self-grading is circular). Denominator
        counts taped trials PLUS untaped failures; untaped ARRIVALS are
        loudly reported as a coverage hole.
  TTA   EVERY non-arrived trial is a failure-before-arrival at a cap.
        Per-condition medians: failures at the CONDITION's cap (60/180 s)
        — descriptive. Mann-Whitney: failures of BOTH arms at the COMMON
        horizon (180 s) — asymmetric caps are anti-conservative for the
        one-sided rank test. Arrivals-only sensitivity variant reported.
        Budgets are hard TOTAL clocks in both conditions (planning and
        search both count).
  Clustering  Trials share layout geometry (6/condition/layout); the
        pooled MW p is reported alongside a pre-specified layout-level
        robustness check: per-layout win table + exact one-sided sign
        test across layouts (memory-median < baseline-median).
  Terminal-error decomposition  per-condition |believed final distance −
        tape| and believed-arrived-but-tape-failed counts — separates
        terminal coordinate quality from search success.
  PE    Path efficiency = actual ground-path length / straight-line
        optimal, arrived trials only, believed-frame. Baseline optimal
        comes from the baseline_acquired event (acquisition pose ->
        chosen target), matching the drive-only tick records.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TIMEOUT_CAP_S = {"rtsm": 60.0, "baseline": 180.0}
COMMON_HORIZON_S = max(TIMEOUT_CAP_S.values())
BUDGET_TOLERANCE_S = 2.0
TCR_SWEEP_CM = (40.0, 50.0, 60.0)
EXCLUDED_RESULTS = {"cancelled", "preempted", "shutdown"}


# ── trial parsing ────────────────────────────────────────────────────────


def parse_trial(path: Path) -> Optional[dict]:
    """One JSONL -> flat dict, or None if malformed/incomplete."""
    try:
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    starts = [r for r in records if r.get("type") == "trial_start"]
    ends = [r for r in records if r.get("type") == "trial_end"]
    if not starts or not ends:
        return None
    start, end = starts[0], ends[-1]
    condition = start.get("condition")
    cfg = start.get("config") or {}
    ticks = [r for r in records if r.get("type") == "tick"]
    events = [r for r in records if r.get("type") == "event"]

    path_len = 0.0
    prev = None
    for t in ticks:
        pose = t.get("pose")
        if not t.get("fresh") or not pose:
            continue
        xyz = pose.get("xyz")
        if prev is not None and xyz is not None:
            path_len += math.hypot(xyz[0] - prev[0], xyz[2] - prev[2])
        prev = xyz

    optimal = None
    plan_pose = start.get("plan_pose")
    target = (start.get("planner") or {}).get("xyz_world")
    if plan_pose and plan_pose.get("xyz") and target:
        optimal = math.hypot(target[0] - plan_pose["xyz"][0],
                             target[2] - plan_pose["xyz"][2])
    search = next((e for e in events if e.get("name") == "baseline_acquired"), None)
    if optimal is None and search is not None:
        ev_pose = (search.get("pose") or {}).get("xyz")
        ev_target = search.get("xyz_world")
        if ev_pose and ev_target:
            optimal = math.hypot(ev_target[0] - ev_pose[0],
                                 ev_target[2] - ev_pose[2])

    budget = cfg.get(
        "timeout_rtsm_s" if condition == "rtsm" else "timeout_baseline_s")
    return {
        "trial_id": start.get("trial_id"),
        "condition": condition,
        "layout_id": start.get("layout_id"),
        "session_id": start.get("session_id"),
        "notes": start.get("notes"),
        "result": end.get("result"),
        "elapsed_s": end.get("elapsed_s"),
        "tta_s": end.get("tta_s"),
        "censored": bool(end.get("censored")),
        "final_dist_m": end.get("final_dist_m"),
        "tape_cm": start.get("tape_cm"),
        "path_len_m": path_len if prev is not None else None,
        "optimal_m": optimal,
        "budget_s": budget,
        "is_calibrated": cfg.get("is_calibrated"),
        "rig_id": cfg.get("rig_id"),
        "search_time_s": search.get("search_time_s") if search else None,
        "planner_path": (start.get("planner") or {}).get("planner_path"),
    }


def load_trials(trials_dir) -> Tuple[List[dict], Dict[str, int]]:
    """-> (analysis_trials, excluded_counts). Every exclusion is counted;
    silent contamination is worse than a smaller n."""
    trials = []
    excluded = {"off_protocol": 0, "invalid_marked": 0,
                "operator_result": 0, "uncalibrated": 0}
    for p in sorted(Path(trials_dir).glob("*.jsonl")):
        t = parse_trial(p)
        if t is None or t["condition"] not in TIMEOUT_CAP_S:
            continue
        notes = t["notes"]
        if isinstance(notes, str) and notes.strip().upper().startswith("INVALID"):
            excluded["invalid_marked"] += 1
            continue
        if t["result"] in EXCLUDED_RESULTS:
            excluded["operator_result"] += 1
            continue
        if not t["is_calibrated"] or not t["rig_id"]:
            excluded["uncalibrated"] += 1
            continue
        cap = TIMEOUT_CAP_S[t["condition"]]
        if (t["budget_s"] is not None
                and abs(float(t["budget_s"]) - cap) > BUDGET_TOLERANCE_S):
            excluded["off_protocol"] += 1
            continue
        trials.append(t)
    return trials, excluded


# ── statistics ───────────────────────────────────────────────────────────


def mann_whitney_u(a, b) -> Dict[str, Optional[float]]:
    """One-sided MW U (alternative: a stochastically SMALLER than b).
    Normal approximation with tie correction + continuity correction.
    Returns None values (never NaN) when a side is empty."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"U": None, "p_one_sided": None}
    combined = np.concatenate([a, b])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1)
    for val in np.unique(combined):
        mask = combined == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    U1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(counts.astype(float) ** 3 - counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"U": U1, "p_one_sided": 1.0 if U1 >= mu else 0.0}
    z = (U1 + 0.5 - mu) / math.sqrt(var)
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))   # Phi(z): small U -> small p
    return {"U": U1, "p_one_sided": p}


def sign_test_by_layout(trials: List[dict]) -> dict:
    """Pre-specified clustering-robust check: per layout, compare the two
    conditions' median common-horizon TTA; exact one-sided binomial on
    'memory wins' across layouts (ties dropped)."""
    layouts: Dict[str, Dict[str, List[float]]] = {}
    for t in trials:
        lid = t["layout_id"]
        if lid is None:
            continue
        layouts.setdefault(lid, {"rtsm": [], "baseline": []})[t["condition"]].append(
            _tta_value(t, COMMON_HORIZON_S))
    table = {}
    wins = losses = 0
    for lid in sorted(layouts):
        r, b = layouts[lid]["rtsm"], layouts[lid]["baseline"]
        if not r or not b:
            table[lid] = {"rtsm_median": float(np.median(r)) if r else None,
                          "baseline_median": float(np.median(b)) if b else None,
                          "winner": None}
            continue
        rm, bm = float(np.median(r)), float(np.median(b))
        winner = "rtsm" if rm < bm else ("baseline" if bm < rm else "tie")
        table[lid] = {"rtsm_median": rm, "baseline_median": bm, "winner": winner}
        if winner == "rtsm":
            wins += 1
        elif winner == "baseline":
            losses += 1
    m = wins + losses
    p = (sum(math.comb(m, i) for i in range(wins, m + 1)) / (2 ** m)
         if m > 0 else None)
    return {"per_layout": table, "memory_wins": wins, "of_layouts": m,
            "p_one_sided": p}


def _tta_value(t: dict, censor_at: float) -> float:
    if t["result"] == "arrived" and t["tta_s"] is not None:
        return min(float(t["tta_s"]), censor_at)
    return censor_at


def _tta_values(trials: List[dict], censor_at: float) -> List[float]:
    """Arrivals at their TTA; EVERY other outcome (timeout, e-stop,
    frame_reset, ...) is a failure-before-arrival entering at censor_at."""
    return [_tta_value(t, censor_at) for t in trials]


def _coerce_tape(t) -> Optional[float]:
    try:
        v = float(t["tape_cm"])
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def summarize(trials_dir, tape_success_cm: float = 50.0) -> dict:
    trials, excluded = load_trials(trials_dir)
    by_cond: Dict[str, List[dict]] = {"rtsm": [], "baseline": []}
    for t in trials:
        by_cond[t["condition"]].append(t)

    summary: dict = {"n_files": len(trials), "excluded": excluded,
                     "off_protocol_n": excluded["off_protocol"],   # back-compat
                     "tape_success_cm": tape_success_cm, "conditions": {}}
    for cond, ts in by_cond.items():
        cap = TIMEOUT_CAP_S[cond]
        arrived = [t for t in ts if t["result"] == "arrived"]
        taped_vals = []
        tape_invalid = 0
        for t in ts:
            if t["tape_cm"] is None:
                continue
            v = _coerce_tape(t)
            if v is None:
                tape_invalid += 1
            else:
                taped_vals.append((t, v))
        untaped = [t for t in ts if t["tape_cm"] is None]
        untaped_failures = [t for t in untaped
                            if t["censored"] or t["result"] != "arrived"]
        untaped_arrivals = [t for t in untaped
                            if not t["censored"] and t["result"] == "arrived"]
        tcr_denom = len(taped_vals) + len(untaped_failures)

        # HEADLINE: completion requires the monitor's arrival verdict AND
        # the tape inside the threshold. Sensitivity sweep + any-verdict
        # secondary reported alongside.
        def _tcr(thr: float, require_arrived: bool) -> Optional[float]:
            if not tcr_denom:
                return None
            good = sum(1 for t, v in taped_vals
                       if v <= thr and (t["result"] == "arrived"
                                        or not require_arrived))
            return good / tcr_denom

        terrs = [abs(float(t["final_dist_m"]) - v / 100.0)
                 for t, v in taped_vals if t["final_dist_m"] is not None]
        flips = sum(1 for t, v in taped_vals
                    if t["result"] == "arrived" and v > tape_success_cm)

        ttas = _tta_values(ts, censor_at=cap)
        pes = [t["path_len_m"] / t["optimal_m"] for t in arrived
               if t["path_len_m"] and t["optimal_m"] and t["optimal_m"] > 0.05]
        searches = [t["search_time_s"] for t in ts
                    if t["search_time_s"] is not None]
        paths: Dict[str, int] = {}
        results: Dict[str, int] = {}
        for t in ts:
            paths[t["planner_path"] or "unknown"] = paths.get(
                t["planner_path"] or "unknown", 0) + 1
            results[t["result"] or "unknown"] = results.get(
                t["result"] or "unknown", 0) + 1
        summary["conditions"][cond] = {
            "n": len(ts),
            "arrived_believed": len(arrived),
            "censored": sum(1 for t in ts if t["censored"]),
            "aborted": sum(1 for t in ts
                           if t["result"] != "arrived" and not t["censored"]),
            "results": results,
            "tcr_tape": _tcr(tape_success_cm, require_arrived=True),
            "tcr_tape_any_verdict": _tcr(tape_success_cm, require_arrived=False),
            "tcr_sweep": {f"{thr:.0f}cm": _tcr(thr, require_arrived=True)
                          for thr in TCR_SWEEP_CM},
            "tcr_tape_denom": tcr_denom,
            "tcr_untaped_arrivals": len(untaped_arrivals),
            "tape_invalid_n": tape_invalid,
            "tcr_believed": (len(arrived) / len(ts)) if ts else None,
            "terminal_err_median_m": (float(np.median(terrs)) if terrs else None),
            "terminal_err_iqr_m": ([float(np.percentile(terrs, 25)),
                                    float(np.percentile(terrs, 75))]
                                   if terrs else None),
            "believed_arrived_tape_failed": flips,
            "tta_median_s": float(np.median(ttas)) if ttas else None,
            "tta_iqr_s": ([float(np.percentile(ttas, 25)),
                           float(np.percentile(ttas, 75))] if ttas else None),
            "tta_n": len(ttas),
            "pe_median": float(np.median(pes)) if pes else None,
            "pe_n": len(pes),
            "search_time_median_s": (float(np.median(searches))
                                     if searches else None),
            "planner_paths": paths,
        }

    mw_all = mann_whitney_u(
        _tta_values(by_cond["rtsm"], censor_at=COMMON_HORIZON_S),
        _tta_values(by_cond["baseline"], censor_at=COMMON_HORIZON_S))
    arrived_r = [float(t["tta_s"]) for t in by_cond["rtsm"]
                 if t["result"] == "arrived" and t["tta_s"] is not None]
    arrived_b = [float(t["tta_s"]) for t in by_cond["baseline"]
                 if t["result"] == "arrived" and t["tta_s"] is not None]
    summary["mann_whitney_tta"] = mw_all
    summary["mann_whitney_tta_arrivals_only"] = mann_whitney_u(arrived_r, arrived_b)
    summary["sign_test_by_layout"] = sign_test_by_layout(trials)
    r_med = summary["conditions"]["rtsm"]["tta_median_s"]
    b_med = summary["conditions"]["baseline"]["tta_median_s"]
    summary["tta_speedup_median"] = (b_med / r_med
                                     if r_med and b_med and r_med > 0 else None)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────


def _fmt_p(mw) -> str:
    if mw["U"] is None:
        return "— (one side has no data)"
    return f"U={mw['U']:.1f}  p={mw['p_one_sided']:.4f}"


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 trial aggregation")
    ap.add_argument("--dir", default="paper/demo2_data")
    ap.add_argument("--tape-success-cm", type=float, default=50.0)
    ap.add_argument("--json", default=None, help="also write summary JSON here")
    args = ap.parse_args()

    s = summarize(args.dir, args.tape_success_cm)
    ex = s["excluded"]
    print(f"trials in analysis: {s['n_files']}")
    if any(ex.values()):
        print(f"EXCLUDED: {ex['invalid_marked']} INVALID-marked, "
              f"{ex['operator_result']} cancelled/preempted/shutdown, "
              f"{ex['uncalibrated']} uncalibrated, "
              f"{ex['off_protocol']} off-protocol budget "
              "(reconcile against the session sheet)")
    for cond, c in s["conditions"].items():
        print(f"\n[{cond}]  n={c['n']}  censored={c['censored']}  aborted={c['aborted']}")
        print(f"  results: {c['results']}")
        print(f"  TCR HEADLINE (arrived AND tape<= {s['tape_success_cm']:.0f}cm): "
              f"{_pct(c['tcr_tape'])} (denom={c['tcr_tape_denom']})"
              f"   any-verdict: {_pct(c['tcr_tape_any_verdict'])}"
              f"   believed: {_pct(c['tcr_believed'])}")
        print(f"  TCR sweep: " + "  ".join(f"{k}={_pct(v)}"
                                           for k, v in c["tcr_sweep"].items()))
        if c["tcr_untaped_arrivals"]:
            print(f"  WARNING: {c['tcr_untaped_arrivals']} arrival(s) not yet "
                  "tape-measured — tape TCR is incomplete until they are")
        if c["tape_invalid_n"]:
            print(f"  WARNING: {c['tape_invalid_n']} non-numeric tape_cm "
                  "entr(ies) skipped")
        if c["terminal_err_median_m"] is not None:
            print(f"  terminal |believed−tape|: median {c['terminal_err_median_m']:.3f} m"
                  f"  IQR {c['terminal_err_iqr_m']}"
                  f"  believed-arrived-but-tape-failed: {c['believed_arrived_tape_failed']}")
        if c["tta_median_s"] is not None:
            print(f"  TTA  median {c['tta_median_s']:.1f} s  IQR {c['tta_iqr_s']}"
                  f"  (n={c['tta_n']}, failures at the {TIMEOUT_CAP_S[cond]:.0f}s cap)")
        if c["pe_median"] is not None:
            print(f"  PE   median {c['pe_median']:.2f} (n={c['pe_n']}, "
                  "arrived only, believed-frame)")
        if c["search_time_median_s"] is not None:
            print(f"  search median {c['search_time_median_s']:.1f} s")
        print(f"  planner paths: {c['planner_paths']}")
    print(f"\nMann-Whitney (TTA, one-sided memory-faster, common {COMMON_HORIZON_S:.0f}s"
          f" horizon): {_fmt_p(s['mann_whitney_tta'])}")
    print(f"  sensitivity (arrivals only): {_fmt_p(s['mann_whitney_tta_arrivals_only'])}")
    st = s["sign_test_by_layout"]
    if st["of_layouts"]:
        print(f"  layout-level sign test: memory wins {st['memory_wins']}/{st['of_layouts']}"
              f"  p={st['p_one_sided']:.4f}")
        for lid, row in st["per_layout"].items():
            print(f"    {lid}: rtsm {row['rtsm_median']}s vs baseline "
                  f"{row['baseline_median']}s -> {row['winner']}")
    if s["tta_speedup_median"] is not None:
        print(f"median-TTA speedup: {s['tta_speedup_median']:.1f}x")
    if args.json:
        Path(args.json).write_text(
            json.dumps(s, indent=2, allow_nan=False), encoding="utf-8")
        print(f"written {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
