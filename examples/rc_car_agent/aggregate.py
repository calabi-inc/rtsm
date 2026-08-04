"""
E1 aggregation — turns a directory of trial JSONLs into the paper's
numbers: TCR, TTA (with conservative censoring), path efficiency, and a
Mann-Whitney U comparison of the two conditions.

    .venv/Scripts/python.exe aggregate.py [--dir paper/demo2_data]
                                          [--tape-success-cm 50] [--json out.json]

Definitions (locked E1 protocol; censoring semantics audited 2026-08-03):
  TCR   Task completion rate judged by the TAPE MEASURE (operator-filled
        `tape_cm` in trial_start), success = tape <= threshold. The
        denominator counts taped trials PLUS untaped failures — dropping
        untaped timeouts would turn the metric into P(success | taped)
        and let the arm with more failures inflate. Untaped ARRIVALS are
        excluded but loudly reported (coverage hole). The robot's own
        arrival belief is reported separately — never the headline.
  TTA   Time-to-arrival. EVERY non-arrived trial (timeout, e-stop,
        frame_reset, ...) is a failure-before-arrival and enters at a cap
        — informative exclusion of one arm's failures would bias the
        very statistic under test.
        * Per-condition medians: failures at the CONDITION's cap
          (60 / 180 s) — descriptive.
        * Mann-Whitney: failures of BOTH arms at the COMMON horizon
          (180 s). Asymmetric caps are anti-conservative for the rank
          test: a failed rtsm trial at 60 would outrank every baseline
          success slower than 60. At the common horizon a memory failure
          ranks at/above every observed baseline arrival — conservative
          against the memory-faster claim.
        * A sensitivity variant over arrivals only is also reported.
  PE    Path efficiency = actual ground-path length (fresh-tick pose
        deltas) / straight-line optimal, arrived trials only, believed-
        frame. For baseline trials the optimal comes from the
        baseline_acquired event (acquisition pose -> chosen target),
        matching the drive-only tick records.
  Protocol conformance: a trial whose recorded budget differs from the
        locked cap by > 2 s (off-protocol pilot run) is EXCLUDED and
        counted — silent contamination is worse than a smaller n.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TIMEOUT_CAP_S = {"rtsm": 60.0, "baseline": 180.0}
COMMON_HORIZON_S = max(TIMEOUT_CAP_S.values())
BUDGET_TOLERANCE_S = 2.0


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

    # PE denominator: memory trials carry plan_pose+target in trial_start;
    # baseline trials carry acquisition pose+target on the acquired event
    # (their trial_start is written BEFORE acquisition, so it has neither).
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

    budget = (start.get("config") or {}).get(
        "timeout_rtsm_s" if condition == "rtsm" else "timeout_baseline_s")
    return {
        "trial_id": start.get("trial_id"),
        "condition": condition,
        "result": end.get("result"),
        "elapsed_s": end.get("elapsed_s"),
        "tta_s": end.get("tta_s"),
        "censored": bool(end.get("censored")),
        "final_dist_m": end.get("final_dist_m"),
        "tape_cm": start.get("tape_cm"),
        "path_len_m": path_len if prev is not None else None,
        "optimal_m": optimal,
        "budget_s": budget,
        "search_time_s": search.get("search_time_s") if search else None,
        "planner_path": (start.get("planner") or {}).get("planner_path"),
    }


def load_trials(trials_dir):
    """-> (on_protocol_trials, off_protocol_count). A trial whose recorded
    budget deviates from the locked cap is excluded, visibly."""
    trials, off_protocol = [], 0
    for p in sorted(Path(trials_dir).glob("*.jsonl")):
        t = parse_trial(p)
        if t is None or t["condition"] not in TIMEOUT_CAP_S:
            continue
        cap = TIMEOUT_CAP_S[t["condition"]]
        if (t["budget_s"] is not None
                and abs(float(t["budget_s"]) - cap) > BUDGET_TOLERANCE_S):
            off_protocol += 1
            continue
        trials.append(t)
    return trials, off_protocol


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


def _tta_values(trials: List[dict], censor_at: float) -> List[float]:
    """Arrivals at their TTA; EVERY other outcome (timeout, e-stop,
    frame_reset, ...) is a failure-before-arrival entering at censor_at."""
    out = []
    for t in trials:
        if t["result"] == "arrived" and t["tta_s"] is not None:
            out.append(min(float(t["tta_s"]), censor_at))
        else:
            out.append(censor_at)
    return out


def _coerce_tape(t) -> Optional[float]:
    try:
        v = float(t["tape_cm"])
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def summarize(trials_dir, tape_success_cm: float = 50.0) -> dict:
    trials, off_protocol = load_trials(trials_dir)
    by_cond: Dict[str, List[dict]] = {"rtsm": [], "baseline": []}
    for t in trials:
        by_cond[t["condition"]].append(t)

    summary: dict = {"n_files": len(trials), "off_protocol_n": off_protocol,
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
        tape_success = [1 for _, v in taped_vals if v <= tape_success_cm]
        tcr_denom = len(taped_vals) + len(untaped_failures)

        ttas = _tta_values(ts, censor_at=cap)
        pes = [t["path_len_m"] / t["optimal_m"] for t in arrived
               if t["path_len_m"] and t["optimal_m"] and t["optimal_m"] > 0.05]
        searches = [t["search_time_s"] for t in ts
                    if t["search_time_s"] is not None]
        paths: Dict[str, int] = {}
        for t in ts:
            key = t["planner_path"] or "unknown"
            paths[key] = paths.get(key, 0) + 1
        summary["conditions"][cond] = {
            "n": len(ts),
            "arrived_believed": len(arrived),
            "censored": sum(1 for t in ts if t["censored"]),
            "aborted": sum(1 for t in ts
                           if t["result"] != "arrived" and not t["censored"]),
            "tcr_tape": (len(tape_success) / tcr_denom) if tcr_denom else None,
            "tcr_tape_denom": tcr_denom,
            "tcr_untaped_arrivals": len(untaped_arrivals),
            "tape_invalid_n": tape_invalid,
            "tcr_believed": (len(arrived) / len(ts)) if ts else None,
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

    # Rank test at the COMMON horizon (conservative, see module docstring).
    mw_all = mann_whitney_u(
        _tta_values(by_cond["rtsm"], censor_at=COMMON_HORIZON_S),
        _tta_values(by_cond["baseline"], censor_at=COMMON_HORIZON_S))
    arrived_r = [float(t["tta_s"]) for t in by_cond["rtsm"]
                 if t["result"] == "arrived" and t["tta_s"] is not None]
    arrived_b = [float(t["tta_s"]) for t in by_cond["baseline"]
                 if t["result"] == "arrived" and t["tta_s"] is not None]
    summary["mann_whitney_tta"] = mw_all
    summary["mann_whitney_tta_arrivals_only"] = mann_whitney_u(arrived_r, arrived_b)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 trial aggregation")
    ap.add_argument("--dir", default="paper/demo2_data")
    ap.add_argument("--tape-success-cm", type=float, default=50.0)
    ap.add_argument("--json", default=None, help="also write summary JSON here")
    args = ap.parse_args()

    s = summarize(args.dir, args.tape_success_cm)
    print(f"trials parsed: {s['n_files']}"
          + (f"  (EXCLUDED {s['off_protocol_n']} off-protocol budget files)"
             if s["off_protocol_n"] else ""))
    for cond, c in s["conditions"].items():
        print(f"\n[{cond}]  n={c['n']}  censored={c['censored']}  aborted={c['aborted']}")
        tape = ("—" if c["tcr_tape"] is None
                else f"{100 * c['tcr_tape']:.0f}% (denom={c['tcr_tape_denom']})")
        print(f"  TCR  tape<= {s['tape_success_cm']:.0f}cm: {tape}"
              f"   believed-arrival: "
              + ("—" if c["tcr_believed"] is None else f"{100 * c['tcr_believed']:.0f}%"))
        if c["tcr_untaped_arrivals"]:
            print(f"  WARNING: {c['tcr_untaped_arrivals']} arrival(s) not yet "
                  "tape-measured — tape TCR is incomplete until they are")
        if c["tape_invalid_n"]:
            print(f"  WARNING: {c['tape_invalid_n']} non-numeric tape_cm "
                  "entr(ies) skipped")
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
    if s["tta_speedup_median"] is not None:
        print(f"median-TTA speedup: {s['tta_speedup_median']:.1f}x")
    if args.json:
        Path(args.json).write_text(
            json.dumps(s, indent=2, allow_nan=False), encoding="utf-8")
        print(f"written {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
