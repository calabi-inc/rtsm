"""
E1 aggregation — turns a directory of trial JSONLs into the paper's
numbers: TCR, TTA (with conservative censoring), path efficiency, and a
Mann-Whitney U comparison of the two conditions.

    .venv/Scripts/python.exe aggregate.py [--dir paper/demo2_data]
                                          [--tape-success-cm 50] [--json out.json]

Definitions (locked E1 protocol):
  TCR   Task completion rate judged by the TAPE MEASURE (operator-filled
        `tape_cm` in each trial_start record), success = tape <= threshold.
        The robot's own arrival belief is reported separately, clearly
        labeled — it is NOT the headline number (self-grading is circular).
  TTA   Time-to-arrival. Censored trials (timeout) enter at their cap —
        60 s (rtsm) / 180 s (baseline) — i.e. as if they had succeeded at
        the buzzer. Conservative: this flatters the baseline, so any
        surviving speedup is a floor.
  PE    Path efficiency = actual ground-path length (sum of fresh-tick
        pose deltas) / straight-line optimal (plan pose -> target), for
        arrived trials. Believed-frame; mapping error shows up in TCR-vs-
        belief disagreement instead.
  Stats Mann-Whitney U on TTA, one-sided (memory faster), normal
        approximation with tie correction — pure numpy, no scipy needed.
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
    return {
        "trial_id": start.get("trial_id"),
        "condition": start.get("condition"),
        "result": end.get("result"),
        "elapsed_s": end.get("elapsed_s"),
        "tta_s": end.get("tta_s"),
        "censored": bool(end.get("censored")),
        "final_dist_m": end.get("final_dist_m"),
        "tape_cm": start.get("tape_cm"),
        "path_len_m": path_len if prev is not None else None,
        "optimal_m": optimal,
        "search_time_s": search.get("search_time_s") if search else None,
        "planner_path": (start.get("planner") or {}).get("planner_path"),
    }


def load_trials(trials_dir) -> List[dict]:
    out = []
    for p in sorted(Path(trials_dir).glob("*.jsonl")):
        t = parse_trial(p)
        if t is not None and t["condition"] in TIMEOUT_CAP_S:
            out.append(t)
    return out


# ── statistics ───────────────────────────────────────────────────────────


def mann_whitney_u(a, b) -> Dict[str, float]:
    """One-sided MW U (alternative: a stochastically SMALLER than b).
    Normal approximation with tie correction + continuity correction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"U": float("nan"), "p_one_sided": float("nan")}
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


def _tta_with_censoring(trials: List[dict], condition: str) -> List[float]:
    cap = TIMEOUT_CAP_S[condition]
    out = []
    for t in trials:
        if t["result"] == "arrived" and t["tta_s"] is not None:
            out.append(float(t["tta_s"]))
        elif t["censored"]:
            out.append(cap)                          # conservative: at the buzzer
        # aborted trials (estop/frame_reset/...) carry no TTA — excluded,
        # but counted in the summary so exclusions are visible.
    return out


def summarize(trials_dir, tape_success_cm: float = 50.0) -> dict:
    trials = load_trials(trials_dir)
    by_cond: Dict[str, List[dict]] = {"rtsm": [], "baseline": []}
    for t in trials:
        by_cond[t["condition"]].append(t)

    summary: dict = {"n_files": len(trials), "tape_success_cm": tape_success_cm,
                     "conditions": {}}
    for cond, ts in by_cond.items():
        arrived = [t for t in ts if t["result"] == "arrived"]
        taped = [t for t in ts if t["tape_cm"] is not None]
        tape_success = [t for t in taped
                        if float(t["tape_cm"]) <= tape_success_cm]
        ttas = _tta_with_censoring(ts, cond)
        pes = [t["path_len_m"] / t["optimal_m"] for t in arrived
               if t["path_len_m"] and t["optimal_m"] and t["optimal_m"] > 0.05]
        searches = [t["search_time_s"] for t in ts if t["search_time_s"] is not None]
        summary["conditions"][cond] = {
            "n": len(ts),
            "arrived_believed": len(arrived),
            "censored": sum(1 for t in ts if t["censored"]),
            "aborted": sum(1 for t in ts
                           if t["result"] not in ("arrived",) and not t["censored"]),
            "tcr_tape": (len(tape_success) / len(taped)) if taped else None,
            "tcr_tape_n": len(taped),
            "tcr_believed": (len(arrived) / len(ts)) if ts else None,
            "tta_median_s": float(np.median(ttas)) if ttas else None,
            "tta_iqr_s": ([float(np.percentile(ttas, 25)),
                           float(np.percentile(ttas, 75))] if ttas else None),
            "tta_n": len(ttas),
            "pe_median": float(np.median(pes)) if pes else None,
            "search_time_median_s": (float(np.median(searches))
                                     if searches else None),
        }

    tta_r = _tta_with_censoring(by_cond["rtsm"], "rtsm")
    tta_b = _tta_with_censoring(by_cond["baseline"], "baseline")
    summary["mann_whitney_tta"] = mann_whitney_u(tta_r, tta_b)
    r_med = summary["conditions"]["rtsm"]["tta_median_s"]
    b_med = summary["conditions"]["baseline"]["tta_median_s"]
    summary["tta_speedup_median"] = (b_med / r_med
                                     if r_med and b_med and r_med > 0 else None)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 trial aggregation")
    ap.add_argument("--dir", default="paper/demo2_data")
    ap.add_argument("--tape-success-cm", type=float, default=50.0)
    ap.add_argument("--json", default=None, help="also write summary JSON here")
    args = ap.parse_args()

    s = summarize(args.dir, args.tape_success_cm)
    print(f"trials parsed: {s['n_files']}")
    for cond, c in s["conditions"].items():
        print(f"\n[{cond}]  n={c['n']}  censored={c['censored']}  aborted={c['aborted']}")
        tape = ("—" if c["tcr_tape"] is None
                else f"{100 * c['tcr_tape']:.0f}% (n={c['tcr_tape_n']})")
        print(f"  TCR  tape<= {s['tape_success_cm']:.0f}cm: {tape}"
              f"   believed-arrival: "
              + ("—" if c["tcr_believed"] is None else f"{100 * c['tcr_believed']:.0f}%"))
        if c["tta_median_s"] is not None:
            print(f"  TTA  median {c['tta_median_s']:.1f} s  IQR {c['tta_iqr_s']}"
                  f"  (n={c['tta_n']}, censored at cap)")
        if c["pe_median"] is not None:
            print(f"  PE   median {c['pe_median']:.2f} (arrived only, believed-frame)")
        if c["search_time_median_s"] is not None:
            print(f"  search median {c['search_time_median_s']:.1f} s")
    mw = s["mann_whitney_tta"]
    print(f"\nMann-Whitney U (TTA, one-sided memory-faster): "
          f"U={mw['U']:.1f}  p={mw['p_one_sided']:.4f}")
    if s["tta_speedup_median"] is not None:
        print(f"median-TTA speedup: {s['tta_speedup_median']:.1f}x")
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"written {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
