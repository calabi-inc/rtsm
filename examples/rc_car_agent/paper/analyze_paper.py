"""Pre-registered secondary analyses for the E1 paper, computed
deterministically from the campaign trial JSONLs. Complements
aggregate.py (which owns the headline TCR/TTA/MW numbers).

Usage: python paper/analyze_paper.py  (from examples/rc_car_agent/)
Writes paper/demo2_data/paper_analyses.json and prints a summary.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).parent / "demo2_data"
HORIZON = 900.0

# Sheet-truth: trial id -> (layout, object, repeat?) from the five manifests.
ROWS = {
    # L1 (session_20260901_L1)
    "t20260901-192904-001": ("L1", "teddy bear", False),
    "t20260901-193125-002": ("L1", "teddy bear", False),
    "t20260901-193406-003": ("L1", "water bottle", False),
    "t20260901-193526-004": ("L1", "water bottle", False),
    "t20260901-193729-005": ("L1", "dumbbell", False),
    "t20260901-193831-006": ("L1", "dumbbell", False),
    "t20260901-195220-002": ("L1", "tissue box", False),
    "t20260901-195410-003": ("L1", "tissue box", False),
    "t20260901-195737-004": ("L1", "scissors", False),
    "t20260901-200333-006": ("L1", "scissors", False),
    "t20260902-004117-002": ("L1", "teddy bear", True),
    "t20260902-004235-003": ("L1", "teddy bear", True),
    # L2
    "t20260902-005340-005": ("L2", "water bottle", False),
    "t20260902-005209-004": ("L2", "water bottle", False),
    "t20260902-005647-007": ("L2", "dumbbell", False),
    "t20260902-005532-006": ("L2", "dumbbell", False),
    "t20260902-010250-009": ("L2", "tissue box", False),
    "t20260902-010138-008": ("L2", "tissue box", False),
    "t20260902-204701-001": ("L2", "scissors", False),
    "t20260902-010440-010": ("L2", "scissors", False),
    "t20260902-205349-003": ("L2", "teddy bear", False),
    "t20260902-205204-002": ("L2", "teddy bear", False),
    "t20260902-205701-005": ("L2", "water bottle", True),
    "t20260902-205530-004": ("L2", "water bottle", True),
    # L3
    "t20260902-212227-001": ("L3", "dumbbell", False),
    "t20260902-212400-002": ("L3", "dumbbell", False),
    "t20260902-213005-003": ("L3", "tissue box", False),
    "t20260902-213132-004": ("L3", "tissue box", False),
    "t20260902-213550-005": ("L3", "scissors", False),
    "t20260902-214503-007": ("L3", "scissors", False),
    "t20260902-215836-009": ("L3", "teddy bear", False),
    "t20260902-215948-010": ("L3", "teddy bear", False),
    "t20260902-220203-011": ("L3", "water bottle", False),
    "t20260902-220317-012": ("L3", "water bottle", False),
    "t20260902-220821-013": ("L3", "dumbbell", True),
    "t20260902-221901-015": ("L3", "dumbbell", True),
    # L4
    "t20260903-152321-001": ("L4", "tissue box", False),
    "t20260903-153951-003": ("L4", "tissue box", False),
    "t20260903-154311-004": ("L4", "scissors", False),
    "t20260903-154518-005": ("L4", "scissors", False),
    "t20260903-154926-006": ("L4", "teddy bear", False),
    "t20260903-160622-001": ("L4", "teddy bear", False),
    "t20260903-161105-002": ("L4", "water bottle", False),
    "t20260903-161429-003": ("L4", "water bottle", False),
    "t20260903-161635-004": ("L4", "dumbbell", False),
    "t20260903-162431-006": ("L4", "dumbbell", False),
    "t20260903-163005-007": ("L4", "tissue box", True),
    "t20260903-170714-011": ("L4", "tissue box", True),
    # L5
    "t20260903-233912-001": ("L5", "scissors", False),
    "t20260903-234029-002": ("L5", "scissors", False),
    "t20260903-235351-003": ("L5", "teddy bear", False),
    "t20260903-235515-004": ("L5", "teddy bear", False),
    "t20260903-235647-005": ("L5", "water bottle", False),
    "t20260903-235827-006": ("L5", "water bottle", False),
    "t20260903-235956-007": ("L5", "dumbbell", False),
    "t20260904-000112-008": ("L5", "dumbbell", False),
    "t20260904-000311-009": ("L5", "tissue box", False),
    "t20260904-000423-010": ("L5", "tissue box", False),
    "t20260904-000601-011": ("L5", "scissors", True),
    "t20260904-001209-013": ("L5", "scissors", True),
}


def load(tid):
    p = DATA / f"{tid}.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    start = json.loads(lines[0])
    end = json.loads(lines[-1])
    ticks = [json.loads(l) for l in lines if '"type": "tick"' in l or '"type":"tick"' in l]
    events = [json.loads(l) for l in lines if '"type": "event"' in l or '"type":"event"' in l]
    return start, end, ticks, events


def eff_tta(end):
    return end.get("tta_s") if end.get("result") == "arrived" else HORIZON


trials = {}
for tid, (layout, obj, rep) in ROWS.items():
    start, end, ticks, events = load(tid)
    cond = start["condition"]  # rtsm | baseline
    trials[tid] = {
        "layout": layout, "object": obj, "repeat": rep,
        "cond": "memory" if cond == "rtsm" else "search",
        "result": end.get("result"), "tta": end.get("tta_s"),
        "eff": eff_tta(end), "tape_cm": start.get("tape_cm"),
        "start": start, "end": end, "ticks": ticks, "events": events,
    }

# ---- pairs (layout, object, repeat) -> {memory, search} ----
pairs = defaultdict(dict)
for tid, t in trials.items():
    pairs[(t["layout"], t["object"], t["repeat"])][t["cond"]] = t

# per-layout win table + per-object gradient
layout_wins = {}
per_object = defaultdict(list)
pair_rows = []
for key in sorted(pairs, key=lambda k: (k[0], k[2], k[1])):
    m, s = pairs[key]["memory"], pairs[key]["search"]
    win = m["eff"] < s["eff"]
    ratio = (s["eff"] / m["eff"]) if m["eff"] > 0 else float("inf")
    pair_rows.append({
        "layout": key[0], "object": key[1], "repeat": key[2],
        "mem_tta": m["tta"], "mem_result": m["result"],
        "search_tta": s["tta"], "search_result": s["result"],
        "mem_eff": m["eff"], "search_eff": s["eff"],
        "memory_faster": win, "ratio_eff": round(ratio, 2),
    })
    layout_wins.setdefault(key[0], []).append(win)
    if not key[2]:
        per_object[key[1]].append((m, s))

layout_table = {L: {"pairs": len(w), "memory_wins": sum(w)} for L, w in sorted(layout_wins.items())}
n_layout_won = sum(1 for L, w in layout_wins.items() if sum(w) > len(w) / 2)
# exact one-sided sign test across 5 layouts (memory wins majority in k of 5)
from math import comb
p_sign = sum(comb(5, k) for k in range(n_layout_won, 6)) / 2 ** 5

obj_table = {}
for obj, ms in sorted(per_object.items()):
    mm = [m["eff"] for m, s in ms]
    ss = [s["eff"] for m, s in ms]
    marr = [m for m, s in ms if m["result"] == "arrived"]
    sarr = [s for m, s in ms if s["result"] == "arrived"]
    obj_table[obj] = {
        "n_pairs": len(ms),
        "mem_arrived": len(marr), "search_arrived": len(sarr),
        "mem_median_eff": round(sorted(mm)[len(mm) // 2], 1),
        "search_median_eff": round(sorted(ss)[len(ss) // 2], 1),
        "median_ratio": round(sorted(ss)[len(ss) // 2] / sorted(mm)[len(mm) // 2], 1),
        "mem_ttas": [m["tta"] for m, s in ms],
        "search_ttas": [s["tta"] for m, s in ms],
    }

# repeat pairs
repeats = []
for key in sorted(pairs):
    if not key[2]:
        continue
    first = pairs[(key[0], key[1], False)]
    rep = pairs[key]
    repeats.append({
        "layout": key[0], "object": key[1],
        "mem_first_tta": first["memory"]["tta"], "mem_repeat_tta": rep["memory"]["tta"],
        "mem_first_result": first["memory"]["result"], "mem_repeat_result": rep["memory"]["result"],
        "search_first_tta": first["search"]["tta"], "search_repeat_tta": rep["search"]["tta"],
        "search_first_result": first["search"]["result"], "search_repeat_result": rep["search"]["result"],
    })

# search decomposition: search_time (from baseline_acquired) vs drive
search_decomp = []
for tid, t in trials.items():
    if t["cond"] != "search" or t["result"] != "arrived":
        continue
    acq = [e for e in t["events"] if e.get("name") == "baseline_acquired"]
    if acq and t["tta"]:
        st = acq[-1].get("search_time_s")
        if st is not None:
            search_decomp.append({"tid": tid, "search_s": st, "drive_s": t["tta"] - st})
sd_search = sorted(x["search_s"] for x in search_decomp)
sd_drive = sorted(x["drive_s"] for x in search_decomp)

# memory acquisition time: time from accept to driving (planning phase) ~ first drive tick t
mem_acq = []
for tid, t in trials.items():
    if t["cond"] != "memory" or t["result"] != "arrived":
        continue
    if t["ticks"]:
        mem_acq.append(t["ticks"][0].get("t", None))
mem_acq = sorted(a for a in mem_acq if a is not None)

# start distances (memory arm first-tick ground_dist as venue geometry proxy)
start_d = []
for tid, t in trials.items():
    if t["cond"] == "memory" and t["ticks"]:
        d0 = t["ticks"][0].get("ground_dist_m")
        if d0:
            start_d.append(d0)
start_d.sort()

# believed-vs-tape terminal decomposition (arrived rows with tape)
term = []
for tid, t in trials.items():
    if t["result"] == "arrived" and t["tape_cm"] is not None and t["end"].get("final_dist_m"):
        term.append(abs(t["end"]["final_dist_m"] - t["tape_cm"] / 100.0))
term.sort()

# coordinate age at plan (memory arm)
ages = []
for tid, t in trials.items():
    if t["cond"] == "memory":
        a = t["start"].get("planner", {}).get("target_age_at_plan_s")
        if a is not None:
            ages.append(a)
ages.sort()

# scan cost / break-even
SCAN_S = 180.0  # conservative stated bound (<= 3 min)
med_m = sorted(t["eff"] for t in trials.values() if t["cond"] == "memory")
med_s = sorted(t["eff"] for t in trials.values() if t["cond"] == "search")
med_m = med_m[len(med_m) // 2 - 1: len(med_m) // 2 + 1]
med_s = med_s[len(med_s) // 2 - 1: len(med_s) // 2 + 1]
med_m = sum(med_m) / 2
med_s = sum(med_s) / 2
saving_med = med_s - med_m
# arrivals-only mean saving (paired, both arrived)
paired_sav = [r["search_tta"] - r["mem_tta"] for r in pair_rows
              if r["mem_result"] == "arrived" and r["search_result"] == "arrived"]
mean_sav = sum(paired_sav) / len(paired_sav)

out = {
    "pair_rows": pair_rows,
    "layout_table": layout_table,
    "layouts_won_by_memory": n_layout_won,
    "sign_test_p_one_sided": round(p_sign, 4),
    "per_object": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("ttas")} for k, v in obj_table.items()},
    "per_object_ttas": {k: {"mem": v["mem_ttas"], "search": v["search_ttas"]} for k, v in obj_table.items()},
    "repeat_pairs": repeats,
    "search_decomposition": {
        "n": len(search_decomp),
        "search_median_s": round(sd_search[len(sd_search) // 2], 1),
        "drive_median_s": round(sd_drive[len(sd_drive) // 2], 1),
    },
    "memory_acquisition_median_s": round(mem_acq[len(mem_acq) // 2], 1) if mem_acq else None,
    "start_dist_m": {"min": round(start_d[0], 2), "max": round(start_d[-1], 2),
                     "median": round(start_d[len(start_d) // 2], 2)},
    "terminal_abs_err_m": {"n": len(term), "median": round(term[len(term) // 2], 3),
                           "max": round(term[-1], 3)},
    "coordinate_age_s": {"n": len(ages),
                         "median": round(ages[len(ages) // 2], 1) if ages else None,
                         "max": round(ages[-1], 1) if ages else None},
    "scan_break_even": {"scan_bound_s": SCAN_S,
                        "median_saving_s": round(saving_med, 1),
                        "tasks_to_break_even_median": round(SCAN_S / saving_med, 1),
                        "mean_paired_saving_arrivals_s": round(mean_sav, 1),
                        "tasks_to_break_even_mean": round(SCAN_S / mean_sav, 1)},
}

with open(DATA / "paper_analyses.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

print(json.dumps({k: v for k, v in out.items() if k not in ("pair_rows", "per_object_ttas")}, indent=1))
