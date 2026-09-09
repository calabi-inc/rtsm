"""Generate the two E1 Results figures from trial JSONLs.
Usage: python paper/make_figures.py  (from examples/rc_car_agent/)
Writes fig_trajectories.pdf and fig_per_object.pdf next to main.tex's
expected figure dir (paper/figs/ -> copy to paper-private manually or
point \graphicspath there).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "demo2_data"
OUT = Path(r"C:\Users\konam\Desktop\calabi-repo\paper-private\figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

MEM_C = "#1a7f37"   # green
SRCH_C = "#b3261e"  # red


def load_ticks(tid):
    p = DATA / f"{tid}.jsonl"
    ticks = []
    start = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if '"type": "trial_start"' in line:
            start = json.loads(line)
        elif '"type": "tick"' in line:
            j = json.loads(line)
            if j.get("pose"):
                ticks.append(j)
    return start, ticks


# ---------- Fig 1: overhead trajectories, L5 scissors pair ----------
mem_start, mem_ticks = load_ticks("t20260903-233912-001")   # memory, 31.3 s
srch_start, srch_ticks = load_ticks("t20260903-234029-002") # search, 486.1 s

fig, ax = plt.subplots(figsize=(3.15, 2.7))
for ticks, color, label, lw in ((srch_ticks, SRCH_C, "memoryless search (486\u2009s)", 0.9),
                                (mem_ticks, MEM_C, "memory (31\u2009s)", 1.6)):
    xs = [t["pose"]["xyz"][0] for t in ticks]
    zs = [t["pose"]["xyz"][2] for t in ticks]
    ax.plot(xs, zs, color=color, lw=lw, label=label, alpha=0.9)
# start + target
sx, sz = mem_ticks[0]["pose"]["xyz"][0], mem_ticks[0]["pose"]["xyz"][2]
t_xyz = mem_start.get("planner", {}).get("xyz_world")
ax.plot(sx, sz, marker="s", color="black", ms=5, ls="none", label="start")
if t_xyz:
    ax.plot(t_xyz[0], t_xyz[2], marker="*", color="#e8a000", ms=11, ls="none",
            label="goal (scissors)", markeredgecolor="black", markeredgewidth=0.4)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_aspect("equal")
ax.legend(loc="best", frameon=False, handlelength=1.4)
ax.set_title("Same goal, one bit masked")
fig.tight_layout()
fig.savefig(OUT / "fig_trajectories.pdf")
print("fig_trajectories.pdf", "mem ticks:", len(mem_ticks), "search ticks:", len(srch_ticks))

# ---------- Fig 2: per-object paired effective TTA ----------
an = json.loads((DATA / "paper_analyses.json").read_text(encoding="utf-8"))
tt = an["per_object_ttas"]
order = ["teddy bear", "water bottle", "tissue box", "dumbbell", "scissors"]
HORIZON = 900.0

fig, ax = plt.subplots(figsize=(3.15, 2.7))
for i, obj in enumerate(order):
    mem = [v if v is not None else HORIZON for v in tt[obj]["mem"]]
    sr = [v if v is not None else HORIZON for v in tt[obj]["search"]]
    for vals, color, dx in ((mem, MEM_C, -0.16), (sr, SRCH_C, 0.16)):
        for v in vals:
            cens = v >= HORIZON
            ax.plot(i + dx, min(v, HORIZON),
                    marker="^" if cens else "o", ms=5 if cens else 3.6,
                    color=color, alpha=0.75, ls="none",
                    markerfacecolor="none" if cens else color)
        med = sorted(vals)[len(vals) // 2]
        ax.plot([i + dx - 0.11, i + dx + 0.11], [med, med], color=color, lw=1.6)
ax.set_yscale("log")
ax.set_ylim(15, 1400)
ax.axhline(HORIZON, color="gray", lw=0.6, ls=":")
ax.text(4.45, 950, "censored", color="gray", fontsize=6.5, ha="right")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["bear", "bottle", "tissue", "dumbbell", "scissors"])
ax.set_ylabel("time-to-arrival (s, log)")
ax.set_title("TTA by object, all five layouts")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=MEM_C, ls="none", label="memory"),
                   Line2D([], [], marker="o", color=SRCH_C, ls="none", label="memoryless"),
                   Line2D([], [], marker="^", color="gray", ls="none", markerfacecolor="none", label="failure (censored)")],
          loc="upper left", frameon=False, handlelength=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_per_object.pdf")
print("fig_per_object.pdf written")

