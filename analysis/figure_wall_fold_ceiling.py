"""R38-99: Figure 7, second pass — clear the annotation, legend and tick collisions.

First render had four layout defects: the "no single-type campaign" note printed inside the hatched bar,
the legend overlapped the dominant Rect bar, panel B's title was clipped, and its fold tick labels ran
together. The annotations move next to the value labels, the legend moves below the axes, and panel B's
labels are shortened to fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
WALL = PROJ / "code/outputs/reproductions/designsafe_prj2430_wall/extracted_wall_data.csv"
FIG = ROOT / "figures"
res = json.loads((ROOT / "analysis" / "out" / "R38_wall_discard_alternative.json").read_text(encoding="utf-8"))
k3 = res["per_K"]["3"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 400, "savefig.bbox": None, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

df = pd.read_csv(WALL, low_memory=False)
AUTH, TYPE = "Authors", "walltype_Shape"
camp_types = {a: set(g[TYPE]) for a, g in df.groupby(AUTH)}
types = ["Rect", "T", "Flanged", "B", "I"]
single = {t: sum(1 for ts in camp_types.values() if ts == {t}) for t in types}
bridging = {t: sum(1 for ts in camp_types.values() if t in ts and len(ts) > 1) for t in types}

W = 6.5
fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(left=0.16, right=0.975, top=0.83, bottom=0.30, wspace=0.38)

ax = axes[0]
y = list(range(len(types)))
ax.barh(y, [single[t] for t in types], 0.55, label="only this type",
        facecolor="#cfe0f3", edgecolor="#1f4e79", linewidth=0.8)
ax.barh(y, [bridging[t] for t in types], 0.55, left=[single[t] for t in types],
        label="also another type", facecolor="#f6d9c9", edgecolor="#8c3b12", linewidth=0.8, hatch="///")
for i, t in enumerate(types):
    tot = single[t] + bridging[t]
    note = "  (no single-type campaign)" if single[t] == 0 else ""
    ax.text(tot + 0.5, i, f"{tot}{note}", va="center", fontsize=8.5,
            color="#8c3b12" if single[t] == 0 else "black")
ax.set_yticks(y)
ax.set_yticklabels([f"type {t}" for t in types])
ax.set_xlabel("campaigns carrying the type")
ax.set_xlim(0, 34)
ax.set_title("(A) B and I appear only in bridging campaigns", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
ax.set_axisbelow(True)

ax = axes[1]
folds = k3["fold_sizes"]
values = list(folds) + [res["records"] - k3["n_retained"]]
labels = [f"fold {i+1}\n{s}" for i, s in enumerate(folds)] + [f"quarantined\n({values[-1]})"]
bars = ax.bar(range(len(values)), values, 0.62,
              color=["#cfe0f3"] * 3 + ["#f6d9c9"],
              edgecolor=["#1f4e79"] * 3 + ["#8c3b12"], linewidth=0.9)
bars[-1].set_hatch("///")
for i, v in enumerate(values):
    ax.text(i, v + 2.5, f"{v}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("records")
ax.set_ylim(0, max(values) * 1.25)
ax.set_title("(B) Three folds keep 29", loc="left")
ax.set_axisbelow(True)

fig.savefig(FIG / "Figure_7_wall_fold_ceiling.pdf")
fig.savefig(FIG / "Figure_7_wall_fold_ceiling.png", dpi=400)
plt.close(fig)
print(f"Figure 7 rewritten | single-type {single} | bridging {bridging} | folds {folds}")
