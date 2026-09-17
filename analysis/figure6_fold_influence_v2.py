"""R38-68: Figure 6, second pass — remove the label collisions and make the co-location visible.

Panel B becomes the source-minus-random difference in released records, one bar per fold. Equal-count
folds then show a zero bar instead of two touching labels, and the point the figure exists to make — that
both the avoided events and the coverage cost concentrate in the same fold — is legible at a glance.
"""
from __future__ import annotations

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
R37 = ROOT.parent / "R37_second_review_repair_2026-09-16"
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 400, "savefig.bbox": None, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

outer = pd.read_csv(R37 / "audit/r37/decision_outer.csv")
beam = outer[(outer.scenario == "beam_release") & (outer.alpha == 0.05)]
tab = beam.pivot_table(index="outer_fold", columns="protocol",
                       values=["n_unsafe", "n_released"], aggfunc="sum").sort_index()
folds = list(tab.index)
unsafe_r, unsafe_s = tab[("n_unsafe", "random")].to_numpy(), tab[("n_unsafe", "source")].to_numpy()
rel_r, rel_s = tab[("n_released", "random")].to_numpy(), tab[("n_released", "source")].to_numpy()
avoided = unsafe_r - unsafe_s
d_rel = rel_s - rel_r

W = 6.5
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharex=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.16, wspace=0.30)
x = list(range(len(folds)))
w = 0.38

ax = axes[0]
ax.bar([i - w / 2 for i in x], unsafe_r, w, label="random inner folds",
       facecolor="#cfe0f3", edgecolor="#1f4e79", linewidth=0.8)
ax.bar([i + w / 2 for i in x], unsafe_s, w, label="source-honouring inner folds",
       facecolor="#f6d9c9", edgecolor="#8c3b12", linewidth=0.8, hatch="///")
for i, (va, vb) in enumerate(zip(unsafe_r, unsafe_s)):
    ax.text(i - w / 2, va + 0.4, f"{int(va)}", ha="center", va="bottom", fontsize=8, color="#1f4e79")
    ax.text(i + w / 2, vb + 0.4, f"{int(vb)}", ha="center", va="bottom", fontsize=8, color="#8c3b12")
for i, av in enumerate(avoided):
    if av > 0:
        ax.annotate(f"$-{int(av)}$", xy=(i, max(unsafe_r[i], unsafe_s[i]) + 3.4), ha="center",
                    fontsize=9, color="black",
                    arrowprops=None)
ax.set_ylim(0, max(unsafe_r.max(), unsafe_s.max()) * 1.34)
ax.set_ylabel("unsafe releases")
ax.set_title("(A) Unsafe releases by outer fold", loc="left")
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)
ax.text(0.03, 0.60, f"{int(avoided.sum())} avoided in total;\n{int(avoided.max())} of them in "
                    f"fold {int(folds[int(avoided.argmax())])}",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.7", linewidth=0.6))

ax = axes[1]
cols = ["#8c3b12" if v < 0 else ("#1f4e79" if v > 0 else "0.75") for v in d_rel]
ax.bar(x, d_rel, 0.5, color=cols, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)
for i, v in enumerate(d_rel):
    if v == 0:
        ax.text(i, 0.35, "0", ha="center", va="bottom", fontsize=8, color="0.35")
    elif v < 0:
        ax.text(i, v - 1.1, f"{int(v)}", ha="center", va="top", fontsize=8.5, color="#8c3b12")
    else:
        ax.text(i, v + 0.8, f"+{int(v)}", ha="center", va="bottom", fontsize=8.5, color="#1f4e79")
ax.set_ylim(min(d_rel) * 1.30, max(d_rel) * 1.6 + 2)
ax.set_ylabel("released records, source $-$ random")
ax.set_title("(B) Coverage cost by outer fold", loc="left")
ax.set_axisbelow(True)

for a in axes:
    a.set_xticks(x)
    a.set_xticklabels([f"fold {int(f)}" for f in folds])

fig.savefig(FIG / "Figure_6_fold_influence.pdf")
fig.savefig(FIG / "Figure_6_fold_influence.png", dpi=400)
plt.close(fig)
print(f"Figure 6 rewritten | avoided {int(avoided.sum())} (max {int(avoided.max())} in fold "
      f"{int(folds[int(avoided.argmax())])}) | coverage differences {[int(v) for v in d_rel]}")
