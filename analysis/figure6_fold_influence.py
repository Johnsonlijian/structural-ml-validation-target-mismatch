"""R38-67: Figure 6 — per-fold influence of the protocol change in the strict beam scenario.

The new subsection claims that 22 of the 24 avoided unsafe releases sit in one outer fold. That claim is
much easier to judge visually than numerically, and the enhancement package's own figure round failed for
want of inputs, so the figure is produced here from the released fold-level records.

Design: two panels sharing the fold axis. Left, unsafe releases per fold for both protocols with the
avoided count annotated; right, releases per fold, which shows what the reduction costs in coverage.
Colour is never the only channel — the two series differ in fill pattern and in edge style as well, and
every bar is labelled with its value.
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
OUT = ROOT / "analysis" / "out"

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
unsafe_r = tab[("n_unsafe", "random")].to_numpy()
unsafe_s = tab[("n_unsafe", "source")].to_numpy()
rel_r = tab[("n_released", "random")].to_numpy()
rel_s = tab[("n_released", "source")].to_numpy()
avoided = unsafe_r - unsafe_s

W = 6.5
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharex=True)
fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16, wspace=0.28)
x = range(len(folds))
w = 0.38

for ax, (a, b), ylab, title in ((axes[0], (unsafe_r, unsafe_s), "unsafe releases",
                                 "(A) Unsafe releases by outer fold"),
                                (axes[1], (rel_r, rel_s), "released records",
                                 "(B) Coverage cost by outer fold")):
    ax.bar([i - w / 2 for i in x], a, w, label="random inner folds",
           facecolor="#cfe0f3", edgecolor="#1f4e79", linewidth=0.8, hatch="")
    ax.bar([i + w / 2 for i in x], b, w, label="source-honouring inner folds",
           facecolor="#f6d9c9", edgecolor="#8c3b12", linewidth=0.8, hatch="///")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va, f"{int(va)}", ha="center", va="bottom", fontsize=8, color="#1f4e79")
        ax.text(i + w / 2, vb, f"{int(vb)}", ha="center", va="bottom", fontsize=8, color="#8c3b12")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"fold {int(f)}" for f in folds])
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    ax.set_axisbelow(True)

for i, av in enumerate(avoided):
    if av > 0:
        axes[0].annotate(f"$-{int(av)}$", xy=(i, max(unsafe_r[i], unsafe_s[i]) + 1.8),
                         ha="center", fontsize=8.5, color="black")
axes[0].set_ylim(0, max(unsafe_r.max(), unsafe_s.max()) * 1.30)
axes[0].legend(loc="upper left", frameon=False, ncol=1)
axes[0].text(0.02, 0.62, f"{int(avoided.sum())} events avoided in total;\n"
                         f"{int(avoided.max())} of them in fold {int(folds[int(avoided.argmax())])}",
             transform=axes[0].transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.7", linewidth=0.6))

fig.savefig(FIG / "Figure_6_fold_influence.pdf")
fig.savefig(FIG / "Figure_6_fold_influence.png", dpi=400)
plt.close(fig)
print(f"Figure 6 written: {len(folds)} folds, avoided total {int(avoided.sum())}, "
      f"max fold {int(avoided.max())} (fold {int(folds[int(avoided.argmax())])})")

tab.to_csv(OUT / "R38_figure6_source.csv")
