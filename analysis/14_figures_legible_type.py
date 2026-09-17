"""R38-14: redraw the data figures so that no type ends up below 7 pt in the printed page.

The geometry check found figure type at 4.8--6.9 pt: the matplotlib figures were drawn with 8 pt base
type at a width slightly larger than the text block, so inclusion scaled them down further. These
figures are redrawn with a 10 pt base and a 6.5 in width (matching \\textwidth), and the feasibility map
keeps the corrected $m<K$ logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
R36 = ROOT.parent / "R36_independent_review_rebuild_2026-09-16"
R37 = ROOT.parent / "R37_second_review_repair_2026-09-16"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": None,
})
W = 6.5


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


PRETTY = {"stub_cfst": "Stub CFST", "corroded_rc_beam": "Corroded RC beam",
          "wall_prj2430_Vmax": "RC wall strength", "wall_prj2430_drift": "RC wall drift",
          "coupling_beams_prj3053": "Coupling beam", "concrete_strength": "Concrete (UCI)"}
SPLIT = {"honoured": "source", "honoured_proxy": "proxy", "wall_combo": "combination",
         "uci_428_ingredients": "7-ingredient", "uci_996_allinputs": "8-input"}
COL = {"source": "#1f4e79", "proxy": "#c55a11", "combination": "#548235", "7-ingredient": "#7030a0",
       "8-input": "#7030a0"}

# ---------------------------------------------------------------- Figure 2: admissible contrasts
pairs = pd.read_csv(R36 / "analysis/out/R36_control_pairs.csv")
d = pairs[(pairs["split_admissible"] == True) & (pairs["delta_vs_sizematch"].notna())].copy()  # noqa: E712
d["splitname"] = d["variant"].map(SPLIT)
d = d.sort_values(["dataset", "variant", "r2_random5"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(W, 4.6))
fig.subplots_adjust(left=0.46, right=0.985, top=0.90, bottom=0.10)
for i, r in d.iterrows():
    col = COL.get(r["splitname"], "#555555")
    ax.plot([r["r2_random5"], r["r2_sizematch"], r["r2_honoured"]], [i, i, i], color=col, lw=1.1)
    ax.scatter(r["r2_random5"], i, s=18, facecolor="white", edgecolor="#333333", lw=0.8, zorder=3)
    ax.scatter(r["r2_sizematch"], i, s=18, marker="s", facecolor="white", edgecolor=col, lw=0.9, zorder=3)
    ax.scatter(r["r2_honoured"], i, s=22, color=col, zorder=3)
ax.axvline(0, color="#999999", lw=0.9, ls="--")
ax.set_yticks(np.arange(len(d)))
ax.set_yticklabels([f"{PRETTY.get(r['dataset'], r['dataset'])} · {r['model']} · {r['splitname']}"
                    for _, r in d.iterrows()], fontsize=7.5)
ax.set_xlabel("Pooled $R^2$ (out-of-fold)")
ax.set_title("Admissible splits only: random (open circle), size-matched (square), honoured (filled)",
             fontsize=9.5)
ax.legend(handles=[Line2D([], [], color=COL["source"], marker="o", lw=1.1, ms=5, label="source-level"),
                   Line2D([], [], color=COL["proxy"], marker="o", lw=1.1, ms=5, label="proxy"),
                   Line2D([], [], color=COL["combination"], marker="o", lw=1.1, ms=5, label="combination-unseen"),
                   Line2D([], [], color=COL["7-ingredient"], marker="o", lw=1.1, ms=5, label="mix relations")],
          fontsize=8, frameon=False, loc="lower left", ncol=2)
save(fig, "Figure_2_admissible_contrasts")

# ---------------------------------------------------------------- Figure 3: feasibility map
feas = pd.read_csv(R36 / "audit/r36/cfst_feasibility_map.csv")
m_families = 3
Ks = sorted(feas["K"].unique())
fmaxes = list(dict.fromkeys(feas.sort_values("f_max_dec")["f_max"].tolist()))
state, reason = {}, {}
for K in Ks:
    for fm in fmaxes:
        row = feas[(feas["K"] == K) & (feas["f_max"] == fm)].iloc[0]
        if bool(row["feasible"]):
            state[(K, fm)], reason[(K, fm)] = 2, "yes"
        elif K > m_families:
            state[(K, fm)], reason[(K, fm)] = 0, "no: $m<K$"
        else:
            state[(K, fm)], reason[(K, fm)] = 1, "no: $\\max w_j>U$"
mat = [[state[(K, fm)] for fm in fmaxes] for K in Ks]
fig, ax = plt.subplots(figsize=(W * 0.72, 3.1))
fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.22)
ax.imshow(mat, cmap=ListedColormap(["#8c1d18", "#f2c14e", "#2e7d32"]), vmin=0, vmax=2, aspect="auto")
for i, K in enumerate(Ks):
    for j, fm in enumerate(fmaxes):
        ax.text(j, i, reason[(K, fm)], ha="center", va="center", fontsize=8.5,
                color="white" if state[(K, fm)] != 1 else "#3d2b00")
ax.set_xticks(range(len(fmaxes)))
ax.set_xticklabels([f"{fm}\n({feas[feas['f_max'] == fm]['f_max_dec'].iloc[0]:.2f})" for fm in fmaxes],
                   fontsize=8.5)
ax.set_yticks(range(len(Ks)))
ax.set_yticklabels([f"$K$ = {k}" for k in Ks], fontsize=9)
ax.set_xlabel("declared fold-upper bound $f_{\\max}$")
ax.set_title("Stub CFST (674 / 396 / 246): is a family-honouring split possible?", fontsize=9.5)
ax.text(0, len(Ks) - 0.35, "green = feasible      amber = blocked only by the fold bound      "
                           "red = structurally impossible ($m<K$)",
        fontsize=8, color="#333333")
save(fig, "Figure_3_feasibility_map")

# ---------------------------------------------------------------- Figure 4: relation coverage
reg = pd.read_csv(R36 / "audit/r36/configuration_registry.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.12)
ax = axes[0]
order = reg.sort_values("relation_kind")
colmap = {"source-level": "#1f4e79", "proxy-family": "#c55a11", "duplication": "#7f7f7f"}
ax.barh(np.arange(len(order)), [1] * len(order), color=[colmap[k] for k in order["relation_kind"]])
for i, r in enumerate(order.itertuples()):
    ax.text(0.02, i, f"{PRETTY.get(r.dataset, r.dataset)} — {r.relation_kind}", va="center",
            fontsize=8.5, color="white")
ax.set_yticks([]); ax.set_xticks([]); ax.invert_yaxis()
ax.set_title("Relation kind per configuration", fontsize=9.5)
ax = axes[1]
ax.bar([0], [142], color="#8c1d18", width=0.5)
ax.text(0, 146, "142 rows\n(100 %)", ha="center", fontsize=8.5, color="#8c1d18")
ax.set_xticks([0]); ax.set_xticklabels(["largest union\ncomponent"], fontsize=8.5)
ax.set_ylabel("rows"); ax.set_ylim(0, 175)
ax.set_title("Wall asset: 1 component from 28 sources × 5 types", fontsize=9.5)
save(fig, "Figure_4_relation_coverage")

# ---------------------------------------------------------------- Figure 5: decision experiment
outer = pd.read_csv(R37 / "audit/r37/decision_outer.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.22)
for ax, scen, title in ((axes[0], "beam_release", "Beam release"),
                        (axes[1], "failure_mode", "Failure mode (release when not dangerous)")):
    sub = outer[outer.scenario == scen]
    if scen == "failure_mode":
        sub = sub[sub.action_set == "release_when_not_dangerous"]
    for strat, col, marker in (("random", "#b03030", "o"), ("source", "#1f4e79", "s")):
        for alpha, ls in ((0.05, "-"), (0.10, "--"), (0.20, ":")):
            g = sub[(sub.protocol == strat) & (sub.alpha == alpha)].sort_values("outer_fold")
            if g.empty:
                continue
            ax.plot(g["inner_r_all"], g["realised_r_all"], ls=ls, lw=0.9, color=col, marker=marker,
                    ms=4.5, alpha=0.9,
                    label=f"{'random' if strat == 'random' else 'source-honouring'}, α={alpha:.2f}")
    lim = max(0.13, float(sub[["inner_r_all", "realised_r_all"]].max().max()) * 1.15)
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"inner estimate $r_{\mathrm{all}}$")
    ax.set_title(title, fontsize=9)
    cov = sub.groupby("protocol")["realised_coverage"].mean()
    ax.text(0.03, 0.97, "mean coverage: " + ", ".join(f"{k} {v:.2f}" for k, v in cov.items()),
            transform=ax.transAxes, fontsize=8, va="top", color="#444444")
axes[0].set_ylabel(r"realised $r_{\mathrm{all}}$ on held-out sources")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8.5, frameon=False, ncol=6, loc="lower center",
           bbox_to_anchor=(0.5, -0.13))
fig.suptitle("Points above the diagonal are optimistic: the estimate understates the rate seen on "
             "unseen sources", fontsize=9.5, y=1.03)
save(fig, "Figure_5_decision_experiment")

print("figures regenerated at a 10 pt base with a 6.5 in width")
