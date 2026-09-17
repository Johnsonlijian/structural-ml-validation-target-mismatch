"""R36-09 (work package E, part 2): rebuilt figures F2-F5 from the corrected artefacts.

  F2  admissible paired contrasts: random folds vs size-matched random vs honoured split
  F3  engineering-decision experiment: estimated vs realised unsafe release, with coverage
  F4  CFST feasibility map over K x fold-upper-bound, showing what blocks each cell
  F5  relation coverage of the corpus and the wall union-graph collapse
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
AUD = ROOT / "audit" / "r36"
OUT = ROOT / "analysis" / "out"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

PRETTY = {
    "stub_cfst": "Stub CFST",
    "corroded_rc_beam": "Corroded RC beam",
    "wall_prj2430_Vmax": "RC wall strength",
    "wall_prj2430_drift": "RC wall drift",
    "coupling_beams_prj3053": "Coupling beam",
    "concrete_strength": "Concrete (UCI)",
}
SPLIT = {"honoured": "source", "honoured_proxy": "proxy", "wall_combo": "combination",
         "uci_428_ingredients": "7-ingredient", "uci_996_allinputs": "8-input"}
COL = {"source": "#1f4e79", "proxy": "#c55a11", "combination": "#548235",
       "7-ingredient": "#7030a0", "8-input": "#7030a0"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------- F2
ctl = pd.read_csv(OUT / "R36_control_pairs.csv")
d = ctl[(ctl["split_admissible"] == True) & (ctl["delta_vs_sizematch"].notna())].copy()   # noqa: E712
d["splitname"] = d["variant"].map(SPLIT)
d = d.sort_values(["dataset", "variant", "r2_random5"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.9, 4.0))
for i, r in d.iterrows():
    col = COL.get(r["splitname"], "#555555")
    ax.plot([r["r2_random5"], r["r2_sizematch"], r["r2_honoured"]], [i, i, i],
            color=col, lw=1.0, zorder=2)
    ax.scatter(r["r2_random5"], i, s=14, facecolor="white", edgecolor="#333333", lw=0.7, zorder=3)
    ax.scatter(r["r2_sizematch"], i, s=14, marker="s", facecolor="white", edgecolor=col, lw=0.8, zorder=3)
    ax.scatter(r["r2_honoured"], i, s=18, color=col, zorder=3)
ax.axvline(0, color="#999999", lw=0.8, ls="--")
ax.set_yticks(np.arange(len(d)))
ax.set_yticklabels([f"{PRETTY.get(r['dataset'], r['dataset'])} · {r['model']} · {r['splitname']}"
                    for _, r in d.iterrows()], fontsize=5.6)
ax.set_xlabel("Pooled $R^2$ (out-of-fold)")
ax.set_title("Admissible splits only: random (open circle), size-matched random (square), "
             "honoured (filled)", fontsize=8)
ax.legend(handles=[
    Line2D([], [], color=COL["source"], marker="o", lw=1.0, ms=4, label="source-level relation"),
    Line2D([], [], color=COL["proxy"], marker="o", lw=1.0, ms=4, label="proxy relation"),
    Line2D([], [], color=COL["combination"], marker="o", lw=1.0, ms=4, label="combination-unseen"),
    Line2D([], [], color=COL["7-ingredient"], marker="o", lw=1.0, ms=4, label="mix relations"),
], fontsize=5.8, frameon=False, loc="lower left", ncol=2)
save(fig, "Figure_2_admissible_contrasts")

# ---------------------------------------------------------------- F3
dec = pd.read_csv(AUD / "D_outer_results.csv").dropna(subset=["realised_unsafe"])
agg = (dec.groupby(["scenario", "alpha_target", "strategy"])
       .agg(est=("est_unsafe", "mean"), real=("realised_unsafe", "mean"),
            cov=("realised_coverage", "mean")).reset_index())
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=False)
for ax, scen, title in ((axes[0], "corroded_beam", "Beam release (corroded RC)"),
                        (axes[1], "failure_mode", "Failure-mode action (shear walls)")):
    sub = agg[agg["scenario"] == scen]
    w = 0.36
    xs = np.arange(len(sorted(sub["alpha_target"].unique())))
    for off, strat, col in ((-w / 2, "random", "#bdbdbd"), (w / 2, "source", "#1f4e79")):
        s = sub[sub["strategy"] == strat].sort_values("alpha_target")
        ax.bar(xs + off, s["real"], width=w * 0.92, color=col,
               label=f"realised ({strat})" if scen == "corroded_beam" else None)
        ax.plot(xs + off, s["est"], ls="none", marker="_", ms=9, color="#c00000",
                label="estimated" if (scen == "corroded_beam" and strat == "random") else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a:.2f}" for a in sorted(sub['alpha_target'].unique())])
    ax.set_xlabel(r"nominal target $\alpha$")
    ax.set_title(title, fontsize=7.5)
    ax.set_ylim(0, max(0.12, sub["real"].max() * 1.35))
axes[0].set_ylabel("unsafe-release rate")
axes[0].legend(fontsize=6, frameon=False, loc="upper left")
axes[1].text(0.02, 0.95, "coverage 0.95 → 0.66 at α = 0.05", transform=axes[1].transAxes,
             fontsize=6, va="top", color="#444444")
fig.suptitle("Estimating the safety of a release with random folds vs source-honouring folds", fontsize=8, y=1.04)
save(fig, "Figure_5_decision_experiment")

# ---------------------------------------------------------------- F4
feas = pd.read_csv(AUD / "cfst_feasibility_map.csv")
Ks = sorted(feas["K"].unique())
fmaxes = list(dict.fromkeys(feas.sort_values("f_max_dec")["f_max"].tolist()))
mat = np.zeros((len(Ks), len(fmaxes)))
for i, K in enumerate(Ks):
    for j, fm in enumerate(fmaxes):
        row = feas[(feas["K"] == K) & (feas["f_max"] == fm)].iloc[0]
        mat[i, j] = 2 if row["feasible"] else (1 if "max_w" in str(row["blocked_by"]) else 0)
fig, ax = plt.subplots(figsize=(4.4, 2.8))
cmap = matplotlib.colors.ListedColormap(["#8c1d18", "#f2c14e", "#2e7d32"])
ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
labels = {0: "no (m<K)", 1: "no (max $w$>U)", 2: "yes"}
for i in range(len(Ks)):
    for j in range(len(fmaxes)):
        ax.text(j, i, labels[int(mat[i, j])], ha="center", va="center", fontsize=6,
                color="white" if mat[i, j] != 1 else "#3d2b00")
ax.set_xticks(range(len(fmaxes)))
ax.set_xticklabels([f"{fm}\n({feas[feas['f_max'] == fm]['f_max_dec'].iloc[0]:.2f})" for fm in fmaxes], fontsize=5.6)
ax.set_yticks(range(len(Ks)))
ax.set_yticklabels([f"K = {k}" for k in Ks], fontsize=6.5)
ax.set_xlabel("declared fold-upper bound $f_{max}$")
ax.set_title("Stub CFST (674 / 396 / 246): where a family-honouring split exists", fontsize=7.5)
ax.text(0, -0.95, "green = feasible   amber = blocked only by the fold-upper bound   red = structurally impossible",
        fontsize=5.8, color="#333333")
save(fig, "Figure_3_feasibility_map")

# ---------------------------------------------------------------- F5
reg = pd.read_csv(AUD / "configuration_registry.csv")
wall = pd.read_csv(PROJ / "code/outputs/reproductions/designsafe_prj2430_wall/extracted_wall_data.csv")
shape_col = next(c for c in wall.columns if c.lower().startswith("walltype"))
rel = pd.DataFrame({"source": wall["Authors"].astype(str), "shape": wall[shape_col].astype(str)})
parent = list(range(len(rel)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for col in ("source", "shape"):
    first = {}
    for i, v in enumerate(rel[col]):
        if v in first:
            ra, rb = find(first[v]), find(i)
            if ra != rb:
                parent[rb] = ra
        else:
            first[v] = i
comp = pd.Series([find(i) for i in range(len(rel))]).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
order = reg.sort_values("relation_kind")
ypos = np.arange(len(order))
colmap = {"source-level": "#1f4e79", "proxy-family": "#c55a11", "duplication": "#7f7f7f"}
ax.barh(ypos, [1] * len(order), color=[colmap[k] for k in order["relation_kind"]])
for i, r in enumerate(order.itertuples()):
    ax.text(0.02, i, f"{PRETTY.get(r.dataset, r.dataset)} — {r.relation_kind}", va="center",
            fontsize=6, color="white")
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Relation kind per configuration", fontsize=7.5)
ax.invert_yaxis()

ax = axes[1]
ax.bar([0], [int(comp.max())], color="#8c1d18", width=0.5)
ax.bar([1], [len(rel) - int(comp.max())], color="#d9d9d9", width=0.5)
ax.text(0, int(comp.max()) * 1.02, f"{int(comp.max())} rows\n({int(comp.max())/len(rel)*100:.0f}%)",
        ha="center", fontsize=6, color="#8c1d18")
ax.set_xticks([0, 1])
ax.set_xticklabels(["largest union\ncomponent", "other rows"], fontsize=6.5)
ax.set_ylabel("rows")
ax.set_title(f"Wall asset: {len(comp)} component from 28 sources x 5 wall types", fontsize=7.5)
save(fig, "Figure_4_relation_coverage")

print("figures written to", FIG)


