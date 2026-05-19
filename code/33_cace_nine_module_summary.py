"""
Aggregate headline RandomForest metrics across the nine reproduction modules.

Outputs:
  - code/outputs/figures/cace_ninemodule_rf_summary.csv
  - figures/draft/figS5_cace_ninemodule_rf_gaps_v0.png (+ .pdf) — submission numbering
  - figures/draft/fig_cace_ninemodule_rf_gaps_v0.png (+ .pdf) — legacy alias
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
REPRO = CODE / "outputs" / "reproductions"
FIG_OUT = PROJECT / "figures" / "draft"
TAB_OUT = CODE / "outputs" / "figures"


def _severity(gap: float, metric_kind: str) -> str:
    if metric_kind == "contrast":
        return "contrast"
    mag = abs(gap)
    if mag < 0.10:
        return "negligible"
    if metric_kind == "classification":
        return "moderate" if mag <= 0.25 else "severe"
    if metric_kind == "regression":
        return "moderate" if mag <= 0.35 else "severe"
    if mag <= 0.30:
        return "moderate"
    return "severe"


def _add_row(rows: list, **kwargs) -> None:
    rows.append(kwargs)


def load_mangalathu(rows: list) -> None:
    df = pd.read_csv(REPRO / "10-1016-j-engstruct-2019-110331" / "results.csv")
    rf = df[df["model"].eq("RandomForest")]
    r = rf[rf["cv"].eq("RandomStratifiedKFold-5")].iloc[0]
    g = rf[rf["cv"].eq("GroupKFold-by-Author-5")].iloc[0]
    gap_acc = float(r["accuracy_mean"] - g["accuracy_mean"])
    if "balanced_accuracy_mean" in r.index and "balanced_accuracy_mean" in g.index:
        gap_bacc = float(r["balanced_accuracy_mean"] - g["balanced_accuracy_mean"])
        rb, gb = float(r["balanced_accuracy_mean"]), float(g["balanced_accuracy_mean"])
    else:
        gap_bacc = np.nan
        rb, gb = np.nan, np.nan
    gap_f1 = float(r["macro_f1_mean"] - g["macro_f1_mean"])
    _add_row(
        rows,
        module_id=1,
        label="Shear-wall failure classification (Mangalathu)",
        task="classification",
        group_variable="Author/source-family proxy",
        n_samples=393,
        n_groups=75,
        model="Random Forest",
        random_split="Stratified 5-fold",
        grouped_split="GroupKFold-by-Author (5 folds)",
        headline_metric="accuracy",
        random_score=r["accuracy_mean"],
        grouped_score=g["accuracy_mean"],
        gap=gap_acc,
        gap_balanced_accuracy=gap_bacc,
        gap_macro_f1=gap_f1,
        secondary_random_balanced_accuracy=rb,
        secondary_grouped_balanced_accuracy=gb,
        secondary_random_macro_f1=r["macro_f1_mean"],
        secondary_grouped_macro_f1=g["macro_f1_mean"],
        severity=_severity(gap_acc, "classification"),
    )


def _pick_rf(df: pd.DataFrame) -> pd.DataFrame:
    m = df["model"].astype(str)
    sel = df[m.isin(["Random Forest", "RandomForest"])].copy()
    if sel.empty:
        raise ValueError("No RandomForest rows in dataframe.")
    return sel


def _rf_regression_gap(df: pd.DataFrame, random_key: str, group_key: str) -> tuple[float, float, float]:
    rf = _pick_rf(df)
    r = rf[rf["validation"].eq(random_key)].iloc[0]
    g = rf[rf["validation"].eq(group_key)].iloc[0]
    return float(r["pooled_r2"]), float(g["pooled_r2"]), float(r["pooled_r2"] - g["pooled_r2"])


def load_sfrc(rows: list) -> None:
    df = pd.read_csv(REPRO / "10-1016-j-engstruct-2020-111743" / "results.csv")
    r2r, r2g, gap = _rf_regression_gap(df, "RandomKFold", "GroupKFold_Source")
    _add_row(
        rows,
        module_id=2,
        label="SFRC shear capacity (Lantsoght DB; Rahman-like task)",
        task="regression",
        group_variable="Source-reference groups",
        n_samples=488,
        n_groups=118,
        model="Random Forest",
        random_split="Random 10-fold",
        grouped_split="GroupKFold by source",
        headline_metric="pooled R2",
        random_score=r2r,
        grouped_score=r2g,
        gap=gap,
        gap_balanced_accuracy=np.nan,
        gap_macro_f1=np.nan,
        secondary_random_balanced_accuracy=np.nan,
        secondary_grouped_balanced_accuracy=np.nan,
        secondary_random_macro_f1=np.nan,
        secondary_grouped_macro_f1=np.nan,
        severity=_severity(gap, "regression"),
    )


def load_uci(rows: list) -> None:
    df = pd.read_csv(REPRO / "10-1016-j-conbuildmat-2020-120950" / "results.csv")
    r2r, r2g, gap = _rf_regression_gap(df, "RandomKFold", "GroupKFold_Mix")
    _add_row(
        rows,
        module_id=3,
        label="UCI concrete strength (mix-family grouping)",
        task="regression",
        group_variable="Identical mixture proportions",
        n_samples=1030,
        n_groups=428,
        model="Random Forest",
        random_split="Random 10-fold",
        grouped_split="GroupKFold by mix",
        headline_metric="pooled R2",
        random_score=r2r,
        grouped_score=r2g,
        gap=gap,
        gap_balanced_accuracy=np.nan,
        gap_macro_f1=np.nan,
        secondary_random_balanced_accuracy=np.nan,
        secondary_grouped_balanced_accuracy=np.nan,
        secondary_random_macro_f1=np.nan,
        secondary_grouped_macro_f1=np.nan,
        severity=_severity(gap, "regression"),
    )


def load_corroded(rows: list) -> None:
    df = pd.read_csv(REPRO / "10-5281-zenodo-8062007" / "results.csv")
    r2r, r2g, gap = _rf_regression_gap(df, "RandomKFold", "GroupKFold_Source")
    _add_row(
        rows,
        module_id=4,
        label="Corroded RC beam moment capacity (Zenodo)",
        task="regression",
        group_variable="Experimental programme (parsed)",
        n_samples=804,
        n_groups=54,
        model="Random Forest",
        random_split="Random 10-fold",
        grouped_split="GroupKFold by source",
        headline_metric="pooled R2",
        random_score=r2r,
        grouped_score=r2g,
        gap=gap,
        gap_balanced_accuracy=np.nan,
        gap_macro_f1=np.nan,
        secondary_random_balanced_accuracy=np.nan,
        secondary_grouped_balanced_accuracy=np.nan,
        secondary_random_macro_f1=np.nan,
        secondary_grouped_macro_f1=np.nan,
        severity=_severity(gap, "regression"),
    )


def load_stub_cfst(rows: list) -> None:
    df = pd.read_csv(REPRO / "10-1038-s41598-024-53352-1" / "results.csv")
    rf = df[df["model"].eq("RandomForest")]
    r = rf[rf["validation"].eq("RandomKFold")].iloc[0]
    g = rf[rf["validation"].eq("GroupKFold_Shape")].iloc[0]
    gap = float(r["pooled_r2"] - g["pooled_r2"])
    _add_row(
        rows,
        module_id=5,
        label="Stub-CFST axial capacity (section-family holdout)",
        task="regression (deployment extrapolation)",
        group_variable="Section family (circular / rectangular / DS)",
        n_samples=1316,
        n_groups=3,
        model="Random Forest",
        random_split="Random 10-fold",
        grouped_split="GroupKFold by section family",
        headline_metric="pooled R2",
        random_score=float(r["pooled_r2"]),
        grouped_score=float(g["pooled_r2"]),
        gap=gap,
        gap_balanced_accuracy=np.nan,
        gap_macro_f1=np.nan,
        secondary_random_balanced_accuracy=np.nan,
        secondary_grouped_balanced_accuracy=np.nan,
        secondary_random_macro_f1=np.nan,
        secondary_grouped_macro_f1=np.nan,
        severity=_severity(gap, "regression"),
    )


def load_mendeley(rows: list) -> None:
    df = pd.read_csv(REPRO / "mendeley_beam_column_joint" / "results.csv")
    specs = [
        (
            "Exterior joint shear strength",
            "shear_strength_regression",
            "regression",
            "Reference-coded authors",
            6,
        ),
        (
            "Exterior joint failure mode",
            "failure_mode_classification",
            "classification",
            "Reference-coded authors",
            6,
        ),
        (
            "Cyclic joint shear strength",
            "joint_shear_strength_regression",
            "regression",
            "Research team blocks",
            6,
        ),
    ]
    for lab, task, kind, grp, mid in specs:
        rrow = df[(df["model"].eq("Random Forest")) & (df["task"].eq(task)) & (df["split"].eq("RandomKFold"))]
        grow = df[(df["model"].eq("Random Forest")) & (df["task"].eq(task)) & (df["split"].eq("GroupKFold"))]
        if rrow.empty or grow.empty:
            continue
        rrow, grow = rrow.iloc[0], grow.iloc[0]
        if kind == "regression":
            gap = float(rrow["pooled_r2"] - grow["pooled_r2"])
            _add_row(
                rows,
                module_id=mid,
                label=f"Mendeley — {lab}",
                task="regression",
                group_variable=grp,
                n_samples=int(rrow["n_samples"]),
                n_groups=int(rrow["n_groups"]),
                model="Random Forest",
                random_split="Random K-fold",
                grouped_split="GroupKFold",
                headline_metric="pooled R2",
                random_score=float(rrow["pooled_r2"]),
                grouped_score=float(grow["pooled_r2"]),
                gap=gap,
                gap_balanced_accuracy=np.nan,
                gap_macro_f1=np.nan,
                secondary_random_balanced_accuracy=np.nan,
                secondary_grouped_balanced_accuracy=np.nan,
                secondary_random_macro_f1=np.nan,
                secondary_grouped_macro_f1=np.nan,
                severity=_severity(gap, "regression"),
            )
        else:
            gap_acc = float(rrow["accuracy"] - grow["accuracy"])
            if (
                "balanced_accuracy" in rrow.index
                and pd.notna(rrow["balanced_accuracy"])
                and pd.notna(grow["balanced_accuracy"])
            ):
                rb, gb = float(rrow["balanced_accuracy"]), float(grow["balanced_accuracy"])
                gap_bacc = float(rb - gb)
            else:
                rb = gb = np.nan
                gap_bacc = np.nan
            gap_f1 = float(rrow["f1_macro"] - grow["f1_macro"])
            _add_row(
                rows,
                module_id=mid,
                label=f"Mendeley — {lab}",
                task="classification",
                group_variable=grp,
                n_samples=int(rrow["n_samples"]),
                n_groups=int(rrow["n_groups"]),
                model="Random Forest",
                random_split="Stratified / grouped CV (see script)",
                grouped_split="GroupKFold",
                headline_metric="accuracy",
                random_score=float(rrow["accuracy"]),
                grouped_score=float(grow["accuracy"]),
                gap=gap_acc,
                gap_balanced_accuracy=gap_bacc,
                gap_macro_f1=gap_f1,
                secondary_random_balanced_accuracy=rb,
                secondary_grouped_balanced_accuracy=gb,
                secondary_random_macro_f1=float(rrow["f1_macro"]),
                secondary_grouped_macro_f1=float(grow["f1_macro"]),
                severity=_severity(gap_acc, "classification"),
            )


def load_designsafe_columns(rows: list) -> None:
    df = pd.read_csv(REPRO / "designsafe_rc_columns" / "results.csv")
    sub = df[(df["case"].eq("DesignSafe RC columns combined")) & (df["model"].eq("Random Forest"))]
    r = sub[sub["split"].eq("RandomKFold")].iloc[0]
    g = sub[sub["split"].eq("GroupKFold")].iloc[0]
    gap = float(r["pooled_r2"] - g["pooled_r2"])
    _add_row(
        rows,
        module_id=7,
        label="DesignSafe RC columns (circular + rectangular combined)",
        task="regression",
        group_variable="Author/source proxy",
        n_samples=int(r["n_samples"]),
        n_groups=int(r["n_groups"]),
        model="Random Forest",
        random_split="Random K-fold",
        grouped_split="GroupKFold",
        headline_metric="pooled R2",
        random_score=float(r["pooled_r2"]),
        grouped_score=float(g["pooled_r2"]),
        gap=gap,
        gap_balanced_accuracy=np.nan,
        gap_macro_f1=np.nan,
        secondary_random_balanced_accuracy=np.nan,
        secondary_grouped_balanced_accuracy=np.nan,
        secondary_random_macro_f1=np.nan,
        secondary_grouped_macro_f1=np.nan,
        severity=_severity(gap, "regression"),
    )


def load_prj2430(rows: list) -> None:
    df = pd.read_csv(REPRO / "designsafe_prj2430_wall" / "results.csv")
    for target, lab in (
        ("Vmax", "PRJ-2430 wall peak lateral strength"),
        ("driftCap", "PRJ-2430 wall drift capacity"),
    ):
        sub = df[(df["target"].eq(target)) & (df["model"].eq("Random Forest"))]
        r = sub[sub["split"].eq("RandomKFold")].iloc[0]
        g = sub[sub["split"].eq("GroupKFold")].iloc[0]
        gap = float(r["pooled_r2"] - g["pooled_r2"])
        _add_row(
            rows,
            module_id=8,
            label=f"DesignSafe — {lab}",
            task="regression",
            group_variable="Author/source groups",
            n_samples=int(r["n_samples"]),
            n_groups=int(r["n_groups"]),
            model="Random Forest",
            random_split="Random K-fold",
            grouped_split="GroupKFold",
            headline_metric="pooled R2",
            random_score=float(r["pooled_r2"]),
            grouped_score=float(g["pooled_r2"]),
            gap=gap,
            gap_balanced_accuracy=np.nan,
            gap_macro_f1=np.nan,
            secondary_random_balanced_accuracy=np.nan,
            secondary_grouped_balanced_accuracy=np.nan,
            secondary_random_macro_f1=np.nan,
            secondary_grouped_macro_f1=np.nan,
            severity=_severity(gap, "regression"),
        )


def load_prj3053(rows: list) -> None:
    df = pd.read_csv(REPRO / "designsafe_prj3053_coupling_beams" / "results.csv")
    for target, short in (
        ("V_m_avg_kips", "peak shear V_m_avg"),
        ("v_m_normalized", "normalized shear"),
        ("CR_capacity_average", "chord rotation capacity"),
    ):
        sub = df[(df["target"].eq(target)) & (df["model"].eq("Random Forest"))]
        r = sub[sub["split"].eq("RandomKFold")].iloc[0]
        g = sub[sub["split"].eq("GroupKFold")].iloc[0]
        gap = float(r["pooled_r2"] - g["pooled_r2"])
        sev = "contrast" if gap < -0.02 else _severity(gap, "regression")
        _add_row(
            rows,
            module_id=9,
            label=f"PRJ-3053 coupling beam — {short}",
            task="regression",
            group_variable="Literature reference groups",
            n_samples=int(r["n_samples"]),
            n_groups=int(r["n_groups"]),
            model="Random Forest",
            random_split="Random K-fold",
            grouped_split="GroupKFold",
            headline_metric="pooled R2",
            random_score=float(r["pooled_r2"]),
            grouped_score=float(g["pooled_r2"]),
            gap=gap,
            gap_balanced_accuracy=np.nan,
            gap_macro_f1=np.nan,
            secondary_random_balanced_accuracy=np.nan,
            secondary_grouped_balanced_accuracy=np.nan,
            secondary_random_macro_f1=np.nan,
            secondary_grouped_macro_f1=np.nan,
            severity=sev,
        )


def make_figure(df: pd.DataFrame) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    order = df["label"].tolist()
    y = np.arange(len(order))
    gaps = df["gap"].to_numpy(dtype=float)
    colors = []
    for s in df["severity"]:
        colors.append({"negligible": "#72B7B2", "moderate": "#F58518", "severe": "#E45756", "contrast": "#B279A2"}.get(s, "#333333"))

    fig, ax = plt.subplots(figsize=(9.2, max(4.5, 0.32 * len(order))))
    ax.barh(y, gaps, color=colors, linewidth=0.8, edgecolor="#333333")
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Headline metric gap (random − grouped/deployment split)")
    # Figure number belongs in the caption; avoid "(draft)" or internal labels on the exported PNG/PDF.
    ax.set_title("Random Forest headline gaps (nine reproduction modules)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color="#72B7B2", markersize=8, label="negligible"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#F58518", markersize=8, label="moderate"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#E45756", markersize=8, label="severe"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#B279A2", markersize=8, label="contrast / reversed"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    for name in (
        "fig_cace_ninemodule_rf_gaps_v0.png",
        "figS5_cace_ninemodule_rf_gaps_v0.png",
    ):
        fig.savefig(FIG_OUT / name, dpi=300, bbox_inches="tight")
    for name in ("fig_cace_ninemodule_rf_gaps_v0.pdf", "figS5_cace_ninemodule_rf_gaps_v0.pdf"):
        fig.savefig(FIG_OUT / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows: list = []
    load_mangalathu(rows)
    load_sfrc(rows)
    load_uci(rows)
    load_corroded(rows)
    load_stub_cfst(rows)
    load_mendeley(rows)
    load_designsafe_columns(rows)
    load_prj2430(rows)
    load_prj3053(rows)

    df = pd.DataFrame(rows)
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    out_csv = TAB_OUT / "cace_ninemodule_rf_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")

    make_figure(df)
    print(f"[OK] {FIG_OUT / 'fig_cace_ninemodule_rf_gaps_v0.png'}")
    print(f"[OK] {FIG_OUT / 'figS5_cace_ninemodule_rf_gaps_v0.png'} (submission numbering)")


if __name__ == "__main__":
    main()
