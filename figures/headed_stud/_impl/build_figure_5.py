from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = ROOT.parent
CASE_ROOT = (
    ROUND_ROOT
    / "experiments"
    / "real_data"
    / "external_stud_family"
)
MULTIRELATION_ROOT = (
    ROUND_ROOT
    / "experiments"
    / "real_data"
    / "real_multirelation_contract_v1"
)
SVG_PATH = ROOT / "output" / "svg" / "Figure_5.svg"
PDF_PATH = ROOT / "output" / "pdf" / "Figure_5.pdf"
PNG_PATH = ROOT / "output" / "png" / "Figure_5.png"

W, H = 2400, 1460
NAVY = "#163A5F"
NAVY_LIGHT = "#EAF1F7"
TEAL = "#147D82"
TEAL_LIGHT = "#DCEFED"
AMBER = "#D98E04"
AMBER_LIGHT = "#FFF0CF"
RED = "#C23B22"
RED_LIGHT = "#FBE8E4"
VIOLET = "#7057A3"
VIOLET_LIGHT = "#EEEAF6"
TEXT = "#17212B"
MUTED = "#5F6B76"
GRID = "#AAB4BD"
PANEL = "#FAFBFC"
WHITE = "#FFFFFF"


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str = "none",
    sw: float = 1.5,
    rx: float = 0,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dashed}/>'
    )


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: float = 22,
    fill: str = TEXT,
    weight: int = 400,
    anchor: str = "start",
    italic: bool = False,
) -> str:
    style = "italic" if italic else "normal"
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>'
    )


def multiline(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: float = 18,
    fill: str = TEXT,
    weight: int = 400,
    anchor: str = "start",
    leading: float | None = None,
) -> str:
    step = leading if leading is not None else size + 8
    return "\n".join(
        text(x, y + index * step, line, size=size, fill=fill, weight=weight,
             anchor=anchor)
        for index, line in enumerate(lines)
    )


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = GRID,
    marker: str = "gray",
    sw: float = 2.6,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="M {x1} {y1} H {x2}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#arrow-{marker})"{dashed}/>'
    )


def elbow_arrow(
    d: str,
    *,
    color: str,
    marker: str,
    sw: float = 2.8,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#arrow-{marker})"{dashed}/>'
    )


def defs() -> str:
    colors = {
        "gray": GRID,
        "navy": NAVY,
        "teal": TEAL,
        "amber": AMBER,
        "red": RED,
        "violet": VIOLET,
    }
    parts = ["<defs>"]
    for name, color in colors.items():
        parts.append(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
    parts.append(
        '<pattern id="aggregate-pattern" width="12" height="12" '
        'patternUnits="userSpaceOnUse"><circle cx="3" cy="3" r="1.7" '
        f'fill="{VIOLET}" opacity="0.55"/><circle cx="9" cy="8" r="1.5" '
        f'fill="{AMBER}" opacity="0.55"/></pattern>'
    )
    parts.append("</defs>")
    return "\n".join(parts)


def panel(x: float, y: float, w: float, h: float, label: str) -> str:
    return "\n".join(
        [
            rect(x, y, w, h, fill=PANEL, stroke="#D5DDE3", sw=2, rx=18),
            text(x + 24, y + 42, label, size=25, fill=TEXT, weight=700),
        ]
    )


def slab_glyph(x: float, y: float, kind: str) -> str:
    parts: list[str] = []
    if kind == "deck":
        parts.append(
            f'<path d="M {x} {y+38} L {x+18} {y+54} L {x+38} {y+38} '
            f'L {x+58} {y+54} L {x+78} {y+38}" fill="none" '
            f'stroke="{AMBER}" stroke-width="5"/>'
        )
        parts.append(rect(x, y + 8, 78, 30, fill=NAVY_LIGHT, stroke=NAVY, sw=1.5, rx=4))
    else:
        fill = "url(#aggregate-pattern)" if kind in {"lwc", "rac"} else NAVY_LIGHT
        stroke = VIOLET if kind == "rac" else NAVY
        parts.append(rect(x, y + 14, 78, 34, fill=fill, stroke=stroke, sw=1.7, rx=4))
        parts.append(
            f'<path d="M {x-5} {y+50} H {x+83}" stroke="{GRID}" '
            'stroke-width="4"/>'
        )
    stud_color = TEAL if kind != "rac" else VIOLET
    parts.extend(
        [
            f'<path d="M {x+39} {y+4} V {y+45}" stroke="{stud_color}" '
            'stroke-width="5"/>',
            f'<path d="M {x+29} {y+5} H {x+49}" stroke="{stud_color}" '
            'stroke-width="5" stroke-linecap="round"/>',
        ]
    )
    return "\n".join(parts)


def object_card(
    x: float,
    y: float,
    title: str,
    subtitle: str,
    *,
    kind: str,
) -> str:
    return "\n".join(
        [
            rect(x, y, 370, 82, fill=WHITE, stroke=NAVY, sw=2, rx=11),
            slab_glyph(x + 18, y + 10, kind),
            text(x + 116, y + 32, title, size=19, fill=NAVY, weight=700),
            text(x + 116, y + 60, subtitle, size=15.5, fill=MUTED),
        ]
    )


def filter_card(
    x: float,
    y: float,
    title: str,
    subtitle: str,
    *,
    color: str,
    light: str,
) -> str:
    return "\n".join(
        [
            rect(x, y, 285, 72, fill=light, stroke=color, sw=2, rx=10),
            text(x + 142.5, y + 29, title, size=17.5, fill=color,
                 weight=700, anchor="middle"),
            text(x + 142.5, y + 54, subtitle, size=13.5, fill=MUTED,
                 anchor="middle"),
        ]
    )


def retained_card(
    x: float,
    y: float,
    main: str,
    subtitle: str,
) -> str:
    return "\n".join(
        [
            rect(x, y, 430, 82, fill=VIOLET_LIGHT, stroke=VIOLET, sw=2.2, rx=11),
            text(x + 215, y + 35, main, size=21, fill=VIOLET,
                 weight=700, anchor="middle"),
            text(x + 215, y + 63, subtitle, size=14.5, fill=MUTED,
                 anchor="middle"),
        ]
    )


def contract_card(
    x: float,
    y: float,
    title: str,
    lines: list[str],
) -> str:
    return "\n".join(
        [
            rect(x, y, 500, 118, fill=NAVY_LIGHT, stroke=NAVY, sw=2.1, rx=12),
            text(x + 22, y + 34, title, size=20, fill=NAVY, weight=700),
            multiline(x + 22, y + 65, lines, size=15.5, fill=TEXT, leading=24),
        ]
    )


def status_card(
    x: float,
    y: float,
    w: float,
    title: str,
    subtitle: str,
    *,
    color: str,
    light: str,
    title_size: float = 18,
) -> str:
    return "\n".join(
        [
            rect(x, y, w, 92, fill=light, stroke=color, sw=2.2, rx=11),
            text(x + w / 2, y + 35, title, size=title_size, fill=color,
                 weight=700, anchor="middle"),
            text(x + w / 2, y + 65, subtitle, size=14, fill=MUTED,
                 anchor="middle"),
        ]
    )


def load_evidence() -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = json.loads((CASE_ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    summaries = {item["parent"]: item for item in manifest["summaries"]}
    contracts = json.loads((CASE_ROOT / "r" / "contracts_v3.json").read_text(encoding="utf-8"))
    statuses = contracts["status_summary"]

    expected_counts = {
        "nwc242": (242, 242, 20),
        "deck464": (464, 413, 21),
        "lwc90": (90, 60, 5),
        "rac27": (27, 27, 1),
    }
    for parent, (raw, retained, recovered_groups) in expected_counts.items():
        item = summaries[parent]
        assert int(item["n_raw"]) == raw
        assert int(item["n_retained"]) == retained
        assert int(item["n_retained_exact_source_groups"]) == recovered_groups
    assert int(summaries["deck464"]["n_source_quarantine"]) == 25
    assert int(summaries["deck464"]["n_missing_target_or_feature"]) == 26
    assert int(summaries["deck464"]["n_unresolved_source_rows"]) == 12
    assert int(summaries["lwc90"]["n_source_quarantine"]) == 30

    expected_statuses = {
        "C_source_claimcut": "compiled_exact",
        "C_source_groupkfold": "realized_plan_invalid_balance",
        "C_deck_claimcut": "certified_partition_infeasible",
        "C_lwc_claimcut": "certified_partition_infeasible",
        "C_rac_claimcut": "certified_partition_infeasible",
        "C_deck_datasail_lossy_plan": "plan_emitted_lossy",
        "C_deck_datasail_lossy_audited": "assignment_audited_lossy",
    }
    for key, value in expected_statuses.items():
        assert statuses[key]["status"] == value
    assert statuses["C_deck_datasail_lossy_audited"]["full_contract_satisfied"] is False
    return summaries, statuses


def load_multirelation_evidence() -> dict[str, object]:
    def read_csv(name: str) -> list[dict[str, str]]:
        with (MULTIRELATION_ROOT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    cells = {
        (row["slab_topology"], row["concrete_family"]): row
        for row in read_csv("cells_v11.csv")
    }
    summaries = read_csv("sum_v11.csv")
    cell_folds = read_csv("cellfold_v11.csv")
    statuses = {row["backend"]: row for row in read_csv("status_v11.csv")}
    freeze = json.loads((MULTIRELATION_ROOT / "freeze_v11.json").read_text(encoding="utf-8"))

    expected_cells = {
        ("solid", "normal_weight"): (242, 20),
        ("solid", "lightweight"): (60, 5),
        ("profiled_deck", "normal_weight"): (401, 21),
    }
    assert set(cells) == set(expected_cells)
    for key, (records, groups) in expected_cells.items():
        assert int(cells[key]["n_records"]) == records
        assert int(cells[key]["n_source_groups"]) == groups
    assert sum(int(row["n_records"]) for row in cells.values()) == 703
    assert sum(int(row["n_source_groups"]) for row in cells.values()) == 46
    assert [int(row["n_records"]) for row in summaries] == [141, 141, 141, 140, 140]
    assert all(float(row["source_novelty"]) == 1.0 for row in summaries)
    assert all(float(row["slab_topology_novelty"]) == 0.0 for row in summaries)
    assert all(float(row["concrete_family_novelty"]) == 0.0 for row in summaries)
    assert all(float(row["regime_novelty"]) == 0.0 for row in summaries)
    assert all(float(row["joint_total_variation"]) == 0.0 for row in summaries)
    assert all(float(row["topology_material_mean_degree"]) == 1.5 for row in summaries)
    lwc_by_fold = [
        int(row["n_records"])
        for row in cell_folds
        if row["structural_material_regime"] == "solid__lightweight"
    ]
    assert min(lwc_by_fold) == 3
    assert statuses["claimcut"]["status"] == "compiled_exact"
    assert statuses["group_kfold"]["status"] == "realized_plan_invalid_balance"
    assert statuses["datasail_c1e_scalar"]["status"] == "plan_emitted_lossy"
    assert freeze["protocol_id"] == "R30-HS-MR-CONTRACT-V1.1"
    assert freeze["strict_file_level_outcome_blind"] is True
    assert freeze["target_values_accessed"] is False
    assert freeze["predictive_results_accessed"] is False

    return {
        "cells": cells,
        "fold_sizes": [int(row["n_records"]) for row in summaries],
        "min_lwc_per_fold": min(lwc_by_fold),
    }


def build_svg() -> str:
    summaries, _ = load_evidence()
    multirelation = load_multirelation_evidence()
    cells = multirelation["cells"]
    deck_total_clusters = (
        int(summaries["deck464"]["n_retained_exact_source_groups"]) + 1
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Headed-stud evidence graph and deployment representability">',
        defs(),
        rect(0, 0, W, H, fill=WHITE),
        text(60, 58,
             "Headed-stud evidence graph: from source audit to deployment representability",
             size=36, fill=TEXT, weight=700),
        text(60, 96,
             "Aggregate-only, outcome-blind row flow and compiler-v3 states",
             size=20, fill=MUTED),
        panel(45, 125, 2310, 560,
              "(a)  Public parent evidence after frozen source and completeness gates"),
        panel(45, 720, 2310, 590,
              "(b)  Development provenance and observed-support contract realization"),
    ]

    lane_y = [220, 330, 440, 550]
    object_x, filter1_x, filter2_x, retained_x, note_x = 85, 520, 885, 1270, 1760
    lanes = [
        (
            "NWC solid slab", "242 records", "solid",
            ("no overlap removal", "development parent", MUTED, "#F2F4F6"),
            ("7-feature complete", "frozen adapter", TEAL, TEAL_LIGHT),
            "242 retained | 20 groups", "development evidence",
            "role: model development", "C_source: split-exact",
        ),
        (
            "Profiled steel deck", "464 records", "deck",
            ("-25 overlap rows", "source quarantine", RED, RED_LIGHT),
            ("-26 missing-feature", "frozen complete-case", AMBER, AMBER_LIGHT),
            f"413 retained | {deck_total_clusters} clusters", "primary external parent",
            "21 recovered + 1 unresolved cluster", "C_deck: partition-infeasible",
        ),
        (
            "LWC solid slab", "90 records", "lwc",
            ("-30 overlap rows", "source quarantine", RED, RED_LIGHT),
            ("0 missing-feature", "after quarantine", TEAL, TEAL_LIGHT),
            "60 retained | 5 groups", "near-domain control",
            "small source-composition sensitivity", "C_LWC: partition-infeasible",
        ),
        (
            "RAC-SCC solid slab", "27 positive-RCA", "rac",
            ("no overlap removal", "declared programme", MUTED, "#F2F4F6"),
            ("7-feature complete", "frozen adapter", TEAL, TEAL_LIGHT),
            "27 retained | 1 programme", "bounded sensitivity only",
            "no source-population inference", "C_RAC: partition-infeasible",
        ),
    ]
    for y, lane in zip(lane_y, lanes):
        title, subtitle, kind, f1, f2, retained, retained_sub, note, contract_tag = lane
        parts.extend(
            [
                object_card(object_x, y, title, subtitle, kind=kind),
                arrow(object_x + 370, y + 41, filter1_x, y + 41,
                      color=GRID, marker="gray"),
                filter_card(filter1_x, y + 5, f1[0], f1[1],
                            color=f1[2], light=f1[3]),
                arrow(filter1_x + 285, y + 41, filter2_x, y + 41,
                      color=GRID, marker="gray"),
                filter_card(filter2_x, y + 5, f2[0], f2[1],
                            color=f2[2], light=f2[3]),
                arrow(filter2_x + 285, y + 41, retained_x, y + 41,
                      color=VIOLET, marker="violet"),
                retained_card(retained_x, y, retained, retained_sub),
                text(note_x, y + 34, note, size=15.5, fill=MUTED, weight=600),
                text(note_x, y + 61, contract_tag, size=13.5,
                     fill=NAVY, weight=700),
            ]
        )

    # Development provenance object.
    parts.extend(
        [
            rect(80, 805, 390, 400, fill=WHITE, stroke=NAVY, sw=2.2, rx=14),
            text(275, 845, "NWC development provenance", size=21, fill=NAVY,
                 weight=700, anchor="middle"),
            slab_glyph(235, 865, "solid"),
            multiline(115, 965,
                      ["242 records", "20 source groups", "slab topology = solid",
                       "concrete family = NWC"], size=17, fill=TEXT, leading=31),
            rect(105, 1100, 340, 72, fill="#EEF3F5", stroke="#A7B3BC",
                 sw=1.5, rx=9),
            text(275, 1129, "compiler target access: NO", size=17,
                 fill=TEXT, weight=700, anchor="middle"),
            text(275, 1155, "provenance fields only", size=14,
                 fill=MUTED, anchor="middle"),
        ]
    )

    contract_x = 555
    contract_y = [800, 945]
    contract_details = [
        ("C_source", ["new publication/source", "same recorded engineering object"]),
        ("C_deck", ["new source + profiled-deck topology", "deployment relation absent in NWC"]),
    ]
    for y, (title_value, lines) in zip(contract_y, contract_details):
        parts.append(contract_card(contract_x, y, title_value, lines))
        parts.append(
            elbow_arrow(
                f"M 470 1005 H 505 V {y+59} H {contract_x}",
                color=NAVY, marker="navy", sw=2.6,
            )
        )

    # Source results.
    parts.extend(
        [
            arrow(1055, 859, 1135, 859, color=TEAL, marker="teal"),
            status_card(1135, 812, 405, "ClaimCut: split-exact*",
                        "recorded source clauses satisfied | executable",
                        color=TEAL, light=TEAL_LIGHT),
            status_card(1580, 812, 650, "GroupKFold: realized plan invalid",
                        "source separated | frozen fold bounds fail",
                        color=RED, light=RED_LIGHT, title_size=17.5),
            elbow_arrow("M 1055 859 H 1090 V 785 H 1905 V 812",
                        color=RED, marker="red", sw=2.5, dash="7,6"),
        ]
    )

    # Deck results and DataSAIL lifecycle.
    parts.extend(
        [
            arrow(1055, 1004, 1135, 1004, color=RED, marker="red"),
            status_card(1135, 957, 405, "ClaimCut: partition-infeasible",
                        "one hard component for five folds",
                        color=RED, light=RED_LIGHT, title_size=17.5),
            status_card(1580, 957, 195, "DataSAIL plan",
                        "emitted lossy", color=AMBER, light=AMBER_LIGHT,
                        title_size=16.5),
            status_card(1810, 957, 195, "Assignment",
                        "audited lossy", color=VIOLET, light=VIOLET_LIGHT,
                        title_size=16.5),
            status_card(2040, 957, 190, "Deck split contract",
                        "NOT SATISFIED", color=RED, light=RED_LIGHT,
                        title_size=16.5),
            elbow_arrow("M 1055 1004 H 1090 V 930 H 1677 V 957",
                        color=AMBER, marker="amber", sw=2.5),
            arrow(1775, 1004, 1810, 1004, color=VIOLET, marker="violet"),
            arrow(2005, 1004, 2040, 1004, color=RED, marker="red"),
        ]
    )

    # Observed-support multi-relation contract: aggregate-only and outcome-blind.
    solid_nwc = cells[("solid", "normal_weight")]
    solid_lwc = cells[("solid", "lightweight")]
    deck_nwc = cells[("profiled_deck", "normal_weight")]
    fold_sizes = ", ".join(str(value) for value in multirelation["fold_sizes"])
    inset_x, inset_y, inset_w, inset_h = 555, 1082, 1675, 135
    matrix_x, matrix_y = 575, 1114
    parts.extend(
        [
            rect(inset_x, inset_y, inset_w, inset_h, fill=WHITE,
                 stroke=NAVY, sw=1.8, rx=11),
            text(inset_x + 18, inset_y + 23,
                 "Observed-support contract | 703 records | 46 sources | 3 regimes",
                 size=16.5, fill=NAVY, weight=700),
            text(matrix_x + 176, matrix_y + 13, "NWC", size=12.5,
                 fill=MUTED, weight=700, anchor="middle"),
            text(matrix_x + 326, matrix_y + 13, "LWC", size=12.5,
                 fill=MUTED, weight=700, anchor="middle"),
            text(matrix_x, matrix_y + 43, "Solid slab", size=12.5,
                 fill=TEXT, weight=700),
            text(matrix_x, matrix_y + 73, "Profiled deck", size=12.5,
                 fill=TEXT, weight=700),
            rect(matrix_x + 105, matrix_y + 22, 142, 25, fill=VIOLET_LIGHT,
                 stroke=VIOLET, sw=1.2, rx=4),
            text(matrix_x + 176, matrix_y + 40,
                 f'{solid_nwc["n_records"]} | {solid_nwc["n_source_groups"]} sources',
                 size=11.8, fill=VIOLET, weight=700, anchor="middle"),
            rect(matrix_x + 255, matrix_y + 22, 142, 25, fill=NAVY_LIGHT,
                 stroke=NAVY, sw=1.2, rx=4),
            text(matrix_x + 326, matrix_y + 40,
                 f'{solid_lwc["n_records"]} | {solid_lwc["n_source_groups"]} sources',
                 size=11.8, fill=NAVY, weight=700, anchor="middle"),
            rect(matrix_x + 105, matrix_y + 52, 142, 25, fill=VIOLET_LIGHT,
                 stroke=VIOLET, sw=1.2, rx=4),
            text(matrix_x + 176, matrix_y + 70,
                 f'{deck_nwc["n_records"]} | {deck_nwc["n_source_groups"]} sources',
                 size=11.8, fill=VIOLET, weight=700, anchor="middle"),
            rect(matrix_x + 255, matrix_y + 52, 142, 25, fill=AMBER_LIGHT,
                 stroke=AMBER, sw=1.3, rx=4, dash="5,4"),
            text(matrix_x + 326, matrix_y + 70, "not observed", size=11.8,
                 fill=AMBER, weight=700, anchor="middle"),
            rect(1000, 1112, 600, 78, fill=TEAL_LIGHT, stroke=TEAL,
                 sw=1.8, rx=8),
            text(1020, 1132, "ClaimCut: split-exact", size=14.5,
                 fill=TEAL, weight=700),
            text(1020, 1151, f"fold n = {fold_sizes}", size=12.2, fill=TEXT),
            text(1020, 1168,
                 "source novel; topology/material/regime supported", size=12.2,
                 fill=TEXT),
            text(1020, 1185, "joint TV = 0 | mean incidence = 1.5",
                 size=12.2, fill=TEXT),
            rect(1620, 1112, 285, 78, fill=RED_LIGHT, stroke=RED,
                 sw=1.8, rx=8),
            text(1762.5, 1137, "GroupKFold", size=14.5, fill=RED,
                 weight=700, anchor="middle"),
            text(1762.5, 1163, "invalid: balance + incidence", size=12.2,
                 fill=TEXT, anchor="middle"),
            rect(1920, 1112, 290, 78, fill=AMBER_LIGHT, stroke=AMBER,
                 sw=1.8, rx=8),
            text(2065, 1137, "DataSAIL", size=14.5, fill=AMBER,
                 weight=700, anchor="middle"),
            text(2065, 1163, "lossy plan; not executed", size=12.2,
                 fill=TEXT, anchor="middle"),
            text(1392.5, 1208,
                 f'Observed union only | deck x LWC absent | min LWC per fold = {multirelation["min_lwc_per_fold"]} | metadata-only split audit',
                 size=12.3, fill=MUTED, weight=600, anchor="middle"),
        ]
    )

    parts.extend(
        [
            rect(80, 1230, 2150, 55, fill="#EEF3F5", stroke="#A7B3BC",
                 sw=1.6, rx=10),
            text(1155, 1265,
                 "Executable source holdout != split-exact deck or material evidence",
                 size=21, fill=TEXT, weight=700, anchor="middle"),
            text(1200, 1350,
                 "Structural glyphs are schematic identifiers; all counts and states are verified aggregates. Solid arrows encode row flow, compilation, execution, or audit.",
                 size=15.5, fill=MUTED, anchor="middle"),
            text(1200, 1380,
                 "DataSAIL is shown as a strong comparator under a different abstraction; lossy status is a task/adapter semantic gap, not an algorithmic defect.",
                 size=15.5, fill=MUTED, anchor="middle"),
            text(1200, 1420,
                 "* Frozen software label: compiled_exact; interpreted only as split-exact under recorded provenance. The inset uses the observed three-regime union only.",
                 size=14.2, fill=MUTED, anchor="middle"),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    for path in (SVG_PATH.parent, PDF_PATH.parent, PNG_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    SVG_PATH.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(PDF_PATH))
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(PNG_PATH),
        output_width=3200,
    )
    print(f"SVG: {SVG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
