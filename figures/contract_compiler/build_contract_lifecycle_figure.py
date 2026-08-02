from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "output" / "svg" / "Figure_1.svg"
PDF_PATH = ROOT / "output" / "pdf" / "Figure_1.pdf"
PNG_PATH = ROOT / "output" / "png" / "Figure_1.png"

W, H = 2400, 1330
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


def rect(x: float, y: float, w: float, h: float, *, fill: str,
         stroke: str = "none", sw: float = 1.5, rx: float = 0,
         dash: str | None = None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def text(x: float, y: float, value: str, *, size: float = 24,
         fill: str = TEXT, weight: int = 400, anchor: str = "start",
         italic: bool = False) -> str:
    style = "italic" if italic else "normal"
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>'
    )


def path(d: str, *, stroke: str = GRID, sw: float = 2.5,
         marker: str = "gray", dash: str | None = None,
         opacity: float = 1) -> str:
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'opacity="{opacity}" marker-end="url(#arrow-{marker})"{ds}/>'
    )


def panel(x: float, y: float, w: float, h: float, label: str,
          color: str) -> str:
    return "\n".join([
        rect(x, y, w, h, fill=PANEL, stroke="#D5DDE3", sw=2, rx=18),
        text(x + 24, y + 42, label, size=24, fill=color, weight=700),
    ])


def card(x: float, y: float, w: float, h: float, title: str,
         lines: list[str], *, color: str, light: str,
         footer: str | None = None, title_size: float = 21,
         line_size: float = 17) -> str:
    header_h = 48
    parts = [
        rect(x, y, w, h, fill=WHITE, stroke=color, sw=2.2, rx=12),
        f'<path d="M {x+12} {y} H {x+w-12} Q {x+w} {y} {x+w} {y+12} '
        f'V {y+header_h} H {x} V {y+12} Q {x} {y} {x+12} {y} Z" fill="{color}"/>',
        text(x + w / 2, y + 32, title, size=title_size, fill=WHITE,
             weight=700, anchor="middle"),
    ]
    yy = y + header_h + 30
    for line_value in lines:
        parts.append(text(x + 18, yy, line_value, size=line_size, fill=TEXT))
        yy += line_size + 12
    if footer:
        parts.extend([
            rect(x + 1, y + h - 42, w - 2, 41, fill=light, stroke="none", rx=10),
            text(x + w / 2, y + h - 15, footer, size=14.5, fill=color,
                 weight=700, anchor="middle"),
        ])
    return "\n".join(parts)


def state_card(x: float, y: float, w: float, h: float, title: str,
               subtitle: str, *, color: str, light: str) -> str:
    return "\n".join([
        rect(x, y, w, h, fill=light, stroke=color, sw=2.2, rx=12),
        text(x + w / 2, y + 35, title, size=19, fill=color,
             weight=700, anchor="middle"),
        text(x + w / 2, y + 68, subtitle, size=14.5, fill=MUTED,
             anchor="middle"),
    ])


def defs() -> str:
    colors = {
        "gray": GRID,
        "navy": NAVY,
        "teal": TEAL,
        "amber": AMBER,
        "red": RED,
        "violet": VIOLET,
    }
    items = ["<defs>"]
    for name, color in colors.items():
        items.append(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
    items.append("</defs>")
    return "\n".join(items)


def build_svg() -> str:
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Typed deployment-validation compiler lifecycle">',
        defs(),
        rect(0, 0, W, H, fill=WHITE),
        text(60, 59, "Typed compilation trace: deployment intent → auditable validation decision",
             size=37, fill=TEXT, weight=700),
        text(60, 96,
             "Outcome-blind knowledge compilation; an emitted backend plan is not a realized fold assignment.",
             size=21, fill=MUTED),
    ]

    # Stage panels.
    p.extend([
        panel(45, 130, 710, 460, "(a)  Declare an engineering knowledge object", NAVY),
        panel(785, 130, 1570, 460, "(b)  Compile representability before fitting", TEXT),
        panel(45, 625, 690, 500, "(c)  Plan layer — specification ≠ folds", NAVY),
        panel(765, 625, 1040, 500, "(d)  Assignment layer — realized folds", VIOLET),
        panel(1835, 625, 520, 500, "(e)  Validation card", TEXT),
    ])

    # Declare.
    p.append(card(
        75, 210, 245, 135, "Deployment use",
        ["prediction unit + target", "decision context"],
        color=NAVY, light=NAVY_LIGHT, title_size=20, line_size=16,
    ))
    p.append(card(
        75, 395, 245, 130, "Provenance table",
        ["records × typed fields", "missing stays missing"],
        color="#6C8194", light="#EFF3F6", title_size=20, line_size=16,
    ))
    p.append(card(
        370, 215, 350, 310, "deployment-contract-v3",
        ["ESTIMAND", "outcome · loss · population", "weighting · risk · decision", "",
         "SPLIT SEMANTICS", "hard · marginal · joint", "incidence · support · order"],
        color=NAVY, light=NAVY_LIGHT, footer="complete object enters the contract hash",
        title_size=19, line_size=15.5,
    ))
    p.extend([
        path("M 320 275 H 350 Q 365 275 365 290 V 320 H 370",
             stroke=NAVY, marker="navy"),
        text(344, 260, "encode", size=13.5, fill=NAVY, anchor="middle"),
        path("M 320 460 H 350 Q 365 460 365 445 V 420 H 370",
             stroke="#6C8194", marker="gray"),
        text(344, 480, "bind", size=13.5, fill=MUTED, anchor="middle"),
    ])

    # Compile.
    p.append(card(
        815, 220, 320, 260, "Static capability analysis",
        ["backend clause coverage", "analytic necessary conditions",
         "ordered-relation check", "frozen search route"],
        color=TEXT, light="#EFF2F4", footer="no features, outcomes, errors, or ranks",
        title_size=18, line_size=15.5,
    ))
    p.append(state_card(
        1180, 205, 340, 100, "FULL CONTRACT EXPOSED",
        "exact-capable route · not yet compiled_exact",
        color=TEAL, light=TEAL_LIGHT,
    ))
    p.append(state_card(
        1180, 350, 340, 100, "LOSSY PROJECTION",
        "semantic gap declared before execution",
        color=AMBER, light=AMBER_LIGHT,
    ))
    p.append(state_card(
        1570, 205, 330, 100, "CERTIFIED INFEASIBLE",
        "independently verifiable predicate",
        color=RED, light=RED_LIGHT,
    ))
    p.append(state_card(
        1940, 205, 375, 100, "SEARCH EXHAUSTED",
        "unresolved · not an infeasibility proof",
        color=AMBER, light="#FFF7E5",
    ))
    p.append(state_card(
        1760, 370, 400, 100, "ORDERED ROUTE REQUIRED",
        "use the predeclared forward alternative",
        color=VIOLET, light=VIOLET_LIGHT,
    ))

    p.extend([
        path("M 720 370 H 815", stroke=TEXT, marker="gray", sw=3),
        text(768, 352, "compile", size=14, fill=TEXT, weight=700, anchor="middle"),
        path("M 1135 280 H 1180", stroke=TEAL, marker="teal"),
        path("M 1135 350 H 1180",
             stroke=AMBER, marker="amber"),
        path("M 1135 235 H 1150 V 185 H 1545 V 255 H 1570",
             stroke=RED, marker="red"),
        path("M 1135 315 H 1155 V 535 H 2330 V 255 H 2315",
             stroke=AMBER, marker="amber", dash="8,7"),
        path("M 1135 420 H 1160 V 515 H 2180 V 420 H 2160",
             stroke=VIOLET, marker="violet"),
    ])

    # Plan layer.
    p.append(card(
        80, 705, 300, 315, "Exact plan emitted",
        ["backend specification", "plan hash", "fidelity = exact",
         "all clauses represented"],
        color=TEAL, light=TEAL_LIGHT, footer="NO FOLD ASSIGNMENT YET",
        title_size=19, line_size=16,
    ))
    p.append(card(
        400, 705, 300, 315, "plan_emitted_lossy",
        ["backend specification", "plan hash", "fidelity = lossy",
         "semantic-gap manifest"],
        color=AMBER, light=AMBER_LIGHT, footer="NOT EXECUTABLE EVIDENCE BY ITSELF",
        title_size=18, line_size=16,
    ))

    # Exact/lossy route into plan layer; use a shared horizontal bus.
    p.extend([
        path("M 1350 305 V 575 H 230 V 705", stroke=TEAL, marker="teal", sw=3),
        text(720, 566, "emit exact plan", size=14, fill=TEAL, weight=700,
             anchor="middle"),
        path("M 1350 450 V 605 H 550 V 705", stroke=AMBER, marker="amber", sw=3),
        text(915, 600, "emit declared projection", size=14, fill=AMBER,
             weight=700, anchor="middle"),
    ])

    # Assignment layer.
    p.append(card(
        795, 705, 250, 170, "Execute backend",
        ["may return folds,", "fail, or return nothing"],
        color="#6C8194", light="#EFF3F6", title_size=18, line_size=15.5,
    ))
    p.append(card(
        1080, 705, 250, 170, "Returned assignment",
        ["record → fold labels", "plan integrity checked"],
        color=VIOLET, light=VIOLET_LIGHT, title_size=17.5, line_size=15.5,
    ))
    p.append(card(
        1365, 690, 405, 205, "Complete realized audit",
        ["nonempty folds + integer balance", "hard separation + support",
         "marginal + joint + incidence", "full contract + projection"],
        color=VIOLET, light=VIOLET_LIGHT, title_size=19, line_size=15.5,
    ))
    p.extend([
        path("M 230 1020 V 1085 H 745 V 775 H 795",
             stroke=TEAL, marker="teal", sw=3),
        path("M 550 1020 V 1105 H 755 V 820 H 795",
             stroke=AMBER, marker="amber", sw=3),
        text(625, 1058, "execute plan", size=14, fill=MUTED,
             weight=700, anchor="middle"),
        path("M 1045 790 H 1080", stroke="#6C8194", marker="gray"),
        text(1062, 772, "if returned", size=12.5, fill=MUTED, anchor="middle"),
        path("M 1330 790 H 1365", stroke=VIOLET, marker="violet"),
        text(1348, 772, "audit", size=12.5, fill=VIOLET, anchor="middle"),
    ])

    p.append(state_card(
        800, 950, 300, 105, "compiled_exact",
        "assignment_state = audited exact",
        color=TEAL, light=TEAL_LIGHT,
    ))
    p.append(state_card(
        1135, 950, 300, 105, "assignment_audited_lossy",
        "projection passes · full contract does not",
        color=AMBER, light=AMBER_LIGHT,
    ))
    p.append(state_card(
        1470, 950, 300, 105, "realized_plan_invalid_*",
        "failed hard, balance, or target audit",
        color=RED, light=RED_LIGHT,
    ))
    p.extend([
        path("M 1480 895 V 920 H 950 V 950", stroke=TEAL, marker="teal"),
        path("M 1565 895 V 925 H 1285 V 950", stroke=AMBER, marker="amber"),
        path("M 1650 895 V 930 H 1620 V 950", stroke=RED, marker="red"),
    ])

    # Validation card.
    p.append(card(
        1870, 700, 450, 355, "Auditable decision record",
        ["contract + provenance hashes", "backend + semantic gaps",
         "plan state + fidelity", "assignment state + residuals",
         "certificate / reason / route"],
        color=TEXT, light="#EFF2F4", footer="FIT IF ADMISSIBLE · OTHERWISE STOP / ROUTE",
        title_size=20, line_size=17,
    ))
    p.extend([
        path("M 950 1055 V 1075 H 1815 V 790 H 1870",
             stroke=TEAL, marker="teal"),
        path("M 1285 1055 V 1095 H 1820 V 860 H 1870",
             stroke=AMBER, marker="amber"),
        path("M 1620 1055 V 1110 H 1825 V 930 H 1870",
             stroke=RED, marker="red"),
    ])

    # Non-executable states bypass plan and assignment.
    p.extend([
        path("M 1735 305 V 560 H 2325 V 735",
             stroke=RED, marker="red", dash="9,7", sw=2.8),
        text(2020, 551, "certificate · no assignment", size=14, fill=RED,
             weight=700, anchor="middle"),
        path("M 2128 305 V 540 H 2340 V 805 H 2320",
             stroke=AMBER, marker="amber", dash="9,7", sw=2.8),
        text(2208, 526, "unresolved · no admissible candidate", size=14,
             fill=AMBER, weight=700, anchor="middle"),
        path("M 2160 420 H 2370 V 875 H 2320",
             stroke=VIOLET, marker="violet", dash="9,7", sw=2.8),
        text(2285, 405, "route alternative", size=14, fill=VIOLET,
             weight=700, anchor="middle"),
    ])

    # Boundary ribbon.
    p.extend([
        rect(45, 1160, 2310, 115, fill="#EEF3F5", stroke="#A7B3BC", sw=2, rx=16),
        text(1200, 1202,
             "COMPILER BOUNDARY",
             size=19, fill=MUTED, weight=700, anchor="middle"),
        text(1200, 1235,
             "Only provenance fields named by the contract enter compilation — not features, observed outcomes, external errors, or model ranks.",
             size=19, fill=TEXT, weight=600, anchor="middle"),
        text(1200, 1263,
             "Exact semantic realization does not guarantee robustness to unrecorded mechanism or distribution shift.",
             size=17, fill=MUTED, anchor="middle"),
        text(1200, 1310,
             "Solid arrows: lifecycle handoff or audited transition. Dashed arrows: status-only path with no realized assignment.",
             size=15.5, fill=MUTED, anchor="middle"),
        "</svg>",
    ])
    return "\n".join(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output",
        help="Path-neutral output directory containing svg/pdf/png subfolders.",
    )
    args = parser.parse_args()
    svg_path = args.output_root / "svg" / "Figure_1.svg"
    pdf_path = args.output_root / "pdf" / "Figure_1.pdf"
    png_path = args.output_root / "png" / "Figure_1.png"
    for path_value in (svg_path.parent, pdf_path.parent, png_path.parent):
        path_value.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(pdf_path))
    raster_svg = svg.replace(
        ' role="img" ',
        ' role="img" text-rendering="geometricPrecision" ',
        1,
    )
    if raster_svg == svg:
        raise RuntimeError("could not apply deterministic PNG text rendering")
    cairosvg.svg2png(
        bytestring=raster_svg.encode("utf-8"),
        write_to=str(png_path),
        output_width=3200,
    )
    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
