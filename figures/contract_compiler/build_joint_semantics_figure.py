from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "output" / "svg" / "Figure_2.svg"
PDF_PATH = ROOT / "output" / "pdf" / "Figure_2.pdf"
PNG_PATH = ROOT / "output" / "png" / "Figure_2.png"

W, H = 2100, 1180
NAVY = "#163A5F"
TEAL = "#147D82"
TEAL_LIGHT = "#DCEFED"
AMBER = "#D98E04"
AMBER_LIGHT = "#FFF0CF"
VIOLET = "#7057A3"
VIOLET_LIGHT = "#EEEAF6"
TEXT = "#17212B"
MUTED = "#5F6B76"
GRID = "#AAB4BD"
ZERO = "#F0F3F5"
PANEL = "#FAFBFC"
WHITE = "#FFFFFF"


def rect(x: float, y: float, w: float, h: float, *, fill: str, stroke: str = "none",
         sw: float = 1.5, rx: float = 0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID,
         sw: float = 2, dash: str | None = None, opacity: float = 1) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}/>'
    )


def text(x: float, y: float, value: str, *, size: float = 28, fill: str = TEXT,
         weight: int = 400, anchor: str = "start", italic: bool = False,
         letter: float = 0) -> str:
    style = "italic" if italic else "normal"
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'letter-spacing="{letter}" text-anchor="{anchor}" fill="{fill}">'
        f'{escape(value)}</text>'
    )


def circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = WHITE,
           sw: float = 3) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def joint_matrix(x: float, y: float, title: str, values: list[list[str]],
                 occupied: set[tuple[int, int]], color: str, light: str) -> str:
    cell = 104
    parts: list[str] = []
    parts.append(text(x + cell, y - 78, title, size=32, fill=color, weight=700, anchor="middle"))
    parts.append(text(x + cell, y - 43, "constructive probability mass", size=17,
                      fill=MUTED, anchor="middle"))
    parts.append(text(x + cell, y - 7, "u_B", size=22, fill=VIOLET, weight=700, anchor="middle"))
    parts.append(text(x - 62, y + cell, "u_A", size=22, fill=NAVY, weight=700,
                      anchor="middle"))
    parts.append(text(x + cell * 0.5, y + 28, "0", size=19, fill=VIOLET,
                      weight=700, anchor="middle"))
    parts.append(text(x + cell * 1.5, y + 28, "1", size=19, fill=VIOLET,
                      weight=700, anchor="middle"))
    parts.append(text(x - 25, y + cell * 0.5 + 7, "0", size=19, fill=NAVY,
                      weight=700, anchor="middle"))
    parts.append(text(x - 25, y + cell * 1.5 + 7, "1", size=19, fill=NAVY,
                      weight=700, anchor="middle"))

    for r in range(2):
        for c in range(2):
            fill = light if (r, c) in occupied else ZERO
            stroke = color if (r, c) in occupied else GRID
            parts.append(rect(x + c * cell, y + r * cell, cell, cell,
                              fill=fill, stroke=stroke, sw=2.5, rx=7))
            parts.append(text(x + (c + 0.5) * cell, y + (r + 0.5) * cell + 10,
                              values[r][c], size=30,
                              fill=color if (r, c) in occupied else MUTED,
                              weight=700 if (r, c) in occupied else 400,
                              anchor="middle"))

    parts.append(text(x + 2 * cell + 31, y - 4, "row sum", size=15,
                      fill=MUTED, anchor="middle"))
    for r in range(2):
        parts.append(text(x + 2 * cell + 31, y + (r + 0.5) * cell + 8,
                          "½", size=23, fill=TEXT, weight=700, anchor="middle"))
    parts.append(text(x + cell, y + 2 * cell + 42, "column sums:  ½     ½",
                      size=18, fill=TEXT, weight=700, anchor="middle"))
    return "\n".join(parts)


def dense_graph(x_a: float, x_b: float, y0: float) -> str:
    parts: list[str] = []
    ays = [y0, y0 + 150]
    bys = [y0, y0 + 150]
    for ay in ays:
        for by in bys:
            parts.append(line(x_a, ay, x_b, by, stroke=TEAL, sw=5, opacity=0.78))
    for i, ay in enumerate(ays, 1):
        parts.append(circle(x_a, ay, 26, fill=NAVY))
        parts.append(text(x_a, ay + 8, f"a{i}", size=19, fill=WHITE, weight=700,
                          anchor="middle"))
    for i, by in enumerate(bys, 1):
        parts.append(circle(x_b, by, 26, fill=VIOLET))
        parts.append(text(x_b, by + 8, f"b{i}", size=19, fill=WHITE, weight=700,
                          anchor="middle"))
    return "\n".join(parts)


def sparse_graph(x_a: float, x_b: float, y0: float) -> str:
    parts: list[str] = []
    ays = [y0, y0 + 75, y0 + 150, y0 + 225]
    bys = [y0 + 38, y0 + 188]
    mapping = [0, 0, 1, 1]
    for ay, j in zip(ays, mapping):
        parts.append(line(x_a, ay, x_b, bys[j], stroke=TEAL, sw=5, opacity=0.78))
    for i, ay in enumerate(ays, 1):
        parts.append(circle(x_a, ay, 23, fill=NAVY))
        parts.append(text(x_a, ay + 7, f"a{i}", size=17, fill=WHITE, weight=700,
                          anchor="middle"))
    for i, by in enumerate(bys, 1):
        parts.append(circle(x_b, by, 26, fill=VIOLET))
        parts.append(text(x_b, by + 8, f"b{i}", size=19, fill=WHITE, weight=700,
                          anchor="middle"))
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Constructive non-identifiability counterexample">',
        rect(0, 0, W, H, fill=WHITE),
        text(70, 62,
             "Constructive counterexample: marginals do not determine joint or incidence semantics",
             size=38, fill=TEXT, weight=700),
        text(70, 101,
             "For a validation fold, u_r = 1 when the record’s relation-r entity is absent from training.",
             size=21, fill=MUTED),
        rect(55, 135, 1210, 770, fill=PANEL, stroke="#D6DDE3", sw=2, rx=18),
        rect(1290, 135, 755, 770, fill=PANEL, stroke="#D6DDE3", sw=2, rx=18),
        text(85, 184, "(a)  Same novelty marginals  ≠  same joint contract",
             size=28, fill=NAVY, weight=700),
        text(1320, 184, "(b)  Same joint pattern  ≠  same incidence topology",
             size=28, fill=VIOLET, weight=700),
    ]

    parts.append(joint_matrix(
        185, 335, "Contract C+",
        [["½", "0"], ["0", "½"]],
        {(0, 0), (1, 1)}, TEAL, TEAL_LIGHT,
    ))
    parts.append(joint_matrix(
        745, 335, "Contract C×",
        [["0", "½"], ["½", "0"]],
        {(0, 1), (1, 0)}, AMBER, AMBER_LIGHT,
    ))
    parts.append(text(625, 455, "≠", size=58, fill=TEXT, weight=700, anchor="middle"))

    parts.extend([
        rect(130, 685, 1050, 78, fill="#EAF1F7", stroke="#9CB3C9", sw=1.8, rx=12),
        text(655, 718, "Both:  P(u_A = 1) = P(u_B = 1) = ½",
             size=25, fill=NAVY, weight=700, anchor="middle"),
        text(655, 748, "Identical marginal novelty targets",
             size=18, fill=MUTED, anchor="middle"),
        rect(130, 785, 1050, 78, fill="#F7F0E2", stroke="#D7B36A", sw=1.8, rx=12),
        text(655, 819, "But:  TV(π(C+) , π(C×)) = 1",
             size=25, fill=TEXT, weight=700, anchor="middle"),
        text(655, 849, "The joint contracts have disjoint support",
             size=18, fill=MUTED, anchor="middle"),
    ])

    parts.extend([
        rect(1330, 215, 675, 72, fill=VIOLET_LIGHT, stroke="#A492C7", sw=1.8, rx=12),
        text(1668, 245, "Both graphs:  four records and π(1,1) = 1",
             size=22, fill=VIOLET, weight=700, anchor="middle"),
        text(1668, 273, "Every record has the same joint novelty pattern",
             size=16, fill=MUTED, anchor="middle"),
        line(1665, 315, 1665, 702, stroke="#D0D6DC", sw=2, dash="6,8"),
        text(1495, 331, "Dense incidence", size=22, fill=TEAL, weight=700, anchor="middle"),
        text(1855, 331, "Sparse incidence", size=22, fill=AMBER, weight=700, anchor="middle"),
        text(1420, 365, "A", size=19, fill=NAVY, weight=700, anchor="middle"),
        text(1585, 365, "B", size=19, fill=VIOLET, weight=700, anchor="middle"),
        text(1760, 365, "A", size=19, fill=NAVY, weight=700, anchor="middle"),
        text(1960, 365, "B", size=19, fill=VIOLET, weight=700, anchor="middle"),
        dense_graph(1420, 1585, 430),
        sparse_graph(1760, 1960, 390),
        rect(1340, 705, 315, 116, fill=TEAL_LIGHT, stroke=TEAL, sw=2, rx=12),
        text(1498, 746, "4 unique A–B edges", size=19, fill=TEAL, weight=700, anchor="middle"),
        text(1498, 779, "mean distinct-B degree / A = 2", size=17, fill=TEXT,
             anchor="middle"),
        rect(1680, 705, 315, 116, fill=AMBER_LIGHT, stroke=AMBER, sw=2, rx=12),
        text(1838, 746, "4 unique A–B edges", size=19, fill=AMBER, weight=700, anchor="middle"),
        text(1838, 779, "mean distinct-B degree / A = 1", size=17, fill=TEXT,
             anchor="middle"),
        text(1668, 857,
             "The record-level joint distribution does not identify cross-relation incidence.",
             size=19, fill=MUTED, weight=600, anchor="middle"),
    ])

    parts.extend([
        rect(55, 935, 1990, 160, fill="#EEF3F5", stroke="#A7B3BC", sw=2, rx=18),
        text(1050, 979, "Logical consequence for the typed contract",
             size=25, fill=TEXT, weight=700, anchor="middle"),
        text(1050, 1023,
             "{ν_A, ν_B} does not identify π_C;  π_C does not identify A–B incidence.",
             size=28, fill=NAVY, weight=700, anchor="middle"),
        text(1050, 1064,
             "Marginal, joint-pattern, and incidence clauses must therefore remain separate and auditable.",
             size=21, fill=TEAL, weight=700, anchor="middle"),
        text(1050, 1140,
             "Constructive proof object — exact constants, not fitted results or a prevalence estimate; other provenance motifs may require additional clauses.",
             size=17, fill=MUTED, anchor="middle"),
        "</svg>",
    ])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output",
        help="Path-neutral output directory containing svg/pdf/png subfolders.",
    )
    args = parser.parse_args()
    svg_path = args.output_root / "svg" / "Figure_2.svg"
    pdf_path = args.output_root / "pdf" / "Figure_2.pdf"
    png_path = args.output_root / "png" / "Figure_2.png"
    for path in (svg_path.parent, pdf_path.parent, png_path.parent):
        path.mkdir(parents=True, exist_ok=True)
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
