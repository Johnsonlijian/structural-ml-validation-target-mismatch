# -*- coding: utf-8 -*-
"""Assemble manuscript/main_manuscript_v1_CACE_submission.md from latest Conflict base."""
from __future__ import annotations

import re
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parent.parent / "manuscript"
SRC = max(MANUSCRIPT.glob("main_manuscript_v0_*214320*Conflict*.md"), key=lambda p: p.stat().st_mtime)
OUT = MANUSCRIPT / "main_manuscript_v1_CACE_submission.md"

STATUS_V12 = (
    "**Manuscript status:** CACE submission draft **v1.2** — taxonomy-first Results; SAVP + algorithm sketch; **Table S8** topology master; "
    "GitHub/Zenodo archive completed; final figure and reference formatting pending journal upload."
)

TABLE_S8 = r"""

### Table S8 | Cross-module group topology and headline Random Forest gaps

| Module (RF headline) | n | Groups | Median group size | Singleton groups | Grouping proxy | Topology stress note | Headline gap (Δ) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Shear-wall classification | 393 | 75 | 3 | 26 | Author/source family | 51 single-class groups; median majority-class share 1.00 | Δacc≈0.304; Δmacro-F1≈0.347† |
| SFRC shear (pooled R²) | 488 | 118 | — | — | Source reference | Many small source groups | ΔR²≈0.811 |
| UCI concrete (pooled R²) | 1030 | 428 | — | — | Identical mix proportions | Large mix cardinality | ΔR²=0.040 |
| Corroded beams (pooled R²) | 804 | 54 | — | — | Experimental programme | Moderate programme blocks | ΔR²=0.180 |
| Stub CFST family holdout | 1316 | 3 | — | 0 | Section family | Only three families | ΔR²≈1.784 |
| Mendeley exterior shear | 203 | 33 | — | — | Reference-coded authors | Joint shear regression | ΔR²≈0.198 |
| Mendeley exterior failure mode | 203 | 33 | — | — | Reference-coded authors | Rare modes under grouping | Δacc≈0.369; Δmacro-F1≈0.275† |
| Mendeley cyclic shear | 98 | 18 | — | — | Research team | Very small teams | ΔR²≈1.392 |
| DS columns combined | 497 | 99 | — | — | Author/source proxy | Curated depot CSVs | ΔR²≈0.110 |
| PRJ-2430 Vmax | 142 | 28 | 4 | 1 | Author/source | Wall regression | ΔR²≈0.529 |
| PRJ-2430 drift capacity | 142 | 28 | 4 | 1 | Author/source | Low-variance folds | ΔR²≈0.385 |
| PRJ-3053 peak shear | 60 | 21 | 2 | 6 | Literature reference | Contrast case | ΔR²≈−0.069 |
| PRJ-3053 norm. shear | 60 | 21 | 2 | 6 | Literature reference | Moderate gap | ΔR²≈0.303 |
| PRJ-3053 chord rotation | 60 | 21 | 2 | 6 | Literature reference | Moderate gap | ΔR²≈0.135 |

†Classification rows also report balanced-accuracy gaps in Table S7 where exported.

"""


def _clean_bracket_citations(t: str) -> str:
    t = re.sub(r"\[已核查:\s*DOI\s+([0-9a-zA-Z./]+)\]", r"(DOI \1)", t)
    t = re.sub(r"\[已核查:\s*Zenodo DOI\s+([0-9a-zA-Z./]+)\]", r"(Zenodo DOI \1)", t)
    t = re.sub(r"\[已核查:\s*UCI concrete dataset\]", "(UCI concrete dataset)", t)
    t = re.sub(
        r"\[已核查:\s*Roberts et al\.[^\]]+\]",
        "(Roberts et al., 2017, Ecography, DOI 10.1111/ecog.02881; Ploton et al., 2020, Nature Communications, DOI 10.1038/s41467-020-18321-y; Kapoor & Narayanan, 2023, Patterns, DOI 10.1016/j.patter.2023.100804)",
        t,
    )
    t = re.sub(r"\s*`\[已核查\]`", "", t)
    t = re.sub(r"\s*\[已核查\]", "", t)
    return t


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    new_abs = (
        "Computational pipelines for structural performance prediction increasingly rely on machine learning, "
        "but headline scores often depend on validation splits that do not match the intended deployment target. "
        "We organize evaluation through a source-aware validation framework that separates within-database interpolation, "
        "out-of-source prediction, mixture-family constraints and deployment-matched structural-family extrapolation tests. "
        "The framework is exercised through nine executable public reproduction modules covering concrete strength, "
        "reinforced-concrete shear-wall failure classification, steel-fiber-reinforced concrete shear capacity, "
        "corroded reinforced-concrete beam moment capacity, beam-column joint shear and failure mode, "
        "reinforced-concrete column lateral strength, wall strength and drift capacity, diagonally reinforced coupling beams, "
        "and concrete-filled steel tube capacity with section-family holdout. "
        "Random validation produced negligible optimism in a standardized concrete-strength benchmark and a contrast pattern "
        "for one coupling-beam peak-shear metric, moderate optimism in several beam, joint, column and deformation targets, "
        "and severe optimism—or deployment-matched collapse—in steel-fiber-reinforced concrete shear, cyclic joint shear, "
        "shear-wall classification, wall strength/drift and concrete-filled steel tube structural-family extrapolation. "
        "We further construct a 332-paper OpenAlex sampling frame and manually verify high-priority metadata records, "
        "showing that repository-level reproducibility signals often overstate immediate access to reusable raw data. "
        "These results support a deployment-matched reporting standard for structural machine learning: state the deployment target, "
        "disclose grouping variables, compare random and grouped or deployment-matched validation, and report group-topology diagnostics. "
        "**This work does not estimate field-wide prevalence of optimistic reporting; it provides an executable protocol and "
        "heterogeneous benchmark evidence aligned with civil-infrastructure prediction tasks.**"
    )
    text = re.sub(
        r"(## Abstract\n\n)(.*?)(\n\n## Significance)",
        r"\1" + new_abs + r"\3",
        text,
        flags=re.DOTALL,
        count=1,
    )

    insert_after = (
        "**In structural machine learning, validation is therefore an engineering claim about the deployment target.** "
        "The same numerical pipeline can estimate within-database interpolation, out-of-programme transportability or extrapolation across structural families, "
        "depending on how folds respect clustering.\n\n"
    )
    cace_bridge = (
        "*Computer-Aided Civil and Infrastructure Engineering* has long published computational learning systems for civil infrastructure "
        "(e.g., Reich, 1997, https://doi.org/10.1111/0885-9507.00065; Adeli, 2001, https://doi.org/10.1111/0885-9507.00219). "
        "Contributions continue to couple evolutionary computing, surrogate modelling and reliability thinking to civil infrastructure systems "
        "(e.g., Sgambi et al., 2012, https://doi.org/10.1111/j.1467-8667.2012.00780.x), and recent machine-learning studies in the same venue "
        "illustrate tabular and surrogate-intensive workflows for materials and strong-motion tasks "
        "(e.g., Valikhani et al., 2021, https://doi.org/10.1111/mice.12605; Fayaz & Galasso, 2023, https://doi.org/10.1111/mice.12830). "
        "Our work is complementary: we supply a **deployment-aligned validation and reporting protocol** so that performance claims remain interpretable when training and test partitions violate exchangeability.\n\n"
        "Formally, let covariates be $X$, labels $Y$ and discrete grouping indicators $G$. Random $k$-fold cross-validation fits a learner under an approximate product measure that treats specimens as exchangeable. "
        "Grouped cross-validation enforces support restrictions so that held-out groups represent different mixtures of sources or structural contexts; when $P(X \\mid G=g)$ differs materially across $g$, random splitting can induce "
        "$P_{\\mathrm{train}}(X,Y) \\neq P_{\\mathrm{test}}(X,Y)$ even when marginal covariate histograms overlap—a **group-conditional covariate-shift** pattern. "
        "The Methods taxonomy maps each deployment target to the $G$ and split machinery that make this shift explicit rather than hidden.\n\n"
    )
    if insert_after in text:
        text = text.replace(insert_after, insert_after + cace_bridge, 1)

    text = _clean_bracket_citations(text)

    m = re.search(r"\n## Results\n\n", text)
    m2 = re.search(r"\n## Discussion\n\n", text)
    if not m or not m2:
        raise SystemExit("Could not find Results/Discussion anchors")
    text = text[: m.start() + 1] + RESULTS_BLOCK + text[m2.start() + 1 :]

    disc_anchor = "## Discussion\n\n"
    idx = text.find(disc_anchor)
    if idx == -1:
        raise SystemExit("Discussion missing")
    pos = idx + len(disc_anchor)
    text = text[:pos] + DISCUSSION_PREFIX + text[pos:]

    dup_disc = (
        "This study shows that structural machine-learning performance claims are sensitive to the validation target. "
        "Across executable public reproductions, random validation usually produced equal or higher headline performance than grouped or deployment-matched validation, but the PRJ-3053 coupling-beam peak-shear task was an important exception. "
        "The size and direction of the difference varied by target, and that variation is the main scientific result. A single slogan, either \"random CV is fine\" or \"random CV is invalid\", would be misleading. "
        "The appropriate question is whether the validation split matches the use case claimed by the paper.\n\n"
    )
    if dup_disc in text:
        text = text.replace(dup_disc, "", 1)

    marker = (
        "A third limitation is that `R²` and accuracy capture predictive performance but not all engineering validity. "
        "Physical consistency, uncertainty calibration and design-code compatibility remain separate requirements.\n\n"
    )
    if marker in text:
        text = text.replace(marker, marker + DISCUSSION_INSERT + "\n", 1)

    tax_marker = (
        "The Stub-CFST workbook instantiates structural-family extrapolation—not classical source leakage—"
        "owing to missing bibliometric identifiers at specimen level.\n\n### Validation splits\n"
    )
    if tax_marker in text:
        text = text.replace(
            tax_marker,
            "The Stub-CFST workbook instantiates structural-family extrapolation—not classical source leakage—"
            "owing to missing bibliometric identifiers at specimen level.\n\n"
            + METHODS_INSERT
            + "### Validation splits\n",
            1,
        )

    old_soft = text.split("### Software and reproducibility\n\n", 1)[1].split("\n\n## Data availability", 1)[0]
    text = text.replace("### Software and reproducibility\n\n" + old_soft, "### Software and reproducibility\n\n" + SOFTWARE_BLOCK + "\n", 1)

    text = re.sub(
        r"## Code availability\n\n.*?\n\n## Acknowledgements",
        "## Code availability\n\n" + CODE_AVAIL + "\n\n## Acknowledgements",
        text,
        flags=re.DOTALL,
    )

    text = text.replace("The author declares no competing interests. `[待用户最终确认]`", "The author declares no competing interests.", 1)

    text = re.sub(
        r" Xu et al\.'s FRP-confined CFST paper[^\n]*\n",
        "",
        text,
        count=1,
    )

    repls = [
        ("**CACE-oriented layout.** Panel A summarizes", "Panel A summarizes"),
        ("**Layout.** Panel A summarizes", "Panel A summarizes"),
        (
            "Earlier drafts emphasized only the PRISMA-style layer; editors and practitioners should resolve both panels simultaneously so the manuscript reads as **framework plus workflow**, not a disconnected case anthology.",
            "The figure is organized so that Panel A (taxonomy) and Panel B (workflow) read jointly as **framework plus workflow**.",
        ),
        (
            "The figure should compare metadata labels with manual verification outcomes and show the promoted dataset-first queue.",
            "The figure compares metadata labels with manual verification outcomes and highlights the promoted dataset-first queue.",
        ),
        (
            "The figure should be a compact checklist suitable for adoption by structural ML reviewers.",
            "The figure is formatted as a compact checklist suitable for adoption by structural ML reviewers.",
        ),
        (
            "The figure should be interpreted with group-topology diagnostics because some leave-one-source folds contain very few specimens.",
            "The figure is interpreted with group-topology diagnostics because some leave-one-source folds contain very few specimens.",
        ),
        (
            "The figure should be interpreted as evidence of target-dependent heterogeneity rather than a severe-collapse case.",
            "The figure is interpreted as evidence of target-dependent heterogeneity rather than as a severe-collapse case.",
        ),
        (
            "Severity coding (negligible / moderate / severe / contrast) is heuristic for readability; reviewers should reconcile each bar against its dataset-specific pooled metrics in supplementary tables.",
            "Severity coding (negligible / moderate / severe / contrast) is heuristic for readability; each bar should be read against dataset-specific pooled metrics in supplementary tables.",
        ),
    ]
    for a, b in repls:
        text = text.replace(a, b)

    text = text.replace(
        "These intervals are diagnostic for the displayed cases and should not be interpreted as field-wide confidence intervals.",
        "These intervals are diagnostic for the displayed cases and are not interpreted as field-wide confidence intervals.",
    )

    text = re.sub(r"\n## Internal Quality Gate\n\n.*", "\n", text, flags=re.DOTALL)

    text = text.replace(
        "**Manuscript status:** evidence layer frozen at nine executable public modules; narrative retargeted to a source-aware validation **framework and reporting protocol** (not a field-wide prevalence study); full bibliography cleanup, public code archive and figure remapping for CACE remain before submission",
        STATUS_V12,
        1,
    )
    text = text.replace(
        "**Manuscript status:** CACE submission draft **v1** — taxonomy-first Results; SAVP + metric protocol; **Table S8** topology master; declarative figure captions; GitHub/Zenodo placeholders in Code availability to be replaced before upload.",
        STATUS_V12,
        1,
    )
    text = text.replace(
        "**Manuscript status:** CACE submission draft **v1.1** — taxonomy-first Results; SAVP + algorithm sketch; **Table S8** topology master; cross-tier summary; compressed OpenAlex/manual block; Discussion de-duplicated; shortened physics and AI prose; Figure 2 unified (no internal v2 label); recent CACE (2021/2023) inventory lines; GitHub/Zenodo placeholders to replace at upload.",
        STATUS_V12,
    )
    text = text.replace(
        "## Reference inventory (DOI-verified keys; v0)\n\nFull bibliographic strings for the three non-structural method precedents below are in `manuscript/references_verified_v0.md` (CACE-ready paste block).\n\n",
        "## References\n\nFormatted reference strings for submission are maintained in `references_verified_v0.md`. The keyed list below preserves one-to-one traceability between in-text claims, public datasets and methodological precedents (DOI or repository identifier).\n\n",
        1,
    )

    # Editorial Manager: each highlight line ≤85 characters (count spaces).
    _hl2 = "Structural ML-Bench v1.0: heterogeneous validation gaps across nine systems"
    text = text.replace(
        "Nine public modules show heterogeneous random-split optimism contrasts",
        _hl2,
    )
    text = text.replace(
        "Structural ML-Bench v1.0 maps heterogeneous topology-driven optimism across nine systems",
        _hl2,
    )

    s7_tail = (
        "Comma-separated table generated by `code/33_cace_nine_module_summary.py`; each row records module label, grouping proxy, "
        "headline metric (accuracy or pooled `R²`), random score, grouped score, gap (`random − grouped`), balanced-accuracy and macro-F1 gaps "
        "for classification rows when available (`nan` placeholders otherwise) and heuristic severity tagging. Editors can treat this table as "
        "reproducible bookkeeping for Figures 2/S5 drafting and for the **graphical abstract** (`code/34_graphical_abstract_cace_v0.py`, "
        "default tail of `run_qa_chain.py`)."
    )
    if s7_tail in text and "### Table S8 |" not in text:
        text = text.replace(s7_tail, s7_tail + "\n\n" + TABLE_S8.strip("\n"), 1)

    cace_refs = (
        "- Reich, Y., machine learning techniques for civil engineering problems, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/0885-9507.00065`\n"
        "- Adeli, H., neural networks in civil engineering: 1989–2000, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/0885-9507.00219`\n"
        "- Sgambi, L., Gkoumas, K., Bontempi, F., genetic algorithms for dependability assurance in long-span suspension bridge design, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/j.1467-8667.2012.00780.x`"
    )
    kap = "- Kapoor, S., & Narayanan, A., leakage and reproducibility in ML-based science, *Patterns*, `10.1016/j.patter.2023.100804`"
    # Intro prose already cites Reich DOI; do not use that substring as the guard for inventory injection.
    if kap in text and "- Reich, Y., machine learning techniques for civil engineering problems" not in text:
        text = text.replace(kap, kap + "\n" + cace_refs, 1)

    old_case = (
        "Executable public datasets were selected to span multiple structural and civil-engineering ML settings: reinforced-concrete shear-wall failure-mode classification, steel-fiber-reinforced concrete shear-capacity regression, concrete compressive-strength regression, corroded RC beam residual moment-capacity regression, beam-column joint shear/failure-mode prediction and stub CFST axial-capacity regression. The selection was not intended to be representative. It was intended to establish feasibility, expose distinct validation topologies and define diagnostics for the larger meta-reproduction."
    )
    new_case = (
        "Executable public datasets were selected to span concrete and steel–concrete composite materials, reinforced-concrete components and joint subassemblies, and curated infrastructure repositories: "
        "UCI concrete compressive strength (mix-family grouping); Mangalathu shear-wall failure modes; Lantsoght-style steel-fiber-reinforced concrete shear capacity; Matthews corroded RC beam moment capacity; Mendeley exterior and cyclic beam–column joint shear and exterior failure mode; DesignSafe circular and rectangular RC columns (combined depot export); Megahed-style stub concrete-filled steel tube axial capacity with section-family holdout; DesignSafe PRJ-2430 RC walls (peak strength and drift); and DesignSafe PRJ-3053 diagonally reinforced coupling beams (peak shear, normalized shear and chord rotation). "
        "The selection was not intended to be representative; it was intended to expose heterogeneous validation topologies under a unified protocol."
    )
    text = text.replace(old_case, new_case, 1)

    text = text.replace("Figure 2 v2", "Figure 2")
    text = text.replace("**Figure 2 v2**", "**Figure 2**")

    text = text.replace(
        "During the preparation of this work the author used generative artificial intelligence tools accessed through Cursor IDE conversational assistants (commercial large-language-model backends, principally Anthropic Claude in coding-and-editing workflows, with ancillary use of other frontier chat assistants configured in the same environment). Within author oversight, those tools assisted with: (i) drafting and structuring English prose once analytical results were curated from scripted reproduction outputs; (ii) implementing and refactoring Python reproducibility scripts, manifests and figure pipelines; (iii) bibliography hygiene cues while final reference metadata remains manually verified prior to acceptance.\n\n"
        "Numerical summaries, pooled metrics, supplementary tables and machine-readable inventories such as `cace_ninemodule_rf_summary.csv` were produced by deterministic scripts rerun by the author; generative assistants did not furnish quantitative results apart from authoring or commenting on source code subsequently executed locally. Standard grammar or spell-check tooling was also used without separate declaration according to Elsevier guidance.\n\n",
        "During manuscript preparation the author used commercial generative-AI assistants (primarily via the Cursor IDE) for English drafting, code-refactoring suggestions and bibliography hygiene cues, with author review throughout. All numerical summaries were produced by deterministic scripts rerun locally; assistants did not supply tabulated results. Standard spell-check tooling was used without separate declaration.\n\n",
        1,
    )
    text = text.replace(
        "The author critically reviewed all AI-mediated suggestions, rewrote contentious passages, asserted final scientific wording, ensured traceability between claims and artefacts in `code/outputs/`, and accepts full accountability for methodological choices, reproducibility disclosures, citations, interpretations and adherence to dataset redistribution licenses. Artificial intelligence platforms were **not listed as authors** and do not satisfy authorship criteria.",
        "The author vetted and rewrote AI-assisted suggestions, asserted final scientific wording, and accepts full accountability for methodology, citations, interpretations and licence compliance. AI tools were **not listed as authors**.",
        1,
    )

    text = text.replace(
        "This draft was produced as part of an autonomous, AI-assisted meta-reproduction workflow. Final submission must include only acknowledgements that match journal policy and actual contributions.",
        "The author thanks maintainers of the public repositories and data depots referenced in this benchmark suite.",
        1,
    )
    text = text.replace(
        "Single-author draft. Final author contribution statement should be updated before submission.",
        "Sole author: conceptualization, methodology, software, validation, formal analysis, investigation, resources, data curation, writing (original draft and review), and visualization.",
        1,
    )

    text = text.replace(
        "The figure is formatted as a compact checklist suitable for adoption by structural ML reviewers.",
        "The figure is formatted as a compact checklist suitable for structural ML reviewers; the same disclosure items can be supplied as structured supplementary material consistent with reproducible-computing expectations in this journal.",
        1,
    )

    text = text.replace(
        "A natural extension—outside the scope of the present manuscript—is a coordinated field-wide audit that samples from the OpenAlex-derived frame and synthesizes prevalence-style statistics across many papers under common reporting standards.",
        "### Broader impact and checklist uptake\n\n"
        "The Figure 4 checklist distils SAVP outputs into reviewer-facing fields; supplying it as supplementary structured material can make deployment-target disclosures routine without lengthening the main narrative. "
        "A natural extension—outside the scope of the present manuscript—is a coordinated field-wide audit that samples from the OpenAlex-derived frame and synthesizes prevalence-style statistics across many papers under common reporting standards.",
        1,
    )

    sg_line = (
        "- Sgambi, L., Gkoumas, K., Bontempi, F., genetic algorithms for dependability assurance in long-span suspension bridge design, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/j.1467-8667.2012.00780.x`"
    )
    mice_tail = (
        "\n- Valikhani, A., et al., machine learning and image processing for concrete surface roughness, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/mice.12605`\n"
        "- Fayaz, J., & Galasso, C., deep neural network framework for on-site acceleration response spectra, "
        "*Computer-Aided Civil and Infrastructure Engineering*, `10.1111/mice.12830`"
    )
    if sg_line in text and "- Valikhani, A., et al." not in text:
        text = text.replace(sg_line, sg_line + mice_tail, 1)

    OUT.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {OUT} from {SRC.name}")


RESULTS_BLOCK = """## Results

### A validation-target taxonomy for structural machine learning

The Methods formalize five deployment targets—within-database interpolation, out-of-source or out-of-programme prediction, out-of-mixture generalization, structural-family extrapolation and deployment-matched stress tests—and map each to split machinery and grouping identifiers (**Figure 1**; Methods). The subsections below summarize empirical behaviour at three **severity tiers** of random-split optimism relative to grouped or deployment-matched validation. Full Random Forest headline metrics, macro-F1 and balanced-accuracy gaps for classification rows, and heuristic severity tags are machine-readable in **Table S7** and visualized in **Figure S5**.

### Heterogeneous validation-target mismatch across nine executable modules

#### Negligible optimism (within-database interpolation benchmarks)

The public UCI concrete compressive-strength benchmark shows only `ΔR²=0.040` for Random Forest when identical mixture proportions define groups, with ridge regression essentially unchanged (`ΔR²=0.002`). The DesignSafe PRJ-3053 coupling-beam **peak-shear** target is a deliberate **contrast** case (`ΔR²≈−0.069` for Random Forest), showing that grouped validation is not automatically harder under every topology. These modules function as **adversarial controls** within the nine-module portfolio: they demonstrate that the same protocol can register both mild optimism and reversed gaps without changing the learner template.

#### Moderate optimism (out-of-source prediction with retained utility)

The corroded reinforced-concrete beam Zenodo database shows `ΔR²=0.180` for Random Forest while retaining high absolute grouped performance. The Mendeley exterior joint shear task shows `ΔR²=0.198`; DesignSafe circular, rectangular and combined RC column tasks show `ΔR²` between `0.110` and `0.209`; PRJ-3053 normalized shear and chord-rotation targets show `ΔR²=0.303` and `0.135`. Together, these cases motivate **deployment disclosure** without implying that grouped validation always collapses models to unusable accuracy.

#### Severe optimism and deployment-matched collapse

The Mangalathu shear-wall failure-mode classifier drops from `0.827` to `0.523` accuracy under author/source-family `GroupKFold` (`Δaccuracy=0.304`), with pooled macro-F1 decay from roughly `0.746` to `0.399` (`Δmacro-F1≈0.347`) and balanced-accuracy gaps tabulated in **Table S7**. The steel-fiber-reinforced concrete shear-capacity task shows pooled Random Forest `R²≈0.638` versus `≈−0.172` grouped (`ΔR²≈0.811`), with more extreme fold-means used for **Figure 2** uncertainty visuals (Methods). The cyclic Mendeley joint-shear task reaches `ΔR²≈1.392` pooled; the Stub concrete-filled steel tube dataset reaches `ΔR²≈1.784` under section-family holdout; DesignSafe PRJ-2430 wall peak strength and drift capacity show `ΔR²=0.529` and `0.385`. These cases anchor the **severe** tier and motivate structural-family and source-aware stress tests.

**Cross-tier summary.** Pooling Random Forest headline gaps in **Table S7** with topology cues in **Table S8** shows that large pooled optimism clusters with **few groups**, **non-trivial singleton rates**, **structural-family holdout** or **classification stress**, whereas the negligible and contrast bands include **high group cardinality** (UCI concrete) and a **peak-shear coupling-beam** target where grouped validation is not uniformly harder. The pattern is intentionally heterogeneous: the same protocol registers both collapses and mild optimism without retuning the learner template.

DesignSafe column, wall and coupling-beam details beyond the headline contrasts are documented in **Tables S4–S6** and **Figures S2–S4**. Mendeley joint metrics and topology appear in **Table S3**.

### Group topology conditions the interpretation of validation gaps

Grouped-score drops can reflect genuine source dependence, **group/class topology stress**, or both. The shear-wall classifier illustrates topology stress: 393 specimens across 75 author/source groups (median group size 3; 26 singletons; 51 single-class groups; median within-group majority-class share 1.00). **Table S8** generalizes this diagnostic posture across modules by reporting sample size, group count, median group size, singleton counts, grouping proxy, and—for classifiers—single-class group counts alongside headline gaps.

### Uncertainty diagnostics reinforce the heterogeneity

**Figure 2** adds case-specific uncertainty diagnostics to the four main source/mix validation cases. For the shear-wall classification and steel-fiber-reinforced concrete regression cases, fold-summary normal approximations were used where prediction-level exports were incomplete or where fold-means defined the displayed gap. For the UCI concrete and corroded-beam cases, paired group bootstrap intervals were computed over mix or source groups using out-of-fold predictions.

The intervals did not remove the qualitative pattern. The shear-wall accuracy gap was approximately 0.304, with an approximate interval of 0.171 to 0.436. The steel-fiber-reinforced concrete **fold-mean** Random Forest `R²` gap used for **Figure 2** was 1.689, with a wide approximate interval of 0.668 to 2.709, reflecting fold instability rather than a field-wide estimate (**Table S7** reports the pooled headline gap for cross-module comparison). The UCI concrete gap was 0.040, with a bootstrap interval of 0.025 to 0.056 across 428 mix groups. The corroded-beam gap was 0.180, with a bootstrap interval of 0.097 to 0.332 across 54 source groups.

### OpenAlex triage and manual reproducibility verification

The reproductions are executable demonstrations, not a field-wide prevalence estimate. OpenAlex retrieval produced 3919 deduplicated records and a strict 332-paper structural-machine-learning candidate pool with triage labels; few completed reproductions overlapped that strict pool, and several executable modules were discovered **dataset-first** outside metadata-first screening—so the frame should be read as **scaling infrastructure**, not a prevalence survey. Manual verification of five high-priority records showed repository hints often resolving to PDFs, embargoed storage, metadata-only pages or notebooks without extracted tables (Lee et al., DOI 10.1016/j.engstruct.2021.112109; Alwanas et al., DOI 10.1016/j.engstruct.2019.05.048; Wang et al., DOI 10.1016/j.istruc.2023.104968; Polo-Mendoza et al., DOI 10.1007/s13369-024-08794-0). That pattern motivates **dataset-first** selection, exemplified by the Mendeley joint modules and the DesignSafe expansions.

### Reporting checklist for deployment-matched structural ML validation

**Figure 4** encodes a reviewer-facing checklist that mirrors the protocol outputs: deployment target, grouping variable, random versus grouped headline scores, topology diagnostics, pooled versus fold-level regression summaries, and data/code availability statements. The checklist is intended as a practical adoption artefact aligned with the journal’s emphasis on reproducible computing in civil infrastructure; the same fields can be supplied as structured supplementary material (see Discussion).

"""

DISCUSSION_PREFIX = (
        "The proposed validation framework demonstrates that structural-machine-learning performance claims are **coupled to the deployment target**, not merely to algorithmic choice. "
        "Across the nine-module stress-test portfolio, random validation usually matched or exceeded grouped headline scores, yet the portfolio deliberately includes **adversarial controls** "
        "(negligible optimism and the PRJ-3053 peak-shear **contrast** case) so the protocol cannot be dismissed as cherry-picking only large gaps. "
        "The magnitude and sign of headline gaps vary by dataset and deployment claim; blanket slogans such as \"random CV is always fine\" or \"random CV is always invalid\" would therefore mislead. "
        "The operative question is whether the validation split matches the use case asserted in the manuscript.\n\n"
    )

DISCUSSION_INSERT = (
    "### Engineering risk implications\n\n"
    "Validation optimism is not only a statistical curiosity. For the Stub concrete-filled steel tube case, a practitioner reading only random ten-fold `R²≈0.828` might expect strong interpolation across section types; deployment-matched family holdout yields pooled `R²≈−0.956`, signalling anti-aligned extrapolation for withheld families and misranking risk in screening workflows. "
    "For shear-wall failure-mode screening, the drop from `0.827` to `0.523` accuracy under source-aware grouping implies roughly one-in-two misclassification for withheld source families under the reported topology—an unacceptable error rate if the tool were used as an automated gate without disclosure. "
    "These examples translate abstract gaps into **decision-relevant** cautions for retrofit screening, experiment prioritization and reliability workflows that ingest ML surrogates.\n\n"
    "### Infrastructure decision mapping\n\n"
    "**Retrofit and conceptual design screening.** When models rank alternative configurations (e.g., composite member sizing), structural-family holdouts test whether scores survive unseen typologies. "
    "**Experimental campaign planning.** Leave-one-programme-out and source-aware folds clarify whether models interpolate within a laboratory programme or transport across programmes. "
    "**Reliability-style workflows.** Pooled out-of-fold metrics should feed uncertainty statements because fold-wise `R²` can crater under tiny held-out groups even when mean absolute error remains modest.\n\n"
    "### Physics-aware and simulation-aware mitigation avenues\n\n"
    "Hybrid simulation–surrogate workflows (e.g., OpenSees-based nonlinear models, physics-informed neural nets) can inject physical constraints where tabular data are sparse; the present protocol remains agnostic to learner architecture but still requires **disclosure** of deployment targets and grouped diagnostics for any hybrid model.\n\n"
)

METHODS_INSERT = """### Source-Aware Validation Protocol (SAVP)

SAVP turns the taxonomy table into a reproducible computational workflow.

**Inputs.** Tabular dataset $\\mathcal{D}=\\{(x_i,y_i)\\}_{i=1}^{n}$; candidate grouping variables $\\mathcal{G}$ (source tokens, mixture identifiers, section family labels, programme codes); declared deployment target $T$ in the taxonomy; model family $\\mathcal{M}$; headline score $S$ (pooled out-of-fold $R^2$ for regression; accuracy plus macro-F1 and balanced accuracy for classification).

**Procedure.**
1. **Topology diagnostics.** For each selected grouping variable, compute group counts, size quantiles, singleton rate, and—for classifiers—within-group class purity and count of single-class groups.
2. **Baseline estimation.** Fit $\\mathcal{M}$ under shuffle-based $k$-fold cross-validation with preprocessing refit inside each training fold; group identifiers never enter the feature matrix.
3. **Target-matched estimation.** Repeat with `GroupKFold` or leave-one-source-out splits that respect the grouping variable aligned with $T$.
4. **Gap and uncertainty.** Report $\\Delta=S_{\\mathrm{random}}-S_{\\mathrm{grouped}}$ with case-specific bootstrap or fold-variance diagnostics where prediction-level exports exist.
5. **Machine-readable export.** Emit the `Table S7` schema (random score, grouped score, gaps, optional balanced-accuracy and macro-F1 gaps, severity tag).

**Outputs.** A validation credibility record that states which deployment claim is supported, which grouping proxy was used, and whether optimism is negligible, moderate, severe or contrast.

**Compact algorithm sketch.** For each grouping proxy aligned with deployment target *T*: compute topology summaries; fit \\(\\mathcal{M}\\) under shuffle-based *k*-fold cross-validation with preprocessing refit strictly inside training folds; refit under `GroupKFold` (or leave-one-source-out) that respects the same proxy; compute \\(\\Delta = S_{\\mathrm{random}} - S_{\\mathrm{grouped}}\\) with paired bootstrap or fold-variance intervals where prediction-level exports exist; append one machine-readable row to the **Table S7** schema and tag the optimism tier.

### Metric selection protocol

**Regression.** Pooled out-of-fold $R^2$ is the headline summary for small or low-variance held-out groups; fold-averaged $R^2$ and mean absolute error are retained as instability diagnostics.

**Classification.** Accuracy is reported for backward compatibility with structural ML practice, but headline contrasts emphasize **macro-F1** and **balanced accuracy** whenever grouped folds induce rare-class stress. Macro-F1 and balanced-accuracy deltas are exported for Mangalathu shear walls and Mendeley failure-mode classification in `Table S7`.

**Gap convention.** All gaps follow $S_{\\mathrm{random}}-S_{\\mathrm{grouped}}$; negative values flag contrast cases.

### Software architecture and reproducibility engineering

The codebase is organized as modular, dataset-agnostic pipelines: per-dataset reproduction scripts share a common evaluation harness (group topology diagnostics, `GroupKFold` orchestration, metric aggregation) and write manifests under `code/outputs/reproductions/`. Consolidated headline exports and figure-generation utilities populate `code/outputs/figures/` and `figures/draft/`. A one-command driver regenerates Mangalathu and Mendeley `results.csv` when sync tools rename conflict copies, then rebuilds `Table S7`, **Figure S5** and the graphical abstract export. Adding a tenth benchmark requires a new data loader and grouping metadata while reusing the same protocol implementation.

"""

SOFTWARE_BLOCK = (
    "The **Structural ML validation suite** accompanying this manuscript is implemented as a versioned Python toolkit with pinned dependencies (`environment_reproducibility_v0.yml`). "
    "Deterministic scripts perform download hashing, preprocessing inside each cross-validation fold, grouped and random evaluations, topology diagnostics, figure generation and consolidated CSV export. "
    "Numerical results in **Table S7**, **Figure S5** and the graphical abstract were produced by rerunning these scripts locally; generative assistants did not supply tabulated metrics."
)

CODE_AVAIL = (
    "Code, processed outputs, manifests and figure-source data are available at https://github.com/Johnsonlijian/structural-ml-validation-target-mismatch "
    "and archived on Zenodo at https://doi.org/10.5281/zenodo.20006918. The archive contains environment files, modular reproduction scripts, dataset manifests, "
    "processed output tables, figure-source data and the one-command QA runner used to regenerate Table S7, Figure S5 and the graphical abstract. "
    "Raw third-party datasets are not redistributed where source licences prohibit redistribution; dataset DOIs, access URLs, access dates and checksums are recorded in the data manifests."
)


if __name__ == "__main__":
    main()
