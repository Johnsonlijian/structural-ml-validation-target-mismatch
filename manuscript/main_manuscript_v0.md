# Validation-target mismatch in structural machine learning

## A reproducible source-aware evaluation framework for structural performance prediction

**Target journal:** Computer-Aided Civil and Infrastructure Engineering (CACE)  
**Manuscript status:** CACE submission draft **v1.1** — taxonomy-first Results; SAVP + algorithm sketch; **Table S8** topology master; cross-tier summary; compressed OpenAlex/manual block; Discussion de-duplicated; shortened physics and AI prose; Figure 2 unified (no internal v2 label); recent CACE (2021/2023) inventory lines; GitHub/Zenodo placeholders to replace at upload.  
**Date:** 2026-05-03  

## Abstract

Computational pipelines for structural performance prediction increasingly rely on machine learning, but headline scores often depend on validation splits that do not match the intended deployment target. We organize evaluation through a source-aware validation framework that separates within-database interpolation, out-of-source prediction, mixture-family constraints and deployment-matched structural-family extrapolation tests. The framework is exercised through nine executable public reproduction modules covering concrete strength, reinforced-concrete shear-wall failure classification, steel-fiber-reinforced concrete shear capacity, corroded reinforced-concrete beam moment capacity, beam-column joint shear and failure mode, reinforced-concrete column lateral strength, wall strength and drift capacity, diagonally reinforced coupling beams, and concrete-filled steel tube capacity with section-family holdout. Random validation produced negligible optimism in a standardized concrete-strength benchmark and a contrast pattern for one coupling-beam peak-shear metric, moderate optimism in several beam, joint, column and deformation targets, and severe optimism—or deployment-matched collapse—in steel-fiber-reinforced concrete shear, cyclic joint shear, shear-wall classification, wall strength/drift and concrete-filled steel tube structural-family extrapolation. We further construct a 332-paper OpenAlex sampling frame and manually verify high-priority metadata records, showing that repository-level reproducibility signals often overstate immediate access to reusable raw data. These results support a deployment-matched reporting standard for structural machine learning: state the deployment target, disclose grouping variables, compare random and grouped or deployment-matched validation, and report group-topology diagnostics. **This work does not estimate field-wide prevalence of optimistic reporting; it provides an executable protocol and heterogeneous benchmark evidence aligned with civil-infrastructure prediction tasks.**

## Significance

Structural and infrastructure machine learning is often evaluated as if observations were independent. In practice, many engineering databases are clustered by literature source, experimental programme, mixture family or structural family. This paper presents a **reproducible source-aware evaluation workflow**—taxonomy, splits, metrics, topology diagnostics and a practical reporting checklist—and shows with nine public modules that performance gaps between random and deployment-matched validation are **heterogeneous**, not uniformly catastrophic. The contribution is methodological and computational: practitioners can reuse the splits and diagnostics to align validation claims with the engineering use case, rather than treating random cross-validation as a universal surrogate for structural generalization.

## Introduction

Machine learning has become a routine modelling tool in structural and civil engineering. Models based on random forests, gradient boosting, support vector machines and neural networks are now used to predict concrete compressive strength, shear strength, axial capacity, seismic failure mode, corrosion-damaged capacity and other response quantities. Many papers report coefficients of determination above 0.9, or classification accuracies high enough to suggest that data-driven models may outperform classical empirical design formulae. These results are attractive because experimental structural databases are expensive to create and because machine learning appears to offer a low-cost way to interpolate across sparse test evidence.

The central validation problem is that structural datasets are rarely independent collections of exchangeable observations. A single laboratory may contribute a block of specimens. A literature database may contain multiple records copied from the same original paper. A concrete-strength benchmark may include repeated mixture designs. A composite-column dataset may combine distinct section families whose extrapolation behaviour is not represented by a random split. Random k-fold cross-validation ignores these dependencies. It can place closely related records in both the training and test folds, so the reported score may measure interpolation among already represented sources rather than prediction for a new experimental programme, source reference or structural family.

**In structural machine learning, validation is therefore an engineering claim about the deployment target.** The same numerical pipeline can estimate within-database interpolation, out-of-programme transportability or extrapolation across structural families, depending on how folds respect clustering.

*Computer-Aided Civil and Infrastructure Engineering* has long published computational learning systems for civil infrastructure (e.g., Reich, 1997, https://doi.org/10.1111/0885-9507.00065; Adeli, 2001, https://doi.org/10.1111/0885-9507.00219). Contributions continue to couple evolutionary computing, surrogate modelling and reliability thinking to civil infrastructure systems (e.g., Sgambi et al., 2012, https://doi.org/10.1111/j.1467-8667.2012.00780.x), and recent machine-learning studies in the same venue illustrate tabular and surrogate-intensive workflows for materials and strong-motion tasks (e.g., Valikhani et al., 2021, https://doi.org/10.1111/mice.12605; Fayaz & Galasso, 2023, https://doi.org/10.1111/mice.12830). Our work is complementary: we supply a **deployment-aligned validation and reporting protocol** so that performance claims remain interpretable when training and test partitions violate exchangeability.

Formally, let covariates be $X$, labels $Y$ and discrete grouping indicators $G$. Random $k$-fold cross-validation fits a learner under an approximate product measure that treats specimens as exchangeable. Grouped cross-validation enforces support restrictions so that held-out groups represent different mixtures of sources or structural contexts; when $P(X \mid G=g)$ differs materially across $g$, random splitting can induce $P_{\mathrm{train}}(X,Y) \neq P_{\mathrm{test}}(X,Y)$ even when marginal covariate histograms overlap—a **group-conditional covariate-shift** pattern. The Methods taxonomy maps each deployment target to the $G$ and split machinery that make this shift explicit rather than hidden.

This issue is well known outside structural engineering. Spatial ecology, remote sensing and other scientific machine-learning domains have shown that random cross-validation can overstate performance when data are spatially, temporally or taxonomically clustered (Roberts et al., 2017, Ecography, DOI 10.1111/ecog.02881; Ploton et al., 2020, Nature Communications, DOI 10.1038/s41467-020-18321-y; Kapoor & Narayanan, 2023, Patterns, DOI 10.1016/j.patter.2023.100804). Yet structural ML has not developed a comparable field-level audit. The absence of such an audit matters because structural-engineering claims often have downstream design implications: a model advertised as accurate for "structural strength prediction" may later be used for screening designs, guiding experiments or supporting reliability discussions.

We therefore treat validation as a statement about the intended generalization target. If the target is interpolation within a known database, random cross-validation may be defensible. If the target is prediction for an unseen source, unseen mixture family, new experimental programme or new structural family, grouped or deployment-matched validation is required. We instantiate this principle through nine executable public reproduction modules that span classification and regression tasks across concrete, reinforced concrete, composites and curated infrastructure databases. We then build a transparent sampling frame from 332 OpenAlex-screened structural-machine-learning papers and manually verify top reproducibility candidates to document metadata-level reuse barriers relevant to computational reproducibility workflows. **The central contribution is an evidence-backed evaluation protocol—not a prevalence estimate.** The reproduction modules demonstrate mechanism and heterogeneity; the sampling frame supports future community scaling without implying that this manuscript completes a representative quantitative audit of the entire literature.

## Results

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

## Discussion

The proposed validation framework demonstrates that structural-machine-learning performance claims are **coupled to the deployment target**, not merely to algorithmic choice. Across the nine-module stress-test portfolio, random validation usually matched or exceeded grouped headline scores, yet the portfolio deliberately includes **adversarial controls** (negligible optimism and the PRJ-3053 peak-shear **contrast** case) so the protocol cannot be dismissed as cherry-picking only large gaps. The magnitude and sign of headline gaps vary by dataset and deployment claim; blanket slogans such as "random CV is always fine" or "random CV is always invalid" would therefore mislead. The operative question is whether the validation split matches the use case asserted in the manuscript.

For structural engineering, this distinction is practical rather than semantic. A model used to interpolate among specimens from a known experimental programme has a different risk profile from a model used to predict the capacity of a new structural family or a new laboratory campaign. Random folds can estimate the first task but not the second. Grouped folds by source reference, experimental programme, mixture family, section family, time block or spatial block are therefore not optional robustness checks. They define what kind of generalization is being claimed.

The results also protect against an anti-ML overcorrection. The corroded-beam case retained high absolute source-aware performance even after a non-trivial gap from random validation. The UCI concrete case showed only mild optimism. These examples demonstrate that group-aware validation can preserve useful predictive ability while calibrating expectations. The goal is not to reject data-driven modelling in structural engineering, but to make model claims more honest and more transportable.

The main limitation is scope. This draft does not yet estimate the field-wide distribution of validation optimism across all reproducible structural ML papers. It provides executable case-study modules, a 332-paper sampling frame and a manual verification protocol. The cases were selected because their data were public and their grouping structures were inspectable; they should be described as motivating reproductions rather than a representative sample. A second limitation is that group proxies differ in strength. Source-reference labels in the SFRC, corroded-beam, Mendeley joint and DesignSafe coupling-beam cases are stronger proxies than author/source family in the shear-wall, DesignSafe column and DesignSafe wall cases, and section-family grouping in the Stub-CFST case tests deployment extrapolation rather than source-reference leakage. A third limitation is that `R²` and accuracy capture predictive performance but not all engineering validity. Physical consistency, uncertainty calibration and design-code compatibility remain separate requirements.

### Engineering risk implications

Validation optimism is not only a statistical curiosity. For the Stub concrete-filled steel tube case, a practitioner reading only random ten-fold `R²≈0.828` might expect strong interpolation across section types; deployment-matched family holdout yields pooled `R²≈−0.956`, signalling anti-aligned extrapolation for withheld families and misranking risk in screening workflows. For shear-wall failure-mode screening, the drop from `0.827` to `0.523` accuracy under source-aware grouping implies roughly one-in-two misclassification for withheld source families under the reported topology—an unacceptable error rate if the tool were used as an automated gate without disclosure. These examples translate abstract gaps into **decision-relevant** cautions for retrofit screening, experiment prioritization and reliability workflows that ingest ML surrogates.

### Infrastructure decision mapping

**Retrofit and conceptual design screening.** When models rank alternative configurations (e.g., composite member sizing), structural-family holdouts test whether scores survive unseen typologies. **Experimental campaign planning.** Leave-one-programme-out and source-aware folds clarify whether models interpolate within a laboratory programme or transport across programmes. **Reliability-style workflows.** Pooled out-of-fold metrics should feed uncertainty statements because fold-wise `R²` can crater under tiny held-out groups even when mean absolute error remains modest.

### Physics-aware and simulation-aware mitigation avenues

Hybrid simulation–surrogate workflows (e.g., OpenSees-based nonlinear models, physics-informed neural nets) can inject physical constraints where tabular data are sparse; the present protocol remains agnostic to learner architecture but still requires **disclosure** of deployment targets and grouped diagnostics for any hybrid model.


Despite these limitations, the evidence is sufficient to justify a reporting change. Structural ML papers should report at least one validation split that matches the intended deployment target, disclose the grouping variable used to form that split, and provide group-topology diagnostics. For regression tasks with small or low-variance held-out groups, pooled out-of-fold `R²` should be reported beside fold-wise `R²`, because fold-wise `R²` can become numerically unstable. For classification tasks, group-level class balance should be reported because single-class groups can make source-aware folds intrinsically difficult.

### Broader impact and checklist uptake

The Figure 4 checklist distils SAVP outputs into reviewer-facing fields; supplying it as supplementary structured material can make deployment-target disclosures routine without lengthening the main narrative. A natural extension—outside the scope of the present manuscript—is a coordinated field-wide audit that samples from the OpenAlex-derived frame and synthesizes prevalence-style statistics across many papers under common reporting standards. Until such an effort exists at scale, the conservative engineering conclusion stands: random cross-validation should not be treated as evidence of out-of-source structural generalization unless clustering by source, mixture, publication programme or structural family has been explicitly addressed and justified against the claimed deployment scenario. The reproducibility artefacts released with this protocol are intended to make that justification routine rather than heroic.

## Methods

### Literature search and sampling frame

We used OpenAlex metadata to identify structural and civil-engineering machine-learning papers published between 2014 and 2026. The search combined terms related to machine learning (`machine learning`, `deep learning`, `neural network`, `random forest`, `gradient boosting`, `XGBoost`, `support vector`) with structural-engineering terms (`concrete-filled steel`, `CFST`, `reinforced concrete`, `shear wall`, `beam-column`, `frame`, `structural`) and performance terms (`strength`, `capacity`, `resistance`, `performance`, `fragility`, `ductility`). Because complex Boolean queries were brittle in the OpenAlex API, the final retrieval used multiple simpler queries and merged the results.

The full metadata retrieval produced 3919 deduplicated OpenAlex records. A stricter structural-engineering candidate pool contained 332 papers. The candidate pool was not treated as a final reproducible sample. Instead, it was converted into a triage table with four labels: completed, high-priority, medium-priority and low-priority. Completed entries matched existing reproductions. High-priority entries contained repository or data-hosting hints in metadata. Medium-priority entries were open or influential enough to inspect manually. Low-priority entries had no immediate open-data or code signal in metadata.

### Manual reproducibility verification

High-priority metadata records were manually verified by inspecting repository APIs, landing pages, full-text pages where available, and data availability statements. A paper was considered immediately reproducible only if a raw data file, code repository or dataset landing page could be resolved without relying on private correspondence. Article PDFs alone were not counted as raw data. Metadata-only records were not counted as raw data. Embargoed or confidential files were excluded from immediate reproduction. Code without the extracted dataset was labeled as partial.

This verification produced `manual_reproducibility_verification_v0.csv` and a markdown audit note. The manual audit was designed to prevent false reproducibility inflation caused by repository hints that point only to article records.

### Case-study selection

Executable public datasets were selected to span concrete and steel–concrete composite materials, reinforced-concrete components and joint subassemblies, and curated infrastructure repositories: UCI concrete compressive strength (mix-family grouping); Mangalathu shear-wall failure modes; Lantsoght-style steel-fiber-reinforced concrete shear capacity; Matthews corroded RC beam moment capacity; Mendeley exterior and cyclic beam–column joint shear and exterior failure mode; DesignSafe circular and rectangular RC columns (combined depot export); Megahed-style stub concrete-filled steel tube axial capacity with section-family holdout; DesignSafe PRJ-2430 RC walls (peak strength and drift); and DesignSafe PRJ-3053 diagonally reinforced coupling beams (peak shear, normalized shear and chord rotation). The selection was not intended to be representative; it was intended to expose heterogeneous validation topologies under a unified protocol.

### Validation-target taxonomy for structural performance prediction

We distinguish five recurring reporting targets. Each target aligns a splitting strategy with an explicit inference claim about structural or infrastructure machine learning.

| Validation target | Typical split machinery | Representative grouping identifiers | Operational claim interrogated |
|---|---|---|---|
| Within-database interpolation | Shuffle-based *k*-fold CV without grouping constraints | — | Fits-and-scores resemble interpolation assuming exchangeable specimens |
| Out-of-source / out-of-programme prediction | Grouped folds or leave-one-source-out | Literature-derived reference codes, experimental programmes | Simulates withheld sources unseen during training |
| Out-of-mixture generalization | Grouped folds keyed by mixture families | Repeated mix proportions (`mix_group`) | Measures optimism when future mixes diverge |
| Structural-family extrapolation | Family-level holdouts | Section/system typology (`circular`, `rectangular`, `double-skin`, …) | Stress-tests deployment contexts with unseen families |
| Deployment-matched stress tests | Investigator-defined withholdings tailored to scenarios | Acquisition campaign-specific metadata etc. | Best operational analogue when labels permit |

The Stub-CFST workbook instantiates structural-family extrapolation—not classical source leakage—owing to missing bibliometric identifiers at specimen level.

### Source-Aware Validation Protocol (SAVP)

SAVP turns the taxonomy table into a reproducible computational workflow.

**Inputs.** Tabular dataset $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n}$; candidate grouping variables $\mathcal{G}$ (source tokens, mixture identifiers, section family labels, programme codes); declared deployment target $T$ in the taxonomy; model family $\mathcal{M}$; headline score $S$ (pooled out-of-fold $R^2$ for regression; accuracy plus macro-F1 and balanced accuracy for classification).

**Procedure.**
1. **Topology diagnostics.** For each selected grouping variable, compute group counts, size quantiles, singleton rate, and—for classifiers—within-group class purity and count of single-class groups.
2. **Baseline estimation.** Fit $\mathcal{M}$ under shuffle-based $k$-fold cross-validation with preprocessing refit inside each training fold; group identifiers never enter the feature matrix.
3. **Target-matched estimation.** Repeat with `GroupKFold` or leave-one-source-out splits that respect the grouping variable aligned with $T$.
4. **Gap and uncertainty.** Report $\Delta=S_{\mathrm{random}}-S_{\mathrm{grouped}}$ with case-specific bootstrap or fold-variance diagnostics where prediction-level exports exist.
5. **Machine-readable export.** Emit the `Table S7` schema (random score, grouped score, gaps, optional balanced-accuracy and macro-F1 gaps, severity tag).

**Outputs.** A validation credibility record that states which deployment claim is supported, which grouping proxy was used, and whether optimism is negligible, moderate, severe or contrast.

**Compact algorithm sketch.** For each grouping proxy aligned with deployment target *T*: compute topology summaries; fit \(\mathcal{M}\) under shuffle-based *k*-fold cross-validation with preprocessing refit strictly inside training folds; refit under `GroupKFold` (or leave-one-source-out) that respects the same proxy; compute \(\Delta = S_{\mathrm{random}} - S_{\mathrm{grouped}}\) with paired bootstrap or fold-variance intervals where prediction-level exports exist; append one machine-readable row to the **Table S7** schema and tag the optimism tier.

### Metric selection protocol

**Regression.** Pooled out-of-fold $R^2$ is the headline summary for small or low-variance held-out groups; fold-averaged $R^2$ and mean absolute error are retained as instability diagnostics.

**Classification.** Accuracy is reported for backward compatibility with structural ML practice, but headline contrasts emphasize **macro-F1** and **balanced accuracy** whenever grouped folds induce rare-class stress. Macro-F1 and balanced-accuracy deltas are exported for Mangalathu shear walls and Mendeley failure-mode classification in `Table S7`.

**Gap convention.** All gaps follow $S_{\mathrm{random}}-S_{\mathrm{grouped}}$; negative values flag contrast cases.

### Software architecture and reproducibility engineering

The codebase is organized as modular, dataset-agnostic pipelines: per-dataset reproduction scripts share a common evaluation harness (group topology diagnostics, `GroupKFold` orchestration, metric aggregation) and write manifests under `code/outputs/reproductions/`. Consolidated headline exports and figure-generation utilities populate `code/outputs/figures/` and `figures/draft/`. A one-command driver regenerates Mangalathu and Mendeley `results.csv` when sync tools rename conflict copies, then rebuilds `Table S7`, **Figure S5** and the graphical abstract export. Adding a tenth benchmark requires a new data loader and grouping metadata while reusing the same protocol implementation.

### Validation splits

Each case was evaluated under a random validation split and at least one stricter grouped or deployment-matched validation split. Random validation used random k-fold or stratified random k-fold cross-validation. Group-aware validation used `GroupKFold` where all records from the same source, source family, experimental programme, mixture family or section family were held out together. Where feasible, leave-one-source-out predictions were also generated. For the Stub-CFST case, section-family grouping was used only as a deployment-matched extrapolation test because public source-reference labels were not available.

### Models and preprocessing

The reproductions relied on ridge, random forest and gradient boosting learners implemented in `scikit-learn`, with preprocessing pipelines assembled from standard scalers, imputers and one-hot encoders as required by each tabular benchmark. Critical controls: preprocessing statistics were recomputed strictly within each training partition of every cross-validation split; group identifiers were reserved exclusively for partitioning folds and were never supplied as explanatory variables; identical hyperparameter templates were reused across protocols per dataset.

### Metrics

Classification summaries foreground accuracy for backward compatibility with the structural ML literature, paired with pooled macro-F1 deltas and balanced-accuracy deltas exported by the regenerated evaluation scripts wherever available. Accuracy alone underreports failure on rare failure modes when grouped splits withhold entire classes-dominated sources.

Regression summaries hinge on pooled out-of-fold `R²`, with fold-averaged `R²` retained as a instability diagnostic whenever groups are extremely small or low-variance. Fold-wise scores can crater even when MAE inflation is modest—hence the dual reporting mirrored in supplementary tables.

The validation gap stays `performance_random − performance_grouped`; positive regression gaps signify optimistic random partitioning, whereas negative gaps flag contrast cases aligned with heterogeneous deployment realism.

### Group topology diagnostics

For each case, we recorded group counts, group-size distributions and grouping rationale. For classification, we additionally inspected class balance by group and majority-class share by group. These diagnostics were used to interpret whether a performance drop reflected source-reference dependence, source-family plus class/topology stress, mix-family optimism or deployment-matched structural-family extrapolation.

### Uncertainty diagnostics

Figure 2 used case-specific uncertainty diagnostics. For Mangalathu shear-wall classification and Rahman/SFRC regression, fold-summary normal approximations were used because prediction-level outputs were incomplete or because the displayed severe SFRC gap was defined using fold-summary metrics. For UCI concrete and corroded RC beams, paired group bootstrap intervals were computed over mix or source groups using out-of-fold predictions. These intervals are diagnostic for the displayed cases and are not interpreted as field-wide confidence intervals.

### Software and reproducibility

The **Structural ML validation suite** accompanying this manuscript is implemented as a versioned Python toolkit with pinned dependencies (`environment_reproducibility_v0.yml`). Deterministic scripts perform download hashing, preprocessing inside each cross-validation fold, grouped and random evaluations, topology diagnostics, figure generation and consolidated CSV export. Numerical results in **Table S7**, **Figure S5** and the graphical abstract were produced by rerunning these scripts locally; generative assistants did not supply tabulated metrics.


## Data availability

The manuscript uses public datasets and public metadata. Dataset-specific download scripts and manifests are included where automated download was possible. Original third-party data are not redistributed unless their source license permits redistribution. For datasets requiring a landing-page download, the manuscript records the DOI and access status.

## Code availability

Code, processed outputs, manifests and figure-source data are available at https://github.com/Johnsonlijian/structural-ml-validation-target-mismatch and archived on Zenodo at https://doi.org/10.5281/zenodo.20006918. The archive contains environment files, modular reproduction scripts, dataset manifests, processed output tables, figure-source data and the one-command QA runner used to regenerate Table S7, Figure S5 and the graphical abstract. Raw third-party datasets are not redistributed where source licences prohibit redistribution; dataset DOIs, access URLs, access dates and checksums are recorded in the data manifests.

## Acknowledgements

The author thanks maintainers of the public repositories and data depots referenced in this benchmark suite.

## Author contributions

Sole author: conceptualization, methodology, software, validation, formal analysis, investigation, resources, data curation, writing (original draft and review), and visualization.

## Competing interests

The author declares no competing interests.

## Highlights (Editorial Manager; each line ≤85 characters)

Structural ML taxonomy maps clustered validation splits to deployments  
Structural ML-Bench v1.0: heterogeneous validation gaps across nine systems  
OpenAlex triage shows metadata overstated reproducible structural ML datasets  
Checklist aligns grouped folds, pooled R2, disclosures, QA scripts  
Python toolkit regenerates validation diagnostics and supplementary figures  

## Declaration of Generative AI and AI-assisted technologies in the writing process

During manuscript preparation the author used commercial generative-AI assistants (primarily via the Cursor IDE) for English drafting, code-refactoring suggestions and bibliography hygiene cues, with author review throughout. All numerical summaries were produced by deterministic scripts rerun locally; assistants did not supply tabulated results. Standard spell-check tooling was used without separate declaration.

Plots were generated with scripted `matplotlib` visualizations of regenerated metrics; **no synthetic photography or generative alteration of forensic imagery was performed.**

The author vetted and rewrote AI-assisted suggestions, asserted final scientific wording, and accepts full accountability for methodology, citations, interpretations and licence compliance. AI tools were **not listed as authors**.

## Reference inventory (DOI-verified keys; v0)

Full bibliographic strings for the three non-structural method precedents below are in `manuscript/references_verified_v0.md` (CACE-ready paste block).

- Mangalathu et al., reinforced-concrete shear-wall failure mode, `10.1016/j.engstruct.2019.110331`
- Rahman et al. / Lantsoght SFRC shear-capacity setting, `10.1016/j.engstruct.2020.111743`
- Nguyen-style concrete-strength modelling, `10.1016/j.conbuildmat.2020.120950`
- Matthews et al. corroded RC beam moment-capacity database, Zenodo `10.5281/zenodo.8062007`
- Megahed et al. Stub-CFST Scientific Reports dataset, `10.1038/s41598-024-53352-1`
- Mendeley exterior RC beam-column joint dataset, `10.17632/8ndgpm7zw7.1`
- Mendeley cyclic beam-column joint shear dataset, `10.17632/rbhfnz32sy.1`
- DesignSafe circular RC column database, `10.17603/ds2-52bz-0n63`
- DesignSafe rectangular RC column database, `10.17603/ds2-7qg0-4303`
- DesignSafe UoA-UW RC wall database, `10.17603/ds2-r12q-t415`
- DesignSafe diagonally reinforced concrete coupling-beam database, `10.17603/ds2-g5n8-4p74`
- Roberts, D.R., et al., cross-validation with spatial/temporal/hierarchical structure, *Ecography*, `10.1111/ecog.02881`
- Ploton, P., et al., spatial validation vs optimistic random CV in large-scale ecological mapping, *Nat. Commun.*, `10.1038/s41467-020-18321-y`
- Kapoor, S., & Narayanan, A., leakage and reproducibility in ML-based science, *Patterns*, `10.1016/j.patter.2023.100804`
- Reich, Y., machine learning techniques for civil engineering problems, *Computer-Aided Civil and Infrastructure Engineering*, `10.1111/0885-9507.00065`
- Adeli, H., neural networks in civil engineering: 1989–2000, *Computer-Aided Civil and Infrastructure Engineering*, `10.1111/0885-9507.00219`
- Sgambi, L., Gkoumas, K., Bontempi, F., genetic algorithms for dependability assurance in long-span suspension bridge design, *Computer-Aided Civil and Infrastructure Engineering*, `10.1111/j.1467-8667.2012.00780.x`
- Valikhani, A., et al., machine learning and image processing for concrete surface roughness, *Computer-Aided Civil and Infrastructure Engineering*, `10.1111/mice.12605`
- Fayaz, J., & Galasso, C., deep neural network framework for on-site acceleration response spectra, *Computer-Aided Civil and Infrastructure Engineering*, `10.1111/mice.12830`

## Figure Captions

### Figure 1 | Validation-target taxonomy and workflow for reproducible structural-ML evaluation

**Layout.** Panel A summarizes the validation-target taxonomy: within-source interpolation (random folds), out-of-source or out-of-programme grouping (literature-aware `GroupKFold`), mixture-family grouping, structural-family extrapolation holds and user-defined deployment-matched splitting. Panel B shows the reproducibility workflow recommended for civil-infrastructure datasets: retrieve OpenAlex-screened structural-ML candidates, triage reproducibility hints, prioritize dataset-first repositories, execute the nine modular benchmarks under identical preprocessing rules, attach group-topology diagnostics and export reporting-checklist artefacts. The figure is organized so that Panel A (taxonomy) and Panel B (workflow) read jointly as **framework plus workflow**.

### Figure 2 | Random validation optimism is heterogeneous and topology-dependent across civil-engineering machine-learning datasets

In random validation, related specimens from the same source family, experimental programme or mixture design can appear in both training and test folds. In group-aware validation, all records from a source or mix family are held out together. Four main reproductions show distinct validation gaps: shear-wall classification decreased from 0.827 to 0.523 accuracy; SFRC shear-capacity regression decreased from fold-mean `R²≈0.642` to `R²≈−1.047` in the panel construction used for Figure 2 (pooled headline values for the same Random Forest run appear in **Table S7**); UCI concrete strength decreased only from pooled `R²=0.909` to `R²=0.868`; and corroded RC beam moment capacity decreased from pooled `R²=0.974` to `R²=0.794`. Error bars show case-specific uncertainty diagnostics. Dataset topology explains the heterogeneity and prevents overinterpretation of any single case.

### Figure 3 | Metadata-level reproducibility signals overstate immediate data availability

The 332-paper OpenAlex candidate pool was triaged into completed, high-priority, medium-priority and low-priority records. Manual verification of the five high-priority records showed that repository hints often pointed to PDFs, metadata-only records, embargoed files or code without raw extracted data. The figure compares metadata labels with manual verification outcomes and highlights the promoted dataset-first queue.

### Figure 4 | Proposed reporting checklist for deployment-matched structural ML validation

The checklist requires authors to report the intended deployment target, grouping variable, random and grouped validation scores, group topology, singleton groups, class balance for classification tasks, pooled and fold-level regression metrics, source/data availability, code availability and a validation-target statement. The figure is formatted as a compact checklist suitable for structural ML reviewers; the same disclosure items can be supplied as structured supplementary material consistent with reproducible-computing expectations in this journal.

### Figure 5 | Source-aware validation in two Mendeley beam-column joint datasets

Two dataset-first Mendeley reproductions extend the case-study evidence to beam-column joint prediction. In the 203-test exterior joint dataset, source-aware grouping by reference-coded authors reduces Random Forest shear-strength pooled `R²` from 0.952 to 0.754 and failure-mode accuracy from 0.635 to 0.266. In the 98-test cyclic joint-shear dataset, grouping by research team reduces Random Forest pooled `R²` from 0.575 to -0.817. The figure is interpreted with group-topology diagnostics because some leave-one-source folds contain very few specimens.

## Supplementary Figure Captions

### Figure S1 | Random validation overstates generalization to unseen CFST section families

In the public Stub-CFST dataset, 1316 specimens are distributed across circular, rectangular and double-skin section families. A Random Forest model achieved pooled `R²=0.828` under random ten-fold validation but decreased to `R²=-0.956` when an entire section family was held out. Because source-reference labels are not included in the public workbooks, this analysis is interpreted as structural-family extrapolation rather than source-reference leakage.

### Figure S2 | Source-aware validation in DesignSafe RC column databases

The DesignSafe circular and rectangular RC column databases provide 497 specimens across 99 source groups after combining the two curated CSV files. Random Forest pooled `R²` decreases from 0.593 to 0.384 for circular columns, from 0.885 to 0.678 for rectangular columns and from 0.600 to 0.490 for the combined task when validation is grouped by author/source.

### Figure S3 | Source-aware validation in the DesignSafe PRJ-2430 RC wall database

The UoA-UW RC wall database includes 142 specimens across 28 author/source groups. Random Forest peak-strength prediction decreases from pooled `R²=0.598` under random validation to `R²=0.069` under GroupKFold. Drift-capacity prediction decreases from `R²=0.300` to `R²=-0.085`. The source-group panel reports the group topology that drives the grouped-validation target.

### Figure S4 | Target-dependent validation gaps in the DesignSafe PRJ-3053 coupling-beam database

The diagonally reinforced concrete coupling-beam database includes 60 specimens across 21 literature-reference groups. Random Forest peak-shear prediction is a contrast case, with grouped validation slightly exceeding random validation (`R²=0.594` random, `R²=0.663` GroupKFold). Normalized shear and chord-rotation capacity show moderate random-minus-group gaps (`ΔR²=0.303` and `0.135`). The figure is interpreted as evidence of target-dependent heterogeneity rather than as a severe-collapse case.

### Figure S5 | Random Forest atlas of headline validation gaps across all reproduced tasks

Horizontal bars report `random − grouped` (or deployment-matched structural-family splits) headline scores for Random Forest benchmarks covering Mangalathu shear-wall classification, SFRC shear capacity, mix-grouped concrete strength, corroded beams, Stub-CFST section-family extrapolation, three Mendeley joint tasks (two regressions plus failure-mode classification), combined DesignSafe RC columns, dual PRJ-2430 wall targets and triple PRJ-3053 coupling-beam outputs. Severity coding (negligible / moderate / severe / contrast) is heuristic for readability; each bar should be read against dataset-specific pooled metrics in supplementary tables.

### Table S1 | Regression validation metrics under random, grouped and leave-one-source-out validation

The table reports both fold-mean and pooled `R²` for SFRC shear capacity, UCI concrete strength, corroded RC beam moment capacity, Mendeley beam-column joint shear tasks and Stub-CFST section-family extrapolation. The table prevents selective reporting and explains why pooled out-of-fold `R²` is used as the headline metric when group folds are small or low-variance.

### Table S2 | Manual reproducibility verification of high-priority metadata records

The table records DOI, topic, metadata priority, checked URL, manual verification status, public-data status, source-label status and next decision for the five metadata-high OpenAlex records and four opportunistic dataset-first candidates.

### Table S3 | Mendeley beam-column joint validation metrics and source topology

The table reports Random Forest headline metrics for the two Mendeley beam-column joint datasets, including random validation, GroupKFold, leave-one-source-out validation and random-minus-group gaps. A companion topology table reports sample counts, source-group counts, median group size, maximum group size and singleton groups. These diagnostics document why pooled prediction-level `R²` is used as the headline metric for the cyclic joint-shear case.

### Table S4 | DesignSafe RC column lateral-strength validation

The table reports random, GroupKFold and leave-one-source-out validation for circular, rectangular and combined reinforced-concrete column lateral-strength prediction. The DesignSafe module is treated as a dataset-first expansion case because public Data Depot APIs expose raw CSV files and author/source labels.

### Table S5 | DesignSafe PRJ-2430 RC wall validation metrics and topology

The table reports peak-strength and drift-capacity regression metrics for the UoA-UW RC wall database under random, GroupKFold and leave-one-source-out validation. It also reports source topology: 142 specimens, 28 source groups, median group size 4, maximum group size 15 and one singleton group.

### Table S6 | DesignSafe PRJ-3053 coupling-beam validation metrics and topology

The table reports peak-shear, normalized-shear and chord-rotation-capacity regression metrics for the diagonally reinforced concrete coupling-beam database. It also reports source topology: 60 specimens, 21 reference groups, median group size 2, maximum group size 9 and six singleton groups.

### Table S7 | Machine-readable Random Forest headline contrasts across nine modules (`cace_ninemodule_rf_summary.csv`)

Comma-separated table generated by `code/33_cace_nine_module_summary.py`; each row records module label, grouping proxy, headline metric (accuracy or pooled `R²`), random score, grouped score, gap (`random − grouped`), balanced-accuracy and macro-F1 gaps for classification rows when available (`nan` placeholders otherwise) and heuristic severity tagging. Editors can treat this table as reproducible bookkeeping for Figures 2/S5 drafting and for the **graphical abstract** (`code/34_graphical_abstract_cace_v0.py`, default tail of `run_qa_chain.py`).

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

