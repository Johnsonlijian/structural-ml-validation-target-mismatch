# R38 verification of the third review

## 1. Rendered page geometry (measured on this project's PDFs)
- main.pdf: pages with text outside the printable area: 2; worst right edge 596.9 pt (page width 595.3 pt)
    page 5: 2 spans outside, max_x 596.9, smallest font 7.97 pt
    page 12: 5 spans outside, max_x 596.2, smallest font 6.8 pt
- si.pdf: pages with text outside the printable area: 5; worst right edge 615.4 pt (page width 612.0 pt)
    page 5: 1 spans outside, max_x 615.4, smallest font 9.96 pt
    page 7: 14 spans outside, max_x 611.2, smallest font 7.97 pt
    page 8: 36 spans outside, max_x 606.5, smallest font 10.91 pt
    page 9: 36 spans outside, max_x 606.5, smallest font 10.91 pt
    page 10: 10 spans outside, max_x 606.5, smallest font 10.91 pt

## 2. Font sizes below 6 pt in main.pdf
- [(4.8, [15]), (4.9, [11]), (5.0, [11, 15]), (5.1, [10, 15]), (5.5, [8]), (5.7, [4])]

## 3. Corpus inventory arithmetic
- inventory rows in the table: 12
    UCI concrete compressive strength       1030
    SFRC beam shear database                 488
    Stub CFST axial capacity                1316
    Corroded RC beam flexure                 804
    RC shear-wall failure mode               393
    UoA-UW RC wall database                  142
    Coupling-beam database                    60
    Circular RC columns                      174
    Rectangular RC columns                   327
    Exterior beam-column joints              203
    Beam-column joints, cyclic                98
- sum of the listed record counts: **5035**
- text claims: 4,933? True; 'nine' present? True
- verdict: CONFIRMED inconsistency

## 4. Relation-stratified contrast table (main Table 4)
- rows: 37; rows printing nan: 16; rows with 'alias': 4
- concrete rows mislabelled as proxy (wall type): 4
- table carries an honoured-R2 column: True
- verdict: alias duplication CONFIRMED; mislabel CONFIRMED; nan display CONFIRMED; missing honoured column not found

## 5. Text claims flagged by the review
- §3.3 still contains the pooled '32 of 36' sentence: CONFIRMED
- §3.3 still claims each other stratum is 4/4 below control: CONFIRMED
- 'largest admissible degradation' wording still used for an R2 value: CONFIRMED
- §2.1 still calls classification a 'seventh dataset': not found

## 6. Decision experiment: protocol comparison per target
- beam release  alpha=0.05: random 0.0499 vs source 0.0199 -> difference -0.0300 (source better)
- beam release  alpha=0.10: random 0.0499 vs source 0.0162 -> difference -0.0336 (source better)
- beam release  alpha=0.20: random 0.0499 vs source 0.0200 -> difference -0.0299 (source better)
- failure mode  alpha=0.05: random 0.0767 vs source 0.0538 -> difference -0.0230 (source better)
- failure mode  alpha=0.10: random 0.0741 vs source 0.0873 -> difference +0.0132 (SOURCE WORSE)
- failure mode  alpha=0.20: random 0.0741 vs source 0.0873 -> difference +0.0132 (SOURCE WORSE)

## 7. Fold-mean versus pooled risk (identity check)
- beam/random/0.05: fold means r_all 0.0499, r_rel 0.1092, C 0.5184 (product 0.0566 != 0.0499)
    pooled: r_all 0.0498, r_rel 0.0962, C 0.5174 (product 0.0498 == r_all)
    counts: n_test 804, n_released 416, n_unsafe 40
- beam/source/0.05: fold means r_all 0.0199, r_rel 0.0311, C 0.4934 (product 0.0153 != 0.0199)
    pooled: r_all 0.0199, r_rel 0.0404, C 0.4925 (product 0.0199 == r_all)
    counts: n_test 804, n_released 396, n_unsafe 16

## 8. Figures carrying stale numbers
- figure 1 still contains '0.021': CONFIRMED
- figure 1 still contains '0.050': CONFIRMED
- figure 1 still contains '0.038': CONFIRMED
- figure 1 still contains '0.026': CONFIRMED
- figure 1 still contains 'cannot move the estimate': CONFIRMED
- figure 1 still contains 'follows the target': CONFIRMED
- feasibility-map code labels amber as max_w>U only: not found

## 9. Abstract and highlights against the AEI indexing limits
- abstract words (word-per-token count): 263
- highlight 1: 85 characters
- highlight 2: 91 characters  **over the 85-character limit**
- highlight 3: 83 characters
- highlight 4: 88 characters  **over the 85-character limit**
- highlight 5: 92 characters  **over the 85-character limit**
