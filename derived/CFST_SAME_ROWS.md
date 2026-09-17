# CFST: comparing on the same evaluated rows

Restricted holdout families: ['double_skin', 'rectangular'] (642 of 1316 records, 48.78%).

| model | random 5-fold, all 1,316 rows | random 5-fold, the same 642 rows | restricted family holdout (642 rows) | difference on the shared rows |
|---|---:|---:|---:|---:|
| Ridge | 0.843 | -0.827 | -1.880 | -1.053 |
| RandomForest | 0.939 | 0.812 | 0.817 | +0.005 |
| ExtraTrees | 0.986 | 0.944 | 0.870 | -0.074 |
| GradientBoosting | 0.972 | 0.928 | 0.894 | -0.034 |

The third and fourth columns share an evaluation set; the first column does not, and is
therefore not a valid comparator for the restricted holdout.
