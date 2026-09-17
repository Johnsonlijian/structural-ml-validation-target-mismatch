# R37 corrected engineering-decision experiment

Unit check (physical path vs log path): PASS [{'margin': 0.0, 'agree': True, 'mismatches': 0}, {'margin': 0.2, 'agree': True, 'mismatches': 0}, {'margin': 0.5, 'agree': True, 'mismatches': 0}]

Risk denominators: `r_all` = unsafe events / all test records (used by the selector);
`r_released` = unsafe events / released records (NA when nothing is released);
`coverage` = released / all test records.

## failure_mode — action set: act_on_confidence

| protocol | alpha | folds | mean estimated r_all | mean realised r_all | gap | folds above own estimate | folds meeting target | mean coverage | mean r_released |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.05 | 5 | 0.0260 | 0.0026 | -0.0234 | 0 | 5 | 0.003 | 1.0000 |
| random | 0.10 | 5 | 0.0415 | 0.0053 | -0.0362 | 0 | 5 | 0.005 | 1.0000 |
| random | 0.20 | 5 | 0.0743 | 0.0579 | -0.0164 | 1 | 4 | 0.063 | 0.9697 |
| source | 0.05 | 5 | 0.0019 | 0.0053 | +0.0034 | 2 | 5 | 0.005 | 1.0000 |
| source | 0.10 | 5 | 0.0524 | 0.0472 | -0.0053 | 1 | 5 | 0.050 | 0.9583 |
| source | 0.20 | 5 | 0.1424 | 0.1602 | +0.0177 | 2 | 4 | 0.203 | 0.6942 |

## failure_mode — action set: release_when_not_dangerous

| protocol | alpha | folds | mean estimated r_all | mean realised r_all | gap | folds above own estimate | folds meeting target | mean coverage | mean r_released |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.05 | 5 | 0.0426 | 0.0767 | +0.0341 | 3 | 2 | 0.477 | 0.1601 |
| random | 0.10 | 5 | 0.0470 | 0.0741 | +0.0271 | 4 | 4 | 0.477 | 0.1553 |
| random | 0.20 | 5 | 0.0470 | 0.0741 | +0.0271 | 4 | 5 | 0.477 | 0.1553 |
| source | 0.05 | 5 | 0.0376 | 0.0538 | +0.0162 | 3 | 2 | 0.403 | 0.1147 |
| source | 0.10 | 5 | 0.0713 | 0.0873 | +0.0160 | 2 | 3 | 0.487 | 0.1879 |
| source | 0.20 | 5 | 0.0772 | 0.0873 | +0.0101 | 2 | 5 | 0.487 | 0.1879 |

## Selection behaviour (is the constraint active?)

| scenario | protocol | fold | alpha | chosen rule | inner r_all | feasible rules |
|---|---|---:|---:|---|---:|---:|
| beam_release | random | 0 | 0.05 | GradientBoosting/0.0 | 0.0140 | 63 |
| beam_release | random | 0 | 0.10 | GradientBoosting/0.0 | 0.0140 | 63 |
| beam_release | random | 0 | 0.20 | GradientBoosting/0.0 | 0.0140 | 63 |
| beam_release | random | 1 | 0.05 | GradientBoosting/0.0 | 0.0125 | 63 |
| beam_release | random | 1 | 0.10 | GradientBoosting/0.0 | 0.0125 | 63 |
| beam_release | random | 1 | 0.20 | GradientBoosting/0.0 | 0.0125 | 63 |
| beam_release | random | 2 | 0.05 | RandomForest/0.0 | 0.0217 | 63 |
| beam_release | random | 2 | 0.10 | RandomForest/0.0 | 0.0217 | 63 |
| beam_release | random | 2 | 0.20 | RandomForest/0.0 | 0.0217 | 63 |
| beam_release | random | 3 | 0.05 | RandomForest/0.0 | 0.0248 | 63 |
| beam_release | random | 3 | 0.10 | RandomForest/0.0 | 0.0248 | 63 |
| beam_release | random | 3 | 0.20 | RandomForest/0.0 | 0.0248 | 63 |
| beam_release | random | 4 | 0.05 | Ridge/0.0 | 0.0264 | 63 |
| beam_release | random | 4 | 0.10 | Ridge/0.0 | 0.0264 | 63 |
| beam_release | random | 4 | 0.20 | Ridge/0.0 | 0.0264 | 63 |
| beam_release | source | 0 | 0.05 | GradientBoosting/0.1 | 0.0389 | 52 |
| beam_release | source | 0 | 0.10 | RandomForest/0.0 | 0.0857 | 63 |
| beam_release | source | 0 | 0.20 | RandomForest/0.0 | 0.0857 | 63 |
| beam_release | source | 1 | 0.05 | GradientBoosting/0.0 | 0.0483 | 55 |
| beam_release | source | 1 | 0.10 | Ridge/0.1 | 0.0966 | 61 |
| beam_release | source | 1 | 0.20 | Ridge/0.0 | 0.1075 | 63 |
| beam_release | source | 2 | 0.05 | GradientBoosting/0.05 | 0.0248 | 51 |
| beam_release | source | 2 | 0.10 | RandomForest/0.2 | 0.0637 | 54 |
| beam_release | source | 2 | 0.20 | RandomForest/0.0 | 0.1630 | 63 |
| beam_release | source | 3 | 0.05 | RandomForest/0.0 | 0.0279 | 56 |
| beam_release | source | 3 | 0.10 | Ridge/0.1 | 0.0994 | 61 |
| beam_release | source | 3 | 0.20 | Ridge/0.0 | 0.1149 | 63 |
| beam_release | source | 4 | 0.05 | GradientBoosting/0.4 | 0.0404 | 42 |
| beam_release | source | 4 | 0.10 | GradientBoosting/0.0 | 0.0606 | 63 |
| beam_release | source | 4 | 0.20 | GradientBoosting/0.0 | 0.0606 | 63 |
| failure_mode | random | 0 | 0.05 | RandomForest/nan | 0.0197 | 13 |
| failure_mode | random | 0 | 0.05 | ExtraTrees/nan | 0.0426 | 13 |
| failure_mode | random | 0 | 0.10 | ExtraTrees/nan | 0.0557 | 14 |
| failure_mode | random | 0 | 0.10 | ExtraTrees/nan | 0.0426 | 14 |
| failure_mode | random | 0 | 0.20 | ExtraTrees/nan | 0.0557 | 14 |
| failure_mode | random | 0 | 0.20 | ExtraTrees/nan | 0.0426 | 14 |
| failure_mode | random | 1 | 0.05 | ExtraTrees/nan | 0.0411 | 14 |
| failure_mode | random | 1 | 0.05 | ExtraTrees/nan | 0.0475 | 14 |
| failure_mode | random | 1 | 0.10 | ExtraTrees/nan | 0.0411 | 14 |
| failure_mode | random | 1 | 0.10 | ExtraTrees/nan | 0.0475 | 14 |
| failure_mode | random | 1 | 0.20 | ExtraTrees/nan | 0.0411 | 14 |
| failure_mode | random | 1 | 0.20 | ExtraTrees/nan | 0.0475 | 14 |
| failure_mode | random | 2 | 0.05 | ExtraTrees/nan | 0.0063 | 12 |
| failure_mode | random | 2 | 0.05 | RandomForest/nan | 0.0379 | 12 |
| failure_mode | random | 2 | 0.10 | ExtraTrees/nan | 0.0063 | 14 |
| failure_mode | random | 2 | 0.10 | ExtraTrees/nan | 0.0505 | 14 |
| failure_mode | random | 2 | 0.20 | RandomForest/nan | 0.1704 | 15 |
| failure_mode | random | 2 | 0.20 | ExtraTrees/nan | 0.0505 | 15 |
| failure_mode | random | 3 | 0.05 | ExtraTrees/nan | 0.0410 | 10 |
| failure_mode | random | 3 | 0.05 | ExtraTrees/nan | 0.0410 | 10 |
| failure_mode | random | 3 | 0.10 | ExtraTrees/nan | 0.0410 | 14 |
| failure_mode | random | 3 | 0.10 | ExtraTrees/nan | 0.0505 | 14 |
| failure_mode | random | 3 | 0.20 | ExtraTrees/nan | 0.0410 | 14 |
| failure_mode | random | 3 | 0.20 | ExtraTrees/nan | 0.0505 | 14 |
| failure_mode | random | 4 | 0.05 | RandomForest/nan | 0.0221 | 13 |
| failure_mode | random | 4 | 0.05 | ExtraTrees/nan | 0.0442 | 13 |
| failure_mode | random | 4 | 0.10 | ExtraTrees/nan | 0.0631 | 14 |
| failure_mode | random | 4 | 0.10 | ExtraTrees/nan | 0.0442 | 14 |
| failure_mode | random | 4 | 0.20 | ExtraTrees/nan | 0.0631 | 14 |
| failure_mode | random | 4 | 0.20 | ExtraTrees/nan | 0.0442 | 14 |
| failure_mode | source | 0 | 0.05 | ExtraTrees/nan | 0.0000 | 9 |
| failure_mode | source | 0 | 0.05 | ExtraTrees/nan | 0.0426 | 9 |
| failure_mode | source | 0 | 0.10 | ExtraTrees/nan | 0.0000 | 12 |
| failure_mode | source | 0 | 0.10 | ExtraTrees/nan | 0.0754 | 12 |
| failure_mode | source | 0 | 0.20 | ExtraTrees/nan | 0.1377 | 16 |
| failure_mode | source | 0 | 0.20 | RandomForest/nan | 0.1049 | 16 |
| failure_mode | source | 1 | 0.05 | ExtraTrees/nan | 0.0063 | 11 |
| failure_mode | source | 1 | 0.05 | ExtraTrees/nan | 0.0443 | 11 |
| failure_mode | source | 1 | 0.10 | ExtraTrees/nan | 0.0981 | 15 |
| failure_mode | source | 1 | 0.10 | RandomForest/nan | 0.0696 | 15 |
| failure_mode | source | 1 | 0.20 | RandomForest/nan | 0.1266 | 16 |
| failure_mode | source | 1 | 0.20 | RandomForest/nan | 0.0696 | 16 |
| failure_mode | source | 2 | 0.05 | ExtraTrees/nan | 0.0032 | 10 |
| failure_mode | source | 2 | 0.05 | ExtraTrees/nan | 0.0410 | 10 |
| failure_mode | source | 2 | 0.10 | ExtraTrees/nan | 0.0663 | 16 |
| failure_mode | source | 2 | 0.10 | RandomForest/nan | 0.0694 | 16 |
| failure_mode | source | 2 | 0.20 | RandomForest/nan | 0.1514 | 17 |
| failure_mode | source | 2 | 0.20 | RandomForest/nan | 0.0694 | 17 |
| failure_mode | source | 3 | 0.05 | ExtraTrees/nan | 0.0000 | 8 |
| failure_mode | source | 3 | 0.05 | ExtraTrees/nan | 0.0158 | 8 |
| failure_mode | source | 3 | 0.10 | RandomForest/nan | 0.0978 | 16 |
| failure_mode | source | 3 | 0.10 | ExtraTrees/nan | 0.0883 | 16 |
| failure_mode | source | 3 | 0.20 | RandomForest/nan | 0.0978 | 16 |
| failure_mode | source | 3 | 0.20 | ExtraTrees/nan | 0.0883 | 16 |
| failure_mode | source | 4 | 0.05 | ExtraTrees/nan | 0.0000 | 10 |
| failure_mode | source | 4 | 0.05 | ExtraTrees/nan | 0.0442 | 10 |
| failure_mode | source | 4 | 0.10 | ExtraTrees/nan | 0.0000 | 14 |
| failure_mode | source | 4 | 0.10 | RandomForest/nan | 0.0536 | 14 |
| failure_mode | source | 4 | 0.20 | ExtraTrees/nan | 0.1987 | 16 |
| failure_mode | source | 4 | 0.20 | RandomForest/nan | 0.0536 | 16 |
