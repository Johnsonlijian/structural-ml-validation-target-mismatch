from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import pandas as pd

EVALUATOR_ROOT = Path(__file__).resolve().parents[2] / "src" / "application_evaluator"
sys.path.insert(0, str(EVALUATOR_ROOT))

from run_public_evaluation import REFERENCE_METRICS, check_frozen_48_cells


class FrozenMetricCheckTests(unittest.TestCase):
    def test_exact_48_cell_check(self) -> None:
        rows = []
        for parent in ("deck464", "lwc90", "rac27"):
            for model in ("Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting"):
                row = {"parent": parent, "model": model}
                row.update({metric: float(len(rows) + index) for index, metric in enumerate(REFERENCE_METRICS)})
                rows.append(row)
        observed = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "external_metrics.csv"
            observed.to_csv(reference, index=False)
            result = check_frozen_48_cells(observed, reference, 1e-12)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metric_cells_checked"], 48)
        self.assertEqual(result["maximum_absolute_difference"], 0.0)

    def test_metric_check_detects_change(self) -> None:
        rows = []
        for parent in ("deck464", "lwc90", "rac27"):
            for model in ("Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting"):
                row = {"parent": parent, "model": model}
                row.update({metric: 1.0 for metric in REFERENCE_METRICS})
                rows.append(row)
        observed = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "external_metrics.csv"
            observed.to_csv(reference, index=False)
            changed = observed.copy()
            changed.loc[0, "external_rmse_kN"] = 1.01
            result = check_frozen_48_cells(changed, reference, 1e-12)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metric_cells_checked"], 48)

    def test_license_separated_projection_reference(self) -> None:
        rows = []
        for parent in ("deck464", "lwc90", "rac27"):
            for model in ("Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting"):
                row = {"parent": parent, "model": model}
                row.update({metric: float(len(rows) + index) for index, metric in enumerate(REFERENCE_METRICS)})
                rows.append(row)
        observed = pd.DataFrame(rows)
        locations = {
            "deck464": "results/cc_by_4_0/deck464/evaluation_external_metrics.csv",
            "lwc90": "results/cc_by_4_0/lwc90/evaluation_external_metrics.csv",
            "rac27": "results/cc_by_nc_sa_4_0/rac27/evaluation_external_metrics.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for parent, logical in locations.items():
                path = root / logical
                path.parent.mkdir(parents=True, exist_ok=True)
                observed.loc[observed["parent"] == parent].to_csv(path, index=False)
            result = check_frozen_48_cells(observed, root, 1e-12)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metric_cells_checked"], 48)
        self.assertEqual(result["reference"]["kind"], "license_separated_public_projection")


if __name__ == "__main__":
    unittest.main()
