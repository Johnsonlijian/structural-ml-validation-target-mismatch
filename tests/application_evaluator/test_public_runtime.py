from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

EVALUATOR_ROOT = Path(__file__).resolve().parents[2] / "src" / "application_evaluator"
sys.path.insert(0, str(EVALUATOR_ROOT))

from public_runtime import assignments_from_splitter, atomic_write_json, build_models


class PublicRuntimeTests(unittest.TestCase):
    def test_build_models_preserves_frozen_factory(self) -> None:
        models = build_models(2026071201)
        self.assertEqual(
            list(models),
            ["Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting"],
        )
        self.assertEqual(models["Ridge"].named_steps["ridge"].alpha, 1.0)
        self.assertEqual(models["RBF_SVR"].named_steps["svr"].C, 10.0)
        self.assertEqual(models["ExtraTrees"].n_estimators, 160)
        self.assertEqual(models["ExtraTrees"].min_samples_leaf, 3)
        self.assertEqual(models["ExtraTrees"].max_features, 0.9)
        self.assertEqual(models["HistGradientBoosting"].max_iter, 180)

    def test_assignments_cover_each_record_once(self) -> None:
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        observed = assignments_from_splitter(
            KFold(n_splits=5, shuffle=True, random_state=7), X, y
        )
        self.assertEqual(len(observed), 10)
        self.assertEqual(np.bincount(observed, minlength=5).tolist(), [2] * 5)

    def test_atomic_json_is_complete_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "receipt.json"
            atomic_write_json(path, {"z": 1, "a": [2, 3]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": [2, 3], "z": 1})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
            self.assertFalse(any(path.parent.glob(".__tmp_*")))


if __name__ == "__main__":
    unittest.main()
