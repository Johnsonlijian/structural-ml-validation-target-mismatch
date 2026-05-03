"""Shared utilities for cross-validation leakage diagnostics.

This module is the heart of P1. Any reproduction script in `code/03_reproduce.py`
should import from here so that the CV strategies are uniform across studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


@dataclass
class CVResult:
    """One row of the per-paper reproduction record."""

    cv_strategy: str          # "random_kfold" | "group_kfold" | "leave_one_source_out" | "temporal"
    k: int                     # number of folds (or n groups for LOSO)
    r2_mean: float
    r2_std: float
    mae_mean: float
    rmse_mean: float
    n_samples: int
    n_groups: int

    def as_dict(self) -> dict:
        return self.__dict__


def random_kfold_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 10,
    seed: int = 0,
) -> CVResult:
    """Vanilla random k-fold — the strategy used by most papers."""
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    return _run_cv(estimator, X, y, cv, strategy="random_kfold", k=k, n_groups=len(y))


def group_kfold_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    k: int = 10,
) -> CVResult:
    """Group-aware k-fold: ensures specimens from one source never split across train/test."""
    n_groups = int(np.unique(groups).size)
    k_eff = min(k, n_groups)
    cv = GroupKFold(n_splits=k_eff)
    return _run_cv(
        estimator, X, y, cv, strategy="group_kfold", k=k_eff, n_groups=n_groups, groups=groups
    )


def leave_one_source_out_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> CVResult:
    """Strictest: each unique group held out once."""
    n_groups = int(np.unique(groups).size)
    cv = GroupKFold(n_splits=n_groups)
    return _run_cv(
        estimator, X, y, cv, strategy="leave_one_source_out", k=n_groups, n_groups=n_groups, groups=groups
    )


def _run_cv(estimator, X, y, cv, *, strategy, k, n_groups, groups=None):
    r2s, maes, rmses = [], [], []
    splitter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for train_idx, test_idx in splitter:
        if test_idx.size == 0:
            continue
        m = _clone_fit(estimator, X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(mean_squared_error(y[test_idx], pred) ** 0.5)
    return CVResult(
        cv_strategy=strategy,
        k=k,
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)),
        mae_mean=float(np.mean(maes)),
        rmse_mean=float(np.mean(rmses)),
        n_samples=len(y),
        n_groups=n_groups,
    )


def _clone_fit(estimator, X, y):
    """Clone the estimator and fit. Works for sklearn-compatible models."""
    from sklearn.base import clone

    return clone(estimator).fit(X, y)


def delta_r2(reference: CVResult, baseline: CVResult) -> float:
    """ΔR² = baseline (random) − reference (group). Positive = leakage inflation."""
    return baseline.r2_mean - reference.r2_mean


def summarize(results: Iterable[CVResult]) -> pd.DataFrame:
    return pd.DataFrame(r.as_dict() for r in results)
