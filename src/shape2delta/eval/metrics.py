"""Δ-prediction metrics and bootstrap gap CIs.

Primary metrics (field-standard for perturbation models, scored on what moves):
  delta_pearson  : per-signature Pearson(Δ_pred, Δ_true) over the 978 genes, mean
  top_k_overlap  : differential-gene recovery — overlap of the top-k |Δ| genes

Headline quantity: the T2−T1 gap. Reported as a bootstrap mean 95% CI over folds.
A gap counts as real only if the CI excludes 0 in the same direction on >=2 split
types (that aggregation is done by the experiment driver, not here).
"""
from __future__ import annotations

import numpy as np


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
    return num / den


def delta_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.nanmean(_pearson_rows(y_true, y_pred)))


def delta_pearson_rows(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return _pearson_rows(y_true, y_pred)


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, y_true.shape[1])
    tt = np.argpartition(-np.abs(y_true), k - 1, axis=1)[:, :k]
    tp = np.argpartition(-np.abs(y_pred), k - 1, axis=1)[:, :k]
    ov = [len(set(tt[i]) & set(tp[i])) / k for i in range(len(tt))]
    return float(np.mean(ov))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap mean + (1-alpha) CI of a 1-D array (e.g. per-fold gaps)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(len(v)),
    }


def gap_summary(per_fold_a, per_fold_b, n_boot: int = 2000, seed: int = 0):
    """Gap = metric(a) − metric(b) per fold (e.g. T2 − T1). Bootstrap over folds.
    excludes_zero is True iff the 95% CI is strictly one side of 0."""
    a = np.asarray(per_fold_a, float)
    b = np.asarray(per_fold_b, float)
    gaps = a - b
    ci = bootstrap_ci(gaps, n_boot=n_boot, seed=seed)
    ci["per_fold_gap"] = [float(x) for x in gaps]
    ci["excludes_zero"] = bool(np.isfinite(ci["lo"]) and (ci["lo"] > 0 or ci["hi"] < 0))
    ci["direction"] = "pos" if ci["mean"] > 0 else "neg"
    return ci
