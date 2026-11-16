"""Shared cross-validation core for the Δ-prediction experiments (E2, E3).

A multi-output ridge predicts the 978-gene Δ from a feature matrix. The same
routine runs every split kind; E2 feeds only baseline (chemistry+cell) features,
E3 feeds baseline ⊕ a target tier (T0/T1/T2) so the T2−T1 gap is isolated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from shape2delta.eval import splits as SP
from shape2delta.eval import metrics as M


def ridge_fold_metrics(X, Y, fold_ids, alpha=100.0, k_top=100):
    """Per-fold metrics for one split (train on fold != i, test on fold == i)."""
    res = []
    folds = sorted(f for f in set(fold_ids) if f >= 0)
    for i in folds:
        te = fold_ids == i
        tr = (~te) & (fold_ids >= 0)
        if tr.sum() < 10 or te.sum() < 5:
            continue
        model = Ridge(alpha=alpha, solver="cholesky")
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])
        res.append({
            "fold": int(i),
            "n_test": int(te.sum()),
            "delta_pearson": M.delta_pearson(Y[te], pred),
            "top100": M.top_k_overlap(Y[te], pred, k=k_top),
        })
    return res


def run_all_splits(X, Y, meta, kinds=SP.SPLIT_KINDS, n_splits=5, seed=0,
                   alpha=100.0):
    out = {}
    for kind in kinds:
        folds = SP.make_folds(meta, kind, n_splits=n_splits, seed=seed)
        fm = ridge_fold_metrics(X, Y, folds, alpha=alpha)
        dp = [r["delta_pearson"] for r in fm]
        t100 = [r["top100"] for r in fm]
        out[kind] = {
            "per_fold": fm,
            "n_folds": len(fm),
            "delta_pearson_mean": float(np.nanmean(dp)) if dp else float("nan"),
            "top100_mean": float(np.nanmean(t100)) if t100 else float("nan"),
            "delta_pearson_folds": dp,
            "top100_folds": t100,
        }
    return out


def drop_ci(random_vals, heldout_vals, n_boot=2000, seed=0):
    """Generalization drop = mean(random) − mean(heldout), CI by independent
    bootstrap of each fold set."""
    r = np.asarray([v for v in random_vals if np.isfinite(v)], float)
    h = np.asarray([v for v in heldout_vals if np.isfinite(v)], float)
    if len(r) == 0 or len(h) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    br = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(1)
    bh = rng.choice(h, size=(n_boot, len(h)), replace=True).mean(1)
    d = br - bh
    return {
        "mean": float(r.mean() - h.mean()),
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "excludes_zero": bool(np.quantile(d, 0.025) > 0 or np.quantile(d, 0.975) < 0),
    }
