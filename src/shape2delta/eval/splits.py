"""Generalization splits — never random-example splits for the headline claim.

Four split kinds over the per-signature table:
  random   : KFold over signatures (the in-distribution reference / E2 baseline)
  target   : GroupKFold grouped by target UniProt accession (held-out target)
  scaffold : GroupKFold grouped by Bemis-Murcko scaffold (held-out chemistry)
  cell     : GroupKFold grouped by cell_iname (held-out cell context)

A whole group lands entirely in train or test, so test targets/scaffolds/cells
are unseen in training — the regime the ensemble feature is meant to repair.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

SPLIT_KINDS = ("random", "target", "scaffold", "cell")
_GROUP_COL = {"target": "accession", "cell": "cell_iname"}


@lru_cache(maxsize=4096)
def _scaffold(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def scaffold_series(meta: pd.DataFrame) -> pd.Series:
    col = "smiles" if "smiles" in meta.columns else None
    if col is None:
        return pd.Series([""] * len(meta), index=meta.index)
    return meta[col].map(_scaffold)


def group_labels(meta: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "scaffold":
        g = scaffold_series(meta).to_numpy()
    else:
        col = _GROUP_COL[kind]
        g = meta[col].astype(str).to_numpy()
    return g


def make_folds(meta: pd.DataFrame, kind: str, n_splits: int = 5,
               seed: int = 0) -> np.ndarray:
    """Return an int fold id per row (0..n_splits-1), or -1 for rows that cannot
    be grouped (empty scaffold / missing group key)."""
    n = len(meta)
    fold = np.full(n, -1, dtype=int)
    if kind == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (_, te) in enumerate(kf.split(np.arange(n))):
            fold[te] = i
        return fold
    if kind not in SPLIT_KINDS:
        raise ValueError(f"unknown split kind {kind!r}")
    g = group_labels(meta, kind)
    valid = np.array([bool(x) and x.lower() not in ("nan", "none") for x in g])
    idx = np.where(valid)[0]
    if len(np.unique(g[idx])) < n_splits:
        # not enough distinct groups to form n folds
        return fold
    gkf = GroupKFold(n_splits=n_splits)
    for i, (_, te) in enumerate(gkf.split(idx, groups=g[idx])):
        fold[idx[te]] = i
    return fold


def fold_table(meta: pd.DataFrame, kinds=SPLIT_KINDS, n_splits: int = 5,
               seed: int = 0) -> pd.DataFrame:
    out = {}
    for k in kinds:
        out[f"fold_{k}"] = make_folds(meta, k, n_splits=n_splits, seed=seed)
    return pd.DataFrame(out, index=meta.index)
