"""Condition-side features for Δ prediction.

A condition = (compound, cell, dose, time). The *non-target* baseline features are:
  - compound chemistry: Morgan/ECFP4 bits from SMILES
  - cell context: one-hot over the cell vocabulary
  - dose/time: log10 dose and time (hours)

Target tier features (T0/T1/T2) are produced elsewhere and concatenated for the
ablation (E3); E2 uses only the baseline block so the gap reflects the predictor's
reliance on seen targets/scaffolds/cells.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=65536)
def ecfp(smiles: str, n_bits: int = 1024, radius: int = 2) -> tuple:
    vec = np.zeros(n_bits, dtype=np.float32)
    if not smiles:
        return tuple(vec)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return tuple(vec)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        on = list(bv.GetOnBits())
        vec[on] = 1.0
    except Exception:
        pass
    return tuple(vec)


def chem_matrix(smiles: pd.Series, n_bits: int = 1024) -> np.ndarray:
    return np.asarray([ecfp(s, n_bits) for s in smiles], dtype=np.float32)


def cell_onehot(cells: pd.Series, vocab: list[str] | None = None):
    if vocab is None:
        vocab = sorted(set(cells.astype(str)))
    idx = {c: i for i, c in enumerate(vocab)}
    M = np.zeros((len(cells), len(vocab)), dtype=np.float32)
    for i, c in enumerate(cells.astype(str)):
        j = idx.get(c)
        if j is not None:
            M[i, j] = 1.0
    return M, vocab


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()


def dose_time_block(meta: pd.DataFrame) -> np.ndarray:
    dose = _num(meta.get("pert_dose", pd.Series([np.nan] * len(meta))))
    time = _num(meta.get("pert_time", pd.Series([np.nan] * len(meta))))
    ldose = np.log10(np.clip(dose, 1e-3, None))
    ldose = np.nan_to_num(ldose, nan=0.0)
    time = np.nan_to_num(time, nan=24.0)
    return np.column_stack([ldose, time]).astype(np.float32)


def baseline_features(meta: pd.DataFrame, cell_vocab: list[str] | None = None,
                      n_bits: int = 1024):
    """Return (X, info). X = [ECFP | cell one-hot | log-dose | time]."""
    chem = chem_matrix(meta["smiles"], n_bits=n_bits)
    cells, vocab = cell_onehot(meta["cell_iname"], vocab=cell_vocab)
    dt = dose_time_block(meta)
    X = np.concatenate([chem, cells, dt], axis=1).astype(np.float32)
    info = {"n_chem": chem.shape[1], "n_cell": cells.shape[1], "n_dt": dt.shape[1],
            "cell_vocab": vocab}
    return X, info
