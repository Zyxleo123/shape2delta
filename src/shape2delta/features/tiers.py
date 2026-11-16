"""Assemble the nested, fixed-length target feature tiers.

T0 ⊂ T1 ⊂ T2 by construction. Every arm is the SAME length D0+D1+D2; only which
blocks are populated changes:
    tier0 = [esm_pca | 0        | 0         ]
    tier1 = [esm_pca | t1_struct | 0         ]
    tier2 = [esm_pca | t1_struct | t2_ensemble]
so the only thing differing between arms is tier membership (the spec's clean
ablation). The comparison set is targets with ESM + T1 + T2 all present.

Also saves the full ESM matrix (for E1's ESM-similarity control) and the raw
standardized T2 block (for E1 / E6).

  python -m shape2delta.features.tiers
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.features import esm_embed

T1_FEATS = ["plddt_mean", "plddt_std", "plddt_frac_vhigh", "plddt_frac_low",
            "rg", "n_res", "ss_helix", "ss_sheet", "ss_coil",
            "pae_mean", "pae_frac_conf", "fpocket_n", "fpocket_top_score",
            "fpocket_top_drugg", "fpocket_top_volume"]
T2_FEATS = ["n_conformers", "rmsf_mean", "rmsf_std", "rmsf_max", "rmsf_p90",
            "frac_flexible", "frac_seq_covered", "rg_mean", "rg_cv"]
OUT = config.FEATURES / "tiers"


def _standardize(df, cols):
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").values.astype(float)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd


def build(d0: int = 64) -> dict:
    from sklearn.decomposition import PCA

    esm_mat, esm_accs = esm_embed.load()
    esm_by = {a: esm_mat[i] for i, a in enumerate(esm_accs)}
    t1 = pd.read_csv(config.FEATURES / "t1_structure" / "t1_structure.csv")
    t2 = pd.read_csv(config.FEATURES / "t2_ensemble.csv")
    t1 = t1.set_index("accession")
    t2 = t2.set_index("accession")

    # comparison set: ESM + T1 + T2 all present
    accs = [a for a in esm_accs if a in t1.index and a in t2.index]
    accs = sorted(set(accs))
    if not accs:
        raise RuntimeError("empty comparison set (need ESM ∩ T1 ∩ T2)")

    esm_full = np.stack([esm_by[a] for a in accs]).astype(np.float32)
    t1b = _standardize(t1.loc[accs].reset_index(), T1_FEATS).astype(np.float32)
    t2b = _standardize(t2.loc[accs].reset_index(), T2_FEATS).astype(np.float32)

    d0 = min(d0, esm_full.shape[1], len(accs))
    esm_pca = PCA(n_components=d0, random_state=0).fit_transform(
        (esm_full - esm_full.mean(0)) / (esm_full.std(0) + 1e-9)).astype(np.float32)

    D0, D1, D2 = esm_pca.shape[1], t1b.shape[1], t2b.shape[1]
    Z1, Z2 = np.zeros((len(accs), D1), np.float32), np.zeros((len(accs), D2), np.float32)
    tier0 = np.concatenate([esm_pca, Z1, Z2], axis=1)
    tier1 = np.concatenate([esm_pca, t1b, Z2], axis=1)
    tier2 = np.concatenate([esm_pca, t1b, t2b], axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "target_tiers.npz", accs=np.array(accs), tier0=tier0,
             tier1=tier1, tier2=tier2, esm_full=esm_full, t2_block=t2b,
             t1_block=t1b, dims=np.array([D0, D1, D2]))
    summ = {"n_targets_comparison_set": len(accs), "D0_esm_pca": int(D0),
            "D1_struct": int(D1), "D2_ensemble": int(D2),
            "esm_full_dim": int(esm_full.shape[1])}
    (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def load():
    z = np.load(OUT / "target_tiers.npz", allow_pickle=True)
    return z


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
