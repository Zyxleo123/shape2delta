"""E5 data: Replogle K562 CRISPRi → pseudobulk Δ per knocked-down target.

For each CRISPRi perturbation (a knocked-down gene), Δ = mean(perturbed) −
mean(control) over the 978 LINCS landmark genes (matched by gene symbol), mapped
to the target's UniProt. Knockdown removes the protein, so its conformation
should carry no signal — the specificity control for E3.

  python -m shape2delta.data.replogle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.data import uniprot as U

H5AD = config.RAW / "replogle" / "ReplogleWeissman2022_K562_essential.h5ad"
OUT = config.FEATURES / "replogle_delta"


def _find_cols(adata):
    pert_col = next((c for c in ["perturbation", "gene", "target_gene", "guide_target"]
                     if c in adata.obs.columns), None)
    if pert_col is None:
        raise RuntimeError(f"no perturbation column in obs: {list(adata.obs.columns)[:20]}")
    return pert_col


def _control_mask(obs, pert_col):
    vals = obs[pert_col].astype(str)
    ctrl_tokens = {"control", "non-targeting", "nontargeting", "ntc", "neg",
                   "safe-harbor", "safe_harbor", "*", "none"}
    return vals.str.lower().isin(ctrl_tokens)


def build() -> dict:
    import scanpy as sc
    ad = sc.read_h5ad(H5AD)
    pert_col = _find_cols(ad)
    # landmark genes by symbol
    lm = S.landmark_genes()  # gene_id, gene_symbol
    lm_syms = {s.upper(): g for g, s in zip(lm["gene_id"], lm["gene_symbol"])}
    var_syms = ad.var.index.astype(str)
    # try to match var to gene symbol (var_names may be symbols or ensembl)
    sym_col = next((c for c in ["gene_symbol", "gene_name", "symbol", "feature_name"]
                    if c in ad.var.columns), None)
    if sym_col is not None:
        var_syms = ad.var[sym_col].astype(str)
    var_up = pd.Index(var_syms.str.upper())
    keep_genes = [i for i, s in enumerate(var_up) if s in lm_syms]
    if len(keep_genes) < 200:
        return {"ok": False, "reason": f"only {len(keep_genes)} landmark genes matched"}

    X = ad[:, keep_genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    obs = ad.obs
    ctrl = _control_mask(obs, pert_col).to_numpy()
    ctrl_mean = X[ctrl].mean(0)

    perts = obs[pert_col].astype(str)
    rows, meta = [], []
    for g, idx in pd.Series(np.arange(len(perts))).groupby(perts.values):
        if _control_mask(pd.DataFrame({pert_col: [g]}), pert_col).iloc[0]:
            continue
        acc, st = U.map_gene(str(g))
        if acc is None:
            continue
        delta = X[idx.values].mean(0) - ctrl_mean
        rows.append(delta)
        meta.append({"perturbation": g, "accession": acc, "n_cells": int(len(idx))})
    if not rows:
        return {"ok": False, "reason": "no mapped perturbations"}

    mat = np.stack(rows).astype(np.float32)
    md = pd.DataFrame(meta)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "delta.npy", mat)
    md.to_csv(OUT / "meta.csv", index=False)
    kept_syms = [str(var_up[i]) for i in keep_genes]
    (OUT / "genes.json").write_text(json.dumps(kept_syms))
    summ = {"ok": True, "n_perturbations": int(len(md)),
            "n_genes_matched": int(len(keep_genes)),
            "n_mapped_targets": int(md["accession"].nunique())}
    (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def load():
    mat = np.load(OUT / "delta.npy")
    md = pd.read_csv(OUT / "meta.csv")
    return mat, md


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
