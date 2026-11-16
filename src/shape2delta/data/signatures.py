"""LINCS L1000 level-5 Δ signatures for the single-target compounds.

Level-5 (MODZ) signatures are the consensus moderated z-score of each compound
condition vs. the plate population baseline — i.e. a per-gene change-from-control
Δ. We slice the 978 landmark genes × the signatures of our single-target
compounds and cache them as a dense matrix so every downstream experiment reads
from disk (figures/metrics redrawable without touching the 33GB GCTX again).

Outputs under $S2D_FEATURES/lincs_delta/:
  delta.npy          float32  [n_sig x n_landmark]   (z-scores = Δ)
  sig_meta.parquet   one row per signature: sig_id, pert_id, cmap_name,
                     cell_iname, pert_dose, pert_idose, pert_time, is_exemplar_sig,
                     gene_symbol, accession, is_enzyme, target_source
  genes.json         landmark gene_ids (column order) + symbols
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from shape2delta import config

GCTX = config.RAW_LINCS / "level5_beta_trt_cp_n720216x12328.gctx"
OUTDIR = config.FEATURES / "lincs_delta"


@lru_cache(maxsize=1)
def landmark_genes() -> pd.DataFrame:
    gi = pd.read_csv(config.RAW_LINCS / "geneinfo_beta.txt", sep="\t", dtype=str).fillna("")
    fs = next(c for c in gi.columns if c.lower() == "feature_space")
    lm = gi[gi[fs].str.lower() == "landmark"].copy()
    lm["gene_id"] = lm["gene_id"].astype(str)
    return lm[["gene_id", "gene_symbol"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def _siginfo() -> pd.DataFrame:
    usecols = ["sig_id", "pert_id", "pert_type", "cmap_name", "cell_iname",
               "pert_dose", "pert_idose", "pert_time", "qc_pass", "is_exemplar_sig"]
    si = pd.read_csv(config.RAW_LINCS / "siginfo_beta.txt", sep="\t",
                     usecols=lambda c: c in usecols, dtype=str).fillna("")
    return si


def select_signatures(compounds: pd.DataFrame, exemplar_only: bool = False) -> pd.DataFrame:
    """QC-pass trt_cp signatures for the given single-target compounds.

    `compounds` must have: compound_id (pert_id), gene_symbol, accession,
    is_enzyme (optional), target_source (optional).
    """
    si = _siginfo()
    si = si[si["pert_type"] == "trt_cp"]
    si = si[si["qc_pass"].isin(["1", "1.0", "TRUE", "True", "true"])]
    if exemplar_only and "is_exemplar_sig" in si.columns:
        si = si[si["is_exemplar_sig"].isin(["1", "1.0", "TRUE", "True", "true"])]
    keep_cols = [c for c in ["compound_id", "name", "smiles", "inchi_key", "moa",
                             "gene_symbol", "accession", "is_enzyme",
                             "target_source"] if c in compounds.columns]
    cmap = compounds[keep_cols].rename(columns={"compound_id": "pert_id"})
    out = si.merge(cmap, on="pert_id", how="inner")
    return out.reset_index(drop=True)


def extract(sig_meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Read the 978 landmark rows (all columns) from the GCTX once, then subset
    to the selected signatures in memory — contiguous row reads are faster and
    simpler than fancy column indexing. Returns (matrix[n_sig x 978] f32, genes).
    """
    from cmapPy.pandasGEXpress.parse import parse  # heavy dep; import locally

    gene_ids = landmark_genes()["gene_id"].tolist()
    sig_ids = sig_meta["sig_id"].tolist()
    g = parse(str(GCTX), rid=gene_ids)              # 978 rows x all sigs (~2.8GB)
    df = g.data_df.reindex(index=gene_ids)          # genes x sigs, landmark order
    present = [s for s in sig_ids if s in df.columns]
    if len(present) != len(sig_ids):
        missing = len(sig_ids) - len(present)
        print(f"[signatures] WARNING: {missing} selected sig_ids absent from GCTX")
    mat = df.reindex(columns=present).values.T.astype(np.float32)  # sigs x genes
    return mat, gene_ids, present


def _extract_aligned(sig_meta: pd.DataFrame):
    mat, gene_ids, present = extract(sig_meta)
    sig_meta = sig_meta.set_index("sig_id").reindex(present).reset_index()
    return mat, gene_ids, sig_meta


def load_delta(exemplar_only: bool = False, require_smiles: bool = True):
    """Load the cached Δ matrix + aligned metadata.

    Returns (delta[n x 978] float32, meta DataFrame, genes dict). Optionally
    restricts to exemplar signatures (one per condition) and to rows with SMILES.
    """
    delta = np.load(OUTDIR / "delta.npy")
    meta = pd.read_csv(OUTDIR / "sig_meta.csv", dtype={"is_exemplar_sig": str}).reset_index(drop=True)
    genes = json.loads((OUTDIR / "genes.json").read_text())
    mask = np.ones(len(meta), dtype=bool)
    if exemplar_only and "is_exemplar_sig" in meta.columns:
        ex = meta["is_exemplar_sig"].astype(str).isin(["1", "1.0", "True", "true", "TRUE"])
        mask &= ex.to_numpy()
    if require_smiles and "smiles" in meta.columns:
        mask &= meta["smiles"].astype(str).str.len().gt(0).to_numpy()
    delta = delta[mask]
    meta = meta[mask].reset_index(drop=True)
    return delta, meta, genes


def build(exemplar_only: bool = False) -> dict:
    from shape2delta.data import targets as T
    comp, _ = T.build_single_target_compounds()
    # attach accession + enzyme flag via the E0 availability table if present
    av_path = config.RESULTS / "E0_target_availability.csv"
    from shape2delta.data import uniprot as U
    accs, enz = [], []
    enzset = U.enzyme_accessions()
    for g in comp["gene_symbol"]:
        a, _ = U.map_gene(g)
        accs.append(a or "")
        enz.append((a in enzset) if a else False)
    comp = comp.assign(accession=accs, is_enzyme=enz)
    comp = comp[comp["accession"] != ""].copy()

    sig_meta = select_signatures(comp, exemplar_only=exemplar_only)
    mat, gene_ids, sig_meta = _extract_aligned(sig_meta)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTDIR / "delta.npy", mat)
    sig_meta.to_csv(OUTDIR / "sig_meta.csv", index=False)
    genes = landmark_genes()
    (OUTDIR / "genes.json").write_text(json.dumps({
        "gene_ids": gene_ids,
        "gene_symbols": genes.set_index("gene_id").loc[gene_ids, "gene_symbol"].tolist(),
    }))
    summary = {
        "n_signatures": int(mat.shape[0]),
        "n_landmark_genes": int(mat.shape[1]),
        "n_compounds": int(sig_meta["pert_id"].nunique()),
        "n_targets": int(sig_meta["accession"].nunique()),
        "n_cells": int(sig_meta["cell_iname"].nunique()),
        "exemplar_only": exemplar_only,
    }
    (OUTDIR / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(build(exemplar_only=False), indent=2))
