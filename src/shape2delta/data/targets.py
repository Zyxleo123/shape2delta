"""Compound -> single primary target, for the LINCS L1000 atlas.

Pipeline:
  LINCS compoundinfo_beta  (pert_id, cmap_name, target, moa, smiles, inchi_key)
    x  DRH repurposing_drugs (authoritative multi-vs-single target: '|'-separated)
    x  LINCS siginfo_beta    (which compounds actually have trt_cp signatures,
                              and in which cell contexts / doses)
  -> per-compound table restricted to a SINGLE annotated primary target.

Every filtering step records a drop count so a thin result is distinguishable
from a real null downstream.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from shape2delta import config

_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_name(s: str) -> str:
    return _ALNUM.sub("", str(s).strip().lower())


def _read_drh(path: Path) -> pd.DataFrame:
    # '!'-prefixed metadata preamble, then a header row.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    start = next(i for i, ln in enumerate(lines) if not ln.startswith("!"))
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines[start:])), sep="\t", dtype=str).fillna("")


@lru_cache(maxsize=1)
def load_compoundinfo() -> pd.DataFrame:
    p = config.RAW_LINCS / "compoundinfo_beta.txt"
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    df["target"] = df["target"].replace('""', "").str.strip()
    df["moa"] = df["moa"].replace('""', "").str.strip()
    return df


@lru_cache(maxsize=1)
def load_drh_drugs() -> pd.DataFrame:
    return _read_drh(config.RAW_DRH / "repurposing_drugs_20200324.txt")


@lru_cache(maxsize=1)
def load_drh_samples() -> pd.DataFrame:
    return _read_drh(config.RAW_DRH / "repurposing_samples_20200324.txt")


def _split_targets(field: str) -> list[str]:
    if not field:
        return []
    out = []
    for tok in re.split(r"[|;,/]", str(field)):
        tok = tok.strip()
        if tok and tok.lower() not in ("na", "nan", "none", '""'):
            out.append(tok)
    # de-dup, keep order
    seen, uniq = set(), []
    for t in out:
        if t.upper() not in seen:
            seen.add(t.upper()); uniq.append(t)
    return uniq


@lru_cache(maxsize=1)
def load_measured_compounds() -> tuple[pd.DataFrame, dict]:
    """Compounds with >=1 QC-pass trt_cp signature, with their cell contexts.

    Returns (df[pert_id, n_sigs, n_cells, cells, n_doses], counts).
    Reads only the columns needed from the 444MB siginfo_beta.
    """
    p = config.RAW_LINCS / "siginfo_beta.txt"
    usecols = ["sig_id", "pert_id", "pert_type", "cmap_name",
               "cell_iname", "pert_dose", "pert_idose", "qc_pass"]
    si = pd.read_csv(p, sep="\t", usecols=lambda c: c in usecols, dtype=str).fillna("")
    counts = {"siginfo_rows": int(len(si))}
    si = si[si["pert_type"] == "trt_cp"]
    counts["trt_cp_sigs"] = int(len(si))
    if "qc_pass" in si.columns:
        si = si[si["qc_pass"].isin(["1", "1.0", "TRUE", "True", "true"])]
    counts["qc_pass_sigs"] = int(len(si))
    g = si.groupby("pert_id")
    agg = pd.DataFrame({
        "n_sigs": g.size(),
        "n_cells": g["cell_iname"].nunique(),
        "cells": g["cell_iname"].apply(lambda s: "|".join(sorted(set(s)))),
        "n_doses": g["pert_dose"].nunique(),
    }).reset_index()
    counts["compounds_with_measured_sig"] = int(len(agg))
    return agg, counts


def build_single_target_compounds() -> tuple[pd.DataFrame, dict]:
    """Core E0 join (LINCS). Returns (df, counts).

    df columns: compound_id, name, smiles, inchi_key, moa, gene_symbol,
                target_source, n_candidate_targets, n_sigs, n_cells, cells, n_doses
    (one row per single-primary-target compound that has measured signatures).
    """
    cinfo = load_compoundinfo()
    drugs = load_drh_drugs()
    measured, mcounts = load_measured_compounds()
    counts = dict(mcounts)

    # restrict to compounds with measured LINCS signatures
    cinfo = cinfo.merge(measured, on="pert_id", how="inner")
    counts["compounds_in_compoundinfo_with_sig"] = int(len(cinfo))

    # DRH target lookup by normalised name
    drugs = drugs.copy()
    drugs["nname"] = drugs["pert_iname"].map(norm_name)
    drh_by_name = {n: t for n, t in zip(drugs["nname"], drugs["target"]) if n}

    rows = []
    drop = {"no_target_annotation": 0, "multi_target": 0}
    for _, r in cinfo.iterrows():
        nname = norm_name(r["cmap_name"])
        drh_t = drh_by_name.get(nname, "")
        drh_list = _split_targets(drh_t)
        lincs_list = _split_targets(r.get("target", ""))
        if drh_list:
            cand, src = drh_list, "drh"
        elif lincs_list:
            cand, src = lincs_list, "lincs"
        else:
            drop["no_target_annotation"] += 1
            continue
        if len(cand) != 1:
            drop["multi_target"] += 1
            continue
        rows.append({
            "compound_id": r["pert_id"],
            "name": r["cmap_name"],
            "smiles": r.get("canonical_smiles", ""),
            "inchi_key": r.get("inchi_key", ""),
            "moa": r.get("moa", ""),
            "gene_symbol": cand[0],
            "target_source": src,
            "n_candidate_targets": 1,
            "n_sigs": int(r["n_sigs"]),
            "n_cells": int(r["n_cells"]),
            "cells": r["cells"],
            "n_doses": int(r["n_doses"]),
        })
    out = pd.DataFrame(rows)
    counts["dropped_no_target_annotation"] = drop["no_target_annotation"]
    counts["dropped_multi_target"] = drop["multi_target"]
    counts["single_target_compounds"] = int(len(out))
    counts["single_target_distinct_genes"] = int(out["gene_symbol"].nunique()) if len(out) else 0
    return out, counts
