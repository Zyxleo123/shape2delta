"""Loaders for the raw metadata sources used by E0 (coverage audit).

Every loader is defensive about column names and returns tidy pandas frames.
Nothing here downloads; files must already be staged by scripts/fetch/fetch_e0.sh.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from .. import config as C


# --------------------------------------------------------------------------- #
# LINCS L1000
# --------------------------------------------------------------------------- #
def load_compoundinfo() -> pd.DataFrame:
    """pert_id, cmap_name, target, moa, canonical_smiles, inchi_key.

    `target` is one gene symbol per row; a compound may span several rows.
    Empty targets are the literal string '' or '""'.
    """
    f = C.RAW_LINCS / "compoundinfo_beta.txt"
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in ("target", "cmap_name", "pert_id", "moa"):
        if c in df:
            df[c] = df[c].str.strip().str.strip('"').replace({"": pd.NA})
    return df


def load_siginfo(usecols=None) -> pd.DataFrame:
    """Signature-level metadata (which compounds were actually measured, where).

    Big file (~465 MB). Read only the columns we need.
    """
    f = C.RAW_LINCS / "siginfo_beta.txt"
    if usecols is None:
        usecols = ["pert_id", "cmap_name", "pert_type", "cell_iname",
                   "pert_idose", "pert_itime", "is_exemplar_sig", "qc_pass",
                   "is_hiq"]
    # only keep columns that actually exist (schema drift safety)
    head = pd.read_csv(f, sep="\t", nrows=0)
    cols = [c for c in usecols if c in head.columns]
    df = pd.read_csv(f, sep="\t", usecols=cols, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Drug Repurposing Hub
# --------------------------------------------------------------------------- #
def load_drh_drugs() -> pd.DataFrame:
    """pert_iname, clinical_phase, moa, target(|-sep), disease_area, indication."""
    f = C.RAW_DRH / "repurposing_drugs_20200324.txt"
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, comment="!")
    df.columns = [c.strip() for c in df.columns]
    df["pert_iname_lc"] = df["pert_iname"].str.strip().str.lower()
    df["drh_targets"] = df["target"].str.strip().replace({"": pd.NA})
    return df


# --------------------------------------------------------------------------- #
# UniProt human reviewed
# --------------------------------------------------------------------------- #
def load_uniprot_human() -> pd.DataFrame:
    """Entry, gene_primary, gene_names, ec, protein_name, catalytic_activity."""
    f = C.RAW_UNIPROT / "human_reviewed.tsv"
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    ren = {
        "Entry": "uniprot",
        "Gene Names (primary)": "gene_primary",
        "Gene Names": "gene_names",
        "EC number": "ec",
        "Protein names": "protein_name",
        "Catalytic activity": "catalytic_activity",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["is_enzyme"] = df.get("ec", "").astype(str).str.strip().ne("")
    return df


def build_gene_to_uniprot(uni: pd.DataFrame) -> dict[str, str]:
    """Map a gene symbol (upper-cased) -> one UniProt accession.

    Primary gene names win; synonyms fill gaps but never overwrite a primary.
    Deterministic on ties (lowest accession).
    """
    primary: dict[str, str] = {}
    for _, r in uni.sort_values("uniprot").iterrows():
        gp = str(r.get("gene_primary", "")).strip().upper()
        if gp and gp not in primary:
            primary[gp] = r["uniprot"]
    syn: dict[str, str] = {}
    for _, r in uni.sort_values("uniprot").iterrows():
        names = str(r.get("gene_names", "")).strip().upper()
        if not names:
            continue
        for g in names.split():
            if g and g not in primary and g not in syn:
                syn[g] = r["uniprot"]
    merged = dict(syn)
    merged.update(primary)  # primary overrides synonym
    return merged


# --------------------------------------------------------------------------- #
# SIFTS  (PDB chain -> UniProt)  -> empirical-ensemble chain counts
# --------------------------------------------------------------------------- #
def load_sifts() -> pd.DataFrame:
    """Columns pdb, chain, uniprot (one row per pdb/chain/segment)."""
    f = C.RAW_SIFTS / "pdb_chain_uniprot.tsv.gz"
    with gzip.open(f, "rt") as fh:
        # first line is a '# ...' comment, second line is the header
        first = fh.readline()
        if not first.startswith("#"):
            fh.seek(0)
        df = pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {"sp_primary": "uniprot"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df[["pdb", "chain", "uniprot"]]


def sifts_chain_counts(sifts: pd.DataFrame) -> pd.DataFrame:
    """Per UniProt: distinct (pdb,chain) and distinct pdb counts."""
    d = sifts.drop_duplicates(["uniprot", "pdb", "chain"])
    g = d.groupby("uniprot")
    out = pd.DataFrame({
        "n_pdb_chains": g.size(),
        "n_pdb_entries": g["pdb"].nunique(),
    }).reset_index()
    return out


# --------------------------------------------------------------------------- #
# Precomputed MD catalogs (preferred T2 source)
# --------------------------------------------------------------------------- #
def load_atlas_uniprots() -> set[str]:
    """UniProt accessions with an ATLAS MD entry (col 'UniProt')."""
    f = C.RAW_ATLAS / "ATLAS_info_2024_11_18.tsv"
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    col = next((c for c in df.columns if c.strip().lower() == "uniprot"), None)
    if col is None:
        return set()
    return {u.strip() for u in df[col] if u and u.strip() not in ("-", "")}


def load_mdcath_pdbchains() -> set[tuple[str, str]]:
    """(pdb_lower, chain) pairs covered by mdCATH domains (e.g. '12asA00')."""
    f = C.RAW_MDCATH / "mdCATH_domains.txt"
    pairs: set[tuple[str, str]] = set()
    for line in Path(f).read_text().splitlines():
        s = line.strip()
        if len(s) >= 5:
            pairs.add((s[:4].lower(), s[4]))
    return pairs
