"""Per-UniProt availability of each feature tier's structural source.

T1 (static)    : AlphaFold DB model. AFDB covers the whole reviewed human
                 proteome, so coverage == reviewed-human membership (validated
                 on a sample via the API at fetch time; kept offline here so the
                 join runs on a compute node with no internet).
T2 (ensemble)  : precomputed MD preferred, else an empirical ensemble from
                 >= MIN_PDB_CHAINS aligned PDB chains of the same UniProt.
   - MD/ATLAS   : ATLAS_info.*.tsv has a UniProt column (+ avg_RMSF, avg_gyration
                  divergence -> usable directly as T2 features later).
   - MD/mdCATH  : domain ids 'ppppCdd' -> (pdb, chain) -> UniProt via SIFTS.
   - empirical  : count distinct (PDB, chain) per UniProt in SIFTS.
"""
from __future__ import annotations

import glob
from functools import lru_cache

import pandas as pd

from shape2delta import config
from shape2delta.data import uniprot


# ------------------------------------------------------------------ SIFTS -----
@lru_cache(maxsize=1)
def load_sifts() -> pd.DataFrame:
    p = config.RAW_SIFTS / "pdb_chain_uniprot.tsv.gz"
    df = pd.read_csv(p, sep="\t", skiprows=1,
                     usecols=["PDB", "CHAIN", "SP_PRIMARY"], dtype=str).fillna("")
    df["PDB"] = df["PDB"].str.lower()
    return df


@lru_cache(maxsize=1)
def sifts_chain_counts() -> pd.Series:
    """uniprot -> number of distinct (PDB, chain) pairs in the PDB."""
    df = load_sifts()
    key = df["PDB"] + "_" + df["CHAIN"]
    return df.assign(_k=key).groupby("SP_PRIMARY")["_k"].nunique()


@lru_cache(maxsize=1)
def sifts_pdbchain_to_uni() -> dict:
    df = load_sifts()
    d: dict[tuple[str, str], str] = {}
    for pdb, ch, up in zip(df["PDB"], df["CHAIN"], df["SP_PRIMARY"]):
        d.setdefault((pdb, ch), up)
    return d


# ------------------------------------------------------------------ ATLAS -----
@lru_cache(maxsize=1)
def atlas_uniprots() -> frozenset[str]:
    files = sorted(glob.glob(str(config.RAW_ATLAS / "ATLAS_info*.tsv")))
    if not files:
        return frozenset()
    ups: set[str] = set()
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t", dtype=str).fillna("")
        except Exception:
            continue
        col = next((c for c in df.columns if c.strip().lower() == "uniprot"), None)
        if col is None:
            continue
        for v in df[col]:
            v = str(v).strip()
            # the ATLAS UniProt column is a clean accession (e.g. P29208); the
            # PDB_chain id lives in a separate PDB column.
            if v and v not in ("-", "nan"):
                ups.add(v)
    return frozenset(ups)


# ------------------------------------------------------------------ mdCATH ----
@lru_cache(maxsize=1)
def mdcath_uniprots() -> frozenset[str]:
    p = config.RAW_MDCATH / "mdCATH_domains.txt"
    if not p.exists():
        return frozenset()
    p2u = sifts_pdbchain_to_uni()
    ups: set[str] = set()
    with open(p) as fh:
        for line in fh:
            dom = line.strip()
            if len(dom) < 5:
                continue
            pdb, chain = dom[:4].lower(), dom[4]
            up = p2u.get((pdb, chain))
            if up:
                ups.add(up)
    return frozenset(ups)


# ------------------------------------------------------------------ T1 --------
@lru_cache(maxsize=1)
def afdb_covered_accessions() -> frozenset[str]:
    # AFDB covers the entire reviewed human proteome (validated via API sample).
    return frozenset(uniprot.load_table()["accession"])


# ------------------------------------------------------- tier availability ----
def tier_availability(accessions, min_chains: int | None = None) -> pd.DataFrame:
    """One row per UniProt accession with per-tier availability flags."""
    if min_chains is None:
        min_chains = config.MIN_PDB_CHAINS
    accs = sorted(set(a for a in accessions if a))
    cc = sifts_chain_counts()
    atlas = atlas_uniprots()
    mdcath = mdcath_uniprots()
    afdb = afdb_covered_accessions()
    enzymes = uniprot.enzyme_accessions()

    rows = []
    for a in accs:
        n_chains = int(cc.get(a, 0))
        t2_emp = n_chains >= min_chains
        t2_atlas = a in atlas
        t2_mdcath = a in mdcath
        t2_md = t2_atlas or t2_mdcath
        rows.append({
            "accession": a,
            "is_enzyme": a in enzymes,
            "t0_identity": True,          # always available
            "t1_afdb": a in afdb,
            "n_pdb_chains": n_chains,
            "t2_empirical": bool(t2_emp),
            "t2_md_atlas": bool(t2_atlas),
            "t2_md_mdcath": bool(t2_mdcath),
            "t2_md_any": bool(t2_md),
            # T2 realisable if MD exists OR an empirical ensemble can be built
            "t2_any": bool(t2_md or t2_emp),
        })
    return pd.DataFrame(rows)
