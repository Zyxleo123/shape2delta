"""UniProt (human, reviewed): gene symbol -> accession, plus enzyme flag.

Source file: $S2D_RAW/uniprot/human_reviewed.tsv  (TSV from the UniProt REST API)
Columns seen: Entry | Gene Names (primary) | Gene Names | EC number | Protein names | ...
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from shape2delta import config


def _find_file() -> Path:
    for name in ("human_reviewed.tsv", "human_reviewed.tsv.gz"):
        p = config.RAW_UNIPROT / name
        if p.exists():
            return p
    raise FileNotFoundError(f"UniProt table not found under {config.RAW_UNIPROT}")


@lru_cache(maxsize=1)
def load_table() -> pd.DataFrame:
    p = _find_file()
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    # normalise the column names we rely on
    ren = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "entry":
            ren[c] = "accession"
        elif cl in ("gene names (primary)", "gene_primary"):
            ren[c] = "gene_primary"
        elif cl in ("gene names", "gene_names"):
            ren[c] = "gene_names"
        elif cl in ("ec number", "ec"):
            ren[c] = "ec"
        elif cl in ("protein names", "protein_name", "protein names "):
            ren[c] = "protein_name"
    df = df.rename(columns=ren)
    for need in ("accession", "gene_primary", "gene_names"):
        if need not in df.columns:
            df[need] = ""
    if "ec" not in df.columns:
        df["ec"] = ""
    df["is_enzyme"] = df["ec"].str.strip().ne("")
    return df


@lru_cache(maxsize=1)
def _maps():
    """Return (primary_map, synonym_map): upper(symbol) -> list[accession]."""
    df = load_table()
    primary: dict[str, list[str]] = {}
    synonym: dict[str, list[str]] = {}
    for acc, prim, names in zip(df["accession"], df["gene_primary"], df["gene_names"]):
        if prim:
            primary.setdefault(prim.strip().upper(), []).append(acc)
        for tok in str(names).replace("|", " ").split():
            synonym.setdefault(tok.strip().upper(), []).append(acc)
    return primary, synonym


@lru_cache(maxsize=1)
def enzyme_accessions() -> frozenset[str]:
    df = load_table()
    return frozenset(df.loc[df["is_enzyme"], "accession"])


def map_gene(symbol: str) -> tuple[str | None, str]:
    """Map a gene symbol to a single reviewed-human accession.

    Returns (accession_or_None, status) where status is one of:
      primary        unique match on primary gene symbol
      primary_multi  symbol is primary for >1 accession (ambiguous) -> first, flagged
      synonym        unique match only via a gene-name synonym
      synonym_multi  synonym maps to >1 accession (ambiguous) -> None
      unmapped       not found
    """
    if not symbol:
        return None, "unmapped"
    key = symbol.strip().upper()
    primary, synonym = _maps()
    if key in primary:
        accs = primary[key]
        return (accs[0], "primary" if len(accs) == 1 else "primary_multi")
    if key in synonym:
        accs = sorted(set(synonym[key]))
        if len(accs) == 1:
            return accs[0], "synonym"
        return None, "synonym_multi"
    return None, "unmapped"
