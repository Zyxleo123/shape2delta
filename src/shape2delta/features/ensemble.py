"""T2 ensemble-feature driver (sharded over targets for a SLURM array).

Each shard profiles its slice of targets from the LOCAL mmCIFs (download done on
the login node) and writes scalar features to a shard CSV. gather() concatenates
shards and attaches ATLAS precomputed MD features as a provenance/robustness
column set. Run:

  shard: python -m shape2delta.features.ensemble <shard_idx> <nshard>
  gather: python -m shape2delta.features.ensemble gather
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import pdb_ensembles as PE

PROFILE_DIR = PE.PROFILE_DIR
SHARD_DIR = PROFILE_DIR / "shards"


def _seqlen(acc: str, manifest_rows) -> int:
    fa = config.RAW_UNIPROT / "fasta" / f"{acc}.fasta"
    if fa.exists():
        seq = "".join(l.strip() for l in fa.read_text().splitlines() if l and not l.startswith(">"))
        if seq:
            return len(seq)
    # fallback: max UniProt residue seen in the manifest segments
    m = 0
    for r in manifest_rows:
        for _, _, _, se in json.loads(r["segs"]):
            m = max(m, int(se))
    return m or 1


def run_shard(shard: int, nshard: int) -> None:
    man = pd.read_csv(PROFILE_DIR / "chain_manifest.csv")
    accs = sorted(man["accession"].unique())
    mine = accs[shard::nshard]
    rows = []
    for acc in mine:
        sub = man[man["accession"] == acc].to_dict("records")
        seqlen = _seqlen(acc, sub)
        try:
            rows.append(PE.profile_target(acc, sub, seqlen))
        except Exception as e:
            rows.append({"accession": acc, "ok": False, "error": str(e)[:120]})
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SHARD_DIR / f"shard_{shard:03d}.csv", index=False)
    print(f"[ensemble] shard {shard}/{nshard}: {len(rows)} targets -> shard_{shard:03d}.csv")


def _atlas_md_features() -> pd.DataFrame:
    """Per-UniProt ATLAS precomputed MD features (avg_RMSF, avg_gyration,
    divergences, SS%), aggregated over systems — a robustness stratum."""
    import glob
    files = sorted(glob.glob(str(config.RAW_ATLAS / "ATLAS_info*.tsv")))
    if not files:
        return pd.DataFrame(columns=["accession"])
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["accession"])
    a = pd.concat(dfs, ignore_index=True)
    col = next((c for c in a.columns if c.strip().lower() == "uniprot"), None)
    if col is None:
        return pd.DataFrame(columns=["accession"])
    a = a.rename(columns={col: "accession"})
    num = {}
    for src, dst in [("avg_RMSF", "atlas_rmsf"), ("avg_gyration", "atlas_rg"),
                     ("div_SE", "atlas_div_se"), ("div_MM", "atlas_div_mm")]:
        if src in a.columns:
            num[dst] = pd.to_numeric(a[src], errors="coerce")
    g = pd.DataFrame(num)
    g["accession"] = a["accession"]
    out = g.groupby("accession").mean(numeric_only=True).reset_index()
    out["has_atlas_md"] = True
    return out


def gather() -> dict:
    shards = sorted(SHARD_DIR.glob("shard_*.csv"))
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True) if shards else pd.DataFrame()
    if "ok" in df.columns:
        df = df[df["ok"].fillna(False).astype(bool)]
    atlas = _atlas_md_features()
    df = df.merge(atlas, on="accession", how="left")
    df["has_atlas_md"] = df["has_atlas_md"].fillna(False)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FEATURES / "t2_ensemble.csv", index=False)
    summ = {"n_targets": int(len(df)),
            "n_with_atlas_md": int(df["has_atlas_md"].sum()),
            "median_conformers": int(df["n_conformers"].median()) if len(df) else 0,
            "median_frac_covered": float(df["frac_seq_covered"].median()) if len(df) else 0.0}
    (config.FEATURES / "t2_ensemble_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "gather":
        print(json.dumps(gather(), indent=2))
    else:
        run_shard(int(sys.argv[1]), int(sys.argv[2]))
