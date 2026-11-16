"""Empirical conformational ensembles from aligned PDB chains → T2 features.

For each target UniProt we take up to K experimental chains (SIFTS), map each
chain's CA atoms onto UniProt residue numbering via the SIFTS segment offsets,
superpose (Kabsch) onto a reference chain, and summarise the per-residue motion:
RMSF profile, radius-of-gyration variance across conformers, fraction flexible.

Split into:
  build_manifest()  -> which (pdb,chain) per target + the unique PDB list
                       (run on the login node to drive the mmCIF download)
  profile_target()  -> per-target npz from LOCAL mmCIF files (SLURM array)
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from shape2delta import config

CIF_DIR = config.RAW / "pdb_cif"
PROFILE_DIR = config.FEATURES / "ensemble_profiles"


@lru_cache(maxsize=1)
def load_sifts_segments() -> pd.DataFrame:
    p = config.RAW_SIFTS / "pdb_chain_uniprot.tsv.gz"
    df = pd.read_csv(p, sep="\t", skiprows=1, dtype=str,
                     usecols=["PDB", "CHAIN", "SP_PRIMARY",
                              "PDB_BEG", "PDB_END", "SP_BEG", "SP_END"]).fillna("")
    df["PDB"] = df["PDB"].str.lower()
    for c in ("PDB_BEG", "PDB_END", "SP_BEG", "SP_END"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["PDB_BEG", "PDB_END", "SP_BEG", "SP_END"])


def _coverage(segs: pd.DataFrame) -> int:
    return int((segs["SP_END"] - segs["SP_BEG"] + 1).clip(lower=0).sum())


def select_chains(acc: str, k: int = 10) -> list[dict]:
    """Up to k (pdb,chain) for a target, preferring broad UniProt coverage and
    distinct PDB entries (diversity). Returns list of dicts with segment maps."""
    sf = load_sifts_segments()
    sub = sf[sf["SP_PRIMARY"] == acc]
    if sub.empty:
        return []
    chains = []
    for (pdb, chain), segs in sub.groupby(["PDB", "CHAIN"]):
        chains.append({"pdb": pdb, "chain": chain, "cov": _coverage(segs),
                       "segs": segs[["PDB_BEG", "PDB_END", "SP_BEG", "SP_END"]].values.tolist()})
    chains.sort(key=lambda c: -c["cov"])
    # greedily prefer distinct PDB ids for conformational diversity
    picked, seen_pdb = [], set()
    for c in chains:
        if c["pdb"] in seen_pdb:
            continue
        picked.append(c); seen_pdb.add(c["pdb"])
        if len(picked) >= k:
            break
    if len(picked) < k:  # top up allowing repeated PDBs (different chains)
        for c in chains:
            if c not in picked:
                picked.append(c)
            if len(picked) >= k:
                break
    return picked


def build_manifest(accs, k: int = 10) -> dict:
    rows, pdbs = [], set()
    for acc in accs:
        for c in select_chains(acc, k=k):
            rows.append({"accession": acc, "pdb": c["pdb"], "chain": c["chain"],
                         "cov": c["cov"], "segs": json.dumps(c["segs"])})
            pdbs.add(c["pdb"])
    man = pd.DataFrame(rows)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(PROFILE_DIR / "chain_manifest.csv", index=False)
    (PROFILE_DIR / "unique_pdbs.txt").write_text("\n".join(sorted(pdbs)))
    return {"n_targets": int(man["accession"].nunique()) if len(man) else 0,
            "n_chains": int(len(man)), "n_unique_pdbs": len(pdbs)}


# ----------------------------------------------------------------- profiling --
def _ca_by_uniprot(cif_path, chain_id, segs, seqlen):
    """CA coords indexed by UniProt residue (1..seqlen); NaN where absent."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    import warnings, gzip
    coords = np.full((seqlen, 3), np.nan, dtype=np.float64)
    opener = gzip.open if str(cif_path).endswith(".gz") else open
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with opener(cif_path, "rt") as fh:
            st = MMCIFParser(QUIET=True).get_structure("s", fh)
    model = next(iter(st))
    if chain_id not in [c.id for c in model]:
        return coords
    ch = model[chain_id]
    # author residue -> uniprot residue, per segment (linear offset)
    def to_uni(auth):
        for pb, pe, sb, se in segs:
            if pb <= auth <= pe:
                return int(sb + (auth - pb))
        return None
    for res in ch:
        if res.id[0] != " " or "CA" not in res:
            continue
        uni = to_uni(res.id[1])
        if uni is not None and 1 <= uni <= seqlen:
            coords[uni - 1] = res["CA"].coord
    return coords


def _kabsch_rmsf(stack):
    """stack: [n_chain x L x 3] with NaN gaps. Superpose on common residues to the
    best-covered chain; return (rmsf[L], n_cover[L], rg_values[n_chain])."""
    n, L, _ = stack.shape
    cover = ~np.isnan(stack[:, :, 0])
    ref = int(cover.sum(axis=1).argmax())
    R = stack[ref]
    aligned = np.full_like(stack, np.nan)
    aligned[ref] = R
    rg = np.full(n, np.nan)
    for i in range(n):
        if i == ref:
            Xi = stack[i][cover[i]]
            rg[i] = _rg(Xi); continue
        common = cover[i] & cover[ref]
        if common.sum() < 4:
            continue
        P = stack[i][common]; Q = R[common]
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        rot = Vt.T @ D @ U.T
        full = stack[i].copy()
        mask = cover[i]
        full[mask] = (full[mask] - P.mean(0)) @ rot.T + Q.mean(0)
        aligned[i] = full
        rg[i] = _rg(stack[i][mask])
    ncover = (~np.isnan(aligned[:, :, 0])).sum(axis=0)
    rmsf = np.full(L, np.nan)
    for j in range(L):
        pts = aligned[~np.isnan(aligned[:, j, 0]), j, :]
        if len(pts) >= 2:
            rmsf[j] = np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean())
    return rmsf, ncover, rg


def _rg(X):
    if len(X) < 2:
        return np.nan
    c = X.mean(0)
    return float(np.sqrt(((X - c) ** 2).sum(1).mean()))


def profile_target(acc, manifest_rows, seqlen, cif_dir=CIF_DIR) -> dict:
    stacks = []
    for r in manifest_rows:
        cif = cif_dir / f"{r['pdb']}.cif.gz"
        if not cif.exists():
            continue
        segs = json.loads(r["segs"])
        try:
            coords = _ca_by_uniprot(cif, r["chain"], segs, seqlen)
        except Exception:
            continue
        if np.isfinite(coords[:, 0]).sum() >= 10:
            stacks.append(coords)
    if len(stacks) < 2:
        return {"accession": acc, "n_conformers": len(stacks), "ok": False}
    stack = np.stack(stacks, axis=0)
    rmsf, ncover, rg = _kabsch_rmsf(stack)
    valid = np.isfinite(rmsf)
    out = {
        "accession": acc, "ok": True, "n_conformers": int(len(stacks)),
        "seqlen": int(seqlen),
        "rmsf_mean": float(np.nanmean(rmsf)) if valid.any() else float("nan"),
        "rmsf_std": float(np.nanstd(rmsf[valid])) if valid.any() else float("nan"),
        "rmsf_max": float(np.nanmax(rmsf[valid])) if valid.any() else float("nan"),
        "rmsf_p90": float(np.nanpercentile(rmsf[valid], 90)) if valid.any() else float("nan"),
        "frac_flexible": float(np.mean(rmsf[valid] > 1.0)) if valid.any() else float("nan"),
        "n_residues_covered": int(valid.sum()),
        "frac_seq_covered": float(valid.sum() / max(seqlen, 1)),
        "rg_mean": float(np.nanmean(rg)),
        "rg_cv": float(np.nanstd(rg) / (np.nanmean(rg) + 1e-9)),
    }
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROFILE_DIR / f"{acc}.npz", rmsf=rmsf, ncover=ncover, rg=rg)
    return out
