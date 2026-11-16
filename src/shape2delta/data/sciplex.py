"""sci-Plex3 (Srivatsan et al. 2020; scPerturb h5ad) — the single-cell atlas.

188 compounds × {A549, K562, MCF7} × {10, 100, 1000, 10000 nM}, 24 h (A549 also
has a 72 h arm, unused). ~800k cells; the matrix mixes human ENSG and mouse
ENSMUSG genes (barnyard) — only human genes are kept and normalised.

Stages
  targets (login node, metadata only):
      python -m shape2delta.data.sciplex targets
    compound -> single target (DRH, else LINCS annotation; salts stripped) ->
    UniProt; SMILES from DRH samples / LINCS. Writes compound_targets.csv and
    target_accessions.txt (fed to fetch_structures.sh / pocket profiling).
  build (SLURM CPU, high-mem):
      python -m shape2delta.data.sciplex build
    pass 1: 120k-cell subsample -> HVGs (+ LINCS landmarks) and a 50-PC PCA.
    pass 2: every 24 h cell, chunked -> per-condition pseudobulk mean of log CP10k
            and per-cell PCA scores (saved, so all single-cell readouts are
            recomputable without touching the 2.5 GB h5ad again).
    readouts: per condition (perturbation, cell line, dose) with >= MIN_CELLS:
            delta (pseudobulk − vehicle of that cell line), E-distance to vehicle,
            fraction of responder cells, dispersion ratio, bimodality, cell-count
            viability proxy.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from shape2delta import config

H5AD = config.RAW_SCIPLEX / "SrivatsanTrapnell2020_sciplex3.h5ad"
OUT = config.FEATURES / "sciplex3"
MIN_CELLS = 30
TIME = 24.0
_SALT = re.compile(
    r"\b(\d?hcl|\d?hbr|dihydrochloride|hydrochloride|tosylate|ditosylate|citrate|"
    r"mesylate|dimesylate|phosphate|diphosphate|sodium|trihydrate|dihydrate|hydrate|"
    r"free base|maleate|fumarate|l-\(\+\)-tartaric acid|tartaric acid|tartrate|sulfate|"
    r"acetate|lactate|malate|succinate|besylate|bromide|chloride|potassium|disodium)\b",
    re.I)


def _candidates(p: str) -> list[str]:
    base = [p, re.sub(r"\s*\(.*?\)\s*", " ", p)] + re.findall(r"\(([^)]+)\)", p)
    out = []
    for n in base:
        for x in [n] + re.split(r"[,/]", n):
            for y in (x, _SALT.sub(" ", x)):
                y = y.strip()
                if y and y not in out:
                    out.append(y)
    return out


def compound_targets() -> pd.DataFrame:
    import anndata as ad
    from shape2delta.data import targets as T
    from shape2delta.data import uniprot as U
    obs = ad.read_h5ad(H5AD, backed="r").obs
    perts = sorted({str(x) for x in obs["perturbation"].dropna().unique()} - {"control"})
    drh = T.load_drh_drugs()
    drh_t = {T.norm_name(n): t for n, t in zip(drh["pert_iname"], drh["target"]) if t}
    smp = T.load_drh_samples()
    drh_s = {T.norm_name(n): s for n, s in zip(smp["pert_iname"], smp.get("smiles", "")) if s}
    ci = T.load_compoundinfo()
    ci_t = {T.norm_name(n): t for n, t in zip(ci["cmap_name"], ci["target"]) if t}
    ci_s = {T.norm_name(n): s for n, s in zip(ci["cmap_name"], ci["canonical_smiles"]) if s}
    rows = []
    for p in perts:
        keys = [T.norm_name(n) for n in _candidates(p)]
        keys = [k for k in keys if k]
        t = next(((drh_t[k], "drh", k) for k in keys if k in drh_t), None) \
            or next(((ci_t[k], "lincs", k) for k in keys if k in ci_t), None)
        smi = next((drh_s[k] for k in keys if k in drh_s), "") \
            or next((ci_s[k] for k in keys if k in ci_s), "")
        tl = T._split_targets(t[0]) if t else []
        acc, st = U.map_gene(tl[0]) if len(tl) == 1 else (None, "not_single")
        rows.append({"perturbation": p, "matched_name": t[2] if t else "",
                     "target_source": t[1] if t else "", "targets": "|".join(tl),
                     "n_targets": len(tl), "gene_symbol": tl[0] if len(tl) == 1 else "",
                     "accession": acc or "", "map_status": st, "smiles": smi})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "compound_targets.csv", index=False)
    accs = sorted(set(df.loc[df["accession"] != "", "accession"]))
    (OUT / "target_accessions.txt").write_text("\n".join(accs) + "\n")
    s = {"n_compounds": len(df), "n_mapped_any": int((df["n_targets"] > 0).sum()),
         "n_single_target": int((df["n_targets"] == 1).sum()),
         "n_single_with_accession": int((df["accession"] != "").sum()),
         "n_single_with_smiles": int(((df["accession"] != "") & (df["smiles"] != "")).sum()),
         "n_target_accessions": len(accs)}
    (OUT / "targets_summary.json").write_text(json.dumps(s, indent=2))
    return df


# --------------------------------------------------------------- build -----
def _lognorm(Xh):
    import scipy.sparse as sp
    tot = np.asarray(Xh.sum(1)).ravel()
    tot[tot == 0] = 1.0
    Xn = sp.diags(1e4 / tot) @ Xh
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def _hvg(mean, var, n_top, n_bins=20):
    """Seurat-v1 style: dispersion = var/mean, z-scored within mean bins."""
    ok = mean > 1e-3
    disp = np.full_like(mean, -np.inf)
    d = np.log(var[ok] / mean[ok] + 1e-12)
    bins = pd.qcut(mean[ok], n_bins, labels=False, duplicates="drop")
    z = np.empty_like(d)
    for b in np.unique(bins):
        m = bins == b
        z[m] = (d[m] - d[m].mean()) / (d[m].std() + 1e-12)
    disp[ok] = z
    return np.argsort(-disp)[:n_top]


def build(n_hvg: int = 2000, n_pcs: int = 50, chunk: int = 40000, seed: int = 0) -> dict:
    import anndata as ad
    from sklearn.decomposition import PCA
    a = ad.read_h5ad(H5AD, backed="r")
    obs = a.obs.reset_index(drop=True)
    pert = obs["perturbation"].astype(object)
    keep = (obs["time"].astype(float) == TIME) & pert.notna()
    ens = a.var["ensembl_id"].astype(str).str.split(".").str[0].to_numpy()
    human = np.where(pd.Series(ens).str.startswith("ENSG") & ~pd.Series(ens).duplicated())[0]
    ens_h = ens[human]
    print(f"[sciplex] cells kept {int(keep.sum())}/{len(obs)}; human genes {len(human)}", flush=True)

    # gene symbols + LINCS landmarks (for a cross-atlas comparable subset)
    gi = pd.read_csv(config.RAW_LINCS / "geneinfo_beta.txt", sep="\t", dtype=str)
    sym = dict(zip(gi.get("ensembl_id", pd.Series(dtype=str)), gi["gene_symbol"]))
    lm = set(gi.loc[gi["feature_space"] == "landmark", "ensembl_id"].dropna())

    # ---- pass 1: subsample -> HVG + PCA (streamed in contiguous row chunks)
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    kidx = np.where(keep.to_numpy())[0]
    insub = np.zeros(len(obs), bool)
    insub[rng.choice(kidx, size=min(120_000, len(kidx)), replace=False)] = True
    blocks = []
    for s in range(0, len(obs), chunk):
        e = min(s + chunk, len(obs))
        m = insub[s:e]
        if m.any():
            blocks.append(_lognorm(a.X[s:e][m][:, human]))
    Xs = sp.vstack(blocks).tocsr()
    mean = np.asarray(Xs.mean(0)).ravel()
    var = np.asarray(Xs.multiply(Xs).mean(0)).ravel() - mean ** 2
    hv = set(_hvg(mean, var, n_hvg).tolist())
    lmi = {i for i, e in enumerate(ens_h) if e in lm and mean[i] > 1e-3}
    G = np.array(sorted(hv | lmi))
    print(f"[sciplex] genes: {len(hv)} HVG ∪ {len(lmi)} landmarks = {len(G)}", flush=True)
    XsG = Xs[:, G].toarray().astype(np.float32)
    mu_g = XsG.mean(0)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(XsG - mu_g)
    del Xs, XsG, blocks

    # ---- pass 2: every kept cell
    cond_key = (pert.astype(str) + "|" + obs["cell_line"].astype(str) + "|"
                + obs["dose_value"].astype(float).map(lambda d: f"{d:g}"))
    cond_key[~keep] = "||-1"                          # dropped cells: one dummy key
    codes, uniq = pd.factorize(cond_key)
    nc = len(uniq)
    S = np.zeros((nc, len(G)), np.float64)
    N = np.zeros(nc, np.int64)
    pcs = np.zeros((len(obs), n_pcs), np.float32)
    for s in range(0, len(obs), chunk):
        e = min(s + chunk, len(obs))
        m = keep.to_numpy()[s:e]
        if not m.any():
            continue
        Xb = _lognorm(a.X[s:e][m][:, human])[:, G].toarray().astype(np.float32)
        c = codes[s:e][m]
        oh = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(nc, len(c)))
        S += oh @ Xb.astype(np.float64)
        N += np.bincount(c, minlength=nc)
        pcs[np.arange(s, e)[m]] = pca.transform(Xb - mu_g).astype(np.float32)
        print(f"[sciplex] pass2 {e}/{len(obs)}", flush=True)
    M = S / np.maximum(N, 1)[:, None]

    cm = pd.DataFrame([k.split("|") for k in uniq], columns=["perturbation", "cell_line", "dose_nM"])
    cm["dose_nM"] = cm["dose_nM"].astype(float)
    cm["n_cells"] = N
    ok = (cm["perturbation"] != "") & (cm["n_cells"] >= MIN_CELLS)

    # ---- per-condition readouts vs the vehicle of the same cell line
    ctrl = {cl: np.where((cm["perturbation"] == "control") & (cm["cell_line"] == cl))[0]
            for cl in cm["cell_line"].unique()}
    cell_cond = codes
    rows, deltas, rng = [], [], np.random.default_rng(seed)
    ctrl_cells = {cl: np.where(np.isin(cell_cond, ix))[0] for cl, ix in ctrl.items()}
    for j in np.where(ok)[0]:
        r = cm.iloc[j]
        if r["perturbation"] == "control":
            continue
        cl = r["cell_line"]
        cix = ctrl[cl]
        mu_c = (S[cix].sum(0) / N[cix].sum())
        deltas.append((M[j] - mu_c).astype(np.float32))
        Pc = pcs[ctrl_cells[cl]]
        Pt = pcs[cell_cond == j]
        rows.append({"cond_idx": len(deltas) - 1, **r.to_dict(), **_sc_readouts(Pt, Pc, rng)})
    meta = pd.DataFrame(rows)
    D = np.stack(deltas)
    # viability proxy: cells recovered vs the same compound/cell line at 10 nM
    lo = meta[meta["dose_nM"] == meta["dose_nM"].min()].set_index(["perturbation", "cell_line"])["n_cells"]
    meta["log2_cells_vs_lowdose"] = [
        np.log2(n / lo.get((p, c), np.nan)) for p, c, n in
        zip(meta["perturbation"], meta["cell_line"], meta["n_cells"])]

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "delta.npy", D)
    meta.to_csv(OUT / "cond_meta.csv", index=False)
    genes = [{"ensembl_id": ens_h[g], "symbol": sym.get(ens_h[g], ""),
              "landmark": ens_h[g] in lm} for g in G]
    (OUT / "genes.json").write_text(json.dumps(genes))
    np.savez_compressed(OUT / "cell_pcs.npz", pcs=pcs[keep.to_numpy()],
                        cond=cell_cond[keep.to_numpy()], uniq=np.array(uniq, dtype=str),
                        evr=pca.explained_variance_ratio_)
    s = {"n_cells_24h": int(keep.sum()), "n_genes": int(len(G)),
         "n_conditions": int(len(meta)), "min_cells": MIN_CELLS,
         "n_perturbations": int(meta["perturbation"].nunique()),
         "pca_var_explained": float(pca.explained_variance_ratio_.sum())}
    (OUT / "build_summary.json").write_text(json.dumps(s, indent=2))
    return s


def _sc_readouts(Pt, Pc, rng, nmax=400):
    """Distribution-level readouts of a condition's cells (Pt) vs vehicle (Pc),
    in PCA space."""
    t = Pt[rng.choice(len(Pt), min(nmax, len(Pt)), replace=False)]
    c = Pc[rng.choice(len(Pc), min(nmax, len(Pc)), replace=False)]

    def md(A, B, within=False):
        d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        n = len(A)                                  # within: unbiased (no diagonal)
        return d.sum() / (n * (n - 1)) if within else d.mean()
    edist = float(2 * md(t, c) - md(t, t, True) - md(c, c, True))
    shift = Pt.mean(0) - Pc.mean(0)
    nrm = np.linalg.norm(shift)
    u = shift / (nrm + 1e-12)
    pt, pc = Pt @ u, Pc @ u
    thr = np.quantile(pc, 0.95)
    from scipy.stats import kurtosis, skew
    g, k = skew(pt), kurtosis(pt)                  # excess kurtosis
    n = len(pt)
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))) if n > 3 else np.nan
    return {"edist": edist, "shift_norm": float(nrm),
            "frac_responders": float(np.mean(pt > thr)),
            "resp_sd_ratio": float(pt.std() / (pc.std() + 1e-12)),
            "dispersion_ratio": float(Pt.var(0).sum() / (Pc.var(0).sum() + 1e-12)),
            "bimodality": float(bc)}


def load():
    D = np.load(OUT / "delta.npy")
    meta = pd.read_csv(OUT / "cond_meta.csv")
    return D, meta


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "targets":
        df = compound_targets()
        print((OUT / "targets_summary.json").read_text())
        print(df.loc[df["accession"] != "", ["perturbation", "gene_symbol", "accession"]].to_string())
    elif cmd == "build":
        print(json.dumps(build(), indent=2))
