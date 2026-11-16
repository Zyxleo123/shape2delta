"""E8 — real MD ensembles (ATLAS / mdCATH) as the T2 source, on LINCS.

Same pocket, same static T1, same T2 summariser as E7 — only the conformers
change (PDB chains -> MD frames). Comparison set = targets whose MD entry
resolves >= 60% of the ligand-defined pocket.

Arms (fixed length [esm | static | pdb_motion | md_motion], absent blocks = 0):
  T0, T1, T2_PDB, T2_MD, T2_MD_perm (md block shuffled across targets), T2_both
Tests: (a) ridge on signature-level Δ (E3 protocol), (b) held-out-target kNN on
target-mean Δ (centred), (c) partial Mantel MD-motion vs Δ | ESM + static (and
the PDB-motion analogue on the same targets, for a like-for-like contrast).

  python -m shape2delta.experiments.e8_md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.eval import metrics as M
from shape2delta.eval import splits as SP
from shape2delta.experiments import cv
from shape2delta.experiments import e7_pocket as E7
from shape2delta.features import esm_embed
from shape2delta.features import md_ensemble as MD
from shape2delta.features import pocket_ensemble as PK
from shape2delta.model import features as F

EFFECT_FLOOR = E7.EFFECT_FLOOR


def _blocks(d0=32):
    from sklearn.decomposition import PCA
    md = pd.read_csv(MD.OUT / "md_features.csv").drop_duplicates("accession").set_index("accession")
    pf = pd.read_csv(PK.OUT / "pocket_features_all.csv").drop_duplicates("accession").set_index("accession")
    esm, esm_accs = esm_embed.load()
    ei = {a: i for i, a in enumerate(esm_accs)}
    sig = set(pd.read_csv(config.FEATURES / "lincs_delta" / "sig_meta.csv", usecols=["accession"])["accession"])
    accs = sorted(a for a in md.index if a in ei and a in pf.index and a in sig)
    E = esm[[ei[a] for a in accs]].astype(float)
    d0 = min(d0, len(accs) - 2)                      # small n: fewer ESM PCs
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
    esm_pca = PCA(n_components=d0, random_state=0).fit_transform(Ez)
    s, s_cols = E7._standardize(pf.loc[accs].reset_index(), PK.T1_COLS)
    m_md, md_cols = E7._standardize(md.loc[accs].reset_index(), PK.T2_COLS)
    has_pdb = pf.loc[accs, "n_conformers_used"].fillna(0).to_numpy() >= 3
    m_pdb, pdb_cols = E7._standardize(pf.loc[accs].reset_index(), PK.T2_COLS)
    return dict(accs=accs, E=E, esm_pca=esm_pca, s=s, m_md=m_md, m_pdb=m_pdb,
                has_pdb=has_pdb, md=md.loc[accs], s_cols=s_cols, md_cols=md_cols,
                pdb_cols=pdb_cols)


def _tiers(b, perm):
    z = np.zeros_like
    e, s, p, m = b["esm_pca"], b["s"], b["m_pdb"], b["m_md"]
    return {"T0": np.hstack([e, z(s), z(p), z(m)]),
            "T1": np.hstack([e, s, z(p), z(m)]),
            "T2_PDB": np.hstack([e, s, p, z(m)]),
            "T2_MD": np.hstack([e, s, z(p), m]),
            "T2_MD_perm": np.hstack([e, s, z(p), m[perm]]),
            "T2_both": np.hstack([e, s, p, m])}


def ridge(b, perm, alpha=100.0):
    delta, meta, _ = S.load_delta(exemplar_only=True)
    idx = {a: i for i, a in enumerate(b["accs"])}
    keep = meta["accession"].isin(set(idx)).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    Y = delta[keep].astype(np.float32)
    rows = meta["accession"].map(idx).to_numpy()
    base, _ = F.baseline_features(meta)
    tiers = _tiers(b, perm)
    res = {arm: cv.run_all_splits(np.concatenate([base, T[rows].astype(np.float32)], 1), Y, meta,
                                  n_splits=config.NFOLDS, seed=config.SEED, alpha=alpha)
           for arm, T in tiers.items()}

    def g(a, c, k):
        return M.gap_summary(res[a][k]["delta_pearson_folds"], res[c][k]["delta_pearson_folds"],
                             config.NBOOT, config.SEED)
    gaps = {k: {"T2MD_minus_T1": g("T2_MD", "T1", k), "T2PDB_minus_T1": g("T2_PDB", "T1", k),
                "T2MD_minus_T2PDB": g("T2_MD", "T2_PDB", k), "T2MD_minus_perm": g("T2_MD", "T2_MD_perm", k),
                "T2both_minus_T1": g("T2_both", "T1", k)} for k in SP.SPLIT_KINDS}
    meaningful = [k for k in SP.SPLIT_KINDS if gaps[k]["T2MD_minus_T1"]["excludes_zero"]
                  and gaps[k]["T2MD_minus_T1"]["mean"] >= EFFECT_FLOOR]
    acc = {arm: {k: {"delta_pearson": res[arm][k]["delta_pearson_mean"],
                     "delta_pearson_folds": res[arm][k]["delta_pearson_folds"]}
                 for k in SP.SPLIT_KINDS} for arm in tiers}
    return {"n_conditions": int(len(meta)), "n_targets": int(meta["accession"].nunique()),
            "accuracy": acc, "gaps": gaps, "meaningful_split_types": meaningful,
            "practical_gain": len(meaningful) >= 2}


def run() -> dict:
    b = _blocks()
    n = len(b["accs"])
    perm = E7._target_perm(n, config.SEED)
    D, n_sig = E7._target_mean_delta(b["accs"])
    have = n_sig > 0
    print(f"[e8] {n} MD targets ({have.sum()} with signatures; {b['has_pdb'].sum()} also "
          f"have a PDB ensemble); md dims {b['m_md'].shape[1]}", flush=True)
    rd = ridge(b, perm)

    sub = {k: (v[have] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in b.items()}
    D_ = D[have]

    def bal(Z):
        return Z / np.sqrt(Z.var(0).sum() + 1e-12)
    b0, bs, bm, bp = bal(sub["esm_pca"]), bal(sub["s"]), bal(sub["m_md"]), bal(sub["m_pdb"])
    perm_ = E7._target_perm(len(D_), config.SEED + 1)
    spaces = {"T0": b0, "T1": np.hstack([b0, bs]), "T2_MD": np.hstack([b0, bs, bm]),
              "T2_MD_permuted": np.hstack([b0, bs, bm[perm_]]), "T2_PDB": np.hstack([b0, bs, bp]),
              "md_only": bm}
    knn, per_t = E7.knn_transfer(D_, spaces, k=min(E7.K_NN, max(3, len(D_) // 6)), seed=config.SEED)
    for sc in ("raw", "cen"):           # extra contrasts E7.knn_transfer does not compute
        for a, c in (("T2_MD", "T1"), ("T2_MD", "T2_MD_permuted"), ("T2_PDB", "T1"), ("T2_MD", "T2_PDB")):
            x = per_t[f"{sc}_{a}"] - per_t[f"{sc}_{c}"]
            rng = np.random.default_rng(config.SEED)
            bt = x[rng.integers(0, len(x), (config.NBOOT, len(x)))].mean(1)
            lo, hi = np.quantile(bt, [0.025, 0.975])
            knn["gaps"][f"{sc}_{a}_minus_{c}"] = {"mean": float(x.mean()), "lo": float(lo),
                                                  "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}
    mt_md, pts = E7.partial_mantel(D_, sub["E"], sub["s"], sub["m_md"], seed=config.SEED)
    mt_pdb, _ = E7.partial_mantel(D_, sub["E"], sub["s"], sub["m_pdb"], n_perm=500, seed=config.SEED)

    # MD vs PDB motion: do the two ensembles even agree on which pockets move?
    from scipy.stats import spearmanr
    agree = {}
    for c in ["e_pk_rmsf_mean", "e_p_rg_cv", "e_fp_drug_cv", "e_fp_vol_cv", "e_rmsd_mean"]:
        pf = pd.read_csv(PK.OUT / "pocket_features_all.csv").set_index("accession")
        x = pd.to_numeric(b["md"][c], errors="coerce").to_numpy()
        y = pd.to_numeric(pf.reindex(b["accs"])[c], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y) & b["has_pdb"]
        if ok.sum() >= 8:
            r, p = spearmanr(x[ok], y[ok])
            agree[c] = {"rho": float(r), "p": float(p), "n": int(ok.sum())}

    kg = knn["gaps"]["cen_T2_MD_minus_T1"]
    knn_gain = kg["excludes_zero"] and kg["mean"] >= EFFECT_FLOOR
    mantel_sig = mt_md["mantel_p"] < 0.05 and mt_md["r_motion_vs_delta_partial_given_esm_static"] > 0
    out = {
        "experiment": "E8_md_ensemble", "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets_md": n, "n_with_pdb_ensemble": int(b["has_pdb"].sum()),
        "md_source_counts": b["md"]["md_source"].value_counts().to_dict(),
        "median_md_pocket_cover": float(b["md"]["md_pocket_cover"].median()),
        "md_cols": b["md_cols"], "ridge": rd, "knn": knn,
        "mantel_md": mt_md, "mantel_pdb_same_targets": mt_pdb, "md_vs_pdb_motion_agreement": agree,
        "verdict": {"ridge_practical_gain": rd["practical_gain"], "knn_practical_gain": bool(knn_gain),
                    "mantel_significant": bool(mantel_sig),
                    "any_signal": bool(rd["practical_gain"] or knn_gain or mantel_sig),
                    "effect_floor": EFFECT_FLOOR},
    }
    R = config.RESULTS
    (R / "E8_md.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(R / "E8_mantel_scatter.npz", **{k: v.astype(np.float32) for k, v in pts.items()})
    pd.DataFrame({"accession": [a for a, h in zip(b["accs"], have) if h], **per_t}).to_csv(
        R / "E8_knn_per_target.csv", index=False)
    _update_summary(out)
    return out


def _update_summary(r):
    f = E7._fmt
    rg, kn, v = r["ridge"]["gaps"], r["knn"]["gaps"], r["verdict"]
    ridge_s = "; ".join(f"{k} {f(rg[k]['T2MD_minus_T1'])}" for k in ("target", "scaffold", "cell", "random"))
    verdict = (
        f"MD ensembles (ATLAS/mdCATH frames, same pocket/T1/summariser as E7) for "
        f"{r['n_targets_md']} targets ({r['md_source_counts']}; median pocket cover "
        f"{r['median_md_pocket_cover']:.2f}). (a) ridge T2_MD−T1 Δρ: {ridge_s}. "
        f"(b) held-out-target kNN centred T2_MD−T1 {f(kn['cen_T2_MD_minus_T1'])}, "
        f"T2_MD−T2_PDB {f(kn['cen_T2_MD_minus_T2_PDB'])}, T2_MD−perm {f(kn['cen_T2_MD_minus_T2_MD_permuted'])}. "
        f"(c) partial Mantel MD-motion vs Δ | ESM+static r="
        f"{r['mantel_md']['r_motion_vs_delta_partial_given_esm_static']:+.3f} "
        f"(p={r['mantel_md']['mantel_p']:.3g}); PDB-motion on the same targets r="
        f"{r['mantel_pdb_same_targets']['r_motion_vs_delta_partial_given_esm_static']:+.3f}. "
        f"→ **{'signal' if v['any_signal'] else 'null'}** (floor {v['effect_floor']}; * = CI excl 0).")
    E7._write_block("E8", "## E8 — Real MD ensembles (ATLAS/mdCATH)  ", r["atlas"], r["generated"], verdict)


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "knn_gaps": r["knn"]["gaps"],
                      "mantel_md": r["mantel_md"], "agreement": r["md_vs_pdb_motion_agreement"]},
                     indent=2, default=float))
