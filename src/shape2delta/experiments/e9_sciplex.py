"""E9 — single-cell readout (sci-Plex3) instead of LINCS bulk.

Same matched pocket tiers as E7 (T1 = AF pocket descriptor; T2 = T1 + its motion
over PDB conformers; plus the MD motion block where an MD entry exists), now
scored against sci-Plex3 rather than LINCS L1000:

  (a) ridge on condition-level pseudobulk Delta (2722 genes = HVGs U L1000
      landmarks), baseline = [ECFP | cell-line one-hot | log dose]. Splits
      random / target / scaffold (5-fold) and cell (3-fold; only 3 lines).
  (b) held-out-target kNN on target-level mean Delta (conditions at >= 1 uM).
  (c) partial Mantel: pocket motion vs Delta similarity | ESM + static pocket.

Plus the part LINCS could not give: single-cell distribution readouts per
condition (E-distance to vehicle, responder fraction, dispersion ratio,
bimodality, viability). (d) asks whether any pocket-motion feature predicts those
better than the static pocket, controlling for dose and cell line — the clean
single-cell test the spec reserved for E4.

  python -m shape2delta.experiments.e9_sciplex
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.eval import metrics as M
from shape2delta.experiments import cv
from shape2delta.experiments import e7_pocket as E7
from shape2delta.features import esm_embed
from shape2delta.features import md_ensemble as MD
from shape2delta.features import pocket_ensemble as PK
from shape2delta.model import features as F

EFFECT_FLOOR = E7.EFFECT_FLOOR
SPLITS = ("random", "target", "scaffold", "cell")
DATA = config.FEATURES / "sciplex3"
MIN_DOSE_FOR_TARGET_MEAN = 1000.0          # nM; target-level Delta averages >= 1 uM


# ----------------------------------------------------------------- data ----
def _load():
    D = np.load(DATA / "delta.npy")
    meta = pd.read_csv(DATA / "cond_meta.csv")
    ct = pd.read_csv(DATA / "compound_targets.csv")
    meta = meta.merge(ct[["perturbation", "accession", "smiles", "gene_symbol"]],
                      on="perturbation", how="left")
    meta["accession"] = meta["accession"].fillna("").astype(str)
    meta["smiles"] = meta["smiles"].fillna("").astype(str)
    # baseline_features / splits expect these column names
    meta["cell_iname"] = meta["cell_line"]
    meta["pert_dose"] = meta["dose_nM"] / 1000.0        # nM -> uM, like LINCS
    meta["pert_time"] = 24.0
    return D, meta


def _pocket_blocks(accs, d0=32):
    """ESM-PCA, static pocket, PDB motion, MD motion — one row per accession in
    `accs`, standardised on this target set. Missing MD rows are zero-filled and
    flagged."""
    from sklearn.decomposition import PCA
    pf = pd.read_csv(PK.OUT / "pocket_features_all.csv").drop_duplicates("accession").set_index("accession")
    md_path = MD.OUT / "md_features.csv"
    md = pd.read_csv(md_path).drop_duplicates("accession").set_index("accession") if md_path.exists() \
        else pd.DataFrame(columns=PK.T2_COLS)
    esm, esm_accs = esm_embed.load()
    ei = {a: i for i, a in enumerate(esm_accs)}
    use = [a for a in accs if a in ei and a in pf.index]
    E = esm[[ei[a] for a in use]].astype(float)
    d0 = min(d0, max(2, len(use) - 2))
    esm_pca = PCA(n_components=d0, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-9))
    s, s_cols = E7._standardize(pf.loc[use].reset_index(), PK.T1_COLS)
    m_pdb, _ = E7._standardize(pf.loc[use].reset_index(), PK.T2_COLS)
    has_pdb = pf.loc[use, "n_conformers_used"].fillna(0).to_numpy() >= 3
    has_md = np.array([a in md.index for a in use])
    md_al = md.reindex(use)
    m_md, _ = E7._standardize(md_al.reset_index(), PK.T2_COLS)
    m_md[~has_md] = 0.0
    return dict(accs=use, E=E, esm_pca=esm_pca, s=s, m_pdb=m_pdb, m_md=m_md,
                has_pdb=has_pdb, has_md=has_md, s_cols=s_cols)


# --------------------------------------------------------------- (a) ridge --
def _ridge(b, meta, D, perm):
    idx = {a: i for i, a in enumerate(b["accs"])}
    keep = meta["accession"].isin(set(idx)).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    Y = D[keep].astype(np.float32)
    rows = meta["accession"].map(idx).to_numpy()
    base, _ = F.baseline_features(meta)
    z = np.zeros_like
    e, s, p, m = b["esm_pca"], b["s"], b["m_pdb"], b["m_md"]
    tiers = {"T0": np.hstack([e, z(s), z(p)]), "T1": np.hstack([e, s, z(p)]),
             "T2_PDB": np.hstack([e, s, p]), "T2_PDB_perm": np.hstack([e, s, p[perm]])}
    res = {}
    for arm, T in tiers.items():
        X = np.concatenate([base, T[rows].astype(np.float32)], 1)
        res[arm] = {}
        for k in SPLITS:
            ns = 3 if k == "cell" else config.NFOLDS
            folds = cv.SP.make_folds(meta, k, n_splits=ns, seed=config.SEED)
            fm = cv.ridge_fold_metrics(X, Y, folds, alpha=100.0)
            res[arm][k] = {"delta_pearson_folds": [r["delta_pearson"] for r in fm],
                           "top100_folds": [r["top100"] for r in fm],
                           "delta_pearson": float(np.nanmean([r["delta_pearson"] for r in fm]) if fm else np.nan)}

    def g(a, c, k):
        return M.gap_summary(res[a][k]["delta_pearson_folds"], res[c][k]["delta_pearson_folds"],
                             config.NBOOT, config.SEED)
    gaps = {k: {"T2_minus_T1": g("T2_PDB", "T1", k), "T1_minus_T0": g("T1", "T0", k),
                "T2_minus_perm": g("T2_PDB", "T2_PDB_perm", k)} for k in SPLITS}
    meaningful = [k for k in SPLITS if gaps[k]["T2_minus_T1"]["excludes_zero"]
                  and gaps[k]["T2_minus_T1"]["mean"] >= EFFECT_FLOOR]
    acc = {a: {k: {"delta_pearson": res[a][k]["delta_pearson"],
                   "delta_pearson_folds": res[a][k]["delta_pearson_folds"]} for k in SPLITS}
           for a in tiers}
    return {"n_conditions": int(len(meta)), "n_targets": int(meta["accession"].nunique()),
            "accuracy": acc, "gaps": gaps, "meaningful_split_types": meaningful,
            "practical_gain": len(meaningful) >= 2}


# ------------------------------------------------------- (d) single-cell ----
def _singlecell_screen(b, meta):
    """Does any pocket-motion feature explain a single-cell readout ACROSS
    TARGETS? Pocket features are constant within a target, so the unit of
    analysis is the target, not the condition — testing at condition level is
    pseudoreplication (n inflated ~10x). So: (1) residualise each readout on
    log-dose + cell line at condition level (stable covariate fit over all
    conditions), (2) average the residual per target -> one value per target,
    (3) Spearman(feature, target-mean residual) across targets, n = #targets.
    BH across readout x feature. A permutation null over target labels gives p
    (the t-approx would still over-trust the rank structure at this small n)."""
    from scipy.stats import rankdata, spearmanr
    pf = pd.read_csv(PK.OUT / "pocket_features_all.csv").drop_duplicates("accession").set_index("accession")
    readouts = ["edist", "frac_responders", "dispersion_ratio", "bimodality",
                "resp_sd_ratio", "log2_cells_vs_lowdose"]
    sub = meta[meta["accession"].isin(set(b["accs"]))].copy().reset_index(drop=True)
    C = np.column_stack([np.ones(len(sub)), np.log10(sub["dose_nM"].clip(lower=1).to_numpy()),
                         pd.get_dummies(sub["cell_line"]).to_numpy()]).astype(float)
    acc_arr = sub["accession"].to_numpy()
    feats = PK.T2_COLS + ["x_cryptic_apo_holo"]
    rng = np.random.default_rng(config.SEED)
    rows = []
    for ro in readouts:
        y = pd.to_numeric(sub[ro], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        res = np.full(len(sub), np.nan)
        res[ok] = y[ok] - C[ok] @ np.linalg.lstsq(C[ok], y[ok], rcond=None)[0]
        # target-mean residual (>= 3 conditions per target to be included)
        tdf = pd.DataFrame({"acc": acc_arr, "res": res}).dropna()
        g = tdf.groupby("acc")["res"].agg(["mean", "size"])
        g = g[g["size"] >= 3]
        for f in feats:
            if f not in pf:
                continue
            fv = pf.reindex(g.index)[f].to_numpy(float)
            yv = g["mean"].to_numpy()
            m = np.isfinite(fv) & np.isfinite(yv)
            nt = int(m.sum())
            if nt < 12 or np.nanstd(fv[m]) < 1e-9:
                continue
            # Spearman = Pearson on ranks; permute target labels (vectorised null)
            fr = rankdata(fv[m]); yr = rankdata(yv[m])
            frc = (fr - fr.mean()) / (fr.std() + 1e-12)
            yrc = (yr - yr.mean()) / (yr.std() + 1e-12)
            r = float(frc @ yrc / nt)
            perm = np.array([rng.permutation(nt) for _ in range(2000)])
            null = (frc @ yrc[perm].T) / nt
            p = float((np.sum(np.abs(null) >= abs(r)) + 1) / 2001)
            rows.append({"readout": ro, "feature": f, "tier": "T2", "n_targets": nt,
                         "partial_rho": float(r), "p": p})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    if len(df):
        mm = len(df)
        q = (df["p"] * mm / (np.arange(mm) + 1)).to_numpy()
        df["q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    return df


# ------------------------------------------------------------------- run ----
def run() -> dict:
    D, meta = _load()
    accs = sorted(a for a in meta["accession"].unique() if a)
    b = _pocket_blocks(accs)
    print(f"[e9] {len(b['accs'])} targets with pocket feats "
          f"({b['has_pdb'].sum()} PDB-ensemble, {b['has_md'].sum()} MD); "
          f"{meta['accession'].isin(set(b['accs'])).sum()} conditions", flush=True)
    perm = E7._target_perm(len(b["accs"]), config.SEED)
    rd = _ridge(b, meta, D, perm)
    print(f"[e9] ridge done: meaningful splits {rd['meaningful_split_types']}", flush=True)

    # target-level Delta (mean over high-dose conditions), for (b)/(c)
    hi = meta[meta["dose_nM"] >= MIN_DOSE_FOR_TARGET_MEAN]
    tmean, order = {}, []
    for a in b["accs"]:
        ix = hi.index[hi["accession"] == a].to_numpy()
        if len(ix):
            tmean[a] = D[ix].mean(0)
            order.append(a)
    keep = np.array([a in tmean for a in b["accs"]])
    Dt = np.stack([tmean[a] for a in b["accs"] if a in tmean])
    sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(b["accs"]) else v)
           for k, v in b.items()}

    def bal(Z):
        return Z / np.sqrt(Z.var(0).sum() + 1e-12)
    b0, bs, bp = bal(sub["esm_pca"]), bal(sub["s"]), bal(sub["m_pdb"])
    perm_ = E7._target_perm(len(Dt), config.SEED + 1)
    spaces = {"T0": b0, "T1": np.hstack([b0, bs]), "T2": np.hstack([b0, bs, bp]),
              "T2_permuted": np.hstack([b0, bs, bp[perm_]]), "motion_only": bp}
    knn, per_t = E7.knn_transfer(Dt, spaces, k=min(E7.K_NN, max(3, len(Dt) // 6)), seed=config.SEED)
    print(f"[e9] kNN done: cen T2-T1 {knn['gaps']['cen_T2_minus_T1']['mean']:+.4f}", flush=True)
    mantel, pts = E7.partial_mantel(Dt, sub["E"], sub["s"], sub["m_pdb"], seed=config.SEED)
    sc = _singlecell_screen(b, meta)
    print(f"[e9] single-cell screen: {len(sc)} tests, "
          f"min q {float(sc['q_bh'].min()) if len(sc) else float('nan'):.3f}", flush=True)

    kg = knn["gaps"]["cen_T2_minus_T1"]
    knn_gain = kg["excludes_zero"] and kg["mean"] >= EFFECT_FLOOR
    mantel_sig = mantel["mantel_p"] < 0.05 and mantel["r_motion_vs_delta_partial_given_esm_static"] > 0
    sc_hit = bool(len(sc) and (sc["q_bh"] < 0.05).any())
    out = {
        "experiment": "E9_sciplex3", "atlas": "sci-Plex3",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets_pocket": len(b["accs"]), "n_targets_target_mean": int(keep.sum()),
        "n_with_pdb": int(b["has_pdb"].sum()), "n_with_md": int(b["has_md"].sum()),
        "ridge": rd, "knn": knn, "mantel": mantel,
        "singlecell_screen_top": sc.head(12).to_dict("records") if len(sc) else [],
        "verdict": {"ridge_practical_gain": rd["practical_gain"], "knn_practical_gain": bool(knn_gain),
                    "mantel_significant": bool(mantel_sig), "singlecell_hit": sc_hit,
                    "any_signal": bool(rd["practical_gain"] or knn_gain or mantel_sig or sc_hit),
                    "effect_floor": EFFECT_FLOOR},
    }
    R = config.RESULTS
    (R / "E9_sciplex.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(R / "E9_mantel_scatter.npz", **{k: v.astype(np.float32) for k, v in pts.items()})
    if len(sc):
        sc.to_csv(R / "E9_singlecell_screen.csv", index=False)
    pd.DataFrame({"accession": [a for a in b["accs"] if a in tmean], **per_t}).to_csv(
        R / "E9_knn_per_target.csv", index=False)
    _update_summary(out)
    return out


def _update_summary(r):
    f = E7._fmt
    rg, kn, mt, v = r["ridge"]["gaps"], r["knn"], r["mantel"], r["verdict"]
    ridge_s = "; ".join(f"{k} {f(rg[k]['T2_minus_T1'])}" for k in SPLITS)
    sc = r["singlecell_screen_top"]
    sc_s = (f"best single-cell assoc {sc[0]['feature']}->{sc[0]['readout']} "
            f"partial-rho {sc[0]['partial_rho']:+.2f} (q={sc[0]['q_bh']:.2f})") if sc else "no single-cell screen"
    verdict = (
        f"Single-cell atlas: {r['n_targets_pocket']} targets w/ pocket feats "
        f"({r['n_with_pdb']} PDB-ensemble, {r['n_with_md']} MD), {r['ridge']['n_conditions']} conditions. "
        f"(a) ridge T2-T1 Delta-rho: {ridge_s}. "
        f"(b) held-out-target kNN centred T2-T1 {f(kn['gaps']['cen_T2_minus_T1'])}, "
        f"T2-perm {f(kn['gaps']['cen_T2_minus_perm'])} ({r['n_targets_target_mean']} targets). "
        f"(c) partial Mantel motion vs Delta | ESM+static r="
        f"{mt['r_motion_vs_delta_partial_given_esm_static']:+.3f} (p={mt['mantel_p']:.3g}). "
        f"(d) {sc_s}. -> **{'signal' if v['any_signal'] else 'null'}** "
        f"(floor {v['effect_floor']}; * = CI excl 0).")
    E7._write_block("E9", "## E9 - Single-cell readout (sci-Plex3)  ", r["atlas"], r["generated"], verdict)


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "knn_gaps": r["knn"]["gaps"],
                      "mantel": r["mantel"], "singlecell_top": r["singlecell_screen_top"][:4]},
                     indent=2, default=float))
