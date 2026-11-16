"""E7 — pocket-centric, MATCHED static-vs-ensemble ablation.

Re-runs the thesis test with the spec's T2 (Deep dive, "Inputs (3)"): per-residue
flexibility AT THE DRUG SITE, state-population summaries and cryptic-pocket
descriptors, with T1 and T2 from ONE extractor (features/pocket_ensemble.py):

    T0 = [esm_pca | 0      | 0     ]
    T1 = [esm_pca | static | 0     ]      static = pocket descriptor of AF model
    T2 = [esm_pca | static | motion]      motion = same descriptor over exp. conformers
    T2_perm = [esm_pca | static | motion[π]]   π = TARGET-level shuffle of the motion
                                                block only (static/ESM untouched)

Three tests, each independently reportable:
  (a) ridge ablation (E3 protocol): signature-level Δ, 4 split kinds, gap CI over
      folds, effect floor 0.01 Δρ, "real" only on >=2 split types.
  (b) target-level kNN transfer: predict a held-out target's mean Δ from its k
      nearest TRAINING targets in each tier's (block-balanced) feature space;
      repeated 5-fold over targets; gap CI by bootstrap over targets. Scored raw
      and "centred" (deviation from the training-mean Δ — the part a target
      feature could possibly explain).
  (c) partial Mantel: motion-similarity vs Δ-deviation similarity, controlling for
      ESM AND static-pocket similarity.

  python -m shape2delta.experiments.e7_pocket
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
from shape2delta.features import esm_embed
from shape2delta.features import pocket_ensemble as PK
from shape2delta.model import features as F

EFFECT_FLOOR = 0.01
K_NN = 10
N_REPEAT = 10


def _standardize(df, cols):
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    r, c = np.where(np.isnan(X))
    X[r, c] = med[c]
    sd = X.std(0)
    keep = sd > 1e-9                                   # drop constant columns
    return ((X[:, keep] - X[:, keep].mean(0)) / sd[keep]), [c for c, k in zip(cols, keep) if k]


def _blocks(d0=64):
    """Comparison set = targets with pocket features (>=3 conformers) ∩ ESM."""
    from sklearn.decomposition import PCA
    pf = pd.read_csv(PK.OUT / "pocket_features.csv")
    esm, esm_accs = esm_embed.load()
    ei = {a: i for i, a in enumerate(esm_accs)}
    pf = pf[pf["accession"].isin(set(ei))].reset_index(drop=True)
    accs = pf["accession"].tolist()
    E = esm[[ei[a] for a in accs]].astype(float)
    d0 = min(d0, E.shape[1], len(accs) - 1)
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
    esm_pca = PCA(n_components=d0, random_state=0).fit_transform(Ez)   # E3 scale
    s, s_cols = _standardize(pf, PK.T1_COLS)
    m, m_cols = _standardize(pf, PK.T2_COLS)
    return accs, pf, E, esm_pca, s, m, s_cols, m_cols


def _target_perm(n, seed):
    rng = np.random.default_rng(seed)
    while True:                                        # derangement: nobody keeps own
        p = rng.permutation(n)
        if (p != np.arange(n)).all():
            return p


# ------------------------------------------------------------ (a) ridge ----
def ridge_ablation(accs, esm_pca, s, m, perm, alpha=100.0):
    delta, meta, _ = S.load_delta(exemplar_only=True)
    idx = {a: i for i, a in enumerate(accs)}
    keep = meta["accession"].isin(set(idx)).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    Y = delta[keep].astype(np.float32)
    rows = meta["accession"].map(idx).to_numpy()
    base, _ = F.baseline_features(meta)
    Zs, Zm = np.zeros_like(s), np.zeros_like(m)
    tiers = {
        "T0": np.concatenate([esm_pca, Zs, Zm], 1),
        "T1": np.concatenate([esm_pca, s, Zm], 1),
        "T2": np.concatenate([esm_pca, s, m], 1),
        "T2_permuted": np.concatenate([esm_pca, s, m[perm]], 1),
        "T0+motion": np.concatenate([esm_pca, Zs, m], 1),
    }
    res = {arm: cv.run_all_splits(np.concatenate([base, T[rows].astype(np.float32)], 1),
                                  Y, meta, n_splits=config.NFOLDS, seed=config.SEED,
                                  alpha=alpha)
           for arm, T in tiers.items()}

    def g(a, b, kind, met="delta_pearson"):
        return M.gap_summary(res[a][kind][f"{met}_folds"], res[b][kind][f"{met}_folds"],
                             config.NBOOT, config.SEED)
    gaps = {k: {"T2_minus_T1_dp": g("T2", "T1", k), "T1_minus_T0_dp": g("T1", "T0", k),
                "T2_minus_perm_dp": g("T2", "T2_permuted", k),
                "T2_minus_T1_top100": g("T2", "T1", k, "top100")}
            for k in SP.SPLIT_KINDS}
    meaningful = [k for k in SP.SPLIT_KINDS if gaps[k]["T2_minus_T1_dp"]["excludes_zero"]
                  and gaps[k]["T2_minus_T1_dp"]["mean"] >= EFFECT_FLOOR]
    acc = {arm: {k: {"delta_pearson": res[arm][k]["delta_pearson_mean"],
                     "top100": res[arm][k]["top100_mean"],
                     "delta_pearson_folds": res[arm][k]["delta_pearson_folds"],
                     "top100_folds": res[arm][k]["top100_folds"]}
                 for k in SP.SPLIT_KINDS} for arm in tiers}
    return {"n_conditions": int(len(meta)), "n_targets": int(meta["accession"].nunique()),
            "accuracy": acc, "gaps": gaps, "meaningful_split_types": meaningful,
            "practical_gain": len(meaningful) >= 2,
            "max_gap": float(max(gaps[k]["T2_minus_T1_dp"]["mean"] for k in SP.SPLIT_KINDS))}


# -------------------------------------------------------------- (b) kNN ----
def _target_mean_delta(accs):
    delta, meta, _ = S.load_delta(exemplar_only=True)
    out, n = np.full((len(accs), delta.shape[1]), np.nan), np.zeros(len(accs), int)
    for i, a in enumerate(accs):
        ix = np.where(meta["accession"].to_numpy() == a)[0]
        if len(ix):
            out[i] = delta[ix].mean(0)
            n[i] = len(ix)
    return out, n


def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12)


def knn_transfer(D, spaces, k=K_NN, n_repeat=N_REPEAT, seed=0):
    from sklearn.model_selection import KFold
    n = len(D)
    raw = {nm: np.zeros((n_repeat, n)) for nm in list(spaces) + ["train_mean"]}
    cen = {nm: np.zeros((n_repeat, n)) for nm in spaces}
    for r in range(n_repeat):
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(np.arange(n)):
            mu = D[tr].mean(0)
            raw["train_mean"][r, te] = _rowcorr(D[te], np.broadcast_to(mu, D[te].shape))
            for nm, Z in spaces.items():
                d2 = ((Z[te, None, :] - Z[None, tr, :]) ** 2).sum(-1)
                nn = np.argsort(d2, 1)[:, :k]
                pred = D[tr][nn].mean(1)
                raw[nm][r, te] = _rowcorr(D[te], pred)
                cen[nm][r, te] = _rowcorr(D[te] - mu, pred - mu)
    per_t = {f"raw_{nm}": v.mean(0) for nm, v in raw.items()}
    per_t.update({f"cen_{nm}": v.mean(0) for nm, v in cen.items()})

    rng = np.random.default_rng(seed)
    B = rng.integers(0, n, size=(config.NBOOT, n))

    def ci(x):
        b = x[B].mean(1)
        lo, hi = np.quantile(b, [0.025, 0.975])
        return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0)}
    summ = {key: ci(v) for key, v in per_t.items()}
    # Standard T0/T1/T2 gaps, only for whichever of those named spaces exist
    # (callers with differently-named spaces, e.g. E8's T2_MD, add their own).
    gaps = {}
    for sc in ("raw", "cen"):
        def g(a, c):
            if f"{sc}_{a}" in per_t and f"{sc}_{c}" in per_t:
                gaps[f"{sc}_{a}_minus_{c}"] = ci(per_t[f"{sc}_{a}"] - per_t[f"{sc}_{c}"])
        g("T2", "T1"); g("T1", "T0"); g("T2", "T2_permuted")
        if f"{sc}_T2_permuted" in per_t:                 # rename for the summary line
            gaps[f"{sc}_T2_minus_perm"] = gaps.get(f"{sc}_T2_minus_T2_permuted")
    return {"k": k, "n_repeat": n_repeat, "n_targets": n, "scores": summ, "gaps": gaps}, per_t


# ----------------------------------------------------------- (c) Mantel ----
def _upper(Mx):
    return Mx[np.triu_indices(Mx.shape[0], 1)]


def _negdist(Z):
    sq = (Z * Z).sum(1)
    return -np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Z @ Z.T, 0))


def _resid(y, C):
    A = np.column_stack([np.ones(len(y))] + list(C))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def partial_mantel(D, E, s, m, n_perm=2000, seed=0):
    Dd = D - D.mean(0)                                 # target-specific deviation
    d_mat = np.corrcoef(Dd)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    c_esm, c_s = _upper(En @ En.T), _upper(_negdist(s))
    e_m = _upper(_negdist(m))
    em_r = _resid(e_m, [c_esm, c_s])
    d = _upper(d_mat)

    def pr(dv):
        return float(np.corrcoef(em_r, _resid(dv, [c_esm, c_s]))[0, 1])
    obs = pr(d)
    rng = np.random.default_rng(seed)
    n = len(D)
    null = np.array([pr(_upper(d_mat[np.ix_(p, p)]))
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    static_partial = float(np.corrcoef(_resid(c_s, [c_esm]), _resid(d, [c_esm]))[0, 1])
    sel = np.random.default_rng(0).choice(len(d), min(4000, len(d)), replace=False)
    pts = {"motion_sim_resid": em_r[sel], "delta_sim_resid": _resid(d, [c_esm, c_s])[sel]}
    return {"n_targets": n, "n_pairs": int(len(d)), "n_perm": n_perm,
            "r_motion_vs_delta_partial_given_esm_static": obs, "mantel_p": p,
            "r_motion_vs_delta_simple": float(np.corrcoef(e_m, d)[0, 1]),
            "r_static_vs_delta_partial_given_esm": static_partial,
            "r_esm_vs_delta_simple": float(np.corrcoef(c_esm, d)[0, 1]),
            "null_abs_p95": float(np.quantile(np.abs(null), 0.95))}, pts


# -------------------------------------------- (d) per-feature screen (desc) --
def feature_screen(pf, D, n_sig):
    """Spearman of each pocket feature with target response strength (mean-Δ norm),
    BH-adjusted. Descriptive only (not a test of the thesis)."""
    from scipy.stats import spearmanr
    strength = np.linalg.norm(D, axis=1)
    rows = []
    for c in PK.T1_COLS + PK.T2_COLS + ["x_cryptic_apo_holo"]:
        v = pd.to_numeric(pf[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 20 or np.nanstd(v[ok]) < 1e-12:
            continue
        rho, p = spearmanr(v[ok], strength[ok])
        rows.append({"feature": c, "tier": "T1" if c in PK.T1_COLS else "T2",
                     "n": int(ok.sum()), "rho": float(rho), "p": float(p)})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    q = (df["p"] * m / (np.arange(m) + 1)).to_numpy()
    df["q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    return df


# ------------------------------------------------------------------- run ----
def run() -> dict:
    accs, pf, E, esm_pca, s, m, s_cols, m_cols = _blocks()
    perm = _target_perm(len(accs), config.SEED)
    D, n_sig = _target_mean_delta(accs)
    have = n_sig > 0
    print(f"[e7] {len(accs)} targets with pocket feats; {have.sum()} with signatures; "
          f"static dims {s.shape[1]}, motion dims {m.shape[1]}", flush=True)

    ridge = ridge_ablation(accs, esm_pca, s, m, perm)
    print(f"[e7] ridge done: max T2−T1 {ridge['max_gap']:+.4f}", flush=True)

    # (b)/(c) on targets that have signatures; block-balanced spaces
    a_, pf_, E_, P0, s_, m_, D_ = ([accs[i] for i in np.where(have)[0]], pf[have].reset_index(drop=True),
                                   E[have], esm_pca[have], s[have], m[have], D[have])
    perm_ = _target_perm(len(a_), config.SEED + 1)
    def bal(Z):                     # each block contributes equal total variance
        return Z / np.sqrt(Z.var(0).sum() + 1e-12)
    b0, bs, bm = bal(P0), bal(s_), bal(m_)
    spaces = {"T0": b0, "T1": np.hstack([b0, bs]), "T2": np.hstack([b0, bs, bm]),
              "T2_permuted": np.hstack([b0, bs, bm[perm_]]), "motion_only": bm}
    knn, per_t = knn_transfer(D_, spaces, seed=config.SEED)
    print(f"[e7] kNN done: cen T2−T1 {knn['gaps']['cen_T2_minus_T1']['mean']:+.4f}", flush=True)
    mantel, pts = partial_mantel(D_, E_, s_, m_, seed=config.SEED)
    print(f"[e7] Mantel partial r {mantel['r_motion_vs_delta_partial_given_esm_static']:+.4f} "
          f"p={mantel['mantel_p']:.3g}", flush=True)
    screen = feature_screen(pf_, D_, n_sig[have])

    kg = knn["gaps"]["cen_T2_minus_T1"]
    knn_gain = kg["excludes_zero"] and kg["mean"] >= EFFECT_FLOOR
    mantel_sig = mantel["mantel_p"] < 0.05 and mantel["r_motion_vs_delta_partial_given_esm_static"] > 0
    out = {
        "experiment": "E7_pocket_matched", "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets_pocket": len(accs),
        "n_enzymes": int(pf["is_enzyme"].sum()) if "is_enzyme" in pf else None,
        "pocket_source": pf["pocket_source"].value_counts().to_dict(),
        "median_conformers": float(pf["n_conformers_used"].median()),
        "static_cols": s_cols, "motion_cols": m_cols,
        "ridge": ridge, "knn": knn, "mantel": mantel,
        "feature_screen_top": screen.head(10).to_dict("records"),
        "verdict": {"ridge_practical_gain": ridge["practical_gain"],
                    "knn_practical_gain": bool(knn_gain), "mantel_significant": bool(mantel_sig),
                    "any_signal": bool(ridge["practical_gain"] or knn_gain or mantel_sig),
                    "effect_floor": EFFECT_FLOOR},
    }
    R = config.RESULTS
    R.mkdir(parents=True, exist_ok=True)
    (R / "E7_pocket.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(R / "E7_mantel_scatter.npz", **{k: v.astype(np.float32) for k, v in pts.items()})
    pd.DataFrame({"accession": a_, "n_signatures": n_sig[have], **per_t}).to_csv(
        R / "E7_knn_per_target.csv", index=False)
    screen.to_csv(R / "E7_feature_screen.csv", index=False)
    _update_summary(out)
    return out


def _fmt(c):
    return f"{c['mean']:+.3f}[{c['lo']:+.3f},{c['hi']:+.3f}]{'*' if c['excludes_zero'] else ''}"


def _update_summary(r):
    sm = config.RESULTS / "SUMMARY.md"
    rg, kn, mt, v = r["ridge"]["gaps"], r["knn"], r["mantel"], r["verdict"]
    ridge_s = "; ".join(f"{k} {_fmt(rg[k]['T2_minus_T1_dp'])}" for k in ("target", "scaffold", "cell", "random"))
    ne = r.get("n_enzymes")
    verdict = (
        f"Matched pocket tiers ({r['n_targets_pocket']} targets"
        f"{f', {ne} enzymes' if ne is not None else ''}; median "
        f"{r['median_conformers']:.0f} conformers; T1 = AF pocket descriptor, T2 = T1 ⊕ its "
        f"motion over experimental conformers). (a) ridge T2−T1 Δρ: {ridge_s}. "
        f"(b) held-out-target kNN, centred Δρ T2−T1 {_fmt(kn['gaps']['cen_T2_minus_T1'])}, "
        f"T2−perm {_fmt(kn['gaps']['cen_T2_minus_perm'])} ({kn['n_targets']} targets). "
        f"(c) partial Mantel motion vs Δ | ESM+static r="
        f"{mt['r_motion_vs_delta_partial_given_esm_static']:+.3f} (p={mt['mantel_p']:.3g}). "
        f"→ **{'signal' if v['any_signal'] else 'null'}** (effect floor {v['effect_floor']}; * = CI excl 0).")
    _write_block("E7", "## E7 — Pocket-centric matched ablation  ", r["atlas"], r["generated"], verdict)


def _write_block(tag, header, atlas, generated, verdict):
    """Replace (or append) the '## <tag> ' section of results/SUMMARY.md."""
    sm = config.RESULTS / "SUMMARY.md"
    block = [header, f"_atlas: {atlas} · {generated}_  ", "", f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    kept, skip = [], False
    for ln in existing.splitlines():
        if ln.startswith(f"## {tag} "):
            skip = True
            continue
        if skip and ln.startswith("## "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "knn_gaps": r["knn"]["gaps"],
                      "mantel": r["mantel"]}, indent=2, default=float))
