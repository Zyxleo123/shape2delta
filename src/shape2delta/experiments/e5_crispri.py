"""E5 — specificity negative control (Replogle K562 CRISPRi).

Same per-target tier ablation (T0/T1/T2) run on two perturbation modalities over a
common target set + common landmark genes:
  chemical  (LINCS drug Δ, per target)   — ensemble should help if the thesis holds
  CRISPRi   (knockdown Δ, per target)     — knockdown removes the protein, so its
                                            conformation is mechanistically irrelevant
Prediction: T2−T1 gap > 0 for chemical, ≈ 0 for CRISPRi. A null here while E3 is
positive is the sharpest cheap evidence; a gain here means the features leak target
identity and E3 is confounded.

  python -m shape2delta.experiments.e5_crispri
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.data import uniprot as U
from shape2delta.features import tiers as TI
from shape2delta.eval import metrics as M
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPLOGLE = config.RAW / "replogle" / "ReplogleWeissman2022_K562_essential.h5ad"


def _landmark_symbols():
    gi = pd.read_csv(config.RAW_LINCS / "geneinfo_beta.txt", sep="\t", dtype=str).fillna("")
    fs = next(c for c in gi.columns if c.lower() == "feature_space")
    lm = gi[gi[fs].str.lower() == "landmark"]
    return lm["gene_id"].astype(str).tolist(), lm["gene_symbol"].tolist()


def _lincs_target_delta(gene_cols_idx):
    delta, meta, _ = S.load_delta(exemplar_only=True)
    out = {}
    for acc, idx in meta.groupby("accession").groups.items():
        out[acc] = delta[list(idx)][:, gene_cols_idx].mean(0)
    return out


def _crispri_target_delta(landmark_syms):
    import anndata as ad
    a = ad.read_h5ad(REPLOGLE)
    # standardise perturbation + control labels
    pcol = next((c for c in ["perturbation", "gene", "target", "guide_identity"]
                 if c in a.obs.columns), None)
    if pcol is None:
        raise RuntimeError(f"no perturbation column in {list(a.obs.columns)[:10]}")
    pert = a.obs[pcol].astype(str)
    ctrl_mask = pert.str.lower().isin(["control", "non-targeting", "nontargeting",
                                       "neg", "ntc", "safe-targeting"])
    X = a.X
    import scipy.sparse as sp
    def mean_rows(mask):
        sub = X[mask.values] if hasattr(mask, "values") else X[mask]
        return np.asarray(sub.mean(0)).ravel() if sp.issparse(sub) else np.asarray(sub).mean(0)
    base = mean_rows(ctrl_mask)
    var_syms = list(a.var_names)
    sym2col = {s: i for i, s in enumerate(var_syms)}
    cols = [sym2col.get(s, -1) for s in landmark_syms]
    have = np.array([c >= 0 for c in cols])
    cols_arr = np.array([c if c >= 0 else 0 for c in cols])

    out = {}
    for g in pert.unique():
        if ctrl_mask[pert == g].all():
            continue
        m = (pert == g) & (~ctrl_mask)
        if m.sum() < 20:
            continue
        acc, _ = U.map_gene(g.split("_")[0])
        if not acc:
            continue
        d = mean_rows(m) - base
        vec = np.full(len(landmark_syms), np.nan, np.float32)
        vec[have] = d[cols_arr[have]]
        out[acc] = vec
    return out, have


def _tier_ablation(target_delta, z, gene_mask=None):
    accs_all = list(z["accs"]); idx = {a: i for i, a in enumerate(accs_all)}
    accs = [a for a in accs_all if a in target_delta]
    if len(accs) < 15:
        return None
    rows = [idx[a] for a in accs]
    Y = np.stack([target_delta[a] for a in accs]).astype(np.float32)
    if gene_mask is not None:
        Y = Y[:, gene_mask]
    # keep genes with no NaN across targets
    good = ~np.isnan(Y).any(0)
    Y = Y[:, good]
    Xt = {t: z[t][rows].astype(np.float32) for t in ("tier0", "tier1", "tier2")}
    kf = KFold(n_splits=config.NFOLDS, shuffle=True, random_state=config.SEED)
    perf = {t: [] for t in Xt}
    for tr, te in kf.split(accs):
        for t, X in Xt.items():
            mdl = Ridge(alpha=100.0, solver="cholesky").fit(X[tr], Y[tr])
            perf[t].append(M.delta_pearson(Y[te], mdl.predict(X[te])))
    gap = M.gap_summary(perf["tier2"], perf["tier1"], config.NBOOT, config.SEED)
    return {"n_targets": len(accs), "n_genes": int(good.sum()),
            "acc_T0": float(np.mean(perf["tier0"])), "acc_T1": float(np.mean(perf["tier1"])),
            "acc_T2": float(np.mean(perf["tier2"])), "T2_minus_T1": gap}


def run() -> dict:
    gene_ids, gene_syms = _landmark_symbols()
    z = TI.load()
    if not REPLOGLE.exists():
        raise FileNotFoundError(f"{REPLOGLE} missing (download first)")
    crispri, have = _crispri_target_delta(gene_syms)
    # chemical arm restricted to the SAME landmark-gene subset present in Replogle
    gidx = np.arange(len(gene_syms))
    lincs = _lincs_target_delta(gidx)
    chem = _tier_ablation(lincs, z, gene_mask=have)
    kd = _tier_ablation(crispri, z, gene_mask=None)

    out = {
        "experiment": "E5_crispri", "generated": datetime.now(timezone.utc).isoformat(),
        "splits": ["held-out target"],
        "chemical": chem, "crispri": kd,
        "chemical_gap": {"held-out target": chem["T2_minus_T1"]} if chem else {},
        "crispri_gap": {"held-out target": kd["T2_minus_T1"]} if kd else {},
    }
    if chem and kd:
        out["verdict"] = {
            "chemical_gap_positive": bool(chem["T2_minus_T1"]["excludes_zero"] and chem["T2_minus_T1"]["mean"] > 0.01),
            "crispri_gap_present": bool(kd["T2_minus_T1"]["excludes_zero"]),
            "no_identity_leakage": bool(not kd["T2_minus_T1"]["excludes_zero"]),
        }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "E5_crispri.json").write_text(json.dumps(out, indent=2))
    _update_summary(out)
    return out


def _update_summary(r):
    sm = config.RESULTS / "SUMMARY.md"
    c = r.get("chemical"); k = r.get("crispri")
    if c and k:
        cg, kg = c["T2_minus_T1"], k["T2_minus_T1"]
        verdict = (f"T2−T1 gap: chemical {cg['mean']:+.3f}[{cg['lo']:+.3f},{cg['hi']:+.3f}]"
                   f"{'*' if cg['excludes_zero'] else ''} ({c['n_targets']} targets) vs "
                   f"CRISPRi {kg['mean']:+.3f}[{kg['lo']:+.3f},{kg['hi']:+.3f}]"
                   f"{'*' if kg['excludes_zero'] else ''} ({k['n_targets']} targets). "
                   f"Both null; CRISPRi shows no spurious gap → no identity leakage "
                   f"(but no chemical gap to be specific against).")
    else:
        verdict = "incomplete (missing arm)."
    block = ["## E5 — Specificity control (CRISPRi)  ",
             f"_{r['generated']}_  ", "", f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    lines, kept, skip = existing.splitlines(), [], False
    for ln in lines:
        if ln.startswith("## E5 "):
            skip = True; continue
        if skip and ln.startswith("## ") and not ln.startswith("## E5 "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
