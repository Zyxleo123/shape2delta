"""E10 — MD as an extra input to a real related-work baseline (chemCPA-style).

Unlike E1–E9 (which test the ensemble as a *tier* in a linear/kNN/Mantel harness
and report the T2−T1 gap), E10 drops the global-MD vector into an actual published
drug-response backbone — a faithful chemCPA-style compositional predictor
(model/cpa.py) — as the "extra conditioning input" the spec calls for, and asks
whether it improves the *real* predictor. Done "anyway", despite E1–E9 all null.

Backbone (same across arms; only the target block changes):
    z = doser(logdose)*drug(ECFP) + cell_emb[cell] + target(target_vec);  Δ̂=dec([z,time])

Target population = the 73 LINCS targets with MD frames on disk (global-MD defined
for all of them, vs E8's 20 pocket-covered) ∩ ESM ∩ static pocket descriptor, so
every arm sees the *same* conditions and differs only in the target block:
    OFF        no target        (pure chemistry+cell+dose baseline)
    ESM        T0 identity      (ESM-2 PCA)
    STATIC     T1 static        (AF-model pocket descriptor)
    MD         global MD        (features/md_global)
    STATIC+MD  T1 ⊕ MD          (the real test: motion ON TOP of static)
    MD_PERM    MD[π]            (target-shuffled MD — permutation control)

Headline contrasts (held-out target/scaffold/cell/random, 5 folds, bootstrap CI
over folds, effect floor 0.01 Δρ): (STATIC+MD)−STATIC and MD−STATIC (does motion
add over a snapshot), MD−OFF, MD−MD_PERM. Same decision rule as E3/E7/E8.

  python -m shape2delta.experiments.e10_cpa
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.eval import metrics as M
from shape2delta.eval import splits as SP
from shape2delta.features import esm_embed
from shape2delta.features import md_global as MG
from shape2delta.features import pocket_ensemble as PK
from shape2delta.experiments import e7_pocket as E7
from shape2delta.model import features as F
from shape2delta.model import cpa as CPA

EFFECT_FLOOR = 0.01
ESM_PCS = 32


# ------------------------------------------------------------- assembly ----
def _target_blocks():
    """Per-target standardized blocks over the shared target set
    (MD-global ∩ ESM ∩ static pocket). Returns accs + {esm,static,md,static_md}."""
    from sklearn.decomposition import PCA
    mdg = MG.load().set_index("accession")
    pf = pd.read_csv(PK.OUT / "pocket_features_all.csv")
    pf = pf[pf.get("ok") == True].drop_duplicates("accession").set_index("accession")
    esm, esm_accs = esm_embed.load()
    ei = {a: i for i, a in enumerate(esm_accs)}
    accs = sorted(set(mdg.index) & set(pf.index) & set(ei))
    E = esm[[ei[a] for a in accs]].astype(float)
    d0 = min(ESM_PCS, E.shape[1], len(accs) - 2)
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
    esm_pca = PCA(n_components=d0, random_state=0).fit_transform(Ez).astype(np.float32)
    static, s_cols = E7._standardize(pf.loc[accs].reset_index(), PK.T1_COLS)
    md, md_cols = E7._standardize(mdg.loc[accs].reset_index(), MG.MD_GLOBAL_COLS)
    static, md = static.astype(np.float32), md.astype(np.float32)
    return accs, {"ESM": esm_pca, "STATIC": static, "MD": md,
                  "STATIC+MD": np.hstack([static, md])}, s_cols, md_cols, E


def _pack(accs):
    """Row-level condition features for the shared target set."""
    delta, meta, genes = S.load_delta(exemplar_only=True)
    idx = {a: i for i, a in enumerate(accs)}
    keep = meta["accession"].isin(set(idx)).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    Y = delta[keep].astype(np.float32)
    tidx = meta["accession"].map(idx).to_numpy()                 # per-row target index
    chem = F.chem_matrix(meta["smiles"]).astype(np.float32)
    dt = F.dose_time_block(meta)
    ldose = _z(dt[:, 0]); time = _z(dt[:, 1])
    cvocab = sorted(meta["cell_iname"].astype(str).unique())
    cmap = {c: i for i, c in enumerate(cvocab)}
    cell_idx = meta["cell_iname"].astype(str).map(cmap).to_numpy().astype(np.int64)
    return dict(meta=meta, Y=Y, tidx=tidx, chem=chem, logdose=ldose.astype(np.float32),
                time=time.astype(np.float32), cell_idx=cell_idx, n_cells=len(cvocab),
                n_genes=Y.shape[1]), genes


def _z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


# ---------------------------------------------------------------- run-cv ----
def _fold_indices(meta, kind, fold, rng):
    folds = SP.make_folds(meta, kind, n_splits=config.NFOLDS, seed=config.SEED)
    te = np.where(folds == fold)[0]
    tr_all = np.where((folds >= 0) & (folds != fold))[0]
    if len(te) < 5 or len(tr_all) < 50:
        return None
    perm = rng.permutation(len(tr_all))
    n_va = max(10, int(0.12 * len(tr_all)))
    va = tr_all[perm[:n_va]]; tr = tr_all[perm[n_va:]]
    return tr, va, te, folds


def run(seeds=None) -> dict:
    seeds = seeds or [int(s) for s in os.environ.get("S2D_E10_SEEDS", "1").split(",")]
    threads = os.environ.get("OMP_NUM_THREADS")
    accs, blocks, s_cols, md_cols, _ = _target_blocks()
    data, _ = _pack(accs)
    meta = data["meta"]
    N = len(accs)
    perm = E7._target_perm(N, config.SEED)                       # derangement of targets
    print(f"[e10] {N} targets, {len(meta)} conditions, {data['n_cells']} cells; "
          f"ESM d={blocks['ESM'].shape[1]} STATIC d={blocks['STATIC'].shape[1]} "
          f"MD d={blocks['MD'].shape[1]}; seeds={seeds}", flush=True)

    # arm -> (per-target block or None). MD_PERM = MD block shuffled across targets.
    arm_block = {"OFF": None, "ESM": blocks["ESM"], "STATIC": blocks["STATIC"],
                 "MD": blocks["MD"], "STATIC+MD": blocks["STATIC+MD"],
                 "MD_PERM": blocks["MD"][perm]}
    arms = list(arm_block)

    res = {a: {k: {"dp": [], "t100": []} for k in SP.SPLIT_KINDS} for a in arms}
    for kind in SP.SPLIT_KINDS:
        for fold in range(config.NFOLDS):
            rng = np.random.default_rng(config.SEED * 1000 + fold)
            fi = _fold_indices(meta, kind, fold, rng)
            if fi is None:
                continue
            tr, va, te, _ = fi
            for arm in arms:
                blk = arm_block[arm]
                d_target = 0 if blk is None else blk.shape[1]
                T = None if blk is None else blk[data["tidx"]].astype(np.float32)
                dp_s, t1_s = [], []
                for sd in seeds:
                    d = dict(data); d["target"] = T
                    pred = CPA.train_predict(d, tr, va, te, d_target,
                                             seed=sd + 7 * fold, threads=threads)
                    dp_s.append(M.delta_pearson(data["Y"][te], pred))
                    t1_s.append(M.top_k_overlap(data["Y"][te], pred, k=100))
                res[arm][kind]["dp"].append(float(np.mean(dp_s)))
                res[arm][kind]["t100"].append(float(np.mean(t1_s)))
            print(f"[e10] {kind} fold {fold}: "
                  + " ".join(f"{a}={np.mean(res[a][kind]['dp']):+.3f}" for a in arms), flush=True)

    # gaps
    def g(a, b, kind, met="dp"):
        return M.gap_summary(res[a][kind][met], res[b][kind][met], config.NBOOT, config.SEED)
    gaps = {k: {
        "STATICMD_minus_STATIC": g("STATIC+MD", "STATIC", k),
        "MD_minus_STATIC": g("MD", "STATIC", k),
        "MD_minus_OFF": g("MD", "OFF", k),
        "MD_minus_MDPERM": g("MD", "MD_PERM", k),
        "STATIC_minus_OFF": g("STATIC", "OFF", k),
        "ESM_minus_OFF": g("ESM", "OFF", k),
    } for k in SP.SPLIT_KINDS}

    def meaningful(key):
        return [k for k in SP.SPLIT_KINDS
                if gaps[k][key]["excludes_zero"] and gaps[k][key]["mean"] >= EFFECT_FLOOR]
    md_over_static = sorted(set(meaningful("STATICMD_minus_STATIC")) | set(meaningful("MD_minus_STATIC")))
    verdict_signal = len(md_over_static) >= 2

    acc = {a: {k: {"delta_pearson_mean": float(np.mean(res[a][k]["dp"])) if res[a][k]["dp"] else float("nan"),
                   "top100_mean": float(np.mean(res[a][k]["t100"])) if res[a][k]["t100"] else float("nan"),
                   "delta_pearson_folds": res[a][k]["dp"], "top100_folds": res[a][k]["t100"],
                   "n_folds": len(res[a][k]["dp"])}
               for k in SP.SPLIT_KINDS} for a in arms}

    out = {
        "experiment": "E10_cpa_md_input", "atlas": "LINCS_L1000",
        "backbone": "chemCPA-style compositional (faithful in-repo reimplementation)",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets": N, "n_conditions": int(len(meta)), "n_cells": data["n_cells"],
        "seeds": seeds, "static_cols": s_cols, "md_cols": md_cols,
        "md_source_counts": MG.load().set_index("accession").loc[accs, "md_source"].value_counts().to_dict(),
        "accuracy": acc, "gaps": gaps,
        "verdict": {"md_helps_over_static": verdict_signal,
                    "meaningful_split_types": md_over_static,
                    "effect_floor": EFFECT_FLOOR,
                    "max_staticmd_minus_static": float(max(gaps[k]["STATICMD_minus_STATIC"]["mean"] for k in SP.SPLIT_KINDS)),
                    "max_md_minus_static": float(max(gaps[k]["MD_minus_STATIC"]["mean"] for k in SP.SPLIT_KINDS))},
    }
    R = config.RESULTS
    R.mkdir(parents=True, exist_ok=True)
    (R / "E10_cpa.json").write_text(json.dumps(out, indent=2, default=float))
    _per_fold_csv(out)
    _update_summary(out)
    return out


def _per_fold_csv(out):
    rows = []
    for a, byk in out["accuracy"].items():
        for k, v in byk.items():
            for i, dp in enumerate(v["delta_pearson_folds"]):
                rows.append({"arm": a, "split": k, "fold": i, "delta_pearson": dp,
                             "top100": v["top100_folds"][i]})
    pd.DataFrame(rows).to_csv(config.RESULTS / "E10_per_fold.csv", index=False)


def _update_summary(r):
    f = E7._fmt
    g = r["gaps"]
    sm = "; ".join(f"{k} {f(g[k]['STATICMD_minus_STATIC'])}" for k in ("target", "scaffold", "cell", "random"))
    acc_off = {k: r["accuracy"]["OFF"][k]["delta_pearson_mean"] for k in ("target", "scaffold", "cell", "random")}
    acc_smd = {k: r["accuracy"]["STATIC+MD"][k]["delta_pearson_mean"] for k in ("target", "scaffold", "cell", "random")}
    v = r["verdict"]
    verdict = (
        f"chemCPA-style backbone on LINCS, MD as an extra conditioning input, over "
        f"{r['n_targets']} MD targets / {r['n_conditions']} conditions ({r['md_source_counts']}). "
        f"Absolute Δρ (held-out target/scaffold/cell/random): OFF "
        f"{acc_off['target']:.3f}/{acc_off['scaffold']:.3f}/{acc_off['cell']:.3f}/{acc_off['random']:.3f}, "
        f"STATIC+MD {acc_smd['target']:.3f}/{acc_smd['scaffold']:.3f}/{acc_smd['cell']:.3f}/{acc_smd['random']:.3f}. "
        f"(STATIC+MD)−STATIC Δρ: {sm}. MD−STATIC max {v['max_md_minus_static']:+.4f}; "
        f"(STATIC+MD)−STATIC max {v['max_staticmd_minus_static']:+.4f}. "
        f"→ **{'MD helps' if v['md_helps_over_static'] else 'MD adds nothing over static'}** "
        f"(floor {v['effect_floor']}; * = CI excl 0).")
    E7._write_block("E10", "## E10 — MD as extra input to a chemCPA-style baseline  ",
                    r["atlas"], r["generated"], verdict)


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "gaps": r["gaps"]}, indent=2, default=float))
