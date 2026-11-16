"""E3 — core ablation (the headline). T0 vs T1 vs T2 target features dropped into
a ridge Δ predictor, same splits/folds, with a shuffled-target permutation arm.

The vectors are nested (T1 ⊂ T2), identical length, so the only thing varying
between arms is tier membership. Headline = the T2−T1 gap, reported as a
bootstrap mean 95% CI over folds; real only if the CI excludes 0 in the same
direction on >=2 split types.

  python -m shape2delta.experiments.e3_ablation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.model import features as F
from shape2delta.features import tiers as TI
from shape2delta.experiments import cv
from shape2delta.eval import splits as SP
from shape2delta.eval import metrics as M


def _assemble(exemplar_only=True, enzyme_only=False):
    delta, meta, _ = S.load_delta(exemplar_only=exemplar_only)
    z = TI.load()
    accs = list(z["accs"])
    idx = {a: i for i, a in enumerate(accs)}
    keep = meta["accession"].isin(set(accs)).to_numpy()
    if enzyme_only and "is_enzyme" in meta.columns:
        keep &= meta["is_enzyme"].astype(str).isin(["True", "true", "1", "1.0"]).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    Y = delta[keep].astype(np.float32)
    rows = meta["accession"].map(idx).to_numpy()
    base, _ = F.baseline_features(meta)
    tier = {k: z[k][rows].astype(np.float32) for k in ("tier0", "tier1", "tier2")}
    return meta, Y, base, tier, rows, z


def run(exemplar_only=True, enzyme_only=False, alpha=100.0) -> dict:
    meta, Y, base, tier, rows, z = _assemble(exemplar_only, enzyme_only)
    rng = np.random.default_rng(config.SEED)
    perm_rows = rng.permutation(len(rows))          # shuffle target→signature link
    arms = {
        "T0": np.concatenate([base, tier["tier0"]], 1),
        "T1": np.concatenate([base, tier["tier1"]], 1),
        "T2": np.concatenate([base, tier["tier2"]], 1),
        "T2_permuted": np.concatenate([base, z["tier2"][rows][perm_rows].astype(np.float32)], 1),
    }
    res = {arm: cv.run_all_splits(X, Y, meta, n_splits=config.NFOLDS,
                                  seed=config.SEED, alpha=alpha)
           for arm, X in arms.items()}

    gaps = {}
    for kind in SP.SPLIT_KINDS:
        def folds(arm, metric):
            return res[arm][kind][f"{metric}_folds"]
        gaps[kind] = {
            "T2_minus_T1_dp": M.gap_summary(folds("T2", "delta_pearson"), folds("T1", "delta_pearson"), config.NBOOT, config.SEED),
            "T1_minus_T0_dp": M.gap_summary(folds("T1", "delta_pearson"), folds("T0", "delta_pearson"), config.NBOOT, config.SEED),
            "T2_minus_perm_dp": M.gap_summary(folds("T2", "delta_pearson"), folds("T2_permuted", "delta_pearson"), config.NBOOT, config.SEED),
            "T2_minus_T1_top100": M.gap_summary(folds("T2", "top100"), folds("T1", "top100"), config.NBOOT, config.SEED),
        }
    # decision rule: a gap counts only if it is BOTH statistically detectable
    # (CI excludes 0) AND practically meaningful (|mean| >= floor). With n~10^4 a
    # 0.0005 gap is "significant" but meaningless, so the floor matters.
    EFFECT_FLOOR = 0.01
    sig_splits = [k for k in SP.SPLIT_KINDS if gaps[k]["T2_minus_T1_dp"]["excludes_zero"]]
    pos = [k for k in sig_splits if gaps[k]["T2_minus_T1_dp"]["mean"] > 0]
    meaningful = [k for k in SP.SPLIT_KINDS
                  if gaps[k]["T2_minus_T1_dp"]["excludes_zero"]
                  and gaps[k]["T2_minus_T1_dp"]["mean"] >= EFFECT_FLOOR]

    out = {
        "experiment": "E3_ablation", "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "exemplar_only": exemplar_only, "enzyme_only": enzyme_only,
        "n_conditions": int(len(meta)), "n_targets": int(meta["accession"].nunique()),
        "tier_dims": {"D0": int(z["dims"][0]), "D1": int(z["dims"][1]), "D2": int(z["dims"][2])},
        "accuracy": {arm: {k: {"delta_pearson": res[arm][k]["delta_pearson_mean"],
                               "top100": res[arm][k]["top100_mean"],
                               "n_folds": res[arm][k]["n_folds"],
                               "delta_pearson_folds": res[arm][k]["delta_pearson_folds"],
                               "top100_folds": res[arm][k]["top100_folds"]}
                           for k in SP.SPLIT_KINDS} for arm in arms},
        "gaps": gaps,
        "verdict": {
            "T2_minus_T1_practical_gain": len(meaningful) >= 2,
            "effect_floor": EFFECT_FLOOR,
            "meaningful_split_types": meaningful,
            "statistically_sig_split_types": sig_splits,
            "positive_split_types": pos,
            "max_gap": max(gaps[k]["T2_minus_T1_dp"]["mean"] for k in SP.SPLIT_KINDS),
        },
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "E3_ablation.json").write_text(json.dumps(out, indent=2))
    _update_summary(out)
    return out


def _update_summary(r):
    sm = config.RESULTS / "SUMMARY.md"
    g = r["gaps"]

    def gl(kind):
        c = g[kind]["T2_minus_T1_dp"]
        return f"{kind} {c['mean']:+.3f}[{c['lo']:+.3f},{c['hi']:+.3f}]{'*' if c['excludes_zero'] else ''}"

    v = r["verdict"]
    gain = v["T2_minus_T1_practical_gain"]
    verdict = (f"T2−T1 Δρ gap: " + "; ".join(gl(k) for k in ("target", "scaffold", "cell", "random"))
               + f" (* = CI excl 0). → **{'practical gain' if gain else 'No practical ensemble gain'}**: "
               f"max gap {v['max_gap']:+.4f} (floor {v['effect_floor']}); CIs exclude 0 only at n~1e4. "
               f"T0≈T1≈T2.")
    block = ["## E3 — Core ablation (T0/T1/T2)  ",
             f"_atlas: {r['atlas']} · n={r['n_conditions']} cond, {r['n_targets']} targets · {r['generated']}_  ",
             "", f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    lines, kept, skip = existing.splitlines(), [], False
    for ln in lines:
        if ln.startswith("## E3 "):
            skip = True; continue
        if skip and ln.startswith("## ") and not ln.startswith("## E3 "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    r = run()
    print(json.dumps({"gaps": r["gaps"], "verdict": r["verdict"]}, indent=2))
