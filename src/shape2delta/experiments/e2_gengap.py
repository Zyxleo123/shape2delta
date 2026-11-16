"""E2 — generalization-gap baseline.

Establishes the failure mode the ensemble feature is meant to repair: a baseline
Δ predictor (chemistry + cell, no target structure) loses accuracy on held-out
target / scaffold / cell splits relative to a random split. This is insurance —
a figure worth having even if every ensemble arm comes back null.

Run: python -m shape2delta.experiments.e2_gengap
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.model import features as F
from shape2delta.experiments import cv


def run(exemplar_only: bool = True, alpha: float = 100.0) -> dict:
    delta, meta, genes = S.load_delta(exemplar_only=exemplar_only)
    X, info = F.baseline_features(meta)
    Y = delta.astype(np.float32)

    res = cv.run_all_splits(X, Y, meta, n_splits=config.NFOLDS, seed=config.SEED,
                            alpha=alpha)

    # generalization drop vs the random split, per metric
    drops = {}
    for held in ("target", "scaffold", "cell"):
        drops[held] = {
            "delta_pearson": cv.drop_ci(res["random"]["delta_pearson_folds"],
                                        res[held]["delta_pearson_folds"],
                                        n_boot=config.NBOOT, seed=config.SEED),
            "top100": cv.drop_ci(res["random"]["top100_folds"],
                                 res[held]["top100_folds"],
                                 n_boot=config.NBOOT, seed=config.SEED),
        }

    out = {
        "experiment": "E2_gengap",
        "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_conditions": int(len(meta)),
        "n_targets": int(meta["accession"].nunique()),
        "n_cells": int(meta["cell_iname"].nunique()),
        "feature_info": {k: info[k] for k in ("n_chem", "n_cell", "n_dt")},
        "alpha": alpha,
        "exemplar_only": exemplar_only,
        "accuracy_by_split": {
            k: {"delta_pearson": res[k]["delta_pearson_mean"],
                "top100": res[k]["top100_mean"],
                "n_folds": res[k]["n_folds"]}
            for k in res
        },
        "generalization_drop_vs_random": drops,
        "per_fold": {k: res[k]["per_fold"] for k in res},
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "E2_gengap.json").write_text(json.dumps(out, indent=2))
    _update_summary(out)
    return out


def _update_summary(r: dict) -> None:
    sm = config.RESULTS / "SUMMARY.md"
    acc = r["accuracy_by_split"]
    d = r["generalization_drop_vs_random"]

    def fmt_drop(h):
        c = d[h]["delta_pearson"]
        star = "*" if c.get("excludes_zero") else ""
        return f"{h} Δρ drop {c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]{star}"

    verdict = (
        f"Baseline Δρ: random {acc['random']['delta_pearson']:.3f} → "
        f"target {acc['target']['delta_pearson']:.3f}, "
        f"scaffold {acc['scaffold']['delta_pearson']:.3f}, "
        f"cell {acc['cell']['delta_pearson']:.3f}. "
        + "; ".join(fmt_drop(h) for h in ("target", "scaffold", "cell"))
        + "  (* = 95% CI excludes 0)"
    )
    block = ["## E2 — Generalization gap  ",
             f"_atlas: {r['atlas']} · n={r['n_conditions']} conditions · {r['generated']}_  ",
             "", f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    lines = existing.splitlines()
    kept, skip = [], False
    for ln in lines:
        if ln.startswith("## E2 "):
            skip = True
            continue
        if skip and ln.startswith("## ") and not ln.startswith("## E2 "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    r = run()
    print(json.dumps({"accuracy_by_split": r["accuracy_by_split"],
                      "generalization_drop_vs_random": r["generalization_drop_vs_random"]},
                     indent=2))
