"""E6 — mechanism-level sanity scatter (standalone, no model).

Across targets, does ensemble flexibility relate to the magnitude/diversity of
the transcriptional response (# DE genes, signature entropy), partialling out
compound potency (dose) and protein size? Speaks to the allostery argument.

  python -m shape2delta.experiments.e6_mechanism
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.data import signatures as S


def _partial_spearman(x, y, controls):
    from scipy.stats import rankdata
    def resid(v, C):
        A = np.column_stack([np.ones(len(v))] + [rankdata(c) for c in C])
        b, *_ = np.linalg.lstsq(A, rankdata(v), rcond=None)
        return rankdata(v) - A @ b
    xr, yr = resid(x, controls), resid(y, controls)
    return float(np.corrcoef(xr, yr)[0, 1])


def run(exemplar_only=True, de_thresh=2.0) -> dict:
    delta, meta, _ = S.load_delta(exemplar_only=exemplar_only)
    t2 = pd.read_csv(config.FEATURES / "t2_ensemble.csv").set_index("accession")
    try:
        t1 = pd.read_csv(config.FEATURES / "t1_structure" / "t1_structure.csv").set_index("accession")
    except Exception:
        t1 = pd.DataFrame()

    # per-signature response descriptors
    nde = (np.abs(delta) > de_thresh).sum(1).astype(float)
    p = np.abs(delta) / (np.abs(delta).sum(1, keepdims=True) + 1e-9)
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    meta = meta.copy()
    meta["nde"], meta["entropy"] = nde, entropy
    meta["ldose"] = np.log10(pd.to_numeric(meta.get("pert_dose"), errors="coerce").clip(lower=1e-3)).fillna(0)

    agg = meta.groupby("accession").agg(nde=("nde", "mean"), entropy=("entropy", "mean"),
                                        ldose=("ldose", "mean"), n_sig=("sig_id", "size"))
    agg = agg.join(t2[["rmsf_mean", "frac_flexible", "rg_cv", "n_conformers"]], how="inner")
    if "n_res" in getattr(t1, "columns", []):
        agg = agg.join(t1[["n_res"]], how="left")
    agg["n_res"] = agg.get("n_res", pd.Series(index=agg.index)).fillna(agg.get("n_res", pd.Series()).median() if "n_res" in agg else 0)
    agg = agg.dropna(subset=["rmsf_mean", "nde"])

    x_flex = agg["rmsf_mean"].to_numpy()
    controls = [agg["ldose"].to_numpy(), np.log(agg["n_res"].to_numpy() + 1) if "n_res" in agg else np.zeros(len(agg))]
    out = {
        "experiment": "E6_mechanism", "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets": int(len(agg)),
        "spearman_flex_vs_nde": float(pd.Series(x_flex).corr(agg["nde"].reset_index(drop=True), method="spearman")),
        "partial_flex_vs_nde_given_dose_size": _partial_spearman(x_flex, agg["nde"].to_numpy(), controls),
        "spearman_flex_vs_entropy": float(pd.Series(x_flex).corr(agg["entropy"].reset_index(drop=True), method="spearman")),
        "partial_flex_vs_entropy_given_dose_size": _partial_spearman(x_flex, agg["entropy"].to_numpy(), controls),
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "E6_mechanism.json").write_text(json.dumps(out, indent=2))
    agg.reset_index()[["accession", "rmsf_mean", "frac_flexible", "nde", "entropy", "ldose", "n_sig"]] \
        .to_csv(config.RESULTS / "E6_points.csv", index=False)
    _update_summary(out)
    return out


def _update_summary(r):
    sm = config.RESULTS / "SUMMARY.md"
    verdict = (f"Flexibility (RMSF) vs #DE genes: Spearman {r['spearman_flex_vs_nde']:+.3f}, "
               f"partial(|dose,size) {r['partial_flex_vs_nde_given_dose_size']:+.3f}; "
               f"vs signature entropy partial {r['partial_flex_vs_entropy_given_dose_size']:+.3f} "
               f"({r['n_targets']} targets).")
    block = ["## E6 — Mechanism scatter  ", f"_atlas: {r['atlas']} · {r['generated']}_  ",
             "", f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    lines, kept, skip = existing.splitlines(), [], False
    for ln in lines:
        if ln.startswith("## E6 "):
            skip = True; continue
        if skip and ln.startswith("## ") and not ln.startswith("## E6 "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
