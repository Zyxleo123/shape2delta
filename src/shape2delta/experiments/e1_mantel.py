"""E1 — feature-only transferability (partial Mantel), no response model.

Do targets whose ENSEMBLE (T2) features are similar have more similar pseudobulk
Δ signatures, controlling for ESM similarity? Zero training; a null here is still
a clean, citable number. Also reports the ESM-only and T1-structure Mantel for
context, and emits per-pair residualized points for the scatter figure.

  python -m shape2delta.experiments.e1_mantel
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from shape2delta import config
from shape2delta.data import signatures as S
from shape2delta.features import tiers as TI


def _target_delta(exemplar_only=True):
    delta, meta, _ = S.load_delta(exemplar_only=exemplar_only)
    df = {}
    for acc, idx in meta.groupby("accession").groups.items():
        df[acc] = delta[list(idx)].mean(0)
    return df  # acc -> mean Δ (978,)


def _upper(m):
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def _sim_corr(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9
    return (Xc / n) @ (Xc / n).T


def _sim_cosine(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return (X / n) @ (X / n).T


def _partial_r(e, d, c):
    """partial corr of e,d controlling c (1-D upper-tri vectors)."""
    def resid(y, x):
        A = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta
    er, dr = resid(e, c), resid(d, c)
    return float(np.corrcoef(er, dr)[0, 1])


def _mantel_p(e_mat, d_mat, c_vec, observed, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = e_mat.shape[0]
    e = _upper(e_mat)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        dperm = _upper(d_mat[np.ix_(p, p)])
        r = _partial_r(e, dperm, c_vec)
        if abs(r) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def run(exemplar_only=True, n_perm=2000) -> dict:
    tdelta = _target_delta(exemplar_only)
    z = TI.load()
    accs = [a for a in z["accs"] if a in tdelta]
    idx = {a: i for i, a in enumerate(list(z["accs"]))}
    rows = [idx[a] for a in accs]

    D = np.stack([tdelta[a] for a in accs])            # targets x 978
    T2 = z["t2_block"][rows]                            # targets x D2
    T1 = z["t1_block"][rows]
    ESM = z["esm_full"][rows]

    d_mat = _sim_corr(D)                                # Δ-signature similarity
    e_mat = _sim_corr(T2)                               # ensemble-feature similarity
    s_mat = _sim_corr(T1)                               # static-structure similarity
    c_mat = _sim_cosine(ESM)                            # ESM similarity (control)

    e, d, s, c = _upper(e_mat), _upper(d_mat), _upper(s_mat), _upper(c_mat)

    r_ens_simple = float(np.corrcoef(e, d)[0, 1])
    r_ens_partial = _partial_r(e, d, c)                 # ENSEMBLE vs Δ | ESM  (the thesis)
    r_esm_simple = float(np.corrcoef(c, d)[0, 1])
    r_struct_partial = _partial_r(s, d, c)              # static-structure vs Δ | ESM

    p_partial = _mantel_p(e_mat, d_mat, c, r_ens_partial, n_perm=n_perm, seed=config.SEED)

    # scatter points (residualized) for the figure — subsample to keep it light
    def resid(y, x):
        A = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
    er, dr = resid(e, c), resid(d, c)
    rng = np.random.default_rng(0)
    k = min(4000, len(er)); sel = rng.choice(len(er), k, replace=False)

    out = {
        "experiment": "E1_mantel", "atlas": "LINCS_L1000",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(accs), "n_pairs": int(len(e)), "n_perm": n_perm,
        "r_ensemble_vs_delta_simple": r_ens_simple,
        "r_ensemble_vs_delta_partial_given_esm": r_ens_partial,
        "r_esm_vs_delta_simple": r_esm_simple,
        "r_struct_vs_delta_partial_given_esm": r_struct_partial,
        "mantel_p_partial": p_partial,
        "significant": bool(p_partial < 0.05 and r_ens_partial > 0),
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "E1_mantel.json").write_text(json.dumps(out, indent=2))
    np.savez(config.RESULTS / "E1_scatter.npz",
             ensemble_sim_resid=er[sel].astype(np.float32),
             delta_sim_resid=dr[sel].astype(np.float32))
    _update_summary(out)
    return out


def _update_summary(r):
    sm = config.RESULTS / "SUMMARY.md"
    verdict = (f"Ensemble-sim vs Δ-sim partial-r (|ESM) = "
               f"{r['r_ensemble_vs_delta_partial_given_esm']:+.3f} "
               f"(Mantel p={r['mantel_p_partial']:.3g}, {r['n_targets']} targets); "
               f"simple r={r['r_ensemble_vs_delta_simple']:+.3f}, "
               f"ESM-only r={r['r_esm_vs_delta_simple']:+.3f}, "
               f"static-struct partial r={r['r_struct_vs_delta_partial_given_esm']:+.3f}. "
               f"→ **{'signal' if r['significant'] else 'null'}**.")
    block = ["## E1 — Feature-only Mantel  ",
             f"_atlas: {r['atlas']} · {r['generated']}_  ", "",
             f"**Verdict.** {verdict}  ", ""]
    existing = sm.read_text() if sm.exists() else "# shape2delta — results SUMMARY\n\n"
    lines, kept, skip = existing.splitlines(), [], False
    for ln in lines:
        if ln.startswith("## E1 "):
            skip = True; continue
        if skip and ln.startswith("## ") and not ln.startswith("## E1 "):
            skip = False
        if not skip:
            kept.append(ln)
    sm.write_text("\n".join(kept).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
