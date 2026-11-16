"""Figures for the pilot (local matplotlib — no GPU, light compute).

Reads only the saved result artefacts in results/, so figures are redrawable
without recomputation. Each function skips cleanly if its result is absent.

  python -m shape2delta.viz.figures          # all available
  python -m shape2delta.viz.figures e3        # one
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from shape2delta import config

R = config.RESULTS
FIG = R / "figures"

# Okabe-Ito colourblind-safe palette, fixed order.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#222222", "grey": "#999999"}
INK = "#222222"; MUTED = "#6b6b6b"; GRID = "#e6e6e6"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 16,
        "axes.edgecolor": MUTED, "axes.linewidth": 1.1, "axes.titlesize": 20,
        "axes.titleweight": "bold", "axes.labelsize": 17, "axes.labelweight": "bold",
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.labelsize": 15, "ytick.labelsize": 14,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.9, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 15, "legend.handlelength": 1.6,
        "legend.handleheight": 1.2, "font.family": "DejaVu Sans",
    })
    return plt


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[fig] {FIG/stem}.png")


# ---------------------------------------------------------------- E0 funnel ---
def fig_e0():
    r = _load("E0_coverage.json")
    if not r:
        return
    plt = _style()
    ta, te = r["targets_per_tier_all"], r["targets_per_tier_enzyme"]
    stages = ["T0\nidentity", "T1\nstatic", "T2\nensemble"]
    allv = [ta["T0_identity"], ta["T1_afdb"], ta["T2_any"]]
    enzv = [te["T0_identity"], te["T1_afdb"], te["T2_any"]]
    md = [ta["T0_identity"], ta["T1_afdb"], ta["T2_md_any"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    x = np.arange(3); w = 0.6
    ax.bar(x, allv, w, color=OI["blue"], label="all single-target targets", zorder=3)
    ax.bar(x, enzv, w, color=OI["orange"], label="enzyme subset", zorder=4)
    ax.plot(x, md, "o-", color=OI["vermillion"], lw=2, ms=7, label="T2 via precomputed MD", zorder=5)
    for xi, v in zip(x, allv):
        ax.text(xi, v + 8, str(v), ha="center", va="bottom", fontsize=13, color=INK)
    for xi, v in zip(x, enzv):
        ax.text(xi, v - 18, str(v), ha="center", va="top", fontsize=12, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("targets reaching tier")
    ax.set_title("E0 — target coverage funnel (T0 → T1 → T2)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(allv) * 1.15); ax.grid(axis="x", visible=False)
    _save(fig, "E0_funnel")


# ---------------------------------------------------------------- E2 bars -----
def fig_e2():
    r = _load("E2_gengap.json")
    if not r:
        return
    plt = _style()
    acc = r["accuracy_by_split"]
    order = ["random", "target", "scaffold", "cell"]
    labels = ["random\n(in-dist)", "held-out\ntarget", "held-out\nscaffold", "held-out\ncell"]
    dp = [acc[k]["delta_pearson"] for k in order]
    t100 = [acc[k]["top100"] for k in order]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    x = np.arange(4); w = 0.38
    b1 = ax.bar(x - w/2, dp, w, color=OI["blue"], label="Δ-Pearson", zorder=3)
    b2 = ax.bar(x + w/2, t100, w, color=OI["orange"], label="top-100 recovery", zorder=3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=12, color=INK)
    ax.axvspan(0.5, 3.5, color=OI["grey"], alpha=0.06, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("prediction accuracy"); ax.set_ylim(0, max(dp+t100)*1.18)
    ax.set_title("E2 — generalization gap (chemistry+cell baseline)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="x", visible=False)
    _save(fig, "E2_gengap")


# ---------------------------------------------------------------- E3 bars -----
def fig_e3():
    r = _load("E3_ablation.json")
    if not r:
        return
    plt = _style()
    arms = ["T0", "T1", "T2", "T2_permuted"]
    colors = [OI["grey"], OI["sky"], OI["vermillion"], OI["black"]]
    splits = ["target", "scaffold", "cell", "random"]
    labels = ["held-out\ntarget", "held-out\nscaffold", "held-out\ncell", "random\n(ref)"]

    def ci(folds):
        a = np.asarray([v for v in folds if np.isfinite(v)], float)
        return 1.96 * a.std() / np.sqrt(max(len(a), 1)) if len(a) else 0.0

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(splits)); w = 0.2
    for i, (arm, c) in enumerate(zip(arms, colors)):
        vals = [r["accuracy"][arm][s]["delta_pearson"] for s in splits]
        err = [ci(r["accuracy"][arm][s].get("delta_pearson_folds", [])) for s in splits]
        ax.bar(x + (i-1.5)*w, vals, w, color=c, label=arm.replace("_", " "),
               yerr=err, capsize=3, ecolor=MUTED, error_kw={"lw": 1.3}, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ-Pearson")
    ax.set_title("E3 — core ablation: T0 ≈ T1 ≈ T2 (no ensemble gain)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              fontsize=14, columnspacing=1.2, handletextpad=0.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.34)
    _save(fig, "E3_ablation")


# ---------------------------------------------------------------- E1 scatter --
def fig_e1():
    r = _load("E1_mantel.json")
    npz = R / "E1_scatter.npz"
    if not r or not npz.exists():
        return
    plt = _style()
    d = np.load(npz)
    x, y = d["ensemble_sim_resid"], d["delta_sim_resid"]
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.scatter(x, y, s=7, color=OI["blue"], alpha=0.25, edgecolors="none", zorder=3)
    # trend line
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b*xs, color=OI["vermillion"], lw=2, zorder=4)
    ax.set_xlabel("ensemble-feature similarity  (residual | ESM)")
    ax.set_ylabel("Δ-signature similarity  (residual | ESM)")
    ax.set_title("E1 — ensemble similarity vs response similarity")
    txt = (f"partial r = {r['r_ensemble_vs_delta_partial_given_esm']:+.3f}\n"
           f"Mantel p = {r['mantel_p_partial']:.3g}\n{r['n_targets']} targets")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color=INK, bbox=dict(boxstyle="round", fc="white", ec=GRID))
    _save(fig, "E1_mantel_scatter")


# ---------------------------------------------------------------- E6 scatter --
def fig_e6():
    r = _load("E6_mechanism.json")
    pts = R / "E6_points.csv"
    if not r or not pts.exists():
        return
    import pandas as pd
    plt = _style()
    df = pd.read_csv(pts)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.scatter(df["rmsf_mean"], df["nde"], s=18, color=OI["green"], alpha=0.5,
               edgecolors="white", linewidths=0.4, zorder=3)
    if len(df) > 2:
        b, a = np.polyfit(df["rmsf_mean"], df["nde"], 1)
        xs = np.linspace(df["rmsf_mean"].min(), df["rmsf_mean"].max(), 50)
        ax.plot(xs, a + b*xs, color=OI["vermillion"], lw=2, zorder=4)
    ax.set_xlabel("ensemble flexibility  (mean RMSF, Å)")
    ax.set_ylabel("# DE genes  (mean per signature)")
    ax.set_title("E6 — flexibility vs transcriptional response")
    txt = (f"Spearman {r['spearman_flex_vs_nde']:+.2f}\n"
           f"partial(|dose,size) {r['partial_flex_vs_nde_given_dose_size']:+.2f}\n{r['n_targets']} targets")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color=INK, bbox=dict(boxstyle="round", fc="white", ec=GRID))
    _save(fig, "E6_mechanism")


# ---------------------------------------------------------------- E5 paired ---
def fig_e5():
    r = _load("E5_crispri.json")
    if not r:
        return
    plt = _style()
    s = r.get("splits", ["held-out target"])[0]
    chem, kd = r["chemical"], r["crispri"]
    cg, kg = chem["T2_minus_T1"], kd["T2_minus_T1"]
    means = [cg["mean"], kg["mean"]]
    lo = [cg["lo"], kg["lo"]]; hi = [cg["hi"], kg["hi"]]
    labels = [f"chemical\n(LINCS, n={chem['n_targets']})",
              f"CRISPRi knockdown\n(Replogle, n={kd['n_targets']})"]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    x = np.arange(2)
    err = [np.array(means) - np.array(lo), np.array(hi) - np.array(means)]
    ax.bar(x, means, 0.55, yerr=err, color=[OI["vermillion"], OI["sky"]],
           capsize=6, ecolor=MUTED, error_kw={"lw": 1.6}, zorder=3)
    ax.axhline(0, color=INK, lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("T2 − T1 gap (Δ-Pearson)")
    ax.set_title("E5 — no ensemble gain in either modality\n(held-out target; knockdown confirms no identity leakage)")
    ax.set_ylim(-0.012, 0.010); ax.grid(axis="x", visible=False)
    _save(fig, "E5_crispri")


# ------------------------------------------------------- E7 pocket-matched ---
def fig_e7():
    r = _load("E7_pocket.json")
    if not r:
        return
    plt = _style()
    sig = r["verdict"]["any_signal"]

    # (1) ridge accuracy bars, E3 layout
    acc = r["ridge"]["accuracy"]
    arms = ["T0", "T1", "T2", "T2_permuted"]
    colors = [OI["grey"], OI["sky"], OI["vermillion"], OI["black"]]
    splits = ["target", "scaffold", "cell", "random"]
    labels = ["held-out\ntarget", "held-out\nscaffold", "held-out\ncell", "random\n(ref)"]

    def ci(folds):
        a = np.asarray([v for v in folds if np.isfinite(v)], float)
        return 1.96 * a.std() / np.sqrt(max(len(a), 1)) if len(a) else 0.0
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(splits)); w = 0.2
    for i, (arm, c) in enumerate(zip(arms, colors)):
        vals = [acc[arm][s]["delta_pearson"] for s in splits]
        err = [ci(acc[arm][s]["delta_pearson_folds"]) for s in splits]
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=arm.replace("_", " "),
               yerr=err, capsize=3, ecolor=MUTED, error_kw={"lw": 1.3}, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ-Pearson")
    ax.set_title("E7 — pocket-matched ablation: "
                 + ("T2 adds signal" if sig else "T0 ≈ T1 ≈ T2"))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=14,
              columnspacing=1.2, handletextpad=0.5)
    ax.grid(axis="x", visible=False)
    top = max(acc[a][s]["delta_pearson"] for a in arms for s in splits)
    ax.set_ylim(0, top * 1.25)
    _save(fig, "E7_ablation")

    # (2) forest plot of every T2 gap with its 95% CI and the effect floor
    rg, kg = r["ridge"]["gaps"], r["knn"]["gaps"]
    rows = [(f"ridge · {s}", rg[s]["T2_minus_T1_dp"], OI["vermillion"]) for s in splits]
    rows += [("ridge · target  (T2 − perm)", rg["target"]["T2_minus_perm_dp"], OI["black"]),
             ("kNN held-out target  (T2 − T1)", kg["cen_T2_minus_T1"], OI["blue"]),
             ("kNN held-out target  (T2 − perm)", kg["cen_T2_minus_perm"], OI["black"])]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for i, (lab, c, col) in enumerate(rows[::-1]):
        ax.errorbar(c["mean"], i, xerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]],
                    fmt="o", ms=10, color=col, ecolor=col, elinewidth=2.4, capsize=5, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r_[0] for r_ in rows[::-1]])
    fl = r["verdict"]["effect_floor"]
    ax.axvline(0, color=INK, lw=1.4)
    for s_ in (-fl, fl):
        ax.axvline(s_, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel("gap in Δ-Pearson  (95% CI; dashed = effect floor)")
    ax.set_title("E7 — does pocket motion add anything?")
    ax.grid(axis="y", visible=False)
    _save(fig, "E7_gaps")

    # (3) partial-Mantel scatter
    npz = R / "E7_mantel_scatter.npz"
    if npz.exists():
        d = np.load(npz); m = r["mantel"]
        x_, y_ = d["motion_sim_resid"], d["delta_sim_resid"]
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        ax.scatter(x_, y_, s=7, color=OI["blue"], alpha=0.25, edgecolors="none", zorder=3)
        b, a = np.polyfit(x_, y_, 1)
        xs = np.linspace(x_.min(), x_.max(), 50)
        ax.plot(xs, a + b * xs, color=OI["vermillion"], lw=2, zorder=4)
        ax.set_xlabel("pocket-motion similarity\n(residual | ESM + static pocket)")
        ax.set_ylabel("Δ-deviation similarity\n(residual | ESM + static pocket)")
        ax.set_title("E7 — motion vs response similarity")
        txt = (f"partial r = {m['r_motion_vs_delta_partial_given_esm_static']:+.3f}\n"
               f"Mantel p = {m['mantel_p']:.3g}\n{m['n_targets']} targets")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=14, color=INK, bbox=dict(boxstyle="round", fc="white", ec=GRID))
        _save(fig, "E7_mantel_scatter")


# ------------------------------------------------- E8 MD / E9 single-cell ----
def _forest(rows, title, stem, floor=0.01, xlabel="gap in Δ-Pearson  (95% CI)"):
    """Horizontal CI forest, shared by E8 and E9. rows = [(label, ci, color)]."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(10.5, 0.7 * len(rows) + 2.0))
    for i, (lab, c, col) in enumerate(rows[::-1]):
        ax.errorbar(c["mean"], i, xerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]],
                    fmt="o", ms=10, color=col, ecolor=col, elinewidth=2.4, capsize=5, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.axvline(0, color=INK, lw=1.4)
    for s in (-floor, floor):
        ax.axvline(s, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel(xlabel + "; dashed = effect floor")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, stem)


def fig_e8():
    r = _load("E8_md.json")
    if not r:
        return
    rg, kg = r["ridge"]["gaps"], r["knn"]["gaps"]
    sp = ["target", "scaffold", "cell", "random"]
    rows = [(f"ridge · {s}  (MD − T1)", rg[s]["T2MD_minus_T1"], OI["vermillion"]) for s in sp]
    rows += [("kNN target  (MD − T1)", kg["cen_T2_MD_minus_T1"], OI["blue"]),
             ("kNN target  (MD − PDB)", kg["cen_T2_MD_minus_T2_PDB"], OI["green"]),
             ("kNN target  (MD − perm)", kg["cen_T2_MD_minus_T2_MD_permuted"], OI["black"])]
    _forest(rows, f"E8 — real MD ensembles add nothing ({r['n_targets_md']} targets)",
            "E8_gaps", floor=r["verdict"]["effect_floor"])


def fig_e9():
    r = _load("E9_sciplex.json")
    if not r:
        return
    plt = _style()
    rg, kg = r["ridge"]["gaps"], r["knn"]["gaps"]
    sp = ["target", "scaffold", "cell", "random"]
    rows = [(f"ridge · {s}  (T2 − T1)", rg[s]["T2_minus_T1"], OI["vermillion"]) for s in sp]
    rows += [("kNN target  (T2 − T1)", kg["cen_T2_minus_T1"], OI["blue"]),
             ("kNN target  (T2 − perm)", kg["cen_T2_minus_perm"], OI["black"])]
    _forest(rows, f"E9 — sci-Plex3 single-cell: no ensemble gain",
            "E9_gaps", floor=r["verdict"]["effect_floor"])

    # single-cell screen: best |partial rho| per readout, BH q as colour
    sc = r.get("singlecell_screen_top") or []
    if sc:
        import pandas as pd
        d = pd.DataFrame(sc)
        best = d.loc[d.groupby("readout")["p"].idxmin()].sort_values("partial_rho")
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        cols = [OI["vermillion"] if q < 0.05 else OI["grey"] for q in best["q_bh"]]
        ax.barh(range(len(best)), best["partial_rho"], color=cols, zorder=3)
        ax.set_yticks(range(len(best)))
        ax.set_yticklabels([f"{ro}\n← {fe}" for ro, fe in zip(best["readout"], best["feature"])],
                           fontsize=11)
        ax.axvline(0, color=INK, lw=1.4)
        ax.set_xlabel("partial Spearman ρ  (feature vs readout | dose, cell)")
        ax.set_title("E9 — single-cell readouts vs pocket motion\n(grey = n.s. after BH; across targets)")
        ax.grid(axis="y", visible=False)
        _save(fig, "E9_singlecell_screen")


def fig_e10():
    r = _load("E10_cpa.json")
    if not r:
        return
    plt = _style()
    sp = ["target", "scaffold", "cell", "random"]
    arms = ["OFF", "ESM", "STATIC", "MD", "STATIC+MD"]
    cols = {"OFF": OI["grey"], "ESM": OI["sky"], "STATIC": OI["blue"],
            "MD": OI["orange"], "STATIC+MD": OI["vermillion"]}
    acc = r["accuracy"]

    # Panel A — absolute Δρ per arm, grouped by split (does MD help the real predictor?)
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    w = 0.16
    x = np.arange(len(sp))
    for j, a in enumerate(arms):
        means = [acc[a][s]["delta_pearson_mean"] for s in sp]
        errs = [np.std(acc[a][s]["delta_pearson_folds"]) if acc[a][s]["delta_pearson_folds"] else 0 for s in sp]
        ax.bar(x + (j - 2) * w, means, w, yerr=errs, color=cols[a], label=a,
               capsize=3, error_kw={"elinewidth": 1.4}, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in sp])
    ax.set_ylabel("Δ-Pearson  (mean ± SD over folds)")
    ax.set_xlabel("held-out split")
    ax.set_title(f"E10 — MD as extra input to a chemCPA-style backbone\n"
                 f"({r['n_targets']} MD targets · {r['n_conditions']} conditions · "
                 f"{len(r['seeds'])} seeds)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    _save(fig, "E10_accuracy")

    # Panel B — the gaps that matter: does motion add over a snapshot?
    g = r["gaps"]
    rows = []
    for s in sp:
        rows.append((f"{s}  (STATIC+MD − STATIC)", g[s]["STATICMD_minus_STATIC"], OI["vermillion"]))
    for s in sp:
        rows.append((f"{s}  (MD − permuted MD)", g[s]["MD_minus_MDPERM"], OI["black"]))
    tail = "MD helps over static" if r["verdict"]["md_helps_over_static"] else "motion adds nothing over static"
    _forest(rows, f"E10 — {tail} in a real backbone",
            "E10_gaps", floor=r["verdict"]["effect_floor"])


ALL = {"e0": fig_e0, "e1": fig_e1, "e2": fig_e2, "e3": fig_e3, "e5": fig_e5, "e6": fig_e6,
       "e7": fig_e7, "e8": fig_e8, "e9": fig_e9, "e10": fig_e10}


def build(which=None):
    for k, fn in ALL.items():
        if which and k not in which:
            continue
        try:
            fn()
        except Exception as e:
            print(f"[fig] {k} failed: {e}")


if __name__ == "__main__":
    build(sys.argv[1:] or None)
