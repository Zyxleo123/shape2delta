"""E10 features — GLOBAL (whole-protein) MD dynamics descriptors.

Motivation (see E10 driver): E8 required an MD entry to resolve >=60% of the
*ligand pocket*, which left only 20 LINCS targets — the one underpowered lever
in the whole harness. The frames for 73 LINCS targets are already on disk
(features/md_ensemble/frames/*.npz, fetched by md_ensemble). Here we drop the
pocket constraint and summarise each target's trajectory GLOBALLY — per-residue
CA flexibility, radius-of-gyration fluctuation, backbone RMSD spread, and a
conformational-state count — so the MD block is defined for as many targets as
have any MD at all. Nothing is simulated or downloaded; this only re-reads the
frames and is pure CPU.

This is deliberately a *different* feature than the E7/E8 pocket T2: it is the
cheap, whole-protein flexibility signal (what ATLAS publishes as avg_RMSF /
avg_gyration, generalised to a distribution), the signal most likely to transfer
across unrelated targets if any does. It feeds the chemCPA-style backbone (E10)
as the "extra input" to a real related-work baseline.

  python -m shape2delta.features.md_global build
  python -m shape2delta.features.md_global build I N     (optional sharding)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.features import md_ensemble as MD
from shape2delta.features import pocket_ensemble as PK

OUT = config.FEATURES / "md_global"
FLEX_THRESH = 1.5          # Å: a residue counts as "flexible" above this RMSF
STATE_CUT = 1.5            # Å: hierarchical-clustering distance cut for states
MAX_PAIR_FRAMES = 60       # cap frames used for the O(n^2) state clustering

# the 13-dim global-MD block the E10 backbone consumes
MD_GLOBAL_COLS = ["g_rmsf_mean", "g_rmsf_median", "g_rmsf_p90", "g_rmsf_max",
                  "g_rmsf_std", "g_rmsf_cv", "g_frac_flexible",
                  "g_rg_mean", "g_rg_cv", "g_rmsd_mean", "g_rmsd_max",
                  "g_n_states", "g_state_entropy"]


# ---------------------------------------------------------------- frame IO --
def _ca_array(key: str, acc: str):
    """CA coordinates [nF, R, 3] in UniProt-residue order for one MD entry, plus
    the mapped UniProt indices. Returns None if the entry maps too few residues.
    Units coerced to Å (xtc/mdCATH can arrive in nm)."""
    if not (MD.FRAMES / f"{key}.npz").exists():
        return None
    z, seq = MD._load_entry(key)
    mp = MD._map_to_uniprot(seq, MD._uniprot_seq(acc))        # md res idx -> uniprot (1-based)
    ri, names = z["res_index"], z["name"]
    ca_atoms = np.where(names == "CA")[0]
    rows = [(int(mp[int(ri[a])]), int(a)) for a in ca_atoms if int(ri[a]) in mp]
    if len(rows) < 10:
        return None
    rows.sort()                                              # by UniProt index
    uidx = np.array([u for u, _ in rows])
    atom = np.array([a for _, a in rows])
    X = z["coords"][:, atom, :].astype(np.float64)           # [nF, R, 3]
    # unit guard: bonded CA-CA ~3.8 Å; ~0.38 means the entry is in nm
    d = np.linalg.norm(np.diff(X[0], axis=0), axis=1)
    if np.median(d) < 1.0:
        X = X * 10.0
    return X, uidx


def _superpose(X: np.ndarray) -> np.ndarray:
    """Kabsch-align every frame onto the running mean (2 passes). X: [nF, R, 3]."""
    ref = X[0]
    A = np.empty_like(X)
    for _ in range(2):
        for i in range(len(X)):
            R, pc, qc = PK._kabsch(X[i], ref)                # X[i] -> ref frame
            A[i] = (X[i] - pc) @ R.T + qc
        ref = A.mean(0)
    return A


def _state_stats(A: np.ndarray):
    """Conformational-state count + entropy from pairwise backbone RMSD on a
    capped, evenly spaced set of frames (hierarchical clustering at STATE_CUT)."""
    nF = len(A)
    sel = np.unique(np.linspace(0, nF - 1, min(MAX_PAIR_FRAMES, nF)).astype(int))
    S = A[sel]
    m = len(S)
    if m < 3:
        return 1.0, 0.0
    D = np.zeros((m, m))
    for i in range(m):
        d = np.sqrt(((S[i] - S) ** 2).sum(-1).mean(-1))      # RMSD(i, all)
        D[i] = d
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    lab = fcluster(linkage(squareform(D, checks=False), "average"),
                   t=STATE_CUT, criterion="distance")
    pops = np.bincount(lab)[1:]
    pops = pops[pops > 0] / pops.sum()
    ent = float(-(pops * np.log(pops)).sum() / np.log(m)) if m > 1 else 0.0
    return float(len(pops)), ent


def _global_features(A: np.ndarray) -> dict:
    """Whole-protein dynamics descriptors from superposed CA frames [nF, R, 3]."""
    mu = A.mean(0)                                           # [R,3] mean structure
    rmsf = np.sqrt(((A - mu) ** 2).sum(-1).mean(0))          # [R] per-residue RMSF
    rg = np.sqrt(((A - A.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))   # [nF]
    rmsd = np.sqrt(((A - mu) ** 2).sum(-1).mean(1))          # [nF] RMSD to mean
    n_states, ent = _state_stats(A)
    return {
        "g_rmsf_mean": float(rmsf.mean()), "g_rmsf_median": float(np.median(rmsf)),
        "g_rmsf_p90": float(np.percentile(rmsf, 90)), "g_rmsf_max": float(rmsf.max()),
        "g_rmsf_std": float(rmsf.std()),
        "g_rmsf_cv": float(rmsf.std() / (rmsf.mean() + 1e-9)),
        "g_frac_flexible": float((rmsf > FLEX_THRESH).mean()),
        "g_rg_mean": float(rg.mean()), "g_rg_cv": float(rg.std() / (rg.mean() + 1e-9)),
        "g_rmsd_mean": float(rmsd.mean()), "g_rmsd_max": float(rmsd.max()),
        "g_n_states": n_states, "g_state_entropy": ent,
    }


def profile_target(acc: str, entries: list[str]) -> dict:
    """Pick the MD entry mapping the most CA residues, summarise it globally."""
    best = None
    for key in entries:
        ca = _ca_array(key, acc)
        if ca is None:
            continue
        X, uidx = ca
        if best is None or len(uidx) > best[0]:
            best = (len(uidx), key, X)
    if best is None:
        return {"accession": acc, "ok": False, "reason": "no_mappable_frames"}
    nres, key, X = best
    A = _superpose(X)
    rec = {"accession": acc, "md_entry": key, "md_source": key.split("_")[0],
           "md_n_frames": int(len(X)), "md_n_res_mapped": int(nres),
           "md_seq_cover": float(nres / max(len(MD._uniprot_seq(acc)), 1))}
    rec.update(_global_features(A))
    rec["ok"] = True
    return rec


# ------------------------------------------------------------------ driver --
def build(i: int = 0, n: int = 1) -> dict:
    man = pd.read_csv(MD.OUT / "md_manifest.csv")
    accs = sorted(man["accession"].unique())[i::n]
    OUT.mkdir(parents=True, exist_ok=True)
    sd = OUT / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    outp = sd / f"shard_{i:03d}.jsonl"
    done = {json.loads(l)["accession"] for l in outp.read_text().splitlines() if l.strip()} \
        if outp.exists() else set()
    with outp.open("a") as fh:
        for acc in accs:
            if acc in done:
                continue
            ents = [f"{s}_{e}" for s, e in man.loc[man["accession"] == acc, ["source", "entry"]].values]
            try:
                rec = profile_target(acc, ents)
            except Exception as e:
                rec = {"accession": acc, "ok": False, "error": repr(e)[:300]}
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(f"[md-global] {acc} ok={rec.get('ok')} {rec.get('md_entry')} "
                  f"res={rec.get('md_n_res_mapped')} rmsf={rec.get('g_rmsf_mean')} "
                  f"{rec.get('reason', rec.get('error', ''))}", flush=True)
    return gather()


def gather() -> dict:
    rows = [json.loads(l) for p in sorted((OUT / "shards").glob("shard_*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "md_global_all.csv", index=False)
    ok = df[df.get("ok") == True].copy() if "ok" in df else df.iloc[:0]
    ok.to_csv(OUT / "md_global_features.csv", index=False)
    s = {"generated": datetime.now(timezone.utc).isoformat(),
         "n_targets": int(len(df)), "n_ok": int(len(ok)),
         "by_source": ok.get("md_source", pd.Series(dtype=str)).value_counts().to_dict(),
         "median_seq_cover": float(ok["md_seq_cover"].median()) if len(ok) else None,
         "median_n_frames": float(ok["md_n_frames"].median()) if len(ok) else None,
         "reasons": df.get("reason", pd.Series(dtype=str)).value_counts().to_dict()}
    (OUT / "gather_summary.json").write_text(json.dumps(s, indent=2))
    return s


def load() -> pd.DataFrame:
    return pd.read_csv(OUT / "md_global_features.csv").drop_duplicates("accession")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "gather":
        print(json.dumps(gather(), indent=2))
    elif args and args[0] == "build":
        i, n = (int(args[1]), int(args[2])) if len(args) >= 3 else (0, 1)
        print(json.dumps(build(i, n), indent=2))
    else:
        print(json.dumps(build(), indent=2))
