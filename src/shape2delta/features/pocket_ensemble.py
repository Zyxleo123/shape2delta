"""E7 features — pocket-centric, MATCHED static vs ensemble target descriptors.

Why this exists (vs. features/ensemble.py, the E1–E6 T2): the spec (Deep dive,
"Inputs (3)") asks for per-residue flexibility, state-population summaries and
cryptic-pocket descriptors, and constraint 3 asks that the static-vs-ensemble
ablation hold structure constant. The E1–E6 T2 was 9 *global* CA-RMSF stats
(plus data-availability artefacts like n_conformers / sequence coverage), and T1
came from a different extractor. Here ONE per-conformer extractor is applied:

  T1 (static)   = descriptor(AlphaFold model)                  one conformer
  T2 (ensemble) = T1  ⊕  statistics of descriptor over the experimental
                  conformers (spread, opening beyond static, pocket RMSF,
                  conformational states, cryptic-pocket indicators)

so T1 ⊂ T2 and the only thing T2 adds is motion. No coverage/count features.

Pocket P (the drug site) = UniProt residues within 4.5 Å of a drug-like ligand
(>= 8 heavy atoms, not a buffer/ion/additive) in >= 1 experimental chain (>= 2
if >= 2 holo chains). Targets with no holo chain fall back to the AlphaFold
model's most druggable fpocket pocket.

Stages:
  python -m shape2delta.features.pocket_ensemble manifest   (login node: + downloads list)
  python -m shape2delta.features.pocket_ensemble shard I N  (SLURM array)
  python -m shape2delta.features.pocket_ensemble gather     (CPU)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from shape2delta import config

OUT = config.FEATURES / "pocket_ensemble"
CIF_DIR = config.RAW / "pdb_cif"
AF_DIR = config.RAW_AFDB / "models"
K_CHAINS = 30

EXCLUDE = set("""HOH DOD SO4 PO4 GOL EDO PEG PGE PG4 1PE P6G ACT ACE DMS MPD BME
TRS EPE MES IMD FMT NAG NDG BMA MAN GAL FUC SIA CL NA K MG ZN CA MN FE FE2 CO NI CU
CD IOD BR SR CS LI HG PT AU NO3 SCN AZI CIT TAR MLI SUC FLC NH4 UNX UNL BOG LDA
C8E OLC MYR PLM STE 12P 15P 2PE P33 PE4 PE8 XPE CXS BTB HEZ IPA EOH MOH DIO
""".split())


# ---------------------------------------------------------------- manifest --
def target_accessions() -> list[str]:
    """Single-target accessions (ANY protein class) with LINCS signatures or a
    sci-Plex3 single-target compound, that have ESM and an AF model.
    (E7's first run was enzyme-only; gather() keeps an is_enzyme flag.)"""
    from shape2delta.features import esm_embed
    m = pd.read_csv(config.FEATURES / "lincs_delta" / "sig_meta.csv", usecols=["accession"])
    accs = set(m["accession"].dropna())
    sp = config.FEATURES / "sciplex3" / "target_accessions.txt"
    if sp.exists():
        accs |= set(sp.read_text().split())
    _, esm_accs = esm_embed.load()
    af = {p.stem for p in AF_DIR.glob("*.pdb")}
    return sorted(accs & set(esm_accs) & af)


def manifest(k: int = K_CHAINS) -> dict:
    from shape2delta.data import pdb_ensembles as PE
    accs = target_accessions()
    rows, pdbs = [], set()
    for acc in accs:
        for c in PE.select_chains(acc, k=k):
            rows.append({"accession": acc, "pdb": c["pdb"], "chain": c["chain"],
                         "cov": c["cov"], "segs": json.dumps(c["segs"])})
            pdbs.add(c["pdb"])
    man = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "chain_manifest.csv", index=False)
    missing = sorted(p for p in pdbs if not (CIF_DIR / f"{p}.cif.gz").exists())
    (OUT / "unique_pdbs.txt").write_text("\n".join(sorted(pdbs)))
    (OUT / "missing_pdbs.txt").write_text("\n".join(missing))
    s = {"n_targets_eligible": len(accs), "n_targets_with_chains": int(man["accession"].nunique()),
         "n_chains": len(man), "n_unique_pdbs": len(pdbs), "n_missing_cifs": len(missing)}
    (OUT / "manifest_summary.json").write_text(json.dumps(s, indent=2))
    return s


# ------------------------------------------------------------- structure IO --
def _load_structure(path: Path):
    import gzip
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if str(path).endswith(".pdb"):
            return PDBParser(QUIET=True).get_structure("s", str(path))
        with gzip.open(path, "rt") as fh:
            return MMCIFParser(QUIET=True).get_structure("s", fh)


def _to_uni_fn(segs):
    def f(auth):
        for pb, pe, sb, se in segs:
            if pb <= auth <= pe:
                return int(sb + (auth - pb))
        return None
    return f


def _chain_record(model, chain_id, segs, L):
    """CA[L,3] (UniProt-indexed, NaN gaps), ligand-contact residue set, bfactor[L],
    and the chain's protein atoms (for fpocket)."""
    from Bio.PDB import NeighborSearch
    ca = np.full((L, 3), np.nan)
    bf = np.full(L, np.nan)
    if chain_id not in [c.id for c in model]:
        return None
    ch = model[chain_id]
    to_uni = _to_uni_fn(segs)
    prot_atoms, res_uni = [], {}
    for res in ch:
        if res.id[0] != " ":
            continue
        u = to_uni(res.id[1])
        if u is None or not (1 <= u <= L):
            continue
        res_uni[id(res)] = u
        if "CA" in res:
            ca[u - 1] = res["CA"].coord
            bf[u - 1] = res["CA"].get_bfactor()
        for a in res:
            if a.element != "H":
                prot_atoms.append(a)
    if np.isfinite(ca[:, 0]).sum() < 10:
        return None
    # drug-like ligands anywhere in the model, contacting this chain
    contacts = set()
    lig_atoms = []
    for c in model:
        for res in c:
            if res.id[0] == " " or res.id[0] == "W":
                continue
            rn = res.get_resname().strip()
            if rn in EXCLUDE:
                continue
            heavy = [a for a in res if a.element != "H"]
            if len(heavy) >= 8:
                lig_atoms.extend(heavy)
    if lig_atoms and prot_atoms:
        ns = NeighborSearch(prot_atoms)
        for la in lig_atoms:
            for a in ns.search(la.coord, 4.5):
                u = res_uni.get(id(a.get_parent()))
                if u is not None:
                    contacts.add(u)
    return {"ca": ca, "bf": bf, "contacts": contacts, "atoms": prot_atoms,
            "res_uni": res_uni}


def atom_line(n, fullname, resname, u, xyz, bfac, el) -> str:
    """One PDB ATOM record, chain A, residue number = UniProt index u."""
    name = fullname if len(fullname) == 4 else (f" {fullname:<3s}" if len(fullname) < 4 else fullname[:4])
    return ("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % (n % 100000, name, resname[:3], u, xyz[0], xyz[1], xyz[2], 1.0,
               bfac, el.strip().upper()[:2].rjust(2)))


def _pdb_lines(atoms, res_uni) -> list[str]:
    """Protein-only chain as PDB lines whose residue numbers ARE the UniProt
    indices (so fpocket's pocket atoms map straight back).

    Written by hand on purpose: Bio.PDB.PDBIO.set_structure() wraps a *copy* of a
    Model, so an id()-based Select rejects every residue and silently writes an
    empty file (fpocket then reports 'no atoms' and every fp_* column came back 0).
    Hand-writing also sidesteps multi-char mmCIF chain ids and insertion codes."""
    lines = []
    for a in atoms:
        r = a.get_parent()
        u = res_uni.get(id(r))
        if u is None or not (1 <= u <= 9999):
            continue
        el = (a.element or a.get_name()[0])
        lines.append(atom_line(len(lines) + 1, a.get_fullname()[:4], r.get_resname(), u,
                               [float(c) for c in a.coord], float(a.get_bfactor()), el))
    return lines


def _fpocket(atoms, res_uni, tag):
    return fpocket_lines(_pdb_lines(atoms, res_uni), set(res_uni.values()), tag)


def fpocket_lines(lines, valid, tag):
    """Run fpocket on UniProt-numbered PDB lines; return list of pockets
    {res:set(uniprot idx), volume, drugg, score, hydro}. Raises on a tool
    failure (instead of returning []), so a broken run can't masquerade as
    'no pocket' — the caller records it per conformer."""
    tool = str(Path(sys.executable).parent / "fpocket")
    pockets = []
    with tempfile.TemporaryDirectory() as td:
        pdb = Path(td) / f"{tag}.pdb"
        if len(lines) < 50:
            raise RuntimeError(f"fpocket input {tag}: <50 atoms written")
        pdb.write_text("\n".join(lines + ["TER", "END"]) + "\n")
        p = subprocess.run([tool, "-f", pdb.name], cwd=td, capture_output=True,
                           text=True, timeout=300)
        od = Path(td) / f"{tag}_out"
        info = od / f"{tag}_info.txt"
        if not info.exists():
            raise RuntimeError(f"fpocket {tag} rc={p.returncode}: "
                               f"{(p.stderr or p.stdout).strip()[-160:]}")
        txt = info.read_text()
        blocks = re.split(r"^Pocket (\d+) :", txt, flags=re.M)
        for i in range(1, len(blocks) - 1, 2):
            pid, body = int(blocks[i]), blocks[i + 1]

            def g(k):
                m = re.search(rf"{k}\s*:\s*([-\d.]+)", body)
                return float(m.group(1)) if m else np.nan
            atm = od / "pockets" / f"pocket{pid}_atm.pdb"
            res = set()
            if atm.exists():
                for ln in atm.read_text().splitlines():
                    if ln.startswith("ATOM"):
                        try:
                            u = int(ln[22:26])
                        except ValueError:
                            continue
                        if u in valid:
                            res.add(u)
            pockets.append({"res": res, "volume": g("Volume"),
                            "drugg": g("Druggability Score"), "score": g("Score"),
                            "hydro": g("Hydrophobicity score")})
    return pockets


# -------------------------------------------------------------- descriptors --
def _rg(X):
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())) if len(X) >= 2 else np.nan


def _hull(X):
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(X).volume) if len(X) >= 4 else np.nan
    except Exception:
        return np.nan


def _span(X):
    if len(X) < 2:
        return np.nan
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(X), 1)].mean())


def _conformer_desc(ca, pockets, P, L):
    """Descriptor of ONE conformer at pocket P (UniProt idx, 1-based)."""
    idx = np.array(sorted(P)) - 1
    X = ca[idx]
    X = X[np.isfinite(X[:, 0])]
    present = len(X) / max(len(idx), 1)
    d = {"p_rg": np.nan, "p_hull": np.nan, "p_span": np.nan}
    if present >= 0.6:
        d.update(p_rg=_rg(X), p_hull=_hull(X), p_span=_span(X))
    d["present"] = present
    if pockets is None:                     # fpocket failed on this conformer
        d.update(fp_vol=np.nan, fp_drug=np.nan, fp_score=np.nan, fp_hyd=np.nan,
                 fp_jacc=np.nan, fp_density=np.nan)
        return d
    best, bj = None, 0.0
    for pk in pockets:
        if not pk["res"]:
            continue
        j = len(pk["res"] & P) / len(pk["res"] | P)
        if j > bj:
            best, bj = pk, j
    if best is not None and bj >= 0.1:
        d.update(fp_vol=best["volume"], fp_drug=best["drugg"],
                 fp_score=best["score"], fp_hyd=best["hydro"], fp_jacc=bj)
    else:
        d.update(fp_vol=0.0, fp_drug=0.0, fp_score=0.0, fp_hyd=np.nan, fp_jacc=0.0)
    n_res = np.isfinite(ca[:, 0]).sum()
    d["fp_density"] = 100.0 * len(pockets) / max(n_res, 1)
    return d


def _kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, dd]) @ U.T
    return R, P.mean(0), Q.mean(0)


def _ensemble_motion(cas, P):
    """Pocket/global RMSF, conformational states, pairwise RMSD over experimental
    conformers (CA, UniProt-indexed). Superposition on the 'core' = residues
    present in >=70% of conformers."""
    if len(cas) < 2:                                   # no ensemble to summarise
        return {k: np.nan for k in ("pk_rmsf_mean", "pk_rmsf_p90", "gl_rmsf_med",
                                    "pk_rmsf_ratio", "n_states", "state_entropy",
                                    "rmsd_max", "rmsd_mean")}
    S = np.stack(cas)                                  # n x L x 3
    cover = np.isfinite(S[:, :, 0])
    core = cover.mean(0) >= 0.7
    n = len(S)
    out = {"pk_rmsf_mean": np.nan, "pk_rmsf_p90": np.nan, "gl_rmsf_med": np.nan,
           "pk_rmsf_ratio": np.nan, "n_states": np.nan, "state_entropy": np.nan,
           "rmsd_max": np.nan, "rmsd_mean": np.nan}
    if core.sum() < 10 or n < 2:
        return out
    ref = int((cover & core).sum(1).argmax())
    A = np.full_like(S, np.nan)
    for i in range(n):
        m = cover[i] & core & cover[ref]
        if m.sum() < 4:
            continue
        R, pc, qc = _kabsch(S[i][m], S[ref][m])
        A[i, cover[i]] = (S[i][cover[i]] - pc) @ R.T + qc
    ok = np.isfinite(A[:, :, 0]).any(1)
    A = A[ok]
    if len(A) < 2:
        return out
    mu = np.nanmean(A, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmsf = np.sqrt(np.nanmean(((A - mu) ** 2).sum(-1), 0))
    cnt = np.isfinite(A[:, :, 0]).sum(0)
    rmsf[cnt < 2] = np.nan
    pidx = np.array(sorted(P)) - 1
    pr = rmsf[pidx]
    pr = pr[np.isfinite(pr)]
    gl = rmsf[np.isfinite(rmsf)]
    if len(pr):
        out["pk_rmsf_mean"] = float(pr.mean())
        out["pk_rmsf_p90"] = float(np.percentile(pr, 90))
    if len(gl):
        out["gl_rmsf_med"] = float(np.median(gl))
        if len(pr):
            out["pk_rmsf_ratio"] = float(pr.mean() / (np.median(gl) + 0.1))
    # pairwise RMSD on core residues present in both, after the common superposition
    m = len(A)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            c = np.isfinite(A[i, :, 0]) & np.isfinite(A[j, :, 0]) & core
            D[i, j] = D[j, i] = (np.sqrt(((A[i, c] - A[j, c]) ** 2).sum(-1).mean())
                                 if c.sum() >= 4 else np.nan)
    D = np.nan_to_num(D, nan=np.nanmax(D) if np.isfinite(D).any() else 0.0)
    iu = np.triu_indices(m, 1)
    out["rmsd_max"] = float(D[iu].max())
    out["rmsd_mean"] = float(D[iu].mean())
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1.5, criterion="distance")
    pops = np.bincount(lab)[1:] / len(lab)
    pops = pops[pops > 0]
    out["n_states"] = float(len(pops))
    out["state_entropy"] = float(-(pops * np.log(pops)).sum() / np.log(len(lab))) if len(lab) > 1 else 0.0
    return out


def _seqlen(acc):
    fa = config.RAW_UNIPROT / "fasta" / f"{acc}.fasta"
    if fa.exists():
        s = "".join(l.strip() for l in fa.read_text().splitlines() if l and not l.startswith(">"))
        if s:
            return len(s)
    return None


def profile_target(acc: str, rows: list[dict]) -> dict:
    af_path = AF_DIR / f"{acc}.pdb"
    af_model = next(iter(_load_structure(af_path)))
    af_chain = next(iter(af_model)).id
    L = _seqlen(acc) or max(r.id[1] for r in af_model[af_chain])
    af = _chain_record(af_model, af_chain, [[1, L, 1, L]], L)
    af_pockets = _fpocket(af["atoms"], af["res_uni"], "af")

    exp = []
    cache = {}
    for r in rows:
        cif = CIF_DIR / f"{r['pdb']}.cif.gz"
        if not cif.exists():
            continue
        try:
            if r["pdb"] not in cache:
                cache = {r["pdb"]: next(iter(_load_structure(cif)))}
            rec = _chain_record(cache[r["pdb"]], r["chain"], json.loads(r["segs"]), L)
        except Exception:
            rec = None
        if rec is None:
            continue
        rec["tag"] = f"{r['pdb']}_{r['chain']}"
        exp.append(rec)

    # pocket definition
    holo = [e for e in exp if e["contacts"]]
    if holo:
        from collections import Counter
        cnt = Counter(u for e in holo for u in e["contacts"])
        need = 2 if len(holo) >= 2 else 1
        P = {u for u, c in cnt.items() if c >= need} or set(cnt)
        src = "ligand"
    else:
        cand = [p for p in af_pockets if p["res"]]
        cand.sort(key=lambda p: -(p["drugg"] if np.isfinite(p["drugg"]) else -1))
        P = set(cand[0]["res"]) if cand else set()
        src = "af_fpocket"
    rec = {"accession": acc, "seqlen": L, "pocket_source": src, "pocket_size": len(P),
           "n_exp_conformers": len(exp), "n_holo": len(holo),
           "pocket_res": sorted(int(u) for u in P)}      # reused by the MD tier
    if not P:
        rec["ok"] = False
        return rec

    # T1: static descriptor of the AlphaFold conformer
    d_af = _conformer_desc(af["ca"], af_pockets, P, L)
    pidx = np.array(sorted(P)) - 1
    rec.update({f"s_{k}": v for k, v in d_af.items() if k != "present"})
    rec["s_p_plddt"] = float(np.nanmean(af["bf"][pidx]))

    # T2 additions: the same descriptor over every experimental conformer
    descs, n_fp_fail = [], 0
    for e in exp:
        idx_p = np.array(sorted(P)) - 1
        if np.isfinite(e["ca"][idx_p, 0]).mean() < 0.6:
            continue                        # pocket not resolved: skip fpocket
        try:
            pk = _fpocket(e["atoms"], e["res_uni"], e["tag"])
        except Exception:
            pk, n_fp_fail = None, n_fp_fail + 1
        dd = _conformer_desc(e["ca"], pk, P, L)
        dd["tag"], dd["holo"] = e["tag"], bool(e["contacts"])
        descs.append(dd)
    rec["n_conformers_used"] = len(descs)
    rec["n_fpocket_fail"] = n_fp_fail
    rec["conf"] = descs                     # per-conformer rows (re-analysable)
    # within-experiment cryptic flag: apo chains undruggable, holo chains druggable.
    # (The AF-vs-experiment e_cryptic below is partly induced fit: holo chains are
    # ligand-stripped, pre-formed pockets. This variant compares like with like,
    # but exists only for targets that have both apo and holo chains.)
    da = [d["fp_drug"] for d in descs if not d["holo"] and np.isfinite(d["fp_drug"])]
    dh = [d["fp_drug"] for d in descs if d["holo"] and np.isfinite(d["fp_drug"])]
    rec["x_n_apo"], rec["x_n_holo_used"] = len(da), len(dh)
    if da and dh:
        rec["x_drug_apo_max"], rec["x_drug_holo_max"] = float(max(da)), float(max(dh))
        rec["x_cryptic_apo_holo"] = float(max(da) < 0.5 <= max(dh))
    rec.update(t2_summary(descs, d_af, [e["ca"] for e in exp], P))
    rec["ok"] = True
    return rec


def t2_summary(descs: list[dict], d_af: dict, cas: list, P: set) -> dict:
    """The T2 block: motion of the per-conformer pocket descriptor around the
    static (AF) one, plus pocket/global RMSF and state statistics. SHARED by the
    PDB-conformer tier (E7) and the MD-frame tier (E8) so the two are matched.
    Conformers whose fpocket run failed / was skipped carry NaN fp_* and are
    ignored by the fp summaries (geometry still uses them)."""
    rec = {}
    if len(descs) >= 2:
        D = pd.DataFrame(descs)
        for k in ["p_rg", "p_hull", "p_span", "fp_vol", "fp_drug"]:
            v = D[k].to_numpy(float)
            if not np.isfinite(v).any():
                continue
            rec[f"e_{k}_cv"] = float(np.nanstd(v) / (abs(np.nanmean(v)) + 1e-6))
            rec[f"e_{k}_open"] = float(np.nanmax(v) - d_af[k]) if np.isfinite(d_af[k]) else np.nan
        dr = D["fp_drug"].to_numpy(float)
        dr = dr[np.isfinite(dr)]
        jc = D["fp_jacc"].to_numpy(float)
        jc = jc[np.isfinite(jc)]
        if len(dr):
            rec["e_frac_druggable"] = float(np.mean(dr >= 0.5))
            rec["e_cryptic"] = float((d_af["fp_drug"] < 0.5) and (dr.max() >= 0.5))
        if len(jc):
            rec["e_frac_pocket_found"] = float(np.mean(jc >= 0.1))
    rec.update({f"e_{k}": v for k, v in _ensemble_motion(cas, P).items()})
    return rec


# ------------------------------------------------------------------ driver --
def run_shard(i: int, n: int) -> None:
    man = pd.read_csv(OUT / "chain_manifest.csv")
    accs = sorted(man["accession"].unique())[i::n]
    sd = OUT / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    outp = sd / f"shard_{i:03d}.jsonl"
    done = set()
    if outp.exists():                                   # resumable
        done = {json.loads(l)["accession"] for l in outp.read_text().splitlines() if l.strip()}
    with outp.open("a") as fh:
        for acc in accs:
            if acc in done:
                continue
            rows = man[man["accession"] == acc].to_dict("records")
            try:
                rec = profile_target(acc, rows)
            except Exception as e:
                rec = {"accession": acc, "ok": False, "error": repr(e)[:200]}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(f"[pocket] {acc} ok={rec.get('ok')} src={rec.get('pocket_source')} "
                  f"P={rec.get('pocket_size')} conf={rec.get('n_conformers_used')}", flush=True)


T1_COLS = ["s_p_rg", "s_p_hull", "s_p_span", "s_fp_vol", "s_fp_drug", "s_fp_score",
           "s_fp_hyd", "s_fp_jacc", "s_fp_density", "s_p_plddt"]
T2_COLS = ([f"e_{k}_{s}" for k in ["p_rg", "p_hull", "p_span", "fp_vol", "fp_drug"]
            for s in ("cv", "open")]
           + ["e_frac_druggable", "e_cryptic", "e_frac_pocket_found",
              "e_pk_rmsf_mean", "e_pk_rmsf_p90", "e_gl_rmsf_med", "e_pk_rmsf_ratio",
              "e_n_states", "e_state_entropy", "e_rmsd_max", "e_rmsd_mean"])


def gather() -> dict:
    rows, conf = [], []
    for p in sorted((OUT / "shards").glob("shard_*.jsonl")):
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            for c in r.pop("conf", None) or []:
                conf.append({"accession": r["accession"], **c})
            rows.append(r)
    from shape2delta.data import uniprot as U
    df = pd.DataFrame(rows)
    enz = U.enzyme_accessions()
    df["is_enzyme"] = df["accession"].isin(enz)
    if "pocket_res" in df:
        json.dump({a: p for a, p in zip(df["accession"], df["pocket_res"]) if isinstance(p, list)},
                  open(OUT / "pocket_residues.json", "w"))
        df = df.drop(columns=["pocket_res"])
    df.to_csv(OUT / "pocket_features_all.csv", index=False)
    pd.DataFrame(conf).to_csv(OUT / "pocket_conformers.csv", index=False)
    ok = df[(df["ok"] == True) & (df.get("n_conformers_used", 0).fillna(0) >= 3)].copy()
    X_COLS = ["x_n_apo", "x_n_holo_used", "x_drug_apo_max", "x_drug_holo_max",
              "x_cryptic_apo_holo", "n_fpocket_fail"]
    for c in T1_COLS + T2_COLS + X_COLS:
        if c not in ok:
            ok[c] = np.nan
    ok[["accession", "is_enzyme", "pocket_source", "pocket_size", "n_exp_conformers", "n_holo",
        "n_conformers_used"] + T1_COLS + T2_COLS + X_COLS].to_csv(
        OUT / "pocket_features.csv", index=False)
    nconf_fp = int(sum(np.isfinite(c.get("fp_drug", np.nan)) for c in conf))
    s = {"n_profiled": int(len(df)), "n_ok": int((df["ok"] == True).sum()),
         "n_usable_ge3_conformers": int(len(ok)),
         "pocket_source": ok["pocket_source"].value_counts().to_dict(),
         "median_conformers": float(ok["n_conformers_used"].median()),
         "n_conformers_total": len(conf), "n_conformers_fpocket_ok": nconf_fp,
         "frac_static_pocket_matched": float((ok["s_fp_jacc"] >= 0.1).mean()),
         "frac_cryptic_af_vs_exp": float(ok["e_cryptic"].mean()),
         "n_with_apo_and_holo": int(ok["x_cryptic_apo_holo"].notna().sum()),
         "frac_cryptic_apo_holo": float(ok["x_cryptic_apo_holo"].mean()),
         "errors": df.get("error", pd.Series(dtype=str)).dropna().str[:60].value_counts().head(5).to_dict(),
         "t1_dims": len(T1_COLS), "t2_extra_dims": len(T2_COLS)}
    (OUT / "gather_summary.json").write_text(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "manifest":
        print(json.dumps(manifest(), indent=2))
    elif cmd == "shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "gather":
        print(json.dumps(gather(), indent=2))
