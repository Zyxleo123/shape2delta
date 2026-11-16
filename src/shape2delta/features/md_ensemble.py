"""E8 features — the T2 descriptor computed over REAL MD frames (ATLAS, mdCATH).

Same pocket P (features/pocket_ensemble.py, ligand-defined), same static T1 (the
AF-model pocket descriptor), and the SAME T2 summariser (pocket_ensemble.
t2_summary) — only the conformer source changes: experimental PDB chains (E7)
-> molecular-dynamics frames (here). So E8 asks "does real dynamics carry what
the PDB conformers did not?", with everything else held fixed.

Sources (both precomputed; nothing is simulated here):
  ATLAS  : 3 × 100 ns, 300 K, CHARMM36m, full PDB chain (MODELLER-completed).
           API zip -> topology PDB + .xtc (decoded by data/xtc.py, validated
           against ATLAS's published RMSF: r >= 0.999).
  mdCATH : CATH domains, 5 replicas at 320 K (lowest of 320–450 K), ~0.5 µs.
           HF h5 (~1.5 GB/domain): streamed, sampled frames kept, h5 deleted.
Residues are mapped to UniProt by sequence alignment (numbering differs by source).
An MD entry is used for a target only if it resolves >= 60% of the pocket.

Stages (all SLURM CPU; compute nodes have internet):
  python -m shape2delta.features.md_ensemble manifest
  python -m shape2delta.features.md_ensemble fetch I N     (download + extract frames)
  python -m shape2delta.features.md_ensemble profile I N   (needs pocket_residues.json)
  python -m shape2delta.features.md_ensemble gather
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from shape2delta import config
from shape2delta.features import pocket_ensemble as PK

OUT = config.FEATURES / "md_ensemble"
FRAMES = OUT / "frames"
TMP = OUT / "tmp"
ATLAS_URL = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{}"
MDCATH_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{}.h5"
MDCATH_TEMP = "320"
FRAMES_PER_REP_MDCATH = 60          # x 5 replicas = 300 frames
ATLAS_EVERY = 10                    # 1001 frames/replica -> ~100 x 3 = 300 frames
FP_FRAMES = 30                      # fpocket on this many evenly spaced frames
MIN_POCKET_COVER = 0.6

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
       "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M"}


# ---------------------------------------------------------------- manifest --
def candidate_accessions() -> set[str]:
    m = pd.read_csv(config.FEATURES / "lincs_delta" / "sig_meta.csv", usecols=["accession"])
    accs = set(m["accession"].dropna())
    sp = config.FEATURES / "sciplex3" / "target_accessions.txt"
    if sp.exists():
        accs |= set(sp.read_text().split())
    return accs


def manifest() -> dict:
    from shape2delta.data import ensembles as EN
    accs = candidate_accessions()
    at = pd.read_csv(sorted(config.RAW_ATLAS.glob("ATLAS_info*.tsv"))[-1], sep="\t", dtype=str)
    rows = [{"accession": u, "source": "atlas", "entry": p}
            for p, u in zip(at["PDB"], at["UniProt"]) if u in accs]
    p2u = EN.sifts_pdbchain_to_uni()
    for d in (config.RAW_MDCATH / "mdCATH_domains.txt").read_text().split():
        u = p2u.get((d[:4].lower(), d[4]))
        if u in accs:
            rows.append({"accession": u, "source": "mdcath", "entry": d})
    man = pd.DataFrame(rows).drop_duplicates()
    OUT.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "md_manifest.csv", index=False)
    s = {"n_entries": len(man), "n_targets": int(man["accession"].nunique()),
         "by_source": man["source"].value_counts().to_dict(),
         "targets_by_source": man.groupby("source")["accession"].nunique().to_dict()}
    (OUT / "manifest_summary.json").write_text(json.dumps(s, indent=2))
    return s


# ------------------------------------------------------------ fetch/extract --
def _curl(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "10",
                        "--max-time", "3600", "-o", str(path), url],
                       capture_output=True, text=True)
    if r.returncode != 0 or not path.exists():
        raise RuntimeError(f"download failed rc={r.returncode}: {r.stderr.strip()[-200:]}")


def _parse_pdb_atoms(text: str) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            name = ln[12:16].strip()
            el = ln[76:78].strip() or name.lstrip("0123456789")[0]
            rows.append({"name": name, "resname": ln[17:20].strip(), "chain": ln[21],
                         "resid": int(ln[22:26]), "icode": ln[26].strip(), "element": el.upper()})
    return pd.DataFrame(rows)


def _save(key, atoms: pd.DataFrame, coords: np.ndarray, rep: np.ndarray, meta: dict):
    heavy = (atoms["element"] != "H").to_numpy()
    a = atoms[heavy].reset_index(drop=True)
    # residue key -> sequential residue index (order of appearance)
    rk = a["chain"].astype(str) + ":" + a["resid"].astype(str) + a["icode"].astype(str)
    ridx, _ = pd.factorize(rk)
    FRAMES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FRAMES / f"{key}.npz", coords=coords[:, heavy].astype(np.float32),
                        name=a["name"].to_numpy(str), resname=a["resname"].to_numpy(str),
                        res_index=ridx.astype(np.int32), element=a["element"].to_numpy(str),
                        rep=rep.astype(np.int16), meta=json.dumps(meta))


def _fetch_atlas(entry: str) -> dict:
    from shape2delta.data.xtc import read_xtc
    td = TMP / f"atlas_{entry}"
    zp = td / "a.zip"
    _curl(ATLAS_URL.format(entry), zp)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        top = z.read(f"{entry}.pdb").decode()
        for r in (1, 2, 3):
            z.extract(f"{entry}_R{r}.xtc", td)
    atoms = _parse_pdb_atoms(top)
    frames, rep = [], []
    for r in (1, 2, 3):
        for _, x in read_xtc(td / f"{entry}_R{r}.xtc", every=ATLAS_EVERY):
            if len(x) != len(atoms):
                raise ValueError(f"xtc atoms {len(x)} != topology {len(atoms)}")
            frames.append(x); rep.append(r)
    _save(f"atlas_{entry}", atoms, np.stack(frames), np.array(rep),
          {"source": "atlas", "entry": entry, "temp_K": 300, "files": names[:3]})
    shutil.rmtree(td, ignore_errors=True)
    return {"n_frames": len(frames), "n_atoms": len(atoms)}


def _fetch_mdcath(dom: str) -> dict:
    import h5py
    td = TMP / f"mdcath_{dom}"
    hp = td / f"{dom}.h5"
    _curl(MDCATH_URL.format(dom), hp)
    with h5py.File(hp, "r") as f:
        g = f[dom]
        keys = list(g.keys())
        if "pdbProteinAtoms" not in g:
            raise KeyError(f"no pdbProteinAtoms in {dom}; keys={keys}")
        raw = g["pdbProteinAtoms"][()]
        atoms = _parse_pdb_atoms(raw.decode() if isinstance(raw, bytes) else str(raw))
        T = g[MDCATH_TEMP]
        frames, rep = [], []
        for r in sorted(T.keys(), key=lambda s: int(s)):
            ds = T[r]["coords"]
            F = ds.shape[0]
            if ds.shape[1] != len(atoms):
                raise ValueError(f"coords atoms {ds.shape[1]} != pdb atoms {len(atoms)}")
            idx = np.unique(np.linspace(0, F - 1, min(FRAMES_PER_REP_MDCATH, F)).astype(int))
            x = ds[idx.tolist()]
            frames.append(x); rep += [int(r)] * len(idx)
    X = np.concatenate(frames)
    # unit check: bonded CA–CA ≈ 3.8 Å (if ~0.38 the file is in nm)
    ca = np.where(atoms["name"].to_numpy() == "CA")[0]
    d = np.linalg.norm(np.diff(X[0, ca], axis=0), axis=1)
    med = float(np.median(d))
    if med < 1.0:
        X = X * 10.0
    _save(f"mdcath_{dom}", atoms, X, np.array(rep),
          {"source": "mdcath", "entry": dom, "temp_K": int(MDCATH_TEMP), "h5_keys": keys,
           "ca_ca_median_raw": med})
    shutil.rmtree(td, ignore_errors=True)
    return {"n_frames": len(X), "n_atoms": len(atoms), "ca_ca_med": med}


def run_fetch(i: int, n: int) -> None:
    man = pd.read_csv(OUT / "md_manifest.csv")
    rows = man.iloc[i::n].to_dict("records")
    log = OUT / "fetch_logs"
    log.mkdir(parents=True, exist_ok=True)
    with (log / f"fetch_{i:03d}.jsonl").open("a") as fh:
        for r in rows:
            key = f"{r['source']}_{r['entry']}"
            if (FRAMES / f"{key}.npz").exists():
                continue                                # resumable
            try:
                info = (_fetch_atlas if r["source"] == "atlas" else _fetch_mdcath)(r["entry"])
                info.update(ok=True)
            except Exception as e:
                info = {"ok": False, "error": repr(e)[:300]}
                shutil.rmtree(TMP / key, ignore_errors=True)
            fh.write(json.dumps({"key": key, **info}) + "\n"); fh.flush()
            print(f"[md-fetch] {key} {info}", flush=True)


# ------------------------------------------------------------------ profile --
def _uniprot_seq(acc):
    fa = config.RAW_UNIPROT / "fasta" / f"{acc}.fasta"
    return "".join(l.strip() for l in fa.read_text().splitlines() if l and not l.startswith(">"))


def _map_to_uniprot(md_seq: str, uni_seq: str) -> dict[int, int]:
    """MD residue index (0-based) -> UniProt index (1-based), via local alignment;
    only identical aligned residues are mapped."""
    from Bio import Align
    al = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-1,
                               open_gap_score=-5, extend_gap_score=-0.5)
    a = al.align(md_seq, uni_seq)[0]
    m = {}
    for (s0, s1), (t0, t1) in zip(*a.aligned):
        for k in range(s1 - s0):
            if md_seq[s0 + k] == uni_seq[t0 + k]:
                m[s0 + k] = t0 + k + 1
    return m


def _load_entry(key):
    z = np.load(FRAMES / f"{key}.npz", allow_pickle=False)
    ri = z["res_index"]
    nres = int(ri.max()) + 1
    resname = np.array([""] * nres, dtype=object)
    resname[ri] = z["resname"]
    seq = "".join(AA3.get(r, "X") for r in resname)
    return z, seq


def profile_target(acc: str, entries: list[str], P: set, s_row: dict) -> dict:
    uni = _uniprot_seq(acc)
    L = len(uni)
    best = None
    for key in entries:
        if not (FRAMES / f"{key}.npz").exists():
            continue
        z, seq = _load_entry(key)
        mp = _map_to_uniprot(seq, uni)
        ri, names = z["res_index"], z["name"]
        has_ca = {int(r) for r, nm in zip(ri, names) if nm == "CA"}
        mapped_u = {mp[r] for r in has_ca if r in mp}
        cov = len(P & mapped_u) / max(len(P), 1)
        if best is None or cov > best[1] + 1e-9:
            best = (key, cov, mp)
    rec = {"accession": acc, "md_candidates": len(entries)}
    if best is None:
        rec.update(ok=False, reason="no_frames")
        return rec
    key, cov, mp = best
    rec.update(md_entry=key, md_source=key.split("_")[0], md_pocket_cover=cov)
    if cov < MIN_POCKET_COVER:
        rec.update(ok=False, reason="pocket_not_covered")
        return rec
    z, _ = _load_entry(key)
    X, ri, names = z["coords"], z["res_index"], z["name"]
    resn, elem = z["resname"], z["element"]
    uidx = np.array([mp.get(int(r), -1) for r in ri])
    keep = uidx > 0
    ca_atoms = np.where(keep & (names == "CA"))[0]
    nF = len(X)
    fp_set = set(np.unique(np.linspace(0, nF - 1, min(FP_FRAMES, nF)).astype(int)).tolist())
    d_af = {k: s_row.get(f"s_{k}", np.nan) for k in ["p_rg", "p_hull", "p_span", "fp_vol", "fp_drug"]}
    descs, cas, n_fail = [], [], 0
    atom_ids = np.where(keep)[0]
    for f in range(nF):
        ca = np.full((L, 3), np.nan)
        ca[uidx[ca_atoms] - 1] = X[f, ca_atoms]
        cas.append(ca)
        pk = None
        if f in fp_set:
            lines = [PK.atom_line(n + 1, str(names[a]), str(resn[a]), int(uidx[a]),
                                  X[f, a].tolist(), 0.0, str(elem[a]))
                     for n, a in enumerate(atom_ids)]
            try:
                pk = PK.fpocket_lines(lines, set(uidx[keep].tolist()), f"md{f}")
            except Exception:
                n_fail += 1
        descs.append(PK._conformer_desc(ca, pk, P, L))
    rec.update(PK.t2_summary(descs, d_af, cas, P))
    meta = json.loads(str(z["meta"]))
    rec.update(md_n_frames=nF, md_temp_K=meta.get("temp_K"), md_fp_frames=len(fp_set),
               md_fpocket_fail=n_fail, ok=True)
    return rec


def run_profile(i: int, n: int) -> None:
    man = pd.read_csv(OUT / "md_manifest.csv")
    pres = json.loads((PK.OUT / "pocket_residues.json").read_text())
    pf = pd.read_csv(PK.OUT / "pocket_features_all.csv").set_index("accession")
    accs = sorted(a for a in man["accession"].unique() if a in pres and a in pf.index)[i::n]
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
                rec = profile_target(acc, ents, set(pres[acc]), pf.loc[acc].to_dict())
            except Exception as e:
                rec = {"accession": acc, "ok": False, "error": repr(e)[:300]}
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(f"[md] {acc} ok={rec.get('ok')} {rec.get('md_entry')} "
                  f"cover={rec.get('md_pocket_cover')} {rec.get('reason', '')}", flush=True)


def gather() -> dict:
    rows = [json.loads(l) for p in sorted((OUT / "shards").glob("shard_*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "md_features_all.csv", index=False)
    ok = df[df["ok"] == True].copy()
    ok.to_csv(OUT / "md_features.csv", index=False)
    s = {"n_targets_profiled": len(df), "n_ok": len(ok),
         "reasons": df.get("reason", pd.Series(dtype=str)).value_counts().to_dict(),
         "errors": df.get("error", pd.Series(dtype=str)).dropna().str[:60].value_counts().head(5).to_dict(),
         "by_source": ok.get("md_source", pd.Series(dtype=str)).value_counts().to_dict(),
         "median_pocket_cover": float(ok["md_pocket_cover"].median()) if len(ok) else None}
    (OUT / "gather_summary.json").write_text(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "manifest":
        print(json.dumps(manifest(), indent=2))
    elif cmd == "fetch":
        run_fetch(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "profile":
        run_profile(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "gather":
        print(json.dumps(gather(), indent=2))
