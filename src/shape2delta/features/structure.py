"""T1 static-structure features from the AlphaFold model (one conformer).

Per target: pLDDT summary (AF stores pLDDT in the B-factor column), radius of
gyration, secondary-structure fractions (mkdssp, phi/psi fallback), PAE summary
(paeDoc json), and fpocket top-pocket descriptors. Fixed-length; missing → NaN.

  SLURM (CPU): python -m shape2delta.features.structure
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from shape2delta import config

_BIN = Path(sys.executable).parent  # the conda env's bin/ (fpocket, mkdssp live here)


def _tool(name: str) -> str:
    return shutil.which(name) or str(_BIN / name)

AFDIR = config.RAW_AFDB / "models"
OUT = config.FEATURES / "t1_structure"

T1_KEYS = ["plddt_mean", "plddt_std", "plddt_frac_vhigh", "plddt_frac_low",
           "rg", "n_res", "ss_helix", "ss_sheet", "ss_coil",
           "pae_mean", "pae_frac_conf", "fpocket_n", "fpocket_top_score",
           "fpocket_top_drugg", "fpocket_top_volume"]


def _ca_atoms(pdb: Path):
    from Bio.PDB import PDBParser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = PDBParser(QUIET=True).get_structure("s", str(pdb))
    model = next(iter(st))
    cas = []
    for ch in model:
        for res in ch:
            if res.id[0] == " " and "CA" in res:
                cas.append(res["CA"])
    return cas


def _plddt_rg(pdb: Path) -> dict:
    cas = _ca_atoms(pdb)
    if not cas:
        return {}
    b = np.array([a.get_bfactor() for a in cas], float)
    xyz = np.array([a.coord for a in cas], float)
    c = xyz.mean(0)
    rg = float(np.sqrt(((xyz - c) ** 2).sum(1).mean()))
    return {"plddt_mean": float(b.mean()), "plddt_std": float(b.std()),
            "plddt_frac_vhigh": float((b > 90).mean()),
            "plddt_frac_low": float((b < 50).mean()),
            "rg": rg, "n_res": float(len(cas))}


def _ss_fractions(pdb: Path) -> dict:
    """Coarse SS fractions from backbone phi/psi (PPBuilder; no external tool).
    mkdssp v4 needs a CCD data dir it can't locate here, so we use geometry."""
    import math
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import PPBuilder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = PDBParser(QUIET=True).get_structure("s", str(pdb))
        h = s = n = 0
        for pp in PPBuilder().build_peptides(next(iter(st))):
            for phi, psi in pp.get_phi_psi_list():
                if phi is None or psi is None:
                    continue
                phid, psid = math.degrees(phi), math.degrees(psi)
                n += 1
                if -160 < phid < -30 and -80 < psid < 20:
                    h += 1                       # right-handed alpha region
                elif -180 < phid < -40 and 90 < psid < 180:
                    s += 1                       # beta region
        if n:
            return {"ss_helix": h / n, "ss_sheet": s / n, "ss_coil": 1 - (h + s) / n}
    except Exception:
        pass
    return {}


def _pae_features(acc: str) -> dict:
    p = AFDIR / f"{acc}.pae.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        d = d[0] if isinstance(d, list) else d
        pae = np.array(d.get("predicted_aligned_error") or d.get("pae"), float)
        if pae.ndim != 2:
            return {}
        off = pae[~np.eye(pae.shape[0], dtype=bool)]
        return {"pae_mean": float(off.mean()), "pae_frac_conf": float((off < 5).mean())}
    except Exception:
        return {}


def _fpocket_features(pdb: Path) -> dict:
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / f"{pdb.stem}.pdb"
            tmp.write_bytes(pdb.read_bytes())
            subprocess.run([_tool("fpocket"), "-f", tmp.name], cwd=td, capture_output=True,
                          text=True, timeout=180)
            info = Path(td) / f"{pdb.stem}_out" / f"{pdb.stem}_info.txt"
            if not info.exists():
                return {"fpocket_n": 0.0}
            txt = info.read_text()
            import re
            n = len(re.findall(r"^Pocket \d+", txt, re.M))
            # top pocket = Pocket 1 block
            blk = txt.split("Pocket 1 :", 1)
            top = blk[1].split("Pocket 2 :", 1)[0] if len(blk) > 1 else ""
            def g(key):
                m = re.search(rf"{key}\s*:\s*([-\d.]+)", top)
                return float(m.group(1)) if m else float("nan")
            return {"fpocket_n": float(n), "fpocket_top_score": g("Score"),
                    "fpocket_top_drugg": g("Druggability Score"),
                    "fpocket_top_volume": g("Volume")}
    except Exception:
        return {"fpocket_n": float("nan")}


def features_for(acc: str) -> dict:
    pdb = AFDIR / f"{acc}.pdb"
    out = {"accession": acc}
    if not pdb.exists():
        return out
    out.update(_plddt_rg(pdb))
    out.update(_ss_fractions(pdb))
    out.update(_pae_features(acc))
    out.update(_fpocket_features(pdb))
    return out


SHARD_DIR = OUT / "shards"


def run_shard(shard: int, nshard: int) -> None:
    import pandas as pd
    accs = sorted(p.stem for p in AFDIR.glob("*.pdb"))
    mine = accs[shard::nshard]
    rows = [features_for(a) for a in mine]
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SHARD_DIR / f"shard_{shard:03d}.csv", index=False)
    print(f"[t1] shard {shard}/{nshard}: {len(rows)} targets")


def gather() -> dict:
    import pandas as pd
    shards = sorted(SHARD_DIR.glob("shard_*.csv"))
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True) if shards else pd.DataFrame()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "t1_structure.csv", index=False)
    cov = {k: int(df[k].notna().sum()) for k in T1_KEYS if k in df.columns}
    summ = {"n_targets": len(df), "coverage": cov}
    (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def build() -> dict:
    import pandas as pd
    accs = sorted(p.stem for p in AFDIR.glob("*.pdb"))
    df = pd.DataFrame([features_for(a) for a in accs])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "t1_structure.csv", index=False)
    cov = {k: int(df[k].notna().sum()) for k in T1_KEYS if k in df.columns}
    summ = {"n_targets": len(df), "coverage": cov}
    (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "gather":
        print(json.dumps(gather(), indent=2))
    elif len(sys.argv) >= 3:
        run_shard(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print(json.dumps(build(), indent=2))
