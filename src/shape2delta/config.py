"""Central path + knob config for shape2delta, read from the environment.

Defaults mirror slurm/preamble.sh exactly so Python code behaves identically
whether launched from a SLURM job (env-file sourced) or a bare shell.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("S2D_ROOT", "/zfsauton2/home/yixiz/shape2delta"))
DATA = Path(os.environ.get("S2D_DATA", "/zfsauton/scratch/yixiz/shape2delta"))
RAW = Path(os.environ.get("S2D_RAW", str(DATA / "raw")))
INTERIM = Path(os.environ.get("S2D_INTERIM", str(DATA / "interim")))
FEATURES = Path(os.environ.get("S2D_FEATURES", str(DATA / "features")))
RESULTS = Path(os.environ.get("S2D_RESULTS", str(ROOT / "results")))

# knobs
MIN_T2_TARGETS = int(os.environ.get("S2D_MIN_T2_TARGETS", "40"))
MIN_PDB_CHAINS = int(os.environ.get("S2D_MIN_PDB_CHAINS", "5"))
NFOLDS = int(os.environ.get("S2D_NFOLDS", "5"))
NBOOT = int(os.environ.get("S2D_NBOOT", "2000"))
SEED = int(os.environ.get("S2D_SEED", "0"))

# raw-source subdirs
RAW_SCIPLEX = RAW / "sciplex3"
RAW_LINCS = RAW / "lincs"
RAW_DRH = RAW / "drh"
RAW_CHEMBL = RAW / "chembl"
RAW_UNIPROT = RAW / "uniprot"
RAW_ATLAS = RAW / "atlas"
RAW_MDCATH = RAW / "mdcath"
RAW_GPCRMD = RAW / "gpcrmd"
RAW_SIFTS = RAW / "pdb_sifts"
RAW_AFDB = RAW / "alphafold"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, FEATURES, RESULTS,
              RAW_SCIPLEX, RAW_LINCS, RAW_DRH, RAW_CHEMBL, RAW_UNIPROT,
              RAW_ATLAS, RAW_MDCATH, RAW_GPCRMD, RAW_SIFTS, RAW_AFDB):
        p.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"ROOT={ROOT}\nDATA={DATA}\nRESULTS={RESULTS}\n"
        f"MIN_T2_TARGETS={MIN_T2_TARGETS} MIN_PDB_CHAINS={MIN_PDB_CHAINS} "
        f"NFOLDS={NFOLDS} NBOOT={NBOOT} SEED={SEED}"
    )


if __name__ == "__main__":
    print(summary())
