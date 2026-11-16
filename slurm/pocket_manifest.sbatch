#!/bin/bash
#SBATCH --job-name=s2d_pocketman
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
# Pocket-profiling manifest over the broadened target set, then download any
# mmCIFs it needs that are not on disk yet (idempotent).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
python -m shape2delta.features.pocket_ensemble manifest
export S2D_CIF_LIST="$S2D_FEATURES/pocket_ensemble/missing_pdbs.txt"
bash "$S2D_ROOT/scripts/fetch/fetch_pdb_cif.sh"
