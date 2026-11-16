#!/bin/bash
#SBATCH --job-name=s2d_e10
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# E10 — MD as an extra input to a chemCPA-style compositional backbone (CPU torch).
# Trains OFF/ESM/STATIC/MD/STATIC+MD/MD_PERM arms x 4 split kinds x 5 folds.
# Configure seeds with S2D_E10_SEEDS=1,2 (default 1). Needs md_global features.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
echo "[e10] S2D_E10_SEEDS=${S2D_E10_SEEDS:-1}"
python -m shape2delta.experiments.e10_cpa
