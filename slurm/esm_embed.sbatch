#!/bin/bash
#SBATCH --job-name=s2d_esm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# T0 ESM-2 (35M) embeddings on CPU. Weights pre-cached on the login node.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
python -m shape2delta.features.esm_embed
echo "[esm] done:"; cat "$S2D_FEATURES/esm/build_summary.json" 2>/dev/null || true
