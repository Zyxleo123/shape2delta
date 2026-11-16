#!/bin/bash
#SBATCH --job-name=s2d_mdglobal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# Build the GLOBAL (whole-protein) MD dynamics features from the frames already
# on disk (features/md_ensemble/frames). Pure CPU, no internet. Prereq for E10.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
echo "[md-global] building whole-protein MD features"
python -m shape2delta.features.md_global build
