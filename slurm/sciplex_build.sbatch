#!/bin/bash
#SBATCH --job-name=s2d_sciplex
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
# E9: sci-Plex3 pseudobulk Δ + per-cell PCA + single-cell readouts.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
export OMP_NUM_THREADS=8
python -m shape2delta.data.sciplex build
