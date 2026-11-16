#!/bin/bash
#SBATCH --job-name=s2d_exp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# Generic experiment runner. Pass S2D_EXP=e1_mantel|e3_ablation|e6_mechanism|...
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
: "${S2D_EXP:?set S2D_EXP=<experiment module>}"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
echo "[exp] running shape2delta.experiments.$S2D_EXP"
python -m "shape2delta.experiments.$S2D_EXP"
