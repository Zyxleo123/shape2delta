#!/bin/bash
#SBATCH --job-name=s2d_ens
#SBATCH --partition=cpu
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%A_%a.err
#
# T2 ensemble profiling, sharded over targets (reads local mmCIFs). NSHARD must
# equal the array width (0-19 -> 20).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
NSHARD=20
python -m shape2delta.features.ensemble "${SLURM_ARRAY_TASK_ID}" "$NSHARD"
