#!/bin/bash
#SBATCH --job-name=s2d_t1
#SBATCH --partition=cpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%A_%a.err
#
# T1 static-structure features (pLDDT/Rg/SS/PAE/fpocket), sharded; NSHARD=10.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
python -m shape2delta.features.structure "${SLURM_ARRAY_TASK_ID}" 10
