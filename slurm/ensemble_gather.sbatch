#!/bin/bash
#SBATCH --job-name=s2d_ensg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
python -m shape2delta.features.ensemble gather
echo "[ens-gather] T2 table:"; ls -l "$S2D_FEATURES/t2_ensemble.csv" 2>/dev/null
cat "$S2D_FEATURES/t2_ensemble_summary.json" 2>/dev/null || true
