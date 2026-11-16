#!/bin/bash
#SBATCH --job-name=s2d_e2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# E2 — generalization-gap baseline (CPU). Needs the cached Δ signatures
# ($S2D_FEATURES/lincs_delta). Gate: if those are missing, print why and exit 0
# so no dependent chain is stranded for an uninformative reason.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner

if [ ! -s "$S2D_FEATURES/lincs_delta/delta.npy" ]; then
  echo "[E2] GATE: $S2D_FEATURES/lincs_delta/delta.npy missing — run build_signatures first. Exiting 0."
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
python -m shape2delta.experiments.e2_gengap
echo "[E2] done."; ls -l "$S2D_RESULTS"/E2_gengap.json 2>/dev/null || true
