#!/bin/bash
#SBATCH --job-name=s2d_e0
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# E0 — coverage audit (the gate). CPU-only, fully offline: reads staged metadata
# and writes results/E0_coverage.json + results/SUMMARY.md. Cannot fail on data;
# a thin result is reported as a number, not an error.
#
# set -euo pipefail in the BODY so a failed source cannot run system python and
# still exit 0 (which would release afterok dependents behind no work).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner

echo "[E0] config: $(python -m shape2delta.config | tr '\n' ' ')"
python -m shape2delta.coverage
echo "[E0] done. Artefacts:"
ls -l "$S2D_RESULTS"/E0_*.{json,csv} 2>/dev/null || true
