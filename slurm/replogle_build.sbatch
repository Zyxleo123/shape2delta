#!/bin/bash
#SBATCH --job-name=s2d_repl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# E5 data: Replogle K562 CRISPRi -> pseudobulk Δ per knockdown target (landmark
# genes). Reads the 1.5GB h5ad; needs headroom for the dense landmark slice.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
python -m shape2delta.data.replogle
echo "[replogle]:"; cat "$S2D_FEATURES/replogle_delta/build_summary.json" 2>/dev/null || true
