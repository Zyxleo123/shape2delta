#!/bin/bash
#SBATCH --job-name=s2d_sig
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# Build LINCS level-5 Δ signatures for the single-target compounds. Reads only
# the 978 landmark rows from the 33GB GCTX (~2.8GB RAM) and caches a dense
# [n_sig x 978] matrix + metadata under $S2D_FEATURES/lincs_delta/.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner

GCTX="$S2D_RAW/lincs/level5_beta_trt_cp_n720216x12328.gctx"
if [ ! -s "$GCTX" ] || [ "$(stat -c%s "$GCTX")" -ne 35518405386 ]; then
  echo "[sig] GCTX missing/incomplete ($GCTX) — run scripts/fetch/fetch_lincs_level5.sh first"; exit 1
fi
python -m shape2delta.data.signatures
echo "[sig] artefacts:"; ls -l "$S2D_FEATURES/lincs_delta/" 2>/dev/null || true
