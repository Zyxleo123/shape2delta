#!/bin/bash
#SBATCH --job-name=s2d_mdprof
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
# E8: matched pocket descriptor over MD frames (needs pocket_residues.json from
# the pocket gather + frames from md_fetch), then gather. Resumable.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
if [ ! -s "$S2D_FEATURES/pocket_ensemble/pocket_residues.json" ]; then
  echo "[md-prof] GATE: pocket_residues.json missing (pocket gather never ran) — nothing to do"; exit 0
fi
NSHARD=40
export OMP_NUM_THREADS=1
pids=()
for i in $(seq 0 $((NSHARD-1))); do
  python -m shape2delta.features.md_ensemble profile "$i" "$NSHARD" \
    > "${S2D_DATA}/logs/md_prof_${SLURM_JOB_ID:-local}_${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[md-prof] shards finished, failures=$fail"
python -m shape2delta.features.md_ensemble gather
[ "$fail" -eq 0 ]
