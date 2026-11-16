#!/bin/bash
#SBATCH --job-name=s2d_mdfetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
# E8: stream ATLAS / mdCATH entries, keep sampled frames (npz), delete raw files.
# NSHARD parallel downloaders; resumable (skips entries whose npz exists).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
NSHARD=8
export OMP_NUM_THREADS=1
pids=()
for i in $(seq 0 $((NSHARD-1))); do
  python -m shape2delta.features.md_ensemble fetch "$i" "$NSHARD" \
    > "${S2D_DATA}/logs/md_fetch_${SLURM_JOB_ID:-local}_${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[md-fetch] shards finished, process failures=$fail"
echo "[md-fetch] frames on disk: $(ls "$S2D_FEATURES/md_ensemble/frames" 2>/dev/null | wc -l)"
cat "$S2D_FEATURES"/md_ensemble/fetch_logs/*.jsonl | grep -c '"ok": false' || true
[ "$fail" -eq 0 ]
