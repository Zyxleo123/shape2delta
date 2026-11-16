#!/bin/bash
#SBATCH --job-name=s2d_pocket
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
#
# E7 pocket-centric matched T1/T2 features. ONE job (the per-user submit limit
# rules out a 40-task array) running NSHARD shard processes in parallel. Each
# shard appends to its own JSONL and skips done targets -> resubmit to resume.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
NSHARD=40
export OMP_NUM_THREADS=1
pids=()
for i in $(seq 0 $((NSHARD-1))); do
  python -m shape2delta.features.pocket_ensemble shard "$i" "$NSHARD" \
    > "${S2D_DATA}/logs/pocket_shard_${SLURM_JOB_ID:-local}_${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[pocket] shards finished, failures=$fail"
[ "$fail" -eq 0 ]
