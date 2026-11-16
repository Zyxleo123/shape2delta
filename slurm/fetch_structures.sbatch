#!/bin/bash
#SBATCH --job-name=s2d_fetchstruct
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/shape2delta/logs/%x_%j.err
# Sequences + AlphaFold models for every target (E0 table ∪ sci-Plex3 targets).
# Compute nodes have internet (probed 2026-11-15, job 111561). Idempotent.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
s2d_banner
bash "$S2D_ROOT/scripts/fetch/fetch_structures.sh"
