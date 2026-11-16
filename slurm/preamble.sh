# shape2delta shared preamble — sourced by every .sbatch body and by local runs.
# Contract (see ~/.claude/CLAUDE.md):
#   * ALL defaults live here, in ONE place. Job scripts set nothing but #SBATCH.
#   * The submitter passes a timestamped env-file PATH as a positional arg (never
#     --export). This preamble sources any arg that is a file, and exports any
#     VAR=value arg. Arguments travel through the job record untouched.
#   * Self-sufficient under `env -i` (sources conda from an absolute path).
#   * `set -euo pipefail` is set in the JOB BODY, not here, so a failed source
#     cannot leave the job running under the system python and still exit 0.
#
# Usage in a .sbatch body:
#   set -euo pipefail
#   source "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh" "$@"
#   s2d_banner

# --- 1. process positional args: source files, export VAR=value -------------
for __arg in "$@"; do
  if [ -f "$__arg" ]; then
    # shellcheck source=/dev/null
    source "$__arg"
  elif [[ "$__arg" == *=* ]]; then
    export "${__arg}"
  fi
done
unset __arg || true

# --- 2. defaults (only if unset) --------------------------------------------
: "${S2D_ROOT:=/zfsauton2/home/yixiz/shape2delta}"
: "${S2D_DATA:=/zfsauton/scratch/yixiz/shape2delta}"
: "${S2D_RAW:=${S2D_DATA}/raw}"
: "${S2D_INTERIM:=${S2D_DATA}/interim}"
: "${S2D_FEATURES:=${S2D_DATA}/features}"
: "${S2D_RESULTS:=${S2D_ROOT}/results}"
: "${S2D_CONDA_SH:=/zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh}"
: "${S2D_ENV:=shape2delta}"
# experiment knobs (overridable per submission)
: "${S2D_MIN_T2_TARGETS:=40}"   # E0 gate: below this, drop MD requirement for T2
: "${S2D_MIN_PDB_CHAINS:=5}"    # empirical ensemble needs >= this many PDB chains
: "${S2D_NFOLDS:=5}"            # >=5 folds, never random example splits
: "${S2D_NBOOT:=2000}"         # bootstrap resamples for CIs
: "${S2D_SEED:=0}"

export S2D_ROOT S2D_DATA S2D_RAW S2D_INTERIM S2D_FEATURES S2D_RESULTS \
       S2D_CONDA_SH S2D_ENV S2D_MIN_T2_TARGETS S2D_MIN_PDB_CHAINS \
       S2D_NFOLDS S2D_NBOOT S2D_SEED
export PYTHONPATH="${S2D_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1   # ignore ~/.local site-packages; use only the env

mkdir -p "$S2D_RAW" "$S2D_INTERIM" "$S2D_FEATURES" "$S2D_RESULTS" 2>/dev/null || true

# --- 3. activate conda env (absolute path => works under env -i) -------------
if [ -f "$S2D_CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$S2D_CONDA_SH"
  conda activate "$S2D_ENV" 2>/dev/null \
    || echo "[preamble] WARN: could not activate conda env '$S2D_ENV'"
else
  echo "[preamble] WARN: conda profile not found at $S2D_CONDA_SH"
fi

# --- 4. startup banner (hostname, GPUs, resource request) -------------------
s2d_banner() {
  echo "================= shape2delta job banner ================="
  echo "date       : $(date -Is)"
  echo "hostname   : $(hostname)"
  echo "job id     : ${SLURM_JOB_ID:-<local>}  name: ${SLURM_JOB_NAME:-<local>}"
  echo "partition  : ${SLURM_JOB_PARTITION:-<local>}  nodelist: ${SLURM_JOB_NODELIST:-<local>}"
  echo "cpus       : ${SLURM_CPUS_ON_NODE:-?}  mem/node(req MB): ${SLURM_MEM_PER_NODE:-?}"
  echo "python     : $(command -v python || echo none)  ($(python --version 2>&1 || true))"
  echo "S2D_ROOT   : $S2D_ROOT"
  echo "S2D_DATA   : $S2D_DATA"
  echo "S2D_RESULTS: $S2D_RESULTS"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi -L ---"; nvidia-smi -L 2>/dev/null || echo "(nvidia-smi -L failed)"
  else
    echo "GPUs       : none visible (CPU job)"
  fi
  echo "=========================================================="
}
