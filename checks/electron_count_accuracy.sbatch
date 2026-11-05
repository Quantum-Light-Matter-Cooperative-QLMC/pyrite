#!/bin/bash
# #278 plan item 5: accuracy vs electron count, Ne (lines) and Ne_brem
# (continuum), on the production default grids. HOPG 1 mm at 45/180 deg,
# One energy per allocation: ECA_ENERGY_KEV=300 or 29999. Five replicas
# at 100/400, plus three independently seeded 1600-electron references.
# Reduced high-energy ladder follows the #278/#286 line-cost decision.
# Resumable: resubmit to continue from the checkpoint JSON.
#SBATCH --job-name=eca278
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=08:00:00
#SBATCH --output=eca278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG" evidence/eca278
uv sync --extra nvidia --locked -q
energy="${ECA_ENERGY_KEV:?set ECA_ENERGY_KEV to 300 or 29999}"
case "$energy" in 300|29999) ;; *) echo "unsupported energy: $energy" >&2; exit 2 ;; esac
cp .pyrite-sync "evidence/eca278/sync-$SLURM_JOB_ID.json"
nvidia-smi > "evidence/eca278/gpu-$SLURM_JOB_ID.txt"
scontrol show job "$SLURM_JOB_ID" > "evidence/eca278/allocation-$SLURM_JOB_ID.txt"
uv run --no-sync python checks/electron_count_accuracy.py \
    --energy-kev "$energy" --counts 100 400 --max-hours 7.5 --grid-check \
    --reference-count 1600 --reference-replicas 3 \
    --output "evidence/eca278/hopg-$energy-replicated.json" > "evidence/eca278/hopg-$energy-$SLURM_JOB_ID.log" 2>&1
