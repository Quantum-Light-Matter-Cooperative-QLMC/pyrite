#!/bin/bash
# CPU-only accuracy references, never input to T(epsilon) comparisons.
# After a compute-node smoke test and pixi install --locked, submit a bounded array:
# sbatch --array=0-LAST%4 accuracy-reference.sbatch PLAN RUNTIME OUTPUT
# RUNTIME contains bin/g4bench, lib/, data/ and SHA256SUMS.
# Plan threads must match --cpus-per-task. No GPU or timing-host assumptions.
#SBATCH --job-name=accuracy1041
#SBATCH --partition=campus
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=accuracy-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
plan=$1 runtime=$2 out=$3
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" PYRITE_CACHE_DIR="$PWD/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export G4LEDATA="$runtime/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$runtime/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$runtime/lib:${LD_LIBRARY_PATH:-}"
threads=$(pixi run --as-is python -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$plan")
[[ $threads == "$SLURM_CPUS_PER_TASK" ]] || { echo "plan/allocation thread mismatch" >&2; exit 2; }
mkdir -p "$out"
provenance="$out/shard-$(printf '%04d' "$SLURM_ARRAY_TASK_ID").provenance.txt"
{
    date -u
    cat .pyrite-sync
    sha256sum "$plan" "$runtime/bin/g4bench" checks/geant4_performance/accuracy_reference.py
    lscpu
    scontrol show job "$SLURM_JOB_ID"
} > "$provenance"
pixi run --as-is python checks/geant4_performance/accuracy_reference.py run \
    "$plan" "$SLURM_ARRAY_TASK_ID" --executable "$runtime/bin/g4bench" --output "$out"
