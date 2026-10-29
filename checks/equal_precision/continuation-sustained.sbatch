#!/bin/bash
#SBATCH --job-name=cont789-sustained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=continuation-sustained-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Branch checkout = submit dir; MAIN = a synced checkout of the comparison revision.
BR="$PWD"; MAIN="${MAIN:?set MAIN to the comparison checkout}"
export PATH="$HOME/.local/bin:$PATH" PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
out="$BR/evidence/continuation/sustained/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$BR/.pyrite-sync" "$out/sync-branch.txt"; cp "$MAIN/.pyrite-sync" "$out/sync-main.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"; nvidia-smi -q > "$out/gpu.txt"
sha256sum "$BR"/checks/equal_precision/{benchmark_continuation.py,pyrite_adapter.py,manifest.json} \
  "$BR"/src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_launch.py,batching.py,continuation.py} \
  "$MAIN"/src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_launch.py,batching.py} > "$out/code.txt"
run_in() {  # dir stem args...
  local dir=$1 stem=$2; shift 2
  ( cd "$dir" && export PYRITE_HOME="$dir" PYRITE_CATALOG="$dir/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$dir/user-catalog" &&
    timeout --signal=TERM --kill-after=5s 600s uv run --no-sync python "$BR/checks/equal_precision/benchmark_continuation.py" "$@" --output "$out/$stem.json" ) > "$out/$stem.log" 2>&1 \
    && echo "DONE $stem" || { echo "{\"status\":\"failed\",\"exit_code\":$?}" > "$out/$stem.failed.json"; echo "FAILED $stem"; }
}
for dir in "$BR" "$MAIN"; do ( cd "$dir" && PYRITE_HOME="$dir" PYRITE_CATALOG="$dir/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$dir/user-catalog" uv run --no-sync python "$BR/checks/equal_precision/prepare_stopping.py" > "$out/tables-$(basename $dir).json" ); done
# Sustained filling: 20000 pending primaries, matched 5000-history concurrency.
BRP=learned32-B5000,cont-Q256-A5000-T32,cont-Q1024-A5000-T32,learned32,cont-Q256-Aall-T32
BRR=cont-Q256-Aall-T32,learned32,cont-Q1024-A5000-T32,cont-Q256-A5000-T32,learned32-B5000
for seed in 78601 78602; do
  pol=$BRP
  if (( seed % 2 == 0 )); then pol=$BRR; fi
  for scene in si-300-100um-0deg w-300-10um-0deg w-100-100um-75deg; do
    run_in "$BR" "branch-$scene-20000-$seed" "$scene" --histories 20000 --seed $seed --policies $pol
  done
done
# Binding scratch budget: 256 MiB holds fewer lanes than histories.
for scene in si-300-100um-0deg w-300-10um-0deg; do
  run_in "$BR" "branch-$scene-20000-256MiB-78601" "$scene" --histories 20000 --seed 78601 \
    --scratch-mib 256 --policies learned32,cont-Q256-Aall-T32,cont-Q64-Aall-T32
done
# Mixed-length stress scene at the matched 5000-history request, with main's reference.
run_in "$BR" "branch-w-100-100um-75deg-5000-78601" w-100-100um-75deg --histories 5000 --seed 78601 \
  --policies reference,learned32,cont-Q256-Aall-T32,cont-Q256-A2500-T32
run_in "$MAIN" "main-w-100-100um-75deg-5000-78601" w-100-100um-75deg --histories 5000 --seed 78601 \
  --policies reference
