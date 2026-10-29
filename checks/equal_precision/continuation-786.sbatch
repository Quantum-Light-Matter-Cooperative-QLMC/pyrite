#!/bin/bash
#SBATCH --job-name=cont789-bench786
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=continuation-786-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Branch checkout = submit dir; MAIN = a synced checkout of the comparison revision.
BR="$PWD"; MAIN="${MAIN:?set MAIN to the comparison checkout}"
export PATH="$HOME/.local/bin:$PATH" PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
out="$BR/evidence/continuation/bench786/$SLURM_JOB_ID"; mkdir -p "$out"
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
BRP=reference,learned32,cont-Q256-Aall-T32,cont-Q1024-Aall-T32,cont-Q64-Aall-T32,cont-Q256-A2500-T32
BRR=cont-Q256-A2500-T32,cont-Q64-Aall-T32,cont-Q1024-Aall-T32,cont-Q256-Aall-T32,learned32,reference
for seed in 78601 78602 78603; do
  pol=$BRP; mp=reference,learned32
  if (( seed % 2 == 0 )); then pol=$BRR; mp=learned32,reference; fi
  for scene in si-300-100um-0deg w-300-10um-0deg; do
    run_in "$BR" "branch-$scene-$seed" "$scene" --histories 5000 --seed $seed --policies $pol
    run_in "$MAIN" "main-$scene-$seed" "$scene" --histories 5000 --seed $seed --policies $mp
  done
done
