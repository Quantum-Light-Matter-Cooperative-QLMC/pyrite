#!/bin/bash
#SBATCH --job-name=cont789-staging
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --output=continuation-staging-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Branch checkout = submit dir; MAIN = a synced checkout of the comparison revision.
BR="$PWD"; MAIN="${MAIN:?set MAIN to the comparison checkout}"
export PATH="$HOME/.local/bin:$PATH" PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
out="$BR/evidence/continuation/staging/$SLURM_JOB_ID"; mkdir -p "$out"
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
( export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog" &&
  uv run --no-sync pyrite-dev test tests/montecarlo/test_transport_continuation_cuda.py \
  tests/montecarlo/test_transport_continuation.py ) > "$out/tests.log" 2>&1 && echo "DONE tests" || echo "FAILED tests"
# Host vs device staging of retained rows, matched and sustained populations.
P20=learned32-B5000,cont-Q256-A5000-T32,cont-Q256-A5000-T32-D,cont-Q256-Aall-T32-D
R20=cont-Q256-Aall-T32-D,cont-Q256-A5000-T32-D,cont-Q256-A5000-T32,learned32-B5000
P5=learned32,cont-Q256-Aall-T32,cont-Q256-Aall-T32-D
R5=cont-Q256-Aall-T32-D,cont-Q256-Aall-T32,learned32
for seed in 78601 78602; do
  p20=$P20; p5=$P5
  if (( seed % 2 == 0 )); then p20=$R20; p5=$R5; fi
  for scene in si-300-100um-0deg w-300-10um-0deg; do
    run_in "$BR" "branch-$scene-5000-$seed" "$scene" --histories 5000 --seed $seed --policies $p5
    run_in "$BR" "branch-$scene-20000-$seed" "$scene" --histories 20000 --seed $seed --policies $p20
  done
done
