#!/bin/bash
#SBATCH --job-name=cont797-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=continuation-797-%j.log
# #797: full-history vs continuation on a covered-material shell soft/hard
# case and an Urban straggling case, #789's methodology (5000 histories,
# same-policy warmup, interleaved policies, order reversed on even seeds).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:$PATH" PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
out="$BR/evidence/continuation/bench797/$SLURM_JOB_ID"; mkdir -p "$out"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"; nvidia-smi -q > "$out/gpu.txt"
sha256sum "$BR"/checks/equal_precision/{benchmark_continuation.py,pyrite_adapter.py,manifest.json} \
  "$BR"/src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_launch.py,batching.py,continuation.py} > "$out/code.txt"
uv sync --package pyrite-mc --no-default-groups --group dev --extra nvidia -q
uv run --no-sync python "$BR/checks/equal_precision/prepare_stopping.py" > "$out/tables.json"
run() {  # stem args...
  local stem=$1; shift
  timeout --signal=TERM --kill-after=5s 900s uv run --no-sync python \
    "$BR/checks/equal_precision/benchmark_continuation.py" "$@" --output "$out/$stem.json" \
    > "$out/$stem.log" 2>&1 && echo "DONE $stem" \
    || { echo "{\"status\":\"failed\",\"exit_code\":$?}" > "$out/$stem.failed.json"; echo "FAILED $stem"; }
}
FWD=reference,learned32,cont-Q256-Aall-T32,cont-Q256-A2500-T32,cont-Q64-Aall-T32
REV=cont-Q64-Aall-T32,cont-Q256-A2500-T32,cont-Q256-Aall-T32,learned32,reference
for seed in 78601 78602 78603; do
  pol=$FWD; (( seed % 2 == 0 )) && pol=$REV
  run "shell-si-300-$seed" si-300-100um-0deg --inelastic shell --histories 5000 --seed $seed --policies $pol
  run "stragg-si-300-$seed" si-300-100um-0deg --straggling --histories 5000 --seed $seed --policies $pol
done
