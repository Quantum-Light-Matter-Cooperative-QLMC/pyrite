#!/bin/bash
#SBATCH --job-name=cont789-cprofile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=continuation-cprofile-%j.log
# Host-time attribution for #789: cProfile of the measured request only, after
# a full unprofiled warmup. Profiled timings are excluded from benchmark summaries.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYRITE_MC_BACKEND=cuda
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
out="$BR/evidence/continuation/cprofile/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"; scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
for n in 5000 20000; do
  pol=learned32,cont-Q256-Aall-T32,cont-Q256-Aall-T32-D
  if (( n == 20000 )); then pol=learned32-B5000,cont-Q256-A5000-T32,cont-Q256-A5000-T32-D; fi
  stem="w$n-cprofile"
  timeout --signal=TERM --kill-after=5s 600s uv run --no-sync python \
    checks/equal_precision/benchmark_continuation.py w-300-10um-0deg --histories $n --seed 78601 \
    --policies $pol --cprofile --output "$out/$stem.json" > "$out/$stem.log" 2>&1 \
    && echo "DONE $stem" || { echo "{\"status\":\"failed\",\"exit_code\":$?}" > "$out/$stem.failed.json"; echo "FAILED $stem"; }
done
