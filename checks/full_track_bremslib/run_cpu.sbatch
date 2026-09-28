#!/bin/bash
# Issue #182: exact per-electron CPU full-track cutoff sweep on the lab box.
# Copy this file and test_benchmark.py to an isolated remote directory before
# submission; the main checkout supplies the pinned PyRITE revision.
#SBATCH --job-name=issue182-bremslib-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_cpu_%j.log

set -euo pipefail
ROOT="${PYRITE_BENCH_ROOT:-$PWD}"
cd "$ROOT/pyrite"
export PYRITE_MC_BACKEND=cpu
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export UV_LINK_MODE=copy
export PYRITE_BENCH_CORE=per-electron

for benchmark_case in w_300kev si_300kev; do
  for cutoff_ev in 1000 5000 10000; do
    export PYRITE_BENCH_CASE="$benchmark_case"
    export PYRITE_BENCH_CUTOFF_EV="$cutoff_ev"
    if [ "$cutoff_ev" = 1000 ]; then
      export PYRITE_BENCH_NE=10000
    else
      export PYRITE_BENCH_NE=3000
    fi
    export PYRITE_BENCH_OUTPUT="$ROOT/${benchmark_case}_${cutoff_ev}ev.json"
    "${PYRITE_REMOTE_UV:-$HOME/.local/bin/uv}" run --no-sync pyrite-dev test \
      "$ROOT/test_benchmark.py" -s -q
  done
done
