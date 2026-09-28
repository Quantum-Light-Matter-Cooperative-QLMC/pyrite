#!/bin/bash
# Issue #182: 100k-primary emission sweep (energy and recoil comparison).
# The compute node has its own home: stage a pinned PyRITE checkout at
# $ROOT/pyrite-fa20e0dd and this directory's test_benchmark.py at
# $ROOT/emission/ first (see README).
#SBATCH --job-name=issue182-emission
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=pyrite_emission_%j.log
set -euo pipefail
ROOT="${PYRITE_BENCH_ROOT:?set PYRITE_BENCH_ROOT to the staged benchmark directory}"
cd "$ROOT/pyrite-fa20e0dd"
export PYRITE_MC_BACKEND=cpu PYRITE_BENCH_CORE=per-electron PYRITE_BENCH_NE=100000
export UV_CACHE_DIR=/tmp/pyrite-uv-cache NUMBA_NUM_THREADS=1
# One wave per cutoff keeps four transports within the node's memory.
for cutoff in 1000 5000 10000; do
  for case in w_300kev si_300kev w_800kev si_800kev; do
    PYRITE_BENCH_CASE=$case PYRITE_BENCH_CUTOFF_EV=$cutoff \
      PYRITE_BENCH_OUTPUT="$ROOT/emission/${case}_${cutoff}ev_emission.json.gz" \
      "${PYRITE_REMOTE_UV:-$HOME/.local/bin/uv}" run --no-sync pyrite-dev test \
      "$ROOT/emission/test_benchmark.py" -s -q > "$ROOT/emission/${case}_${cutoff}ev.out" 2>&1 &
  done
  wait
done
