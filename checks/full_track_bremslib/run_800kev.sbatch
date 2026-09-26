#!/bin/bash
# Issue #182: second-energy full-track check and W elastic-model control.
#SBATCH --job-name=issue182-800kev
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-ace
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aamador/.local/opt/pyrite-issue-182/benchmark_800kev_%j.log
set -euo pipefail
cd /home/aamador/.local/opt/pyrite-issue-182/pyrite
export PYRITE_MC_BACKEND=cpu
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export UV_LINK_MODE=copy
export PYRITE_BENCH_CORE=per-electron
export PYRITE_BENCH_CUTOFF_EV=1000
export PYRITE_BENCH_NE=10000
for entry in w_800kev:mott w_800kev:sr si_800kev:mott; do
  export PYRITE_BENCH_CASE="${entry%%:*}"
  export PYRITE_BENCH_ELASTIC="${entry##*:}"
  if [ "$PYRITE_BENCH_ELASTIC" = sr ]; then
    export PYRITE_BENCH_OUTPUT="/home/aamador/.local/opt/pyrite-issue-182/${PYRITE_BENCH_CASE}_sr_1000ev.json"
  else
    export PYRITE_BENCH_OUTPUT="/home/aamador/.local/opt/pyrite-issue-182/${PYRITE_BENCH_CASE}_1000ev.json"
  fi
  /home/aamador/.local/bin/uv run --no-sync pyrite-dev test \
    /home/aamador/.local/opt/pyrite-issue-182/test_benchmark.py -s -q
done
