#!/bin/bash
#SBATCH --job-name=issue182-sr-control
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-ace
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aamador/.local/opt/pyrite-issue-182/sr_control_%j.log
set -euo pipefail
cd /home/aamador/.local/opt/pyrite-issue-182/pyrite
export PYRITE_MC_BACKEND=cpu
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export UV_LINK_MODE=copy
export PYRITE_BENCH_CORE=per-electron
export PYRITE_BENCH_ELASTIC=sr
export PYRITE_BENCH_CASE=w_300kev
export PYRITE_BENCH_CUTOFF_EV=1000
export PYRITE_BENCH_NE=10000
export PYRITE_BENCH_OUTPUT=/home/aamador/.local/opt/pyrite-issue-182/w_300kev_sr_1000ev.json
/home/aamador/.local/bin/uv run --no-sync pyrite-dev test /home/aamador/.local/opt/pyrite-issue-182/test_benchmark.py -s -q
