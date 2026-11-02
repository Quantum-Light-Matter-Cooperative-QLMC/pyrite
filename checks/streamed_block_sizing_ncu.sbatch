#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o s907ncu-%j.log
# #907 Nsight Compute over a window of transport launches spanning the warmup
# tail and the measured run's first (full-population, 16,384-history) rounds.
# usage (from an isolated synced checkout root): sbatch checks/streamed_block_sizing_ncu.sbatch
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=/home/aamador/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda PROBE_SHA=$(cat CODE_SHA)
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
OUT=s907_out; mkdir -p $OUT
ncu --set full --kernel-name regex:transport --launch-skip 24 --launch-count 12 \
  -o $OUT/ncu-window -f \
  uv run --no-sync python -u checks/streamed_block_sizing.py transport 16384 16384 2048 32 0 $OUT/ncu-window-run
ncu --import $OUT/ncu-window.ncu-rep --page details --csv > $OUT/ncu-window.csv
ncu --import $OUT/ncu-window.ncu-rep --page raw --csv \
  --metrics launch__grid_size,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,launch__occupancy_limit_registers,sm__maximum_warps_per_active_cycle_pct \
  > $OUT/ncu-window-raw.csv
cat $OUT/ncu-window-raw.csv
echo "done rc=0"
