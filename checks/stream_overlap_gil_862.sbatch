#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o overlapgil862-%j.log
# #862 GIL attribution of live streamed runs on the #872 harness: Si 10 um slab,
# 20000 electrons / 4096 blocks, serial and PYRITE_STREAM_OVERLAP=1. Nsight traces the
# GIL (python-gil), samples Python backtraces and records thread context switches, so the
# GIL holder and its functions are known per thread while the other thread waits.
# usage (from an isolated synced checkout root): sbatch -J og862-<sha> checks/stream_overlap_gil_862.sbatch [OUTDIR]
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
OUT=${1:-og862_out}; mkdir -p "$OUT"
"${B[@]}" run streamed 1000 1000 "$OUT/warmup.json" silicon 300 cuda 1e5 1 > /dev/null
for ARM in serial overlap; do
  [ "$ARM" = overlap ] && export PYRITE_STREAM_OVERLAP=1 || unset PYRITE_STREAM_OVERLAP
  echo "== silicon 20000 4096 $ARM gil"
  STREAM_PHASES=0 STREAM_NVTX=1 nsys profile --trace=cuda,nvtx,osrt,python-gil \
    --python-sampling=true --python-sampling-frequency=1000 --sample=process-tree \
    --cpuctxsw=process-tree -o "$OUT/$ARM" -f true \
    "${B[@]}" run streamed 20000 4096 "$OUT/$ARM.json" silicon 300 cuda 1e5 1 > /dev/null
  nsys export --type sqlite -o "$OUT/$ARM.sqlite" -f true "$OUT/$ARM.nsys-rep" > /dev/null
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wall_s"])' "$OUT/$ARM.json"
done
"${B[@]}" compare "$OUT/serial.json" "$OUT/overlap.json" | tail -1
