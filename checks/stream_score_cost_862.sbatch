#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o scorecost862-%j.log
# #862 host-cost attribution of serial live streamed scoring on the #872 harness, by
# function (cProfile, main thread): Si 10 um 20000 electrons / 4096 blocks and one
# 4096-electron W 10 um block. Prints the top pyrite functions by cumulative and own time.
# MODE nsys instead runs each arm once under Nsight with NVTX phase ranges and prints
# `gpu-busy` (per-estimator wall, GPU busy and top kernels); W runs 8192 / 4096 there.
# usage (from an isolated synced checkout root): sbatch -J sc862-<sha> checks/stream_score_cost_862.sbatch [OUTDIR] [cprofile|nsys]
set -uo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
pixi install --locked -e nvidia
B=(pixi run --as-is -e nvidia python checks/streamed_scoring_memory.py)
OUT=${1:-sc862_out}; MODE=${2:-cprofile}; mkdir -p "$OUT"
[ "$MODE" = nsys ] && export TMPDIR="$HOME/tmp-nsys" && mkdir -p "$TMPDIR"
"${B[@]}" run streamed 1000 1000 "$OUT/warmup.json" silicon 300 cuda 1e5 1 > /dev/null
ARMS=("silicon 20000 4096" "tungsten 4096 4096")
# nsys: W gets a steady-state block after the 4096-electron selection prefix.
[ "$MODE" = nsys ] && ARMS=("silicon 20000 4096" "tungsten 8192 4096")
for ARM in "${ARMS[@]}"; do
  read -r MAT NE BLOCK <<< "$ARM"
  STEM=$OUT/$MAT-$NE-$BLOCK
  if [ "$MODE" = nsys ]; then
    echo "== $ARM nsys"
    STREAM_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$STEM" -f true \
      "${B[@]}" run streamed "$NE" "$BLOCK" "$STEM.json" "$MAT" 300 cuda 1e5 1 > /dev/null
    nsys export --type sqlite -o "$STEM.sqlite" -f true "$STEM.nsys-rep" > /dev/null
    "${B[@]}" gpu-busy "$STEM.sqlite" | tee "$STEM-gpu_busy.json"
    continue
  fi
  echo "== $ARM cProfile"
  STREAM_PHASES=0 STREAM_CPROFILE="$STEM.pstats" "${B[@]}" run streamed "$NE" "$BLOCK" "$STEM.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  pixi run --as-is -e nvidia python -c '
import pstats, sys
for key in ("cumulative", "tottime"):
    print(f"-- by {key}")
    pstats.Stats(sys.argv[1]).sort_stats(key).print_stats("pyrite|numpy|cupy", 45)
' "$STEM.pstats" | grep -v -E "^\s*$" | sed -e "s#$PWD/##" -e "s#.venv/lib/python3.14/site-packages/##"
done
