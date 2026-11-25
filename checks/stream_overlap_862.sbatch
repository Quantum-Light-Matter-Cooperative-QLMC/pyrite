#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH -o overlap862-%j.log
# #862 transport/scoring phase split and GPU idle time of live streamed runs on the #872
# harness: W and Si 10 um slabs, 20000 electrons / 4096 blocks, plus W 65536 / 8192.
# Per arm: two unsynchronized production-wall reps (STREAM_PHASES=0), one synchronized
# phase split, and one Nsight run with NVTX phase ranges reduced by `gpu-busy`; then two
# PYRITE_STREAM_OVERLAP=1 production reps (bitwise `compare` against serial) and an
# overlapped Nsight run whose `run` busy fraction shows the filled device idle time.
# usage (from an isolated synced checkout root): sbatch -J o862-<sha> checks/stream_overlap_862.sbatch [OUTDIR]
set -uo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
pixi install --locked -e nvidia
B=(pixi run --as-is -e nvidia python checks/streamed_scoring_memory.py)
ROOT=${1:-o862_out}
summary() { pixi run --as-is -e nvidia python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r[k] for k in ("wall_s", "phase_s", "n_segments", "host_peak_MiB", "device_pool_peak_above_base_MiB")}))' "$1"; }
for ARM in "silicon 20000 4096" "tungsten 20000 4096" "tungsten 65536 8192"; do
  read -r MAT NE BLOCK <<< "$ARM"
  OUT=$ROOT/$MAT-$NE-$BLOCK; mkdir -p "$OUT"
  "${B[@]}" run streamed 1000 1000 "$OUT/warmup.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  for REP in 1 2; do
    echo "== $ARM production rep $REP"
    STREAM_PHASES=0 "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/prod-$REP.json" "$MAT" 300 cuda 1e5 1 > /dev/null
    summary "$OUT/prod-$REP.json"
  done
  echo "== $ARM phase split"
  "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/split.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  summary "$OUT/split.json"
  "${B[@]}" compare "$OUT/prod-1.json" "$OUT/prod-2.json" "$OUT/split.json" | tail -1
  echo "== $ARM nsys"
  STREAM_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$OUT/nsys" -f true \
    "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/nsys.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  nsys export --type sqlite -o "$OUT/nsys.sqlite" -f true "$OUT/nsys.nsys-rep" > /dev/null
  "${B[@]}" gpu-busy "$OUT/nsys.sqlite" | tee "$OUT/gpu_busy.json"
  for REP in 1 2; do
    echo "== $ARM overlap rep $REP"
    PYRITE_STREAM_OVERLAP=1 STREAM_PHASES=0 "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/overlap-$REP.json" "$MAT" 300 cuda 1e5 1 > /dev/null
    summary "$OUT/overlap-$REP.json"
  done
  "${B[@]}" compare "$OUT/prod-1.json" "$OUT/overlap-1.json" "$OUT/overlap-2.json" | tail -1
  echo "== $ARM overlap nsys"
  PYRITE_STREAM_OVERLAP=1 STREAM_PHASES=0 STREAM_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$OUT/nsys-overlap" -f true \
    "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/nsys-overlap.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  nsys export --type sqlite -o "$OUT/nsys-overlap.sqlite" -f true "$OUT/nsys-overlap.nsys-rep" > /dev/null
  "${B[@]}" gpu-busy "$OUT/nsys-overlap.sqlite" | tee "$OUT/gpu_busy_overlap.json"
done
