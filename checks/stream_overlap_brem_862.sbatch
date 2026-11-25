#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o overlap-brem862-%j.log
# #862 brem-only (STREAM_CXR=none) serial vs PYRITE_STREAM_OVERLAP=1 production wall on a
# 16 GiB device: the large-block cases where molab (96 GiB) showed overlap gains, at block
# sizes whose pool peak fits. Per case: interleaved serial/overlap reps (STREAM_PHASES=0),
# one synchronized phase split, one #868 memory-sizer serial rep, then bitwise `compare`.
# CASES overrides the case list: ';'-separated "MAT NE BLOCK [CXR [REPS]]", CXR none (default) or
# mc (lines on), REPS interleaved pairs (default 3); CXR=mc cases skip the sizer rep.
# usage (from an isolated synced checkout root): sbatch -J ob862-<sha> checks/stream_overlap_brem_862.sbatch [OUTDIR]
set -uo pipefail
shopt -s nullglob
export PATH=$HOME/.pixi/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
CASES=${CASES:-"silicon 2097152 131072;silicon 16777216 1048576;tungsten 2097152 65536;tungsten 8388608 65536"}
echo "code $(cat CODE_SHA)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
pixi install --locked -e nvidia
B=(pixi run --as-is -e nvidia python checks/streamed_scoring_memory.py)
ROOT=${1:-ob862_out}; mkdir -p "$ROOT"
summary() { pixi run --as-is -e nvidia python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r.get(k) for k in ("wall_s", "phase_s", "n_segments", "host_peak_MiB", "device_pool_peak_above_base_MiB")}))' "$1"; }
STREAM_CXR=none "${B[@]}" run streamed 1000 1024 "$ROOT/warmup.json" tungsten 300 cuda > /dev/null
IFS=';' read -ra CASE_LIST <<< "$CASES"
for CASE in "${CASE_LIST[@]}"; do
  read -r MAT NE BLOCK CXR REPS <<< "$CASE"
  export STREAM_CXR=${CXR:-none}
  OUT=$ROOT/$MAT-$NE-$BLOCK-$STREAM_CXR; mkdir -p "$OUT"
  for REP in $(seq "${REPS:-3}"); do
    for OV in 0 1; do
      echo "== $CASE overlap=$OV rep $REP"
      PYRITE_STREAM_OVERLAP=$OV STREAM_PHASES=0 "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/ov$OV-$REP.json" "$MAT" 300 cuda > /dev/null
      echo "rc=$?"; summary "$OUT/ov$OV-$REP.json"
    done
  done
  echo "== $CASE phase split"
  "${B[@]}" run streamed "$NE" "$BLOCK" "$OUT/split.json" "$MAT" 300 cuda > /dev/null; summary "$OUT/split.json"
  if [ "$STREAM_CXR" = none ]; then
    echo "== $CASE memory sizer mem:12288"
    STREAM_PHASES=0 "${B[@]}" run streamed "$NE" mem:12288 "$OUT/sizer.json" "$MAT" 300 cuda > /dev/null; summary "$OUT/sizer.json"
  fi
  "${B[@]}" compare "$OUT/ov0-1.json" "$OUT"/ov0-[!1]*.json "$OUT"/ov1-*.json "$OUT/split.json" | tail -3
done
echo ALLDONE
