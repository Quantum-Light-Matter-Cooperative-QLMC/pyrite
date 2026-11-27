#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH -t 12:00:00
#SBATCH -o fiber1480-%j.log
# #1480 research diagnostics. Default: FP64 CPU on qlmc-queen. For the NVIDIA
# GPU: sbatch -p gpu --nodelist=DESKTOP-QNIHO3D --gres=gpu:1 ... (device REAL is
# float32 unless PYRITE_FP64=1 is exported). Run from a synced checkout.
# Usage:
#   sbatch checks/fiber_nonbasal_cxr_1480.sbatch OUT_DIR frozen CRYSTAL E0 THICK TILT [opts]
#   sbatch checks/fiber_nonbasal_cxr_1480.sbatch OUT_DIR paired [fiber_paired_1480.py opts]
# `frozen` writes OUT_DIR/result.{json,npz}; `paired` gets --output OUT_DIR/paired.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if (( $# < 2 )); then
    echo "usage: $0 OUT_DIR {frozen|paired} [options]" >&2
    exit 2
fi
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
MODE=$2
shift 2
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OMP_NUM_THREADS=1
ENV=default
if [[ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null; then
    export PYRITE_MC_BACKEND=cuda
    ENV=nvidia
else
    export PYRITE_MC_BACKEND=cpu
fi
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
SHA=$(cat CODE_SHA 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)
printf '%s\n' "$SHA" > "$OUT/CODE_SHA"
printf '%q ' "$MODE" "$@" > "$OUT/arguments.txt"
printf '\n' >> "$OUT/arguments.txt"
echo "code $SHA host $(hostname) job $SLURM_JOB_ID mode $MODE backend $PYRITE_MC_BACKEND fp64=${PYRITE_FP64:-0}"
pixi install --locked -e "$ENV"
case "$MODE" in
    frozen)
        /usr/bin/time -v pixi run --as-is -e "$ENV" python -u \
            checks/fiber_nonbasal_cxr_1480.py frozen "$@" --json "$OUT/result.json" \
            > "$OUT/run.log" 2>&1
        files=(CODE_SHA arguments.txt run.log result.json result.npz)
        ;;
    paired)
        /usr/bin/time -v pixi run --as-is -e "$ENV" python -u \
            checks/fiber_paired_1480.py "$@" --output "$OUT/paired" > "$OUT/run.log" 2>&1
        files=(CODE_SHA arguments.txt run.log paired/spectra.npz paired/summary.json)
        ;;
    *)
        echo "unknown mode $MODE" >&2
        exit 2
        ;;
esac
(cd "$OUT" && sha256sum "${files[@]}" > SHA256SUMS)
