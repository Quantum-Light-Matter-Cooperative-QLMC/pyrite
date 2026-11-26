#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 01:30:00
#SBATCH -o fr1296-%j.log
# #1296 Fourier bin-mean reduction phase split: capture flushes on one case, then bench.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch [-p gpu-amd] -J fr1296-<sha> checks/fourier_reduction_phases_1296.sbatch \
#       OUT_DIR MATERIAL E0_KEV TILT HISTORIES [REPEATS]
# ARMSET=variants replays saved flushes through the candidate-fix arms instead of bench.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; MATERIAL=$2; E0=$3; TILT=$4; HISTORIES=$5; REPEATS=${6:-3}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/fourier_reduction_phases_1296.py
if ! ls "$OUT"/flush-*.npz > /dev/null 2>&1; then
    pixi run --as-is -e amd python -u $H capture "$MATERIAL" "$E0" "$TILT" "$HISTORIES" "$OUT"
    echo "capture rc=$?"
fi
pixi run --as-is -e amd python -u $H "${ARMSET:-bench}" "$OUT" "$REPEATS"
echo "bench rc=$?"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null
