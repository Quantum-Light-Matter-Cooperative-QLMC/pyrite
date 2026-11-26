#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 01:30:00
#SBATCH -o fs1297-%j.log
# #1297 warp-gather ES spreader on the device Fourier line reduction: device
# tests, production spreader bench on saved #1296 flushes, #1143 margins
# against the v6 (atomic) route, and the end-to-end mos2 30 keV / 15 deg capture.
# usage (from an isolated synced checkout root holding CODE_SHA, tests/ and the
# shipped user config under .xdg-config/pyrite; FLUSHES holds flush-*.npz):
#   sbatch -J fs1297-<sha> checks/fourier_spread_1297.sbatch FLUSHES OUT_DIR
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}" PROBE_SHA=$(cat CODE_SHA)
FLUSHES=$1; OUT=$2
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
H=checks/fourier_reduction_phases_1296.py
PYRITE_TEST_BACKEND=$PYRITE_MC_BACKEND pixi run --as-is -e "$PYRITE_PIXI_ENV" pyrite-dev test \
    tests/montecarlo/spectrum/test_fourier_cuda.py; echo "tests rc=$?"
pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H spread "$FLUSHES" 5; echo "spread rc=$?"
cp "$FLUSHES/spread_bench.json" "$OUT/" 2>/dev/null
MARGIN_REF=v6 pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H margins "$OUT/margins.json" all 64 642501 0.375,3
echo "margins rc=$?"
CAPTURE=0 pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H capture mos2 30 15 1024 "$OUT"; echo "capture rc=$?"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null
