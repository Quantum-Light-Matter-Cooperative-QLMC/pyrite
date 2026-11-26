#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 02:00:00
#SBATCH -o fm1320-%j.log
# #1320 Fourier mode-count census on one case, then replay of saved flushes.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch -J fm1320-<sha> checks/fourier_mode_cap_1320.sbatch OUT_DIR MATERIAL E0_KEV TILT HISTORIES
# CENSUS_ONLY=0 runs the production route on every flush (acceptance run).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; MATERIAL=$2; E0=$3; TILT=$4; HISTORIES=$5
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/fourier_mode_cap_1320.py
pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H capture "$MATERIAL" "$E0" "$TILT" "$HISTORIES" "$OUT" > "$OUT/capture.log" 2>&1
echo "capture rc=$?"
if ls "$OUT"/flush-*.npz > /dev/null 2>&1; then
    pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H replay "$OUT" > "$OUT/replay.log" 2>&1
    echo "replay rc=$?"
fi
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null
