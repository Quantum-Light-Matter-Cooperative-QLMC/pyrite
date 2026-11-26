#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 03:00:00
#SBATCH -o fiber1480-%j.log
# Run from an isolated, synced checkout. One job = one paired diagnostic.
# Usage: sbatch checks/fiber_nonbasal_cxr_1480.sbatch OUT_DIR \
#   CRYSTAL E0_KEV THICK_NM TILT_DEG [additional frozen options]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if (( $# < 5 )); then
    echo "usage: $0 OUT_DIR CRYSTAL E0_KEV THICK_NM TILT_DEG [options]" >&2
    exit 2
fi
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
shift
# Source the same backend/environment selection used by existing GPU checks.
. checks/rocm/env.sh
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export XDG_CONFIG_HOME="$PWD/.xdg-config"
export PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite/catalog"
if [[ -f CODE_SHA ]]; then
    SHA=$(cat CODE_SHA)
else
    SHA=$(git rev-parse HEAD)
fi
printf '%s\n' "$SHA" > "$OUT/CODE_SHA"
printf '%q ' "$@" > "$OUT/arguments.txt"
printf '\n' >> "$OUT/arguments.txt"
echo "code $SHA host $(hostname) backend $PYRITE_MC_BACKEND job $SLURM_JOB_ID"
pyrite_gpu_sync -q
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u \
    checks/fiber_nonbasal_cxr_1480.py frozen "$@" --json "$OUT/result.json" \
    2>&1 | tee "$OUT/run.log"
(cd "$OUT" && sha256sum CODE_SHA arguments.txt gpu-info.txt run.log result.json result.npz > SHA256SUMS)
