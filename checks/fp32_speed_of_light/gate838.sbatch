#!/bin/bash
# #838 FP32-FP64 refinement gate: fp64 and mixed transport_precision on the
# #821 anchors, paired by seed; gate838.py evaluates the #510 one-third bound.
# Usage (from the synced checkout root):
#   sbatch checks/fp32_speed_of_light/gate838.sbatch "<scene:histories ...>" "<seeds>" "<estimators>"
#SBATCH --job-name=gate838
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=08:00:00
#SBATCH --output=gate838-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=$HOME/.cache/uv PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
scenes=${1:?scene:histories list}; seeds=${2:-"201 202 203"}; estimators=${3:-discrete}
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
out="evidence/gate838/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; nvidia-smi > "$out/gpu.txt"
for spec in $scenes; do
    s=${spec%%:*} h=${spec##*:}
    for est in $estimators; do
        extra=()
        [[ $est == continuous ]] && extra=(--estimator continuous --cone-mode point)
        for seed in $seeds; do
            # Alternate precision order per seed so drift cannot favour one arm.
            ps=(fp64 mixed); (( seed % 2 )) && ps=(mixed fp64)
            for p in "${ps[@]}"; do
                stem="$out/$p-$est-$s-$seed"
                uv run --no-sync python checks/geant4_performance/pyrite_bench.py "$s" \
                    --histories "$h" --seed "$seed" --rows-per-round 1024 --device-staging \
                    --transport-precision "$p" --no-audit "${extra[@]}" --output "$stem.json" \
                    > "$stem.log" 2>&1 && echo "DONE $p $est $s $seed" || echo "FAIL $p $est $s $seed"
            done
        done
    done
done
uv run --no-sync python checks/fp32_speed_of_light/gate838.py "$out" --output "$out/summary.json" \
    > "$out/summary.md"
