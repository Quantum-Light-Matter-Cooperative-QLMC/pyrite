#!/bin/bash
# #1013 kernel cost: production exact-kernel time per retained row, tree against tree.
# Usage (from this checkout's root; each tree is a synced checkout with its own venv):
#   sbatch checks/fp32_speed_of_light/kernel_time.sbatch "<tree dirs>" "<scene:histories ...>" \
#       "<seeds>" "<precisions>"
# Runs every (tree, precision) arm per scene and seed, rotating the arm order per seed.
#SBATCH --job-name=ktime1013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=ktime1013-%j.log
set -euo pipefail
here="$SLURM_SUBMIT_DIR"
read -r -a trees <<< "${1:?tree dirs}"
scenes=${2:?scene:histories list}; seeds=${3:-"1 2 3"}; read -r -a precisions <<< "${4:-mixed}"
out="$here/evidence/ktime1013/$SLURM_JOB_ID"; mkdir -p "$out"
for tree in "${trees[@]}"; do
    (cd "$tree" && . checks/rocm/env.sh && pyrite_gpu_sync -q \
        && uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null)
    echo "$tree $(cat "$tree/SHA" 2>/dev/null || echo unknown)" >> "$out/trees.txt"
done
arms=()
for tree in "${trees[@]}"; do for p in "${precisions[@]}"; do arms+=("$tree:$p"); done; done
for spec in $scenes; do
    s=${spec%%:*} h=${spec##*:}
    for seed in $seeds; do
        r=$(( seed % ${#arms[@]} )); order=("${arms[@]:r}" "${arms[@]:0:r}")
        for arm in "${order[@]}"; do
            tree=${arm%:*} p=${arm##*:} tag="$(basename "$tree")-$p-$s-$seed"
            (cd "$tree" && . checks/rocm/env.sh \
                && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
                && uv run --no-sync python "$here/checks/fp32_speed_of_light/kernel_time.py" \
                    --stats "$out/$tag.stats.json" "$s" --histories "$h" --seed "$seed" \
                    --rows-per-round 1024 --device-staging --transport-precision "$p" --no-audit \
                    --warmup-histories -1 --output "$out/$tag.json" > "$out/$tag.log" 2>&1) \
                && echo "DONE $tag" || echo "FAIL $tag"
        done
    done
done
