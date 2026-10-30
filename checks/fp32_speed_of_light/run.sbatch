#!/bin/bash
# #831 FP32 speed-of-light: exact / fp32 / mixed transport kernels on the #821 scenes.
# Usage (from the synced checkout root, with the #821 harness copied to bench821/):
#   sbatch checks/fp32_speed_of_light/run.sbatch "<variants>" "<scene:histories ...>" "<seeds>"
#SBATCH --job-name=fp32sol831
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=fp32sol831-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=$HOME/.cache/uv PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
variants=${1:-"exact fp32 mixed mixed_math"}; scenes=${2:-"si-300keV-100um:4000"}; seeds=${3:-"1 2 3"}
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
out="evidence/fp32sol831/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
export PV_DUMP_DIR="$out/source"
for spec in $scenes; do
    s=${spec%%:*} h=${spec##*:}
    for seed in $seeds; do
        # Rotate variant order per seed so drift cannot favour one arm.
        vs=($variants); r=$(( seed % ${#vs[@]} )); vs=("${vs[@]:r}" "${vs[@]:0:r}")
        for v in "${vs[@]}"; do
            uv run --no-sync python checks/fp32_speed_of_light/precision_variants.py \
                --variant "$v" --bench bench821/pyrite_bench.py --stats "$out/$v-$s-$seed.stats.json" \
                "$s" --histories "$h" --seed "$seed" --rows-per-round 1024 --device-staging \
                --output "$out/$v-$s-$seed.json" > "$out/$v-$s-$seed.log" 2>&1 \
                && echo "DONE $v $s $seed" || echo "FAIL $v $s $seed"
        done
    done
done
