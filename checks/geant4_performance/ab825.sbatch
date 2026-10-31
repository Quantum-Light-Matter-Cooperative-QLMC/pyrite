#!/bin/bash
# #825 A/B: continuation transport at a base tree (AB_BASE, default
# ~/pyrite-825-base) vs this tree, no probes. W 1 MeV / Si 300 keV, 8,000
# histories, host and device staging, seeds 1-3, trees alternated per seed.
# Usage: sbatch checks/geant4_performance/ab825.sbatch
#SBATCH --job-name=ab825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:15:00
#SBATCH --output=ab825-%j.log
set -uo pipefail
new="$SLURM_SUBMIT_DIR"; base="${AB_BASE:-$HOME/pyrite-825-base}"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$new/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$new/user-catalog" XDG_CONFIG_HOME="$new/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="$new/evidence/ab825/$SLURM_JOB_ID"; mkdir -p "$out"
for t in "$base" "$new"; do
    (cd "$t" && uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q \
        && uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null)
    echo "rc=$? sync $t $(cat "$t/CODE_SHA")"
done
for seed in 1 2 3; do
    trees=(base new); (( seed % 2 == 0 )) && trees=(new base)
    for s in w-1MeV-100um si-300keV-100um; do
        for mode in host staging; do
            for tree in "${trees[@]}"; do
                dir=$new; [[ $tree == base ]] && dir=$base
                stem="$out/$s-$mode-$tree-s$seed"
                flags=()
                [[ $mode == staging ]] && flags+=(--device-staging)
                (cd "$dir" && PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
                    /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
                    checks/fp32_speed_of_light/precision_variants.py \
                    --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
                    "$s" --histories 8000 --block 8000 --seed "$seed" "${flags[@]}" --no-audit \
                    --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
                    --output "$stem.json" > "$stem.log" 2>&1)
                echo "rc=$? $s $mode $tree s$seed"
            done
        done
    done
done
