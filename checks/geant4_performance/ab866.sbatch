#!/bin/bash
# #866 A/B: host reassembly at a base tree (AB_BASE, default ~/pyrite-866-base,
# main) vs device staging (run-copy join, pinned D2H) in this tree, no probes.
# Si 300 keV, W 300 keV, W 1 MeV, Si 1 MeV; 8,000 histories; seeds 1-3; arm
# order alternated per seed. Compare transport time, assembly wall, peak RSS,
# pool peak, spill decision and raw-score digests.
# Usage: sbatch checks/geant4_performance/ab866.sbatch
#SBATCH --job-name=ab866
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=ab866-%j.log
set -uo pipefail
new="$SLURM_SUBMIT_DIR"; base="${AB_BASE:-$HOME/pyrite-866-base}"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$new/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$new/user-catalog" XDG_CONFIG_HOME="$new/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="$new/evidence/ab866/$SLURM_JOB_ID"; mkdir -p "$out"
for t in "$base" "$new"; do
    (cd "$t" && uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q \
        && uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null)
    echo "rc=$? sync $t $(cat "$t/CODE_SHA")"
done
for seed in 1 2 3; do
    arms=(host staging); (( seed % 2 == 0 )) && arms=(staging host)
    for s in si-300keV-100um w-300keV-10um w-1MeV-100um si-1MeV-1mm; do
        for mode in "${arms[@]}"; do
            dir=$new; flags=(--device-staging)
            [[ $mode == host ]] && dir=$base && flags=()
            stem="$out/$s-$mode-s$seed"
            (cd "$dir" && PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
                /usr/bin/time -v -o "$stem.time" timeout 900 uv run --no-sync python \
                checks/fp32_speed_of_light/precision_variants.py \
                --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
                "$s" --histories 8000 --block 8000 --seed "$seed" "${flags[@]}" --no-audit \
                --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
                --output "$stem.json" > "$stem.log" 2>&1)
            echo "rc=$? $s $mode s$seed"
        done
    done
done
