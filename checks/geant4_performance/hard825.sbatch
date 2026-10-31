#!/bin/bash
# #825 hard-photon fields A/B: base tree (AB_BASE, default ../base) vs this
# tree. W 1 MeV / 100 um, 8,000 histories, host continuation; per-phase peak
# RSS and wall (hard825_probe.py), seeds 1-2 timing arms alternated per seed,
# then one digest arm per tree (field SHA-256 plus raw score digest).
# Usage: sbatch checks/geant4_performance/hard825.sbatch
#SBATCH --job-name=hard825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=hard825-%j.log
set -uo pipefail
new="$SLURM_SUBMIT_DIR"; base="${AB_BASE:-$new/../base}"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$new/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$new/user-catalog" XDG_CONFIG_HOME="$new/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="$new/evidence/hard825/$SLURM_JOB_ID"; mkdir -p "$out"
for t in "$base" "$new"; do
    (cd "$t" && uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q \
        && uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null)
    echo "rc=$? sync $t $(cat "$t/CODE_SHA")" | tee -a "$out/trees.txt"
done
run() {  # tree seed tag [env...]
    local tree=$1 seed=$2 tag=$3; shift 3
    local dir=$new; [[ $tree == base ]] && dir=$base
    local stem="$out/w-1MeV-100um-$tree-$tag-s$seed"
    (cd "$dir" && env "$@" HARD825_OUT="$stem.probe.json" \
        PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
        /usr/bin/time -v -o "$stem.time" timeout 900 uv run --no-sync python \
        checks/geant4_performance/hard825_probe.py \
        --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        w-1MeV-100um --histories 8000 --block 8000 --seed "$seed" --no-audit \
        --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
        --output "$stem.json" > "$stem.log" 2>&1)
    echo "rc=$? $tree $tag s$seed"
}
for seed in 1 2; do
    trees=(base new); (( seed % 2 == 0 )) && trees=(new base)
    for tree in "${trees[@]}"; do run "$tree" "$seed" time; done
done
for tree in base new; do run "$tree" 1 digest HARD825_DIGEST=1; done
