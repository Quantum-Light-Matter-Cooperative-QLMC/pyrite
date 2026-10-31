#!/bin/bash
# #825 device vs host staging after the #861 run copy and the spill pool
# release: continuation (t32, exact kernel, 512 MiB scratch), 8,192 histories,
# seeds 1-3, modes alternated per seed. Fitting scenes (Si/W 300 keV, the #842
# gate's) and a spilling one (W 1 MeV). Raw-score digests must match per seed.
# Usage: sbatch checks/geant4_performance/staging825.sbatch
#SBATCH --job-name=staging825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=staging825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="evidence/staging825/$SLURM_JOB_ID"; mkdir -p "$out"
cp CODE_SHA "$out/" 2>/dev/null || true
uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
N=${STAGING_HISTORIES:-8192}
for seed in 1 2 3; do
    modes=(staging host); (( seed % 2 == 0 )) && modes=(host staging)
    for s in si-300keV-100um w-300keV-10um w-1MeV-100um; do
        for mode in "${modes[@]}"; do
            stem="$out/$s-$mode-s$seed"
            flags=()
            [[ $mode == staging ]] && flags+=(--device-staging)
            PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
                /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
                checks/fp32_speed_of_light/precision_variants.py \
                --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
                "$s" --histories "$N" --block "$N" --seed "$seed" "${flags[@]}" --no-audit \
                --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
                --output "$stem.json" > "$stem.log" 2>&1
            echo "rc=$? $s $mode s$seed"
        done
    done
done
