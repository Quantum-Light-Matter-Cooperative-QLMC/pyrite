#!/bin/bash
# #825 check of the column-arena round downloads: CUDA continuation tests
# (slow included), then per-phase RSS peaks and timing (W 1 MeV / Si 300 keV,
# 8,000 histories; host with arena and per-round np.empty downloads, device
# staging; seeds 1-3).
#SBATCH --job-name=verify825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=verify825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="evidence/verify825/$SLURM_JOB_ID"; mkdir -p "$out"
cp CODE_SHA "$out/" 2>/dev/null || true
uv sync --package pyrite-mc --no-default-groups --group test --extra nvidia --locked -q
PYRITE_SLOW_TESTS=1 uv run --no-sync python -m pytest -q -p no:cacheprovider \
    tests/montecarlo/test_transport_continuation_cuda.py tests/montecarlo/test_transport_continuation.py \
    > "$out/tests.log" 2>&1
echo "rc=$? tests"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
for seed in 1 2 3; do
    modes=(host host-numpy staging); (( seed % 2 == 0 )) && modes=(staging host-numpy host)
    for s in w-1MeV-100um si-300keV-100um; do
        for mode in "${modes[@]}"; do
            stem="$out/$s-$mode-s$seed"
            flags=()
            env=()
            [[ $mode == staging ]] && flags+=(--device-staging)
            [[ $mode == host-numpy ]] && env+=(SPILL_PROBE_NUMPY_DST=1)
            env "${env[@]}" SPILL_PROBE_RSS="$stem.rss.json" PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
                /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
                checks/geant4_performance/spill825_probe.py \
                --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
                "$s" --histories 8000 --block 8000 --seed "$seed" "${flags[@]}" --no-audit \
                --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
                --output "$stem.json" > "$stem.log" 2>&1
            echo "rc=$? $s $mode s$seed"
        done
    done
done
