#!/bin/bash
# #825 spill follow-up: D2H-vs-pool microbenchmark, then W 1 MeV probe arms
# (staging x3, staging+free-pool x3, host+malloc_trim x2).
#SBATCH --job-name=spill825b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=spill825b-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
out="evidence/spill825/$SLURM_JOB_ID"; mkdir -p "$out"
cp CODE_SHA "$out/" 2>/dev/null || true
uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q
uv run --no-sync python checks/geant4_performance/d2h825.py > "$out/d2h.jsonl" 2> "$out/d2h.err"
echo "rc=$? d2h"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
arms=(staging staging-freepool host-trim staging staging-freepool host-trim staging staging-freepool)
i=0
for arm in "${arms[@]}"; do
    stem="$out/w-1MeV-100um-$arm-r$i"; i=$((i + 1))
    flags=() env=()
    [[ $arm == staging* ]] && flags+=(--device-staging)
    [[ $arm == *freepool ]] && env+=(SPILL_PROBE_FREEPOOL=1)
    [[ $arm == *trim ]] && env+=(SPILL_PROBE_TRIM=1)
    env "${env[@]}" SPILL_PROBE_OUT="$stem.jsonl" PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
        /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
        checks/geant4_performance/spill825_probe.py \
        --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        w-1MeV-100um --histories 8000 --block 8000 --seed 1 "${flags[@]}" --no-audit \
        --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
        --cprofile "$stem.prof" --output "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $arm"
done
