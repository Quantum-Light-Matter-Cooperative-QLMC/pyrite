#!/bin/bash
# #825 host-side retained-row attribution: continuation transport with device
# staging on/off, cProfile of the warm request plus the continuation assembly
# wall. Exact kernel, 32 threads/block, production scratch (512 MiB).
# Usage: sbatch checks/geant4_performance/host825.sbatch  (HOST_HISTORIES=8000)
#SBATCH --job-name=host825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=host825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/host825/$SLURM_JOB_ID"; mkdir -p "$out"
cp CODE_SHA "$out/" 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.csv"
uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
N=${HOST_HISTORIES:-8000}
arms=(
  w-1MeV-100um:staging
  w-1MeV-100um:host
  si-300keV-100um:staging
  si-300keV-100um:host
)
for a in "${arms[@]}"; do
    IFS=: read -r s mode <<< "$a"
    stem="$out/$s-$mode-n$N"
    flags=()
    [[ $mode == staging ]] && flags+=(--device-staging)
    PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
        /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
        checks/fp32_speed_of_light/precision_variants.py \
        --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        "$s" --histories "$N" --block "$N" --seed 1 "${flags[@]}" --no-audit \
        --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
        --cprofile "$stem.prof" --output "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $a"
    uv run --no-sync python -c "import pstats,sys; p=pstats.Stats(sys.argv[1]); \
p.sort_stats('cumulative').print_stats(45); p.sort_stats('tottime').print_stats(30)" \
        "$stem.prof" > "$stem.pstats.txt" 2>&1
done
