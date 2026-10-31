#!/bin/bash
# #825 device-staging spill diagnosis: W 1 MeV continuation (8,000 histories)
# with device staging on/off under spill825_probe.py, which logs per-round
# download wall, RSS and kernel reclaim/compaction/THP/cgroup counters.
# Usage: sbatch checks/geant4_performance/spill825.sbatch  (SPILL_HISTORIES=8000)
#SBATCH --job-name=spill825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=spill825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/spill825/$SLURM_JOB_ID"; mkdir -p "$out"
cp CODE_SHA "$out/" 2>/dev/null || true
{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  for f in enabled defrag; do echo "thp_$f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done
  free -b; cat /proc/self/cgroup
} > "$out/host.txt" 2>&1
uv sync --package pyrite-mc --no-default-groups --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
N=${SPILL_HISTORIES:-8000}
arms=(${SPILL_ARMS:-staging host staging host})
i=0
for mode in "${arms[@]}"; do
    stem="$out/w-1MeV-100um-$mode-n$N-r$i"; i=$((i + 1))
    flags=()
    [[ $mode == staging ]] && flags+=(--device-staging)
    SPILL_PROBE_OUT="$stem.jsonl" PYRITE_MC_TRANSPORT_SCHEDULE=continuation PV_NTHREADS=32 \
        /usr/bin/time -v -o "$stem.time" timeout 600 uv run --no-sync python \
        checks/geant4_performance/spill825_probe.py \
        --variant exact --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        w-1MeV-100um --histories "$N" --block "$N" --seed 1 "${flags[@]}" --no-audit \
        --warmup-histories 200 --scratch-mib 512 --seg-capacity 512 --probe-electrons 1024 \
        --output "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $mode"
done
