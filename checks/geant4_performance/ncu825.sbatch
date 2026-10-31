#!/bin/bash
# #825/#796 hardware-counter baseline of the exact CUDA transport kernel.
# Step 1: one-kernel ncu probe (fails fast on ERR_NVGPUCTRPERM).
# Step 2: `--set full` on a few steady-state transport launches per #821 scene,
#         production schedule (auto continuation), pilot-3 flags.
# Usage: sbatch ncu825.sbatch [skip] [count]   (from the synced checkout root)
#SBATCH --job-name=ncu825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=ncu825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
skip=${1:-10} count=${2:-4}
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/ncu825/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; nvidia-smi > "$out/gpu.txt"; ncu --version > "$out/ncu-version.txt"
uv sync --extra nvidia --locked -q
py=(uv run --no-sync python)

echo "== probe"
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_active \
    "${py[@]}" -c "import cupy as cp; x = cp.arange(1 << 20); print(int((x + 1).sum()))" \
    > "$out/probe.txt" 2>&1
cat "$out/probe.txt"
if grep -q -E "ERR_NVGPUCTRPERM|does not have permission" "$out/probe.txt"; then
    echo "counters refused"; exit 3
fi

for sh in si-1MeV-1mm:2500 w-1MeV-100um:1250; do
    s=${sh%%:*} n=${sh#*:}
    echo "== $s"
    ncu --target-processes all --kernel-name "regex:transport_kernel" \
        --launch-skip "$skip" --launch-count "$count" --set full --import-source yes \
        --replay-mode kernel -f -o "$out/$s" \
        "${py[@]}" checks/geant4_performance/pyrite_bench.py "$s" --histories "$n" --seed 1 \
        --rows-per-round 1024 --device-staging --warmup-histories 200 \
        --output "$out/$s.json" > "$out/$s.log" 2>&1
    echo "rc=$?"; tail -3 "$out/$s.log"
    ncu --import "$out/$s.ncu-rep" --page details --csv > "$out/$s-details.csv" 2>&1
    ncu --import "$out/$s.ncu-rep" --page raw --csv > "$out/$s-raw.csv" 2>&1
    ncu --import "$out/$s.ncu-rep" --page details --section WarpStateStats \
        --section Occupancy --section ComputeWorkloadAnalysis --section LaunchStats \
        --section SourceCounters --section MemoryWorkloadAnalysis | head -400
done
