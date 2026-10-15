#!/bin/bash
#SBATCH --job-name=precision510
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=precision-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg PYRITE_MC_BACKEND=cuda
out="evidence/precision/$SLURM_JOB_ID"
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
nvidia-smi -q > "$out/gpu.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/gpu-processes.txt"
lscpu > "$out/cpu.txt"
cp .pyrite-sync "$out/code.txt"
cp checks/equal_precision/{precision_source.py,precision_checks.py,benchmark_precision.py,probe_precision.py,run_precision.py,precision-experiment.json,manifest.json,pyrite_adapter.py,precision.sbatch} "$out/source/"
cp src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_launch.py,_jit_radiative_device.py,_jit_device.py} "$out/source/"
if [ "$1" = preflight ]; then
    PYRITE_TEST_BACKEND=cuda "$HOME/.local/bin/uv" run --no-sync pyrite-dev test tests/montecarlo/test_hard_radiative_cuda.py --durations=0 > "$out/reference-tests.log" 2>&1
    "$HOME/.local/bin/uv" run --no-sync pyrite-dev test tests/performance/test_precision_experiment.py --durations=0 > "$out/harness-tests.log" 2>&1
fi
extra=()
if [ "$1" != preflight ]; then extra=(--preflight "$2"); fi
"$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/run_precision.py "$1" "$out" "${extra[@]}"
nvidia-smi -q > "$out/gpu-after.txt"
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
