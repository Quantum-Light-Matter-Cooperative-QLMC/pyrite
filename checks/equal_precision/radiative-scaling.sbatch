#!/bin/bash
#SBATCH --job-name=precision510radlarge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=radiative-scaling-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
OUT="evidence/radiative-optimization/$SLURM_JOB_ID"
mkdir -p "$OUT/source"
scontrol show job "$SLURM_JOB_ID" > "$OUT/job.txt"
nvidia-smi -q > "$OUT/gpu.txt"
cp checks/equal_precision/{benchmark_radiative.py,diagnose_kernel.py,pyrite_adapter.py,manifest.json,radiative-scaling.json,run_radiative_scaling.py,radiative-scaling.sbatch,radiative_baseline.py} "$OUT/source/"
cp src/pyrite/montecarlo/transport/_jit_radiative_device.py "$OUT/source/candidate.py"
sha256sum "$OUT"/source/* > "$OUT/code.txt"
"$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/run_radiative_scaling.py "$1" "$OUT"
