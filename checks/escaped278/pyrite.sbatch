#!/bin/bash
#SBATCH --job-name=py-escape278
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=04:00:00
#SBATCH --output=py-wbenchmark278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
trap 'touch "evidence/hold-release-$1"' EXIT
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p evidence/py-wbenchmark278
scontrol show node DESKTOP-QNIHO3D > evidence/py-wbenchmark278/allocation.txt
squeue -w DESKTOP-QNIHO3D > evidence/py-wbenchmark278/jobs.txt
nvidia-smi > evidence/py-wbenchmark278/gpu.txt
sha256sum checks/escaped278/*.py checks/escaped278/*.cc checks/escaped278/*.sbatch checks/escaped278/CMakeLists.txt > evidence/py-wbenchmark278/harness-sha256.txt
uv run --no-sync python checks/escaped278/run.py pyrite --smoke --output evidence/py-wbenchmark278-smoke
uv run --no-sync python checks/escaped278/run.py pyrite --output evidence/py-wbenchmark278 --max-hours 3.3
