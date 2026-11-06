#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o scratch/roulette879/bench-%j.log
# Run from a clean isolated checkout at the pushed SHA. Arguments: output root and
# frozen line grid. Each arm is its own process, so a device failure stops only
# that arm. REAL and the transport settings are read at import.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/scratch/roulette879/empty-user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda
export PYRITE_TEST_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG"
git check-ignore scratch/roulette879/probe || exit 1
sha="$(git rev-parse HEAD)"
root="$1"
grid="$2"
nvidia-smi
uv sync -q --locked --no-default-groups --extra nvidia --group dev || exit 1

transport_arm() {  # label electrons block core fp64 precision schedule
  echo "=== transport $1"
  PYRITE_FP64="$5" PYRITE_MC_TRANSPORT_PRECISION="$6" PYRITE_MC_TRANSPORT_SCHEDULE="$7" \
    uv run --no-sync python checks/line_roulette_tail_879.py --out "$root/transport-$1" \
    --code-sha "$sha" --ne "$2" --block "$3" --core "$4" || echo "FAILED transport $1 rc=$?"
}

request_arm() {  # label core fp64
  echo "=== request $1"
  PYRITE_FP64="$3" uv run --no-sync python checks/line_roulette_core_bench_879.py \
    --out "$root/request-$1" --code-sha "$sha" --grid "$grid" --core "$2" \
    || echo "FAILED request $1 rc=$?"
}

transport_arm cpu-b4096 16384 4096 per-electron 1 auto auto
transport_arm cuda-fp64-fp64-b4096 65536 4096 cuda 1 fp64 auto
transport_arm cuda-mixed-fp64-b4096 65536 4096 cuda 1 mixed auto
transport_arm cuda-mixed-fp32-b4096 65536 4096 cuda 0 mixed auto
transport_arm cuda-mixed-fp32-b16384 65536 16384 cuda 0 mixed auto
transport_arm cuda-mixed-fp32-b65536 65536 65536 cuda 0 mixed auto
transport_arm cuda-mixed-fp32-b16384-full 65536 16384 cuda 0 mixed full-history

request_arm cpu-fp64 per-electron 1
request_arm cuda-fp64 cuda 1
request_arm cuda-fp32 cuda 0

echo "=== float32 mass jitter"
PYRITE_FP64=0 uv run --no-sync python checks/line_roulette_mass_jitter_879.py \
  || echo "FAILED jitter rc=$?"
echo "=== float32 roulette tests"
PYRITE_FP64=0 uv run --no-sync pyrite-dev test tests/montecarlo/spectrum/test_line_roulette.py \
  || echo "FAILED float32 tests rc=$?"
echo "=== complete"
