#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o scratch/roulette879/fom-%j.log
# Run from a clean isolated checkout at the pushed SHA. First argument is the
# output root; the rest pass verbatim to checks/line_roulette_fom_879.py.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/scratch/roulette879/empty-user-catalog"
export PYRITE_MC_BACKEND=cuda
export PYRITE_TEST_BACKEND=cuda
export PYRITE_FP64="${PYRITE_FP64:-0}"  # production float32 scoring; 1 forces float64
# fp64 transport: mixed saved 4-10% in job 4017, and its MeV tail effect is unresolved.
export PYRITE_MC_TRANSPORT_PRECISION="${PYRITE_MC_TRANSPORT_PRECISION:-fp64}"
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG"
git check-ignore scratch/roulette879/probe
sha="$(git rev-parse HEAD)"
nvidia-smi
pixi install --locked -e nvidia
# Recorded, not fatal: the float32 results are themselves what these two steps measure.
pixi run --as-is -e nvidia python checks/line_roulette_mass_jitter_879.py || echo "FAILED jitter rc=$?"
pixi run --as-is -e nvidia pyrite-dev test tests/montecarlo/spectrum/test_line_roulette.py --durations=5 \
  || echo "FAILED roulette tests rc=$?"
output_root="$1"; shift
check="${CHECK:-checks/line_roulette_fom_879.py}"  # or checks/line_roulette_blocks_879.py
pixi run --as-is -e nvidia python "$check" --out "$output_root" --code-sha "$sha" "$@"
