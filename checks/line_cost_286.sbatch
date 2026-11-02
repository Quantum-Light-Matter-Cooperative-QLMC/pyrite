#!/bin/bash
# #286 bin-mean line-phase attribution at 29,999 keV on current main code.
# Usage: sbatch checks/line_cost_286.sbatch   (from the synced checkout root)
#SBATCH --job-name=line286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=line286-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/line286/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
uv run --no-sync pytest -q -p no:cacheprovider tests/montecarlo/spectrum/test_sinc_bin_integration_cuda.py tests/montecarlo/spectrum/test_sinc_bin_near_far.py > "$out/tests.log" 2>&1
echo "pytest rc=$?"
py=(uv run --no-sync python -W ignore checks/line_cost_286.py --quadrature bin-mean)
"${py[@]}" --energy-kev 300 --counts 100 400 --repeat 2 --output "$out/hopg-300.json"
echo "rc=$?"
"${py[@]}" --energy-kev 29999 --counts 100 400 --repeat 1 --output "$out/hopg-29999.json"
echo "rc=$?"
