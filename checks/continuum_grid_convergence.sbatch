#!/bin/bash
# #278 plan item 2: continuum grid candidates on fixed tracks, production path
# (attenuation + Timepix3/EagleXO). HOPG 1 mm at 45/180 deg (the
# hopg_high_energy geometry) at 300 keV and 29,999 keV, plus tungsten 100 um at
# 1 MeV (L/K edges in the band). Reference: uniform 25 eV; truth: dense refined
# geometric grid. Run from a synced checkout root.
#SBATCH --job-name=cgc278
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --output=cgc278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
uv sync --extra nvidia --locked -q
out="evidence/cgc278/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
grids=(--geometric-nodes 257 --geometric-nodes 513 --geometric-nodes 1025
       --geometric-nodes 2049 --geometric-nodes 4097
       --gl-order 2 --gl-order 4 --gl-order 8 --gl-ratio 2 --gl-ratio 1.25)
uv run --no-sync python checks/continuum_grid_convergence.py --material hopg \
    --energy-kev 300 --energy-kev 29999 --n-electrons 150 "${grids[@]}" \
    --output "$out/hopg.json" > "$out/hopg.log" 2>&1 || echo "FAIL hopg"
echo "DONE hopg"
uv run --no-sync python checks/continuum_grid_convergence.py --material tungsten \
    --energy-kev 1000 --thickness-ang 1e6 --tilt-deg 5 --tilt-azim-deg 0 \
    --n-electrons 150 "${grids[@]}" \
    --output "$out/tungsten.json" > "$out/tungsten.log" 2>&1 || echo "FAIL tungsten"
echo "DONE tungsten"
