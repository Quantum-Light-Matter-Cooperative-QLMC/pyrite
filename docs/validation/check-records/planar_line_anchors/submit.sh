#!/bin/bash
# Planar line anchors (#1057): default arm per case plus convergence,
# divergence and history-reuse control arms.
#SBATCH --job-name=1057-anchors
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --output=/home/aamador/pyrite-1057/scratch/anchors/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-1057
export PATH=$HOME/.local/bin:$PATH
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_SOURCE_REVISION=b7d61539e
mkdir -p scratch/anchors
genz=()
for plane in 110 111; do for e in 5.2 9; do for z in 13 20 30 55; do
  genz+=("--datasets genz --cases genz_${plane}_${e}_${z}"); done; done; done
arms=("${genz[@]}"
  "--datasets kephart --cases kephart_110"
  "--datasets kephart --cases kephart_100"
  "--datasets azadegan --cases azadegan_110_14.6"
  "--datasets azadegan --cases azadegan_110_17_"
  "--datasets azadegan --cases azadegan_110_30"
  "--datasets azadegan --cases azadegan_110_34"
  "--datasets genz --cases genz_110_9_13 --cells 16 --orders 24 --dz 50 --bins 12 --grid 4000"
  "--datasets genz --cases genz_111_9_55 --cells 16 --orders 24 --dz 50 --bins 12 --grid 4000"
  "--datasets genz --cases genz_110_5.2_55 --cells 16 --orders 24 --dz 50 --bins 12 --grid 4000"
  "--datasets genz --cases genz_110_9_13 genz_110_9_55 genz_111_9_13 genz_111_9_55 --divergence-scale 2"
  "--datasets genz --cases genz_110_5.2_13 --models kernel --bins 3 --no-history-reuse"
  "--datasets kephart --cases kephart_110 --cells 16 --orders 24 --dz 50 --bins 12"
  "--datasets azadegan --cases azadegan_110_30 --cells 16 --orders 24 --dz 50 --bins 12 --grid 4000"
  "--datasets azadegan --cases azadegan_110_17_ --cells 16 --orders 24 --dz 50 --bins 12 --grid 4000")
id=$SLURM_ARRAY_TASK_ID
echo "arm $id: ${arms[$id]}"
read -r -a argv <<< "${arms[$id]}"
/usr/bin/time -v uv run --no-sync python checks/planar_line_anchors.py "${argv[@]}" \
  --out "scratch/anchors/arm-$id.json"
