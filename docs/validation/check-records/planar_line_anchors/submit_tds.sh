#!/bin/bash
# Planar line anchors (#1057): uniform TDS x1.3 sensitivity (correlation stand-in).
#SBATCH --job-name=1057-tds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/aamador/pyrite-1057-tds/scratch/anchors/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-1057-tds
export PATH=$HOME/.local/bin:$PATH
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_SOURCE_REVISION=64c1b6a9d
mkdir -p scratch/anchors
arms=(
  "--datasets genz --cases genz_110_9_13 genz_110_9_55 genz_111_9_13 genz_111_9_55 --tds-scale 1.3 --models kernel"
  "--datasets genz --cases genz_110_5.2_13 genz_111_5.2_13 --tds-scale 1.3 --models kernel"
  "--datasets azadegan --cases azadegan_110_14.6 azadegan_110_17_ --tds-scale 1.3 --models kernel")
id=$SLURM_ARRAY_TASK_ID
echo "arm $id: ${arms[$id]}"
read -r -a argv <<< "${arms[$id]}"
/usr/bin/time -v uv run --no-sync python checks/planar_line_anchors.py "${argv[@]}" \
  --out "scratch/anchors/arm-$id.json"
