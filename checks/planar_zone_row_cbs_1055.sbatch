#!/bin/bash
# #1055 planar zone-row CBS matrix; submit from ~/pyrite-1055:
#   sbatch --array=0-6 checks/planar_zone_row_cbs_1055.sbatch
#SBATCH --job-name=zone-cbs-1055
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/aamador/pyrite-1055/runs/%x-%A-%a.log
export PATH=$HOME/.local/bin:$PATH
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$HOME/pyrite-1055/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_MC_BACKEND=cpu NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $HOME/pyrite-1055
mkdir -p runs
CASES=(
  "silicon 0.0 60"
  "silicon 0.8 60"
  "silicon 2.2 60"
  "silicon 4.0 100"
  "diamond 0.0 100"
  "diamond 1.0 100"
  "diamond 3.0 120"
)
read -r CRYSTAL PSI EMAX <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
echo "sha $(cat SHA) case $CRYSTAL psi=$PSI mrad"
uv run --frozen python checks/planar_zone_row_cbs.py --crystal "$CRYSTAL" --psi-mrad "$PSI" \
  --energy-MeV 17 --thickness-um 20 --electrons 2000 --chunks 20 \
  --e-min-keV 5 --e-max-keV "$EMAX" --n-energy $(( (EMAX - 5) * 10 + 1 )) \
  --band cr:10:25 --band cbs:25:40 \
  --out "runs/zone-cbs-1055-$CRYSTAL-$PSI.json"
