#!/bin/bash
#SBATCH --job-name=issue341-matrix-reduce-981c1141
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=reduce-%j.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/.local/opt/pyrite-341-cascade-981c1141
export PYRITE_HOME=$HOME/.local/opt/pyrite-issue-341/home
export PYRITE_CATALOG=$ROOT/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$ROOT/empty-user-catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1
cd "$ROOT"
overall=0
for case in si_0p3mev si_1mev si_3mev si_10mev w_1mev w_3mev w_10mev; do
  /usr/bin/time -v -o "matrix/$case.resources" uv run --locked python checks/photon_full_track/compare.py reduce-geant4-cascade "$case" "matrix/$case.json" matrix/results/"$case"_s*_cascade.log.gz
  uv run --locked python checks/photon_full_track/compare.py compact "matrix/$case.json" "matrix/$case.json.gz"
  status=0
  uv run --locked python checks/photon_full_track/compare.py compare-cascade "$case" "matrix/$case.json.gz" --output "matrix/$case-report.json" || status=$?
  printf '%s\n' "$status" > "matrix/$case-exit-code.txt"
  if [ "$status" -ne 0 ]; then overall=1; fi
done
sha256sum matrix/*.json matrix/*.json.gz > matrix/reduction-sha256.txt
exit "$overall"
