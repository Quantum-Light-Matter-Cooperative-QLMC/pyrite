#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=g4-queen278-%j.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" UV_CACHE_DIR=/tmp/pyrite-uv-cache
prefix="$HOME/.local/opt/pyrite-issue-182/geant4"
export G4LEDATA="$prefix/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$prefix/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --no-sync python checks/continuum_geant4_precision.py --executable "$PWD/g4bench-build/g4bench" --pilot evidence/g4-precision-plan.json --output evidence/g4-precisionqueen278 --threads 16
