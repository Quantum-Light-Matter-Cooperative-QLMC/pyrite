#!/bin/bash
#SBATCH --job-name=g4-escape278
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=g4-wbenchmark278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
prefix="$HOME/.local/opt/pyrite-issue-182/geant4"
export G4LEDATA="$prefix/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$prefix/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
mkdir -p evidence/g4-wbenchmark278
scontrol show node qlmc-queen > evidence/g4-wbenchmark278/allocation.txt
sha256sum checks/escaped278/*.py checks/escaped278/*.cc checks/escaped278/*.sbatch checks/escaped278/CMakeLists.txt g4escaped-build/g4bench > evidence/g4-wbenchmark278/harness-sha256.txt
arm=${1:-tune}
case "$arm" in
  tune)
    uv run --no-sync python checks/escaped278/audit_native.py --output evidence/g4-wbenchmark278-audit
    uv run --no-sync python checks/escaped278/run.py geant4 --smoke --output evidence/g4-wbenchmark278-smoke
    uv run --no-sync python checks/escaped278/tune.py --msc 0 --output evidence/g4-wbenchmark278-tune-ss
    uv run --no-sync python checks/escaped278/tune.py --msc 1 --output evidence/g4-wbenchmark278-tune-msc
    ;;
  ss-analogue)
    uv run --no-sync python checks/escaped278/run.py geant4 --output evidence/g4-wbenchmark278-ss-analogue --max-hours 3.5
    ;;
  ss-split)
    uv run --no-sync python checks/escaped278/run.py geant4 --output evidence/g4-wbenchmark278-ss-split --split-plan evidence/g4-wbenchmark278-tune-ss/plan.json --max-hours 3.5
    ;;
  msc-split)
    uv run --no-sync python checks/escaped278/run.py geant4 --msc 1 --output evidence/g4-wbenchmark278-msc-split --split-plan evidence/g4-wbenchmark278-tune-msc/plan.json --max-hours 3.5
    ;;
  *) exit 2 ;;
esac
