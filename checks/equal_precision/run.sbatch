#!/bin/bash
#SBATCH --job-name=precision510
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=benchmark-%j.log
set -euo pipefail
# Submit from the isolated checkout; no production checkpoints are read or written.
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export G4LEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$HOME/.local/opt/pyrite-issue-182/geant4/lib:${LD_LIBRARY_PATH:-}"
UV="$HOME/.local/bin/uv"
mkdir -p evidence
scontrol show job "$SLURM_JOB_ID" > "evidence/job-$SLURM_JOB_ID.txt"
lscpu > "evidence/cpu-$SLURM_JOB_ID.txt"
nvidia-smi -q > "evidence/gpu-$SLURM_JOB_ID.txt"
cat .pyrite-sync > "evidence/code-$SLURM_JOB_ID.txt"
if [ ! -x benchmark-geant4-v2/build/TestEm5 ]; then
    mkdir -p benchmark-geant4-v2
    cp -a "$HOME/.local/opt/pyrite-issue-182/source-emission/examples/extended/electromagnetic/TestEm5" benchmark-geant4-v2/source
    "$UV" run --no-sync python checks/equal_precision/prepare_geant4.py benchmark-geant4-v2/source
    cmake -S benchmark-geant4-v2/source -B benchmark-geant4-v2/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGeant4_DIR="$HOME/.local/opt/pyrite-issue-182/geant4/lib/cmake/Geant4" \
        -DCMAKE_PREFIX_PATH="$HOME/.local/opt/pyrite-issue-182/geant4" \
        -DCMAKE_CXX_COMPILER="$HOME/.local/opt/pyrite-issue-182/geant4/bin/x86_64-conda-linux-gnu-g++"
    cmake --build benchmark-geant4-v2/build -j 1
fi
"$UV" run --no-sync python checks/equal_precision/run_pilot.py smoke
if [ "${1:-smoke}" != smoke ]; then
    "$UV" run --no-sync python checks/equal_precision/run_pilot.py "$1"
fi
