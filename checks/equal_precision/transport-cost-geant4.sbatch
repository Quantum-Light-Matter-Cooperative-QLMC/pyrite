#!/bin/bash
#SBATCH --job-name=transport786-g4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=transport-cost-geant4-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" PYRITE_MC_BACKEND=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
g4="$HOME/.local/opt/pyrite-issue-182/geant4"
export G4LEDATA="$g4/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$g4/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$g4/lib:${LD_LIBRARY_PATH:-}"
out="evidence/transport-cost-geant4/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
lscpu > "$out/cpu.txt"
if [[ ! -x benchmark-geant4-v2/build/TestEm5 ]]; then
    mkdir -p benchmark-geant4-v2
    cp -a "$HOME/.local/opt/pyrite-issue-182/source-emission/examples/extended/electromagnetic/TestEm5" \
        benchmark-geant4-v2/source
    uv run --no-sync python checks/equal_precision/prepare_geant4.py benchmark-geant4-v2/source
    cmake -S benchmark-geant4-v2/source -B benchmark-geant4-v2/build \
        -DCMAKE_BUILD_TYPE=Release -DGeant4_DIR="$g4/lib/cmake/Geant4" \
        -DCMAKE_PREFIX_PATH="$g4" -DCMAKE_CXX_COMPILER="$g4/bin/x86_64-conda-linux-gnu-g++"
    cmake --build benchmark-geant4-v2/build -j 1
fi
cp benchmark-geant4-v2/source/benchmark-patch.json "$out/patch.json"
sha256sum benchmark-geant4-v2/build/TestEm5 checks/equal_precision/{geant4_adapter.py,manifest.json} \
    > "$out/code.txt"
for seed in 78601 78602 78603; do
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        stem="$out/$scene-$seed"
        timeout --signal=TERM --kill-after=5s 60s uv run --no-sync python \
            checks/equal_precision/geant4_adapter.py "$scene" --histories 5000 --seed "$seed" \
            --physics emstandardSS --executable benchmark-geant4-v2/build/TestEm5 \
            --output "$stem.json" > "$stem.log" 2>&1
        printf 'DONE %s\n' "$stem"
    done
done
