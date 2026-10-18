#!/bin/bash
#SBATCH --job-name=cheap-validation510
#SBATCH --cpus-per-task=1
#SBATCH --output=cheap-validation-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg
export G4LEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$HOME/.local/opt/pyrite-issue-182/geant4/lib:${LD_LIBRARY_PATH:-}"
out="scratch/cheap-validation/$SLURM_JOB_ID"
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
lscpu > "$out/cpu.txt"
cp .pyrite-sync "$out/code.txt"
cp "$0" "$out/source/submitted.sbatch"
cp checks/equal_precision/{cheap-validation.json,benchmark_validation.py,run_cheap_validation.py,run_pilot.py,pyrite_adapter.py,geant4_adapter.py,precision_checks.py,paired_equivalence.py,manifest.json} "$out/source/"
cp -r src/pyrite/montecarlo/transport "$out/source/transport"
find "$out/source" -type d -name __pycache__ -prune -exec rm -rf {} +
if [ "$1" = gpu ]; then
  export PYRITE_MC_BACKEND=cuda
  nvidia-smi -q > "$out/gpu.txt"
else
  export PYRITE_MC_BACKEND=cpu
  g4=/home/aamador/pyrite-510-benchmarks/benchmark-geant4-v2
  cp "$g4/source/benchmark-patch.json" "$out/source/benchmark-patch.json"
  sha256sum "$g4/build/TestEm5" > "$out/executable-sha256.txt"
  tar -czf "$out/geant4-source.tar.gz" -C "$g4" source
fi
set +e
"$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/run_cheap_validation.py "$1" "$out"
status=$?
set -e
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
exit "$status"
