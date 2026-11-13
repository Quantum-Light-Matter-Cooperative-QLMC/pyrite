#!/bin/bash
#SBATCH --job-name=strag642s2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=straggling-s2-%j.log
# Stage 2 of the #642 collision-straggling study: one material at one energy,
# all protocol range fractions, straggling off then on.
# Usage: sbatch run_stage2.sbatch MATERIAL ENERGY_KEV [ARM_TIMEOUT_S]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_USER_CATALOG
export PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
material="${1:?material}"
energy="${2:?energy keV}"
arm_timeout="${3:-900}"
protocol=checks/straggling_defaults/protocol-stage2.json
out="evidence/straggling-s2/$SLURM_JOB_ID-$material-${energy}keV"
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
nvidia-smi -q > "$out/gpu.txt"
lscpu > "$out/cpu.txt"
cp SOURCE_SHA "$out/source_sha.txt"
cp checks/straggling_defaults/{study.py,analyze.py,protocol-stage2.json,run_stage2.sbatch} "$out/source/"
sha256sum src/pyrite/data/conduction_band.toml > "$out/conduction-band.txt"
fractions=$(uv run --no-sync python -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['range_fractions'])))" "$protocol")
for fraction in $fractions; do
  for switch in off on; do
    name="$fraction-$switch"
    rc=0
    timeout "$arm_timeout" uv run --no-sync python checks/straggling_defaults/study.py \
      "$material" "$fraction" "$switch" "$out/$name" --protocol "$protocol" --energy "$energy" \
      > "$out/$name.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "$name exit $rc (124 = timeout)" | tee -a "$out/failures.txt"
    fi
  done
done
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
