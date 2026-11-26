#!/bin/bash
#SBATCH -p gpu-amd
#SBATCH --nodelist=qlmc-ace
#SBATCH --gres=gpu:radeon8060s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o cone203-unbiased-%j.log
# #203 unbiasedness gate: analog vs split, line/brem/characteristic; measurement only.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export OPENBLAS_NUM_THREADS=1
export CUPY_INSTALL_USE_HIP=1
source src/pyrite/remote/rocm_build_env.sh
_pyrite_rocm_env "$PWD"
export PYRITE_MC_BACKEND=rocm
echo "code $(cat CODE_SHA)"
pixi install --locked -e amd
OUT=${1:-cone203_unbiased_out}
mkdir -p "$OUT"
for E in ${ENERGIES:-1000 2000}; do
  pixi run --as-is -e amd python checks/detector_cone_unbiasedness.py \
    --energy-keV "$E" "${@:2}" --out "$OUT/unbiased-$E${TAG:-}.json"
done
