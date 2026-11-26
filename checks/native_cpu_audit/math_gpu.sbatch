#!/bin/bash
#SBATCH --partition=gpu-amd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=runs/mathgpu1469-%j.log
# #1469 math-mode validation on one GPU: each case in CASES (default `layered`)
# under `PYRITE_MATH=reproducible` and `fast`, one fastmath_arms.py process per
# (case, mode), output arrays kept for the comparison. The backend comes from
# the node (ROCm on gpu-amd through rocm_build_env.sh, CUDA otherwise); the
# backend refuses a device without FP64.
#   sbatch [-p gpu] checks/native_cpu_audit/math_gpu.sbatch
set -euo pipefail
export PATH=$HOME/.pixi/bin:$PATH
export PYRITE_HOME=$PWD PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$PWD/user-catalog XDG_CONFIG_HOME=$PWD/.xdg-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs "$XDG_CONFIG_HOME"
if [ -e /dev/kfd ]; then
    env_name=amd
    . src/pyrite/remote/rocm_build_env.sh && _pyrite_rocm_env "$PWD"
    export PYRITE_MC_BACKEND=rocm
else
    env_name=nvidia
    export PYRITE_MC_BACKEND=cuda
fi
pixi install --locked -e "$env_name"
py=(pixi run --as-is -e "$env_name" python)
"${py[@]}" -c 'from pyrite._backend import BACKEND; print("backend", BACKEND.name, BACKEND.device)'
read -r -a cases <<< "${CASES:-layered}"
for c in "${cases[@]}"; do
    for mode in reproducible fast; do
        export PYRITE_MATH=$mode NUMBA_CACHE_DIR=$PWD/runs/nc-gpu-$mode-$c
        tag=$c-$env_name-gpu-math_$mode
        timeout "${ARM_TIMEOUT_S:-3000}" "${py[@]}" checks/native_cpu_audit/fastmath_arms.py \
            --case "$c" --repeats "${REPEATS:-1}" --out "runs/$tag.json" --arrays "runs/$tag.npz" \
            || echo "ARM_FAILED $c $mode $?"
    done
done
