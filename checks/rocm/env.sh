# shellcheck shell=bash
# Shared job environment for #888 evaluation scripts, sourced from the synced
# checkout root on either GPU partition. AMD nodes get the same CuPy-HIP build
# environment as `pyrite run --remote` (src/pyrite/remote/rocm_build_env.sh)
# and the pixi `amd` environment; NVIDIA nodes use `nvidia`. Pass
# `-e "$PYRITE_PIXI_ENV"` to `pixi run`.
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
if [ -e /dev/kfd ] && command -v rocminfo >/dev/null 2>&1; then
    export PYRITE_GPU_VENDOR=amd PYRITE_MC_BACKEND=rocm
    export CUPY_INSTALL_USE_HIP=1 CUPY_NUM_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
    # shellcheck source=src/pyrite/remote/rocm_build_env.sh
    . src/pyrite/remote/rocm_build_env.sh && _pyrite_rocm_env "$PWD"
else
    export PYRITE_GPU_VENDOR=nvidia PYRITE_MC_BACKEND=cuda
fi

# pixi GPU environment of this node: `nvidia` or `amd` (#1462).
export PYRITE_PIXI_ENV="$PYRITE_GPU_VENDOR"

pyrite_gpu_sync() {
    pixi install --locked -e "$PYRITE_PIXI_ENV" "$@"
}

pyrite_gpu_info() {
    echo "c++: $(command -v c++ || echo none); CXX: ${CXX:-unset}"
    if [ "$PYRITE_GPU_VENDOR" = amd ]; then
        rocminfo | grep -E "Marketing Name|Name: +gfx"
        hipconfig --full 2>&1
    else
        nvidia-smi
    fi
    # Informational only: never fail a caller running under set -e.
    return 0
}
