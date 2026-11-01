#!/bin/bash
# #875 launch-shape / compile-option retune (checks/launch_retune_875.py).
# Usage (from the synced checkout root):
#   sbatch checks/launch_retune_875.sbatch <tag> attributes "<arms>"
#   sbatch checks/launch_retune_875.sbatch <tag> timing "<arms>" "<seeds>" "<schedules>" "<scenes>"
#SBATCH --job-name=retune875
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --output=retune875-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=$HOME/.cache/uv PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
tag=${1:?tag}; mode=${2:?attributes|timing}; arms=${3:?arms}
seeds=${4:-"1 2 3"}; schedules=${5:-"full-history continuation"}
scenes=${6:-"si-300keV-100um w-300keV-10um si-1MeV-1mm w-1MeV-100um"}
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
out="evidence/retune875/$SLURM_JOB_ID-$tag"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; nvidia-smi > "$out/gpu.txt"
export CUPY_CACHE_DIR="$out/cupy-cache"
read -r -a arm_list <<< "$arms"; read -r -a seed_list <<< "$seeds"
read -r -a sched_list <<< "$schedules"; read -r -a scene_list <<< "$scenes"
if [[ $mode == attributes ]]; then
    uv run --no-sync python checks/launch_retune_875.py attributes --arms "${arm_list[@]}" \
        --output "$out/$mode.json" > "$out/$mode.log" 2>&1
else
    uv run --no-sync python checks/launch_retune_875.py timing --arms "${arm_list[@]}" \
        --seeds "${seed_list[@]}" --schedules "${sched_list[@]}" --scenes "${scene_list[@]}" \
        --output "$out/$mode.json" > "$out/$mode.log" 2>&1
fi
echo "rc=$? $out"
