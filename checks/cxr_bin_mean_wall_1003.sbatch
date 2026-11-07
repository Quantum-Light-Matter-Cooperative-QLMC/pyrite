#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o scratch/bin1003/wall-%j.log
# #1003: warm wall per primary and full spectra, before (MAIN_DIR, a clean
# checkout of the base commit) and after (this checkout), on frozen grids:
# h-BN from scratch/bin1003/grid-hbn-seed0.npy, silicon frozen here first.
# Run from a clean isolated checkout at the pushed SHA.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
here="$PWD"
out="$here/scratch/bin1003"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda PYRITE_FP64=0 PYRITE_MC_TRANSPORT_PRECISION=fp64
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,driver_version --format=csv
script="$here/checks/cxr_bin_mean_profile_1003.py"
# Same script in both checkouts; it only uses APIs present on the base.
cp "$script" "$MAIN_DIR/scratch/wall_1003.py"

checkout_env() {
  export PYRITE_HOME="$1" PYRITE_CATALOG="$1/src/pyrite/data/catalog"
  export PYRITE_USER_CATALOG="$1/scratch/empty-user-catalog" XDG_CONFIG_HOME="$1/scratch/xdg"
  mkdir -p "$PYRITE_USER_CATALOG" "$XDG_CONFIG_HOME"
  cd "$1"
  git rev-parse HEAD
  git status --short
  uv sync -q --locked --no-default-groups --extra nvidia --group dev
}

checkout_env "$here"
uv run --no-sync python "$script" grid --material silicon --core cuda \
  --out "$out/grid-silicon-n128.npy"

cases=(hbn silicon)
grids=("$out/grid-hbn-seed0.npy" "$out/grid-silicon-n128.npy")
for side in before after; do
  if [ "$side" = before ]; then
    checkout_env "$MAIN_DIR"
    run=("$MAIN_DIR/scratch/wall_1003.py")
  else
    checkout_env "$here"
    run=("$script")
  fi
  for i in 0 1; do
    uv run --no-sync python "${run[@]}" wall --material "${cases[$i]}" --core cuda \
      --grid "${grids[$i]}" --out "$out/wall-${cases[$i]}-$side.npz" \
      || echo "FAILED wall ${cases[$i]} $side rc=$?"
  done
done

cd "$here"
for i in 0 1; do
  uv run --no-sync python "$script" compare "$out/wall-${cases[$i]}-before.npz" \
    "$out/wall-${cases[$i]}-after.npz" --grid "${grids[$i]}" \
    --out "$out/wall-${cases[$i]}-compare.json" || echo "FAILED compare ${cases[$i]} rc=$?"
done
echo DONE
