#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o char1455-%j.log
# #1455 option B: fused per-chunk characteristic kernel vs the grouped chain (A).
# Gates B vs A and B vs the NumPy twin at rtol 1e-12 plus bitwise rerun, records
# warm wall per arm, then (NSYS=1) counts kernels per arm and chunk under Nsight.
# usage (from a synced checkout root): sbatch checks/characteristic_chunk_kernel_1455.sbatch [OUTDIR] [NSYS] [SEGMENTS]
# Set PYRITE_MC_BACKEND=rocm at submit for an AMD node.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=${PYRITE_MC_BACKEND:-cuda}
EXTRA=$([ "$PYRITE_MC_BACKEND" = rocm ] && echo amd || echo nvidia)
[ -f CODE_SHA ] && echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra "$EXTRA"
B=(uv run --no-sync python checks/characteristic_chunk_kernel_1455.py)
ROOT=${1:-c1455_out}
NSYS=${2:-1}
SEGMENTS=${3:-2000000}
mkdir -p "$ROOT"
"${B[@]}" check "$ROOT/check.json" --segments "$SEGMENTS"
STATUS=$?
echo "check exit $STATUS"
if [ "$NSYS" = 1 ] && [ "$PYRITE_MC_BACKEND" = cuda ]; then
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$ROOT/nsys" -f true \
    "${B[@]}" check "$ROOT/nsys-check.json" --segments "$SEGMENTS" --repeats 1 > /dev/null
  nsys export --type sqlite -o "$ROOT/nsys.sqlite" -f true "$ROOT/nsys.nsys-rep" > /dev/null
  "${B[@]}" launches "$ROOT/nsys.sqlite" | tee "$ROOT/launches.json"
fi
exit $STATUS
