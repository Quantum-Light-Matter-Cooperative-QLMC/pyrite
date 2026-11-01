#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o streamed872-%j.log
# #872 per-estimator split of streamed scoring (W, Si control), 20000 electrons / 4096 blocks,
# plus a joined W 4000 reference and an Nsight kernel summary of the streamed W arm.
# usage (from the synced checkout root): sbatch -J s872-<sha> checks/streamed_split_872.sbatch [OUTDIR]
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=/home/aamador/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
ROOT=${1:-s872_out}
for MAT in silicon tungsten; do
  OUT=$ROOT/$MAT; mkdir -p $OUT
  "${B[@]}" run streamed 1000 1000 $OUT/warmup.json $MAT 300 cuda 1e5 1
  for REP in 1 2; do
    echo "== $MAT streamed rep $REP"
    "${B[@]}" run streamed 20000 4096 $OUT/streamed-$REP.json $MAT 300 cuda 1e5 1
  done
  "${B[@]}" compare $OUT/streamed-1.json $OUT/streamed-2.json
done
echo "== tungsten joined 4000"
"${B[@]}" run joined 4000 0 $ROOT/tungsten/joined-4000.json tungsten 300 cuda 1e5 1
echo "== nsys tungsten streamed"
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o $ROOT/tungsten/nsys -f true \
  "${B[@]}" run streamed 20000 4096 $ROOT/tungsten/nsys.json tungsten 300 cuda 1e5 1
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv -o $ROOT/tungsten/nsys $ROOT/tungsten/nsys.nsys-rep >/dev/null
head -25 $ROOT/tungsten/nsys_cuda_gpu_kern_sum.csv
head -6 $ROOT/tungsten/nsys_cuda_gpu_mem_time_sum.csv
