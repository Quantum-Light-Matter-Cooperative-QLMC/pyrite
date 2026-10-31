#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o prof611-%j.log
# #611 Nsight Systems profiles of joined / joined-device / streamed scoring at 16k.
# usage (from the synced checkout root): sbatch -J prof611-<sha> checks/streamed_scoring_profile.sbatch MAT
# nsys 2025.6 rejects -q and needs a writable TMPDIR (/tmp/nvidia is not writable on the lab box).
set -uo pipefail
MAT=$1
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=/home/aamador/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA) mat $MAT"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
OUT=prof611_out/$MAT; mkdir -p $OUT
B=(uv run --no-sync python checks/streamed_scoring_memory.py run)
"${B[@]}" streamed 1000 1000 $OUT/warmup.json $MAT 300 cuda 1e5 1
for A in "joined 0" "joined-device 0" "streamed 4000"; do
  set -- $A
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o $OUT/$1 -f true \
    "${B[@]}" $1 16000 $2 $OUT/$1.json $MAT 300 cuda 1e5 1
  nsys stats --report cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv -o $OUT/$1 $OUT/$1.nsys-rep >/dev/null
  echo "== $1"; head -8 $OUT/$1_cuda_gpu_mem_size_sum.csv; head -8 $OUT/$1_cuda_gpu_mem_time_sum.csv
  grep -iE "Synchronize|Memcpy" $OUT/$1_cuda_api_sum.csv | head -8
done
