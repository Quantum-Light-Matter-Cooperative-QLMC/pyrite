#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH -t 03:00:00
#SBATCH -o s907-%j.log
# #907 streamed-block sizing on the RTX 5080 (16 GB): block sweep at 262,144 W
# electrons, transport-only population probe, scoring/transport split timers,
# then Nsight Systems (streamed 16k block) and Nsight Compute (one
# full-population transport launch).
# usage (from an isolated synced checkout root, see remote-gpu-jobs):
#   sbatch -J s907-<sha> checks/streamed_block_sizing.sbatch
# nsys needs a writable TMPDIR (/tmp/nvidia is not writable on the lab box).
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=/home/aamador/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda PROBE_SHA=$(cat CODE_SHA)
echo "code $PROBE_SHA"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
H=checks/streamed_block_sizing.py
OUT=s907_out; mkdir -p $OUT
cat > $OUT/rtx5080.arms <<'EOF'
streamed 262144 4096 512 32 0
streamed 262144 8192 1024 32 0
streamed 262144 16384 2048 32 0
streamed 262144 32768 4096 32 0
streamed 262144 16384 2048 32 0
transport 262144 16384 2048 32 0
transport 262144 32768 4096 32 0
transport 262144 65536 8192 32 0
streamed 262144 16384 2048 32 0 PROBE_SPLIT=1
transport 262144 16384 2048 32 0 PROBE_SPLIT=1
joined 32768 0 4096 32 0
streamed 32768 8192 4096 32 0
EOF
uv run --no-sync python -u $H sweep $OUT/rtx5080.arms $OUT/sweep
echo "== nsys streamed 65536/16384"
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o $OUT/nsys-streamed -f true \
  uv run --no-sync python -u $H streamed 65536 16384 2048 32 0 $OUT/nsys-streamed
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
  --format csv -o $OUT/nsys-streamed $OUT/nsys-streamed.nsys-rep >/dev/null
for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum; do
  echo "-- $r"; head -15 $OUT/nsys-streamed_$r.csv
done
echo "== ncu one full-population transport launch (block 16384)"
# The warmup run launches ~30 transport rounds; skip past them into the
# measured run's first (full-population) continuation round.
ncu --set full --kernel-name regex:transport --launch-skip 40 --launch-count 1 \
  -o $OUT/ncu-transport -f \
  uv run --no-sync python -u $H transport 16384 16384 2048 32 0 $OUT/ncu-run
ncu --import $OUT/ncu-transport.ncu-rep --page details --csv > $OUT/ncu-transport.csv
ncu --import $OUT/ncu-transport.ncu-rep --page details | head -200
echo "done rc=0"
