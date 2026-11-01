#!/bin/bash
# #821 block-size calibration (2026-11-01). Warm request per arm, seed 1, discrete
# estimator, no audit: ns/row, launches, peak VRAM (pool) and peak host RSS per
# block size, rows_per_round 256 vs 1024 at the mid block, and #889 small passes
# (full-history vs continuation). Then the CUDA continuation regressions (#889).
# Usage: sbatch calib.sbatch
#SBATCH --job-name=g4bench821-calib
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=calib-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/calib/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; nvidia-smi > "$out/gpu.txt"
run() {  # stem, then pyrite_bench args
    local stem=$1; shift
    PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
        "$@" --seed 1 --warmup-histories -1 --device-staging --no-audit \
        --output "$out/$stem.json" > "$out/$stem.log" 2>&1 || echo "FAIL $stem"
    echo "DONE $stem"
}
# scene histories block...
while read -r s n blocks; do
    read -r -a bs <<< "$blocks"
    for b in "${bs[@]}"; do
        run "$s-n$n-b$b-q1024" "$s" --histories "$n" --block "$b" --rows-per-round 1024
    done
done <<'ARMS'
w-1MeV-100um 10000 1250 2500 5000
w-3MeV-100um 9600 2400 4800 9600
si-1MeV-1mm 10000 2500 5000 10000
si-300keV-100um 80000 20000 40000 80000
w-300keV-10um 40000 10000 20000 40000
ARMS
run w-1MeV-100um-n10000-b5000-q256 w-1MeV-100um --histories 10000 --block 5000 --rows-per-round 256
run si-300keV-100um-n80000-b40000-q256 si-300keV-100um --histories 80000 --block 40000 \
    --rows-per-round 256
for s in w-1MeV-100um si-300keV-100um; do
    for n in 250 500; do
        for sched in full-history continuation; do
            PYRITE_MC_TRANSPORT_SCHEDULE=$sched run "$s-n$n-$sched" "$s" --histories "$n" \
                --block "$n" --rows-per-round 256
        done
    done
done
PYRITE_TEST_BACKEND=cuda PYRITE_MC_BACKEND=cuda uv run --no-sync pyrite-dev test --slow \
    tests/montecarlo/test_transport_continuation_cuda.py > "$out/cuda-tests.log" 2>&1
echo "cuda tests rc=$?"
