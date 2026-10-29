#!/bin/bash
#SBATCH --job-name=elsepa-rate792
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=elsepa-rate792-%j.log
# #792 profiling gate: marginal cost of exact ELSEPA flight-rate sums on CPU
# (per-electron Numba, one thread) and CUDA, via probe_elsepa_rate_cost.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec 9>.pyrite-sync.lock
flock -s -w 30 9
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/elsepa-rate792/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
lscpu > "$out/cpu.txt"
deadline=$((SECONDS + 3480))
scenes=(si-300-100um-0deg w-300-10um-0deg sio2-800kev-tr0.3-0deg-compound silicon-5000kev-tr0.3-0deg)
declare -A histories=([per-electron]=1000 [cuda]=10000)
for repeat in 1 2 3; do
    amps=(0 1 2 4)
    if (( repeat == 2 )); then amps=(4 2 1 0); fi
    for core in cuda per-electron; do
        backend=cpu
        if [[ $core == cuda ]]; then backend=cuda; fi
        for scene in "${scenes[@]}"; do
            for k in "${amps[@]}"; do
                if (( SECONDS > deadline - 120 )); then
                    printf 'BOUNDED_STOP %s %s %s %s\n' "$repeat" "$core" "$scene" "$k"
                    exit 0
                fi
                stem="$out/$core-$scene-k$k-r$repeat"
                PYRITE_MC_BACKEND=$backend timeout --signal=TERM --kill-after=5s 300s \
                    uv run --no-sync python checks/equal_precision/probe_elsepa_rate_cost.py \
                    "$k" "$stem-audit.json" "$scene" --histories "${histories[$core]}" \
                    --seed 79201 --core "$core" --arm transport --radiative-loss none \
                    --probe-electrons 0 --scratch-mib 4096 --max-steps 200000 --output "$stem.json" \
                    > "$stem.log" 2>&1 || printf 'FAILED %s\n' "$stem"
            done
        done
    done
done
