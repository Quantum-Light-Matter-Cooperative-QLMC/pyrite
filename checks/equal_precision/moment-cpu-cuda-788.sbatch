#!/bin/bash
#SBATCH --job-name=moment-cpucuda788
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:15:00
#SBATCH --output=moment-cpucuda788-%j.log
# #788 same-table CPU/CUDA comparison and launch-work audit. CUDA reruns the
# CPU held-out seeds/arguments, then pairs them with CPU=<dir> records.
# Protocol: moment-cpu-cuda-788.json.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec 9>.pyrite-sync.lock
flock -s -w 30 9
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
policy=checks/equal_precision/moment-cpu-cuda-788.json
cpu="${CPU:?CPU held-out evidence directory}"
# OUT resumes an earlier directory; complete records there are not rerun.
out="${OUT:-evidence/moment-cpucuda788/$SLURM_JOB_ID}"
mkdir -p "$out/pairs" "$out/work"
cp .pyrite-sync "$out/sync-$SLURM_JOB_ID.txt"
cp "$policy" "$out/policy-$SLURM_JOB_ID.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job-$SLURM_JOB_ID.txt"
nvidia-smi -q > "$out/gpu-$SLURM_JOB_ID.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables-$SLURM_JOB_ID.json"
deadline=$((SECONDS + 7800))
field() { uv run --no-sync python -c "import json; v=json.load(open('$policy'))$1; print('\n'.join(map(str, v)) if isinstance(v, list) else v)"; }
run() {
    local case=$1 variant=$2 seed=$3 histories=$4 stem=$5
    local scene=${case%%:*} cutoff=()
    if [[ $case == *:* ]]; then cutoff=(--cutoff-ev "${case#*:}"); fi
    if [[ -s $stem.json && -s $stem-audit.json ]]; then return 0; fi
    if (( SECONDS > deadline - 900 )); then
        printf 'BOUNDED_STOP %s %s %s\n' "$case" "$seed" "$variant"
        exit 0
    fi
    timeout --signal=TERM --kill-after=5s 900s uv run --no-sync python \
        checks/equal_precision/benchmark_moment_production.py "$variant" "$stem-audit.json" \
        "$scene" "${cutoff[@]}" --histories "$histories" --seed "$seed" --core cuda \
        --batch-histories "$histories" --scratch-mib 4096 --probe-electrons 0 \
        --output "$stem.json" > "$stem.log" 2>&1
}
mapfile -t cases < <(field '["cases"]')
mapfile -t seeds < <(field '["ensemble"]["seeds"]')
histories=$(field '["ensemble"]["histories_per_seed"]')
for seed in "${seeds[@]}"; do
    variants=(exact table refined)
    if (( seed % 2 == 0 )); then variants=(refined table exact); fi
    for case in "${cases[@]}"; do
        for variant in "${variants[@]}"; do
            run "$case" "$variant" "$seed" "$histories" "$out/pairs/${case/:/-cut}-$seed-$variant"
        done
    done
done
mapfile -t work_cases < <(field '["work_audit"]["cases"]')
seed=$(field '["work_audit"]["seed"]')
for case in "${work_cases[@]}"; do
    count=$(uv run --no-sync python -c "import json; s=json.load(open('$policy'))['work_audit']; print(s['histories_override'].get('$case', s['histories']))")
    for variant in exact table; do
        run "$case" "$variant" "$seed" "$count" "$out/work/${case/:/-cut}-$seed-$variant"
    done
done
uv run --no-sync python checks/equal_precision/analyze_moment_cpu_cuda.py \
    "$cpu" "$out/pairs" "$out/analysis.json"
echo COMPLETE
