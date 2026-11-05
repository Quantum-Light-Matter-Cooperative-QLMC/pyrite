#!/bin/bash
#SBATCH --job-name=moment-prod788
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:15:00
#SBATCH --output=moment-prod788-%j.log
# #788 production-path CUDA check. PHASE=ensemble (default) runs the hardware
# tests, replay preflights and the frozen fresh-seed ensemble; PHASE=benchmark
# runs the repeated timing matrix. Protocol: moment-production-788.json.
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
policy=checks/equal_precision/moment-production-788.json
phase=${PHASE:-ensemble}
# OUT resumes an earlier directory; complete records there are not rerun.
out="${OUT:-evidence/moment-prod788/$SLURM_JOB_ID-$phase}"
mkdir -p "$out"
cp .pyrite-sync "$out/sync-$SLURM_JOB_ID.txt"
cp "$policy" "$out/policy-$SLURM_JOB_ID.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job-$SLURM_JOB_ID.txt"
nvidia-smi -q > "$out/gpu-$SLURM_JOB_ID.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables-$SLURM_JOB_ID.json"
deadline=$((SECONDS + 7800))
field() { uv run --no-sync python -c "import json,sys; v=json.load(open('$policy'))$1; print('\n'.join(map(str, v)) if isinstance(v, list) else v)"; }
# Per-case history count: the section default unless the protocol overrides it.
count() { uv run --no-sync python -c "import json; s=json.load(open('$policy'))['$1']; print(s.get('histories_override', {}).get('${2%%:*}', s['$3']))"; }
run() {
    local case=$1 variant=$2 seed=$3 histories=$4 stem=$5; shift 5
    local scene=${case%%:*} cutoff=()
    if [[ $case == *:* ]]; then cutoff=(--cutoff-ev "${case#*:}"); fi
    if [[ -s $stem.json && -s $stem-audit.json ]]; then return 0; fi
    if (( SECONDS > deadline - 900 )); then
        printf 'BOUNDED_STOP %s %s %s\n' "$case" "$seed" "$variant"
        exit 0
    fi
    timeout --signal=TERM --kill-after=5s 900s uv run --no-sync python \
        checks/equal_precision/benchmark_moment_production.py "$variant" "$stem-audit.json" \
        "$@" "$scene" "${cutoff[@]}" --histories "$histories" --seed "$seed" --core cuda \
        --batch-histories "$histories" --scratch-mib 4096 --probe-electrons 0 \
        --output "$stem.json" > "$stem.log" 2>&1
}
if [[ $phase == ensemble ]]; then
    uv run --no-sync pyrite-dev test --slow tests/montecarlo/cascades/test_radiative_moments_cuda.py \
        tests/montecarlo/cascades/test_hard_radiative_cuda.py --durations=0 > "$out/cuda-tests.log" 2>&1
    mapfile -t cases < <(field '["cases"]')
    mapfile -t variants < <(field '["variants"]')
    mapfile -t seeds < <(field '["ensemble"]["seeds"]')
    for case in "${cases[@]}"; do
        for variant in "${variants[@]}"; do
            run "$case" "$variant" 78830 256 "$out/preflight-${case/:/-cut}-$variant" --replay
        done
    done
    for seed in "${seeds[@]}"; do
        order=("${variants[@]}")
        if (( seed % 2 == 0 )); then order=(refined table exact); fi
        for case in "${cases[@]}"; do
            for variant in "${order[@]}"; do
                run "$case" "$variant" "$seed" "$(count ensemble "$case" histories_per_seed)" \
                    "$out/${case/:/-cut}-$seed-$variant"
            done
        done
    done
    uv run --no-sync python checks/equal_precision/analyze_moment_transport.py \
        --policy "$policy" "$out" "$out/analysis.json"
elif [[ $phase == benchmark ]]; then
    mapfile -t cases < <(field '["benchmark"]["cases"]')
    mapfile -t seeds < <(field '["benchmark"]["seeds"]')
    for index in "${!seeds[@]}"; do
        order=(exact table)
        if (( index % 2 == 1 )); then order=(table exact); fi
        for case in "${cases[@]}"; do
            for variant in "${order[@]}"; do
                run "$case" "$variant" "${seeds[$index]}" "$(count benchmark "$case" histories)" \
                    "$out/${case/:/-cut}-r$index-${seeds[$index]}-$variant"
            done
        done
    done
else
    echo "unknown PHASE=$phase" >&2
    exit 2
fi
echo COMPLETE
