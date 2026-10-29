#!/bin/bash
#SBATCH --job-name=pyrite795-photons
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=completion-795-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_FP64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/pyrite-uv-cache}"
UV="${UV:-$HOME/.local/bin/uv}"
if ! command -v "$UV" >/dev/null; then UV=uv; fi
BASE_REF="${PYRITE_795_BASE_REF:-42931cbb2b5fe71a2a8afd3fd00a69033a92c9ad}"
BASE_FILE='src/pyrite/montecarlo/transport/hard_radiative.py'
if ! git cat-file -e "${BASE_REF}^{commit}" 2>/dev/null; then
    echo "Missing pinned baseline commit ${BASE_REF}; set PYRITE_795_BASE_REF to a known pre-#795 commit" >&2
    exit 2
fi
OUT="${PYRITE_795_OUT:-evidence/completion-795/${SLURM_JOB_ID:-manual}}"
if [ -e "$OUT/baseline-hard_radiative.py" ]; then
    echo "Refusing to overwrite existing benchmark directory $OUT" >&2; exit 2
fi
mkdir -p "$OUT"
git show "${BASE_REF}:${BASE_FILE}" > "$OUT/baseline-hard_radiative.py"
if ! grep -q 'for index in np.flatnonzero(energies > 0.0)' "$OUT/baseline-hard_radiative.py"; then
    echo "Selected baseline does not contain the original scalar completion" >&2; exit 2
fi
if grep -q '_complete_hard_radiative_chunk' "$BASE_FILE" && \
   grep -q '_complete_hard_radiative_chunk' "$OUT/baseline-hard_radiative.py"; then
    echo "Baseline unexpectedly contains new completion" >&2; exit 2
fi
{
    git rev-parse HEAD
    git status --short
    sha256sum "$BASE_FILE" "$OUT/baseline-hard_radiative.py" \
        checks/equal_precision/{benchmark_completion_795.py,pyrite_adapter.py,manifest.json}
    "$UV" --version
    python --version
} > "$OUT/provenance.txt"
if command -v scontrol >/dev/null && [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job "$SLURM_JOB_ID" > "$OUT/job.txt"
fi
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/gpu.txt"; fi
# Validation gates before timing; each backend is selected in a fresh process.
PYRITE_MC_BACKEND=cpu "$UV" run --no-sync pyrite-dev test \
    tests/montecarlo/test_hard_radiative_completion_795.py \
    tests/montecarlo/test_hard_radiative.py > "$OUT/tests-cpu.log" 2>&1
PYRITE_MC_BACKEND=cuda "$UV" run --no-sync pyrite-dev test \
    tests/montecarlo/test_hard_radiative_completion_795.py \
    tests/montecarlo/test_hard_radiative_cuda.py > "$OUT/tests-cuda.log" 2>&1
"$UV" run --no-sync pyrite-dev complexity --check > "$OUT/complexity.log" 2>&1
SCRIPT=checks/equal_precision/benchmark_completion_795.py
# Interleave variants and reverse order for alternating seeds to reduce time drift.
# Each variant is a fresh Python process, with one equal-workload warmup.
for seed in 79501 79502 79503; do
    if (( seed % 2 )); then variants=(baseline candidate); else variants=(candidate baseline); fi
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for histories in 5000 20000; do
            batch_histories=$((histories > 5000 ? 5000 : histories))
            for variant in "${variants[@]}"; do
                stem="$OUT/${scene}-${histories}-${seed}-${variant}"
                echo "$(date -Is) $stem"
                timeout --signal=TERM --kill-after=5s 600s "$UV" run --no-sync python "$SCRIPT" run \
                    --variant "$variant" --baseline-source "$OUT/baseline-hard_radiative.py" \
                    "$scene" --histories "$histories" --seed "$seed" --core cuda \
                    --batch-histories "$batch_histories" --scratch-mib 4096 --probe-electrons 0 \
                    --output "$stem.json" > "$stem.log" 2>&1 || {
                    echo "FAILED $stem (see .log)" >&2
                    exit 1
                }
            done
            base="$OUT/${scene}-${histories}-${seed}-baseline.json"
            candidate="$OUT/${scene}-${histories}-${seed}-candidate.json"
            "$UV" run --no-sync python "$SCRIPT" compare "$base" "$candidate" \
                > "$OUT/${scene}-${histories}-${seed}-parity.json" || {
                echo "STOP: parity failure for $scene / $histories / $seed" >&2; exit 1;
            }
        done
    done
done
"$UV" run --no-sync python "$SCRIPT" summarize "$OUT" > "$OUT/summary.json"
echo "Completed #795 benchmark: $OUT/summary.json"
