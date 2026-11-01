#!/bin/bash
# #888 GPU backend evaluation: build, CuPy smoke, exact-transport hardware
# suite, and #821 warm-transport bench. Vendor-neutral (checks/rocm/env.sh), so
# the same SHA runs on qlmc-ace (ROCm) and the RTX 5080 (CUDA).
# Usage, from the synced checkout root (tests/ rsynced; `pyrite remote sync`
# skips it):
#   sbatch -p gpu-amd -w qlmc-ace --gres=gpu:radeon8060s:1 checks/rocm/evaluate.sbatch [stages]
#   sbatch -p gpu --gres=gpu:1 checks/rocm/evaluate.sbatch bench
# stages: any of "build smoke suite bench" (default: all, in that order).
#SBATCH --job-name=gpu-eval888
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=gpu-eval888-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
. checks/rocm/env.sh
stages=${1:-"build smoke suite bench"}
out="evidence/gpu-eval888/$PYRITE_GPU_VENDOR/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json" 2>/dev/null || true
env | grep -E '^(PYRITE_|ROCM_|HCC_|CC=|CXX=|CPATH|LIBRARY_PATH|LD_LIBRARY_PATH|UV_PROJECT)' | sort > "$out/env.txt"
pyrite_gpu_info > "$out/gpu.txt" 2>&1
step() { echo "=== $1 ($(date +%T))"; }
has() { [[ " $stages " == *" $1 "* ]]; }
status=0

if has build; then
    step build
    # --no-cache: rebuild CuPy from source rather than reuse a cached wheel, so
    # the stage proves the build path itself.
    start=$SECONDS
    pyrite_gpu_sync -v --no-cache --reinstall-package cupy > "$out/build.log" 2>&1; rc=$?
    echo "build wall $((SECONDS - start)) s"
    echo "build rc=$rc"
    (( rc == 0 )) || { grep -iE "error|fatal|not found" "$out/build.log" | head -40; exit 1; }
fi
pyrite_gpu_sync -q || { echo "STOP: environment not built"; exit 1; }
uv run --no-sync python -c "import pyrite._backend as b; print(b.BACKEND.name, b.BACKEND.vendor, b.BACKEND.device.name)" \
    | tee "$out/backend.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"

if has smoke; then
    step smoke
    uv run --no-sync python checks/rocm/cupy_smoke.py > "$out/cupy-smoke.txt" 2>&1 || status=1
    cat "$out/cupy-smoke.txt"
fi

if has suite; then
    # Pass A: every hardware module in an ordinary (CPU-pinned) session; device
    # paths are requested explicitly and driver tests spawn device children.
    step "suite A: hardware modules"
    mapfile -t modules < <(find tests -name 'test_*cuda*.py' -o -name 'test_gpu_oom_retry.py' | sort)
    modules+=(tests/montecarlo/transport/test_multilayer.py)
    uv run --no-sync pyrite-dev test --slow "${modules[@]}" -p no:randomly -rfEs \
        --junitxml="$out/suite-a.xml" --durations=30 > "$out/suite-a.log" 2>&1 || status=1
    tail -40 "$out/suite-a.log"
    # Pass B: modules whose bodies need the session itself on the device.
    step "suite B: device session"
    PYRITE_TEST_BACKEND=$PYRITE_MC_BACKEND uv run --no-sync pyrite-dev test --slow \
        tests/montecarlo/transport/test_photon_production_cuda.py \
        tests/montecarlo/transport/test_photon_reentry_cuda.py \
        tests/montecarlo/transport/test_photon_reentry_pair_cuda.py \
        tests/montecarlo/spectrum/test_optical_cuda_parity.py \
        tests/montecarlo/spectrum/test_optical_source_cuda.py \
        tests/montecarlo/spectrum/test_spectrum_cuda_cheap_hoists.py \
        "tests/montecarlo/runner/test_trajectory_capture.py::test_cuda_resident_run_captures_device_segments" \
        -p no:randomly -rfEs --junitxml="$out/suite-b.xml" > "$out/suite-b.log" 2>&1 || status=1
    tail -40 "$out/suite-b.log"
fi

if has bench; then
    # #821 scenes <= 1 MeV, run.sbatch's discrete-estimator settings. Histories
    # are sized for >= ~2 s warm transport on the RTX 5080, not for statistics.
    step bench
    specs=(si-300keV-100um:20000 w-300keV-10um:10000 si-1MeV-1mm:2000 w-1MeV-100um:1000)
    for seed in 301 302 303; do
        for spec in "${specs[@]}"; do
            s=${spec%%:*} h=${spec##*:}
            for p in fp64 mixed; do
                stem="$out/bench-$p-$s-$seed"
                if uv run --no-sync python checks/geant4_performance/pyrite_bench.py "$s" \
                    --histories "$h" --seed "$seed" --rows-per-round 1024 --device-staging \
                    --transport-precision "$p" --no-audit --output "$stem.json" > "$stem.log" 2>&1; then
                    echo "DONE $p $s $seed"
                else
                    echo "FAIL $p $s $seed"; status=1
                fi
            done
        done
    done
    uv run --no-sync python checks/rocm/bench_table.py "$out" > "$out/bench.md" && cat "$out/bench.md"
fi
step "done status=$status"
exit "$status"
