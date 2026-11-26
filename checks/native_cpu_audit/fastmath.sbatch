#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=runs/fastmath1469-%j.log
# #1469 fastmath / reduced-accuracy arms on four OS-bound CPUs (bind.sh).
# Numba threads from AUDIT1432_NUMBA_THREADS (default 1); BLAS/OpenMP stay 1.
# PIXI_ENV selects the pixi environment (default `default`; `svml` for SVML Numba).
#   sbatch [--nodelist=N] checks/native_cpu_audit/fastmath.sbatch MODE [ARGS]
# MODE:
#   cases ARM...   every case in CASES (default "photon-case layered transport-si300")
#                  under each ARM, one process per (case, arm), each arm with its own
#                  cold Numba cache. ARM is FLAGSET (global njit fastmath, see
#                  fastmath_arms.py), `asm` (strict plus vectorization records), or
#                  `reducer=EDIT[:FLAGSET]`. REPEATS (default 2).
#   replay ARM...  capture photon-case reducer inputs once, then sin_micro.py and
#                  reducer_replay.py over EDIT:FLAGSET arms.
set -euo pipefail
mode=${1:?mode}; shift
source checks/native_cpu_audit/bind.sh
export PATH=$HOME/.pixi/bin:$PATH
export PYRITE_HOME=$PWD PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$PWD/user-catalog PYRITE_MC_BACKEND=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
threads=${AUDIT1432_NUMBA_THREADS:-1}
env_name=${PIXI_ENV:-default}
export NUMBA_NUM_THREADS=$threads
mkdir -p runs
audit=checks/native_cpu_audit
pixi install --locked -e "$env_name"
run() { pixi run --as-is -e "$env_name" python "$@"; }
run -c 'from pyrite.validation.check_records import current_revision; print(current_revision())'
run -c 'import numba; print("numba", numba.__version__, "svml", numba.config.USING_SVML)'
tag=$env_name-t$threads
case $mode in
cases)
    read -r -a cases <<< "${CASES:-photon-case layered transport-si300}"
    for arm in "$@"; do
        opts=(); name=$arm
        case $arm in
        asm) opts=(--asm) ;;
        reducer=*) opts=(--reducer "${arm#reducer=}"); name=r-${arm#reducer=}; name=${name//:/-} ;;
        *) opts=(--fastmath "$arm") ;;
        esac
        for c in "${cases[@]}"; do
            export NUMBA_CACHE_DIR=$PWD/runs/nc-$tag-$name-$c
            rm -rf "$NUMBA_CACHE_DIR"
            timeout "${ARM_TIMEOUT_S:-1500}" run $audit/fastmath_arms.py --case "$c" "${opts[@]}" \
                --repeats "${REPEATS:-2}" --out "runs/$c-$tag-$name.json" \
                --arrays "runs/$c-$tag-$name.npz" || echo "ARM_FAILED $c $arm $?"
        done
    done ;;
replay)
    export NUMBA_CACHE_DIR=$PWD/runs/nc-$tag-replay
    inputs=runs/win_inputs/photon-case
    [ -f "$inputs/calls.json" ] || run $audit/capture_window_inputs.py photon-case "$inputs" > runs/capture.log
    run $audit/sin_micro.py "$(hostname)-$tag" > "runs/sin_micro-$tag.jsonl" || echo "ARM_FAILED sin_micro $?"
    run $audit/reducer_replay.py "$(hostname)-$tag" "$inputs" "$@" > "runs/replay-$tag.jsonl" \
        || echo "ARM_FAILED replay $?" ;;
*) echo "unknown mode $mode" >&2; exit 2 ;;
esac
