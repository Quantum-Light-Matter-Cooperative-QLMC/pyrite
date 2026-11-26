#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o fiber1485-%j.log
# #1485 research diagnostics: one `fiber_domain_structure_1485.py` invocation
# as an FP64 CPU job on any cpu-partition node. Pin a node on the command
# line, for example `sbatch --nodelist=qlmc-ace ...`. Run from a synced checkout.
# Usage:
#   sbatch checks/fiber_domain_structure_1485.sbatch OUT_DIR SUBCOMMAND [options]
# Writes OUT_DIR/result.{json,npz} (frozen) or OUT_DIR/result.json.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if (( $# < 2 )); then
    echo "usage: $0 OUT_DIR {profile|single-flight|frozen} [options]" >&2
    exit 2
fi
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
shift
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OMP_NUM_THREADS=1
export PYRITE_MC_BACKEND=cpu
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
SHA=$(cat CODE_SHA 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)
printf '%s\n' "$SHA" > "$OUT/CODE_SHA"
printf '%q ' "$@" > "$OUT/arguments.txt"
printf '\n' >> "$OUT/arguments.txt"
echo "code $SHA host $(hostname) job $SLURM_JOB_ID args $*"
pixi install --locked -e default
/usr/bin/time -v pixi run --as-is -e default python -u \
    checks/fiber_domain_structure_1485.py "$@" --json "$OUT/result.json" > "$OUT/run.log" 2>&1
files=(CODE_SHA arguments.txt run.log result.json)
[[ -f "$OUT/result.npz" ]] && files+=(result.npz)
(cd "$OUT" && sha256sum "${files[@]}" > SHA256SUMS)
