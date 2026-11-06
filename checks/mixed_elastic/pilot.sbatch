#!/bin/bash
# Planning pilot for the mixed elastic equivalence study (#894). Its seeds stay
# out of the held-out family: it sizes populations and freezes margins only.
# Usage: sbatch pilot.sbatch <git-sha>
#SBATCH --job-name=894-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=pilot-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
sha=$1
export PATH="$HOME/.local/bin:$PATH" PYRITE_HOME="$HOME/pyrite" PYRITE_MC_BACKEND=cuda
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export XDG_CONFIG_HOME="$(mktemp -d)" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --locked --package pyrite-mc --no-default-groups --extra nvidia -q || exit 1
out="evidence/pilot/$SLURM_JOB_ID"; mkdir -p "$out"
echo "$sha" > "$out/sha.txt"
run() {  # tag scene histories [bench options...]
    local tag=$1 scene=$2 n=$3; shift 3
    .venv/bin/python checks/geant4_performance/pyrite_bench.py "$scene" --histories "$n" \
        --warmup-histories -1 --seed 91001 --rows-per-round 1024 --device-staging \
        --no-audit --output "$out/$tag-$scene.json" "$@" > "$out/$tag-$scene.log" 2>&1
    echo "DONE $tag $scene rc=$?"
}
scenes=(si-300keV-100um w-300keV-10um si-1MeV-1mm w-1MeV-100um si-3MeV-1mm w-3MeV-100um)
for s in "${scenes[@]}"; do
    run det "$s" 10000
    run c05 "$s" 10000 --mixed-c1 0.05 --mixed-c2 0.05
    run c20 "$s" 10000 --mixed-c1 0.2 --mixed-c2 0.2
    run shdet "$s" 5000 --inelastic-cutoff-ev 50
    run shc05 "$s" 5000 --inelastic-cutoff-ev 50 --mixed-c1 0.05 --mixed-c2 0.05
done
