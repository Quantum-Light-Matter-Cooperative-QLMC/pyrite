#!/bin/bash
# Geant4 mixed-scheme arms of the #821 benchmark (#894): single scattering (the
# reference), WentzelVI plus single Coulomb, and Goudsmit-Saunderson, same node.
# Usage: sbatch --dependency=afterok:<build> g4-mixed.sbatch <git-sha> <histories> <seeds...>
#SBATCH --job-name=894-g4
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=g4-mixed-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
sha=$1 histories=$2; shift 2; seeds=("$@")
export PYRITE_HOME="$HOME/pyrite" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog" XDG_CONFIG_HOME="$(mktemp -d)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G4_PREFIX="$HOME/.local/opt/pyrite-issue-182/geant4"
export G4LEDATA="$G4_PREFIX/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$G4_PREFIX/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$G4_PREFIX/lib:${LD_LIBRARY_PATH:-}"
G4BENCH="$PWD/g4bench-build/g4bench"
out="evidence/g4/$SLURM_JOB_ID"; mkdir -p "$out"
echo "$sha" > "$out/sha.txt"; lscpu > "$out/cpu.txt"
sha256sum "$G4BENCH" checks/geant4_performance/g4bench/g4bench.cc > "$out/code.txt"
scenes=(si-300keV-100um w-300keV-10um si-1MeV-1mm w-1MeV-100um si-3MeV-1mm w-3MeV-100um)
for seed in "${seeds[@]}"; do
    for s in "${scenes[@]}"; do
        for arm in ss wvi gs; do
            stem="$PWD/$out/$arm-$s-$seed"
            .venv/bin/python checks/geant4_performance/geant4_bench.py "$s" \
                --histories "$histories" --warmup $((histories / 100)) --seed "$seed" \
                --threads 32 --executable "$G4BENCH" --elastic "$arm" \
                --output "$stem.json" > "$stem.log" 2>&1
            echo "DONE $arm $s $seed rc=$?"
        done
    done
done
