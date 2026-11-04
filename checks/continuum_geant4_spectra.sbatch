#!/bin/bash
# Submit from the staged checkout; override node/CPU allocation with sbatch.
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --job-name=g4-hopg278-spectra
#SBATCH --output=g4-hopg278-spectra-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
mkdir -p evidence/g4-hopg278-spectra
sha256sum "$G4BENCH" checks/geant4_performance/g4bench/g4bench.cc > evidence/g4-hopg278-spectra/code.txt
for energy in 300 3000 30000 100000; do
    for thickness in 100 1000; do
        stem="$PWD/evidence/g4-hopg278-spectra/${energy}-${thickness}"
        mkdir -p "$stem"
        "$G4BENCH" --material C --energy-kev "$energy" --thickness-um "$thickness" \
            --histories 100000 --warmup 1000 --seed 1 --threads "$SLURM_CPUS_PER_TASK" \
            --spectral 1 --output "$stem/rec" --cone-center-deg 90 --cone-half-deg 10 \
            > "$stem/run.log" 2>&1
    done
done
