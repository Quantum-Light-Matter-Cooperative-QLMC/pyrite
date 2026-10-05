#!/bin/bash
# Geant4 DPWA single-scattering reference (#183, #317): 10 shards x 10,000 primaries
# per case, shard i seeded "12345+i 67890". Task t runs case t/10, shard t%10.
# Submit from a directory holding the TestEm5 build and the four *_ssdpwa.mac files:
#   sbatch --array=0-39 run_ssdpwa.sbatch
# G4_RUN: command prefix that runs inside the Geant4 environment (e.g. a
# micromamba run -p PREFIX); empty runs ./TestEm5 directly.
#SBATCH --job-name=ssdpwa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=ssdpwa-%a.log
set -euo pipefail
CASES=(si_300kev si_800kev w_300kev w_800kev)
CASE=${CASES[$((SLURM_ARRAY_TASK_ID / 10))]}
I=$((SLURM_ARRAY_TASK_ID % 10))
mkdir -p out
MAC="${CASE}_ssdpwa_s${I}.mac"
sed -e "s|^/random/setSeeds .*|/random/setSeeds $((12345 + I)) 67890|" \
    -e "s|^/analysis/setFileName .*|/analysis/setFileName out/${CASE}_ssdpwa_s${I}|" \
    -e "s|^/run/beamOn .*|/run/beamOn 10000|" "${CASE}_ssdpwa.mac" > "$MAC"
${G4_RUN:-} ./TestEm5 "$MAC" > "out/${CASE}_ssdpwa_s${I}.log" 2>&1
