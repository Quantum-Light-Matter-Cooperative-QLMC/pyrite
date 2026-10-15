#!/bin/bash
#SBATCH --job-name=precision510audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --output=work-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export G4LEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$HOME/.local/opt/pyrite-issue-182/geant4/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$HOME/.local/opt/pyrite-issue-182/geant4/lib:${LD_LIBRARY_PATH:-}"
mkdir -p evidence
scontrol show job "$SLURM_JOB_ID" > "evidence/job-$SLURM_JOB_ID.txt"
lscpu > "evidence/cpu-$SLURM_JOB_ID.txt"
sha256sum checks/equal_precision/{manifest.json,scaling.json,geant4_adapter.py,run_single_scattering.py,run_pilot.py,single-scattering.sbatch} benchmark-geant4-v2/build/TestEm5 > "evidence/code-$SLURM_JOB_ID.txt"
mkdir -p "evidence/work-audit/$SLURM_JOB_ID"
sha256sum checks/equal_precision/{pyrite_adapter.py,audit_transport_work.py,work-audit.sbatch} > "evidence/work-audit/$SLURM_JOB_ID/code.txt"
for scene in si-300-100um-0deg w-300-10um-0deg; do
    PYRITE_MC_BACKEND=cuda "$HOME/.local/bin/uv" run --no-sync python \
        checks/equal_precision/audit_transport_work.py \
        "evidence/work-audit/$SLURM_JOB_ID/$scene-work.json" "$scene" \
        --histories 1000 --seed 53000 --core cuda --batch-histories 1000 \
        --scratch-mib 4096 --probe-electrons 0 \
        --output "evidence/work-audit/$SLURM_JOB_ID/$scene.json"
done
