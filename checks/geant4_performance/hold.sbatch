#!/bin/bash
# Holds the cpu partition's 16 CPUs on the GPU host while run.sbatch (gpu
# partition, 8 CPUs) measures. Partition caps leave the other 8 CPUs
# unallocatable, so the host is otherwise idle. Exits on the release file.
#SBATCH --job-name=g4bench821-hold
#SBATCH --partition=cpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=hold-%j.log
release="$SLURM_SUBMIT_DIR/evidence/hold-release-$SLURM_JOB_ID"
while [[ ! -e "$release" ]]; do sleep 10; done
