#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/home/aamador/pyrite-993/runs/%x-%a.log
export PATH=$HOME/.local/bin:$PATH
export PYRITE_HOME=$HOME/pyrite-993 PYRITE_CATALOG=$HOME/pyrite-993/src/pyrite/data/catalog PYRITE_USER_CATALOG=$HOME/pyrite-993/user-catalog
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $HOME/pyrite-993
mkdir -p runs/$SLURM_JOB_NAME
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
echo "sha $(cat SYNCED_SHA) script $SCRIPT start $START chunk $CHUNK"
uv run --frozen python checks/$SCRIPT chunk "$(cat args/$ARGS.json)" $START $CHUNK runs/$SLURM_JOB_NAME/part-$SLURM_ARRAY_TASK_ID.json
