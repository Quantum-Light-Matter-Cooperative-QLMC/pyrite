#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o cone379-acc-%j.log
# #379 end-to-end acceptance through the production run path; measurement only.
# Site flags at submit: qlmc `-p cpu -w qlmc-queen`; Hoffman2 `-p campus -A pi_qlmc`
# with PYRITE_DATA_DIR=$HOME/pyrite exported (tables).
# GPU=1 scores the monitor and spectra on CUDA (split transport stays on the CPU
# per-electron core); add `--gres=gpu:1` (Hoffman2: `--constraint 'A100|H200'`) at submit.
# ARGS: the full argument list of checks/detector_cone_acceptance_379.py (mode first).
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
if [ "${GPU:-0}" = 1 ]; then
  if [ -f /u/local/Modules/default/init/modules.sh ]; then
    . /u/local/Modules/default/init/modules.sh
    module load cuda
  fi
  sync_extra=(--extra nvidia)
else
  sync_extra=()
fi
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "code $(cat CODE_SHA 2>/dev/null || echo unknown)"
pixi install --locked -e nvidia "${sync_extra[@]}"
nvidia-smi -L 2>/dev/null || true
read -r -a argv <<<"${ARGS:?set ARGS}"
pixi run --as-is -e nvidia python checks/detector_cone_acceptance_379.py "${argv[@]}"
