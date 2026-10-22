#!/bin/bash
#SBATCH --job-name=522-overflow
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/aamador/pyrite-522-overflow/scratch/overflow/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-522-overflow
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=bcc966aaab7d4d2ae319b5f9fd7a47b6d8ead0dc+dirty
sha256sum --check <<'HASHES'
3b391ec5b0ee62d1060947858123f12c5239e8ec2f123568335e36aa7e6ad29e  checks/channeling_overflow.py
80545cadd49473b0f72a556b831e082cf4482385bb0a000aa245cbffd120eb7f  src/pyrite/montecarlo/channeling/overflow.py
9f7aecd724f8a75ece3efe007ad71a5590932103cfc16cab13798a659b06d23d  src/pyrite/montecarlo/channeling/density.py
812fa48f1c3193b1cbc4ccabac7ebfcd91464b8df86eb89bf7cdebde95fc6345  src/pyrite/montecarlo/channeling/operators.py
HASHES
id=$SLURM_ARRAY_TASK_ID
case "$id" in
  0) plane=110; energy=5; H=16; dz=125; tails=(32 64);;
  1) plane=110; energy=5; H=16; dz=62.5; tails=(32 64);;
  2) plane=100; energy=54.5; H=48; dz=31.25; tails=(64 96);;
  3) plane=110; energy=54.5; H=48; dz=31.25; tails=(64 96);;
esac
set +e
/usr/bin/time -v ~/.local/bin/uv run --no-default-groups python checks/channeling_overflow.py \
  --plane "$plane" --energy-MeV "$energy" --orders "$H" --cells 16 \
  --tail-orders "${tails[@]}" --dz "$dz" --thickness-um 20 \
  --out "scratch/overflow/run-$id.json"
status=$?
echo "Diagnostic exit status: $status (1 means a reported closure failure)"
exit "$status"
