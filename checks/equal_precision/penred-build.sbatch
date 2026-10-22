#!/bin/bash
#SBATCH --job-name=penred510build
#SBATCH --partition=cpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=penred-build-%j.log
# Build pinned PenRed with the #510 tally on the node that times it
# (PenRed's default WITH_NATIVE=ON targets the build host).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PIN=145a8a357f08affa551f592300fa6dfcab98a762
DEST="$HOME/.local/opt/penred-510/$PIN"
if [ -e "$DEST" ]; then echo "refusing to overwrite $DEST" >&2; exit 1; fi
mkdir -p "$DEST"
git clone --quiet https://github.com/PenRed/PenRed.git "$DEST/source"
git -C "$DEST/source" checkout --quiet "$PIN"
test "$(git -C "$DEST/source" rev-parse HEAD)" = "$PIN"
"$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/prepare_penred.py "$DEST/source"
mkdir "$DEST/source/src/build"
cd "$DEST/source/src/build"
cmake -DWITH_DICOM=OFF -DWITH_MPI=OFF -DWITH_LB=OFF -DBUILD_UTILITIES=OFF \
  -DCMAKE_BUILD_TYPE=Release .. > "$DEST/cmake.log" 2>&1
cmake --build . --target install -j "$SLURM_CPUS_PER_TASK" > "$DEST/build.log" 2>&1
EXE="$DEST/source/src/compiled/mains/penmain/pen_main"
{
  echo "pin $PIN"
  echo "host $(hostname)"
  g++ --version | head -1
  cmake --version | head -1
  grep -E '^(WITH_NATIVE|CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS)' CMakeCache.txt
  sha256sum "$EXE"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'
} | tee "$DEST/build-info.txt"
