# Sourced by the #821 job scripts from the synced checkout root.
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
G4_PREFIX="$HOME/.local/opt/pyrite-issue-182/geant4"
export G4LEDATA="$G4_PREFIX/share/Geant4/data/EMLOW8.8"
export G4ENSDFSTATEDATA="$G4_PREFIX/share/Geant4/data/ENSDFSTATE3.0"
export LD_LIBRARY_PATH="$G4_PREFIX/lib:${LD_LIBRARY_PATH:-}"
G4BENCH="$PWD/g4bench-build/g4bench"
if [[ ! -f g4bench-build/CMakeCache.txt ]]; then
    cmake -S checks/geant4_performance/g4bench -B g4bench-build -DCMAKE_BUILD_TYPE=Release \
        -DGeant4_DIR="$G4_PREFIX/lib/cmake/Geant4" -DCMAKE_PREFIX_PATH="$G4_PREFIX" \
        -DCMAKE_CXX_COMPILER="$G4_PREFIX/bin/x86_64-conda-linux-gnu-g++" > g4bench-cmake.log
fi
cmake --build g4bench-build -j 8 > g4bench-build.log
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
