# shellcheck shell=bash
# Sourced by the AMD SLURM prelude (pyrite.remote.scripts._amd_prelude) before
# `uv sync --extra amd` builds CuPy from source with CUPY_INSTALL_USE_HIP=1.
#
# Every step only detects and fills a gap, so a node with AMD's full ROCm under
# /opt/rocm and a host C++ compiler passes through unchanged. The gaps are the
# ones a distribution ROCm leaves (Ubuntu 26.04 ROCm 7.1 on qlmc-ace, #888):
#
# 1. ROCm root under /usr, not /opt/rocm.
# 2. No host C++ compiler: use the HIP toolchain's clang/clang++.
# 3. Math-library headers under /usr/include/<lib>/ only: CuPy includes the
#    flat names (hipblas.h, ...), so add one-line wrappers.
# 4. roctx and roctracer, which CuPy links for profiler markers only, are each
#    stubbed when absent. Ubuntu packages roctx (libroctx-dev) but not
#    roctracer; a real roctx is always used, with a wrapper for whichever of
#    <roctx.h> / <roctracer/roctx.h> it lacks. A stubbed roctx makes
#    cupy.cuda.nvtx markers inert; a stubbed roctracer makes
#    cudaProfilerStart/Stop inert. Compute is unaffected either way.
#
# Wrappers and stubs live in $PYRITE_ROCM_SHIM (default: <checkout>/.rocm-shim),
# are rebuilt only when missing, and are announced on stderr. Stale roctx stubs
# are removed once a real roctx is installed, so they can never shadow it.

_pyrite_rocm_lib() {
  # Print the first unversioned link library $1 under the ROCm root, if any.
  local dir
  for dir in "$ROCM_HOME/lib" "$ROCM_HOME/lib64" "$ROCM_HOME/lib/x86_64-linux-gnu"; do
    if [ -e "$dir/$1" ]; then
      echo "$dir/$1"
      return 0
    fi
  done
  return 1
}

_pyrite_rocm_stub() {
  # _pyrite_rocm_stub <cc> <shim> <library> <C source>: build one no-op library.
  printf '%s\n' "$4" > "$2/src/$3.c"
  if ! "$1" -shared -fPIC -O2 -o "$2/lib/lib${3}64.so" "$2/src/$3.c"; then
    echo "rocm-env: $3 stub build failed with $1" >&2
    rm -f "$2/lib/lib${3}64.so"
    return 1
  fi
}

_pyrite_rocm_env() {
  local checkout=$1 shim hip_clang cc lib header roctx_header=""
  local need_flat=0 stub_roctx=0 stub_roctracer=0 wrap_roctx=""
  if [ -z "${ROCM_HOME:-}" ]; then
    if [ -d /opt/rocm ]; then
      ROCM_HOME=/opt/rocm
    else
      ROCM_HOME=$(hipconfig --rocmpath 2>/dev/null || echo /usr)
    fi
  fi
  export ROCM_HOME ROCM_PATH="${ROCM_PATH:-$ROCM_HOME}"
  if [ -z "${HCC_AMDGPU_TARGET:-}" ]; then
    HCC_AMDGPU_TARGET=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
    [ -n "$HCC_AMDGPU_TARGET" ] && export HCC_AMDGPU_TARGET
  fi
  if [ -z "${CXX:-}" ] && ! command -v c++ >/dev/null 2>&1; then
    hip_clang=$(hipconfig --hipclangpath 2>/dev/null || true)
    if [ -x "$hip_clang/clang++" ]; then
      export CC="$hip_clang/clang" CXX="$hip_clang/clang++"
      echo "rocm-env: no host c++; building with $CXX" >&2
    else
      echo "rocm-env: no host c++ and no HIP clang found; CuPy build will fail" >&2
    fi
  fi
  cc=${CC:-cc}
  for lib in hipblas hipsparse hipfft rocsolver; do
    if [ ! -e "$ROCM_HOME/include/$lib.h" ] && [ -e "$ROCM_HOME/include/$lib/$lib.h" ]; then
      need_flat=1
    fi
  done
  for header in roctracer/roctx.h roctx.h; do
    [ -e "$ROCM_HOME/include/$header" ] && roctx_header=${roctx_header:-$header}
  done
  if [ -n "$roctx_header" ] && _pyrite_rocm_lib libroctx64.so >/dev/null; then
    # Real roctx: wrap only the spelling the install lacks.
    if [ ! -e "$ROCM_HOME/include/roctx.h" ]; then
      wrap_roctx=roctx.h
    elif [ ! -e "$ROCM_HOME/include/roctracer/roctx.h" ]; then
      wrap_roctx=roctracer/roctx.h
    fi
  else
    stub_roctx=1
  fi
  [ -e "$ROCM_HOME/include/roctracer/roctracer_ext.h" ] || stub_roctracer=1
  shim=${PYRITE_ROCM_SHIM:-$checkout/.rocm-shim}
  if [ "$stub_roctx" = 0 ] && [ -d "$shim" ]; then
    # A shim built before roctx was installed must not shadow the real one.
    rm -f "$shim/lib/libroctx64.so" "$shim/include/roctx.h" "$shim/include/roctracer/roctx.h"
  fi
  [ "$need_flat$stub_roctx$stub_roctracer" = "000" ] && [ -z "$wrap_roctx" ] && return 0
  mkdir -p "$shim/include/roctracer" "$shim/lib" "$shim/src" || return 1
  if [ "$need_flat" = 1 ]; then
    for lib in hipblas hipsparse hipfft rocsolver; do
      header="$shim/include/$lib.h"
      if [ ! -e "$ROCM_HOME/include/$lib.h" ] && [ -e "$ROCM_HOME/include/$lib/$lib.h" ] \
        && [ ! -e "$header" ]; then
        printf '#pragma once\n#include <%s/%s.h>\n' "$lib" "$lib" > "$header"
      fi
    done
  fi
  if [ -n "$wrap_roctx" ]; then
    printf '#pragma once\n#include <%s>\n' "$roctx_header" > "$shim/include/$wrap_roctx"
  fi
  if [ "$stub_roctx" = 1 ] && [ ! -e "$shim/lib/libroctx64.so" ]; then
    cat > "$shim/include/roctx.h" <<'H'
#pragma once
#define ROCTX_VERSION_MAJOR 4
#define ROCTX_VERSION_MINOR 1
#ifdef __cplusplus
extern "C" {
#endif
void roctxMarkA(const char* message);
int roctxRangePushA(const char* message);
int roctxRangePop(void);
#ifdef __cplusplus
}
#endif
H
    cp "$shim/include/roctx.h" "$shim/include/roctracer/roctx.h"
    _pyrite_rocm_stub "$cc" "$shim" roctx "$(printf '%s\n' \
      'void roctxMarkA(const char* m){(void)m;}' \
      'int roctxRangePushA(const char* m){(void)m;return 0;}' \
      'int roctxRangePop(void){return 0;}')" || return 1
  fi
  if [ "$stub_roctracer" = 1 ] && [ ! -e "$shim/lib/libroctracer64.so" ]; then
    cat > "$shim/include/roctracer/roctracer_ext.h" <<'H'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void roctracer_start(void);
void roctracer_stop(void);
#ifdef __cplusplus
}
#endif
H
    _pyrite_rocm_stub "$cc" "$shim" roctracer \
      "$(printf '%s\n' 'void roctracer_start(void){}' 'void roctracer_stop(void){}')" || return 1
  fi
  export CPATH="$shim/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$shim/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$shim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "rocm-env: using evaluation shim $shim (flat headers=$need_flat," \
    "no-op roctx=$stub_roctx, no-op roctracer=$stub_roctracer)" >&2
}
