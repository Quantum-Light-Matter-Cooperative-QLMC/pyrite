"""Compile a ``cupyx.jit`` kernel's generated source with extra options (#796).

``cupyx.jit.rawkernel`` transpiles a Python kernel to CUDA C++ and compiles it
with fixed NVRTC options; it accepts no compile options or launch bounds. This
module takes the source the JIT generated for the launch's argument types (the
kernel's own ``_cached_codes`` entry, so the C++ is byte-for-byte the JIT's),
optionally rewrites it, and compiles it through the JIT's own entry point,
``cupy._core.core.compile_with_cache``, under the JIT's options plus a
:class:`KernelBuild`'s. ``cupy.RawModule`` is not used: it omits the CuPy
kernel header the generated source relies on (``CUDART_INF``). With the
default build the source, options and therefore the CuPy cache key equal the
JIT's, so the cubin is the JIT's. Launch geometry, argument conversion and
stream are the JIT's.

Experimental: it reads private CuPy state (``_JitRawKernel._func``, ``_mode``,
``_cached_codes``, ``cupyx.jit._compile.transpile`` and
``cupy._core.core.compile_with_cache``). See "Register cap and
read-only loads" in ``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Like its siblings, this module imports ``cupy`` at module scope and stays out
of the package ``__init__``.

Validation: gpu-transport-core
"""

import re
from typing import NamedTuple

import cupy
import numpy as np
from cupy._core import core
from cupyx.jit import _compile, _cuda_typerules, _cuda_types


class KernelBuild(NamedTuple):
    """Compile choices layered on the JIT's own NVRTC options.

    ``maxrregcount`` (0: compiler default) caps registers per thread;
    ``launch_bounds`` ``(max_threads, min_blocks)`` emits ``__launch_bounds__``
    on the kernel (``(0, 0)``: none); ``readonly`` names array parameters the
    kernel never writes, whose loads are then emitted as ``__ldg``
    (``ld.global.nc``).
    """

    maxrregcount: int = 0
    launch_bounds: tuple[int, int] = (0, 0)
    readonly: frozenset[str] = frozenset()

    def options(self):
        """NVRTC options this build adds to the JIT's."""
        return (f"--maxrregcount={self.maxrregcount}",) if self.maxrregcount else ()


# Array parameters of `_jit_kernel._transport_kernel` that hold tables or
# per-history inputs the kernel only reads. A write to one of them would not
# compile under the read-only build (its element type is `const`).
TRANSPORT_READONLY = frozenset(
    {
        "stream_key",
        "alive",
        "internal_bounds",
        "E_cut_by_electrons",
        "L_Js",
        "L_Zs",
        "L_ks",
        "L_coeffs",
        "L_E_cross",
        "L_ncm3",
        "L_sr_rate_numer",
        "L_mott_numer",
        "L_mott_denom1",
        "L_mott_denom2",
        "L_sr_joy_numer",
        "L_sbethe_n",
        "L_sbethe_logE",
        "L_sbethe_logS",
        "L_nel",
        "L_top",
        "L_bot",
        "mott_has_table",
        "mott_start",
        "mott_len",
        "mott_logE_flat",
        "mott_logA_flat",
        "el_start",
        "el_len",
        "el_logE",
        "el_log_rate",
        "el_cdf",
        "el_pdf",
        "el_mu",
        "el_guide",
        "il_keys",
        "il_n",
        "il_logE",
        "il_rate",
        "il_omega2",
        "il_nch",
        "il_ch_rate",
        "il_ch_U",
        "il_ch_W",
        "il_ch_branch",
        "il_ch_code",
        "rad_keys",
        "rad_nel",
        "rad_nt",
        "rad_z",
        "rad_ncm3",
        "rad_incident",
        "rad_nominal",
        "rad_top",
        "rad_chi",
        "rad_plasma",
        "cont_slots",
    }
)

# Element reads of a `CArray<const T, ...>` go through `__ldg` (ld.global.nc).
# Same layout as CuPy's `CArray`, so the kernel argument is passed unchanged;
# indexing arithmetic copies `carray.cuh`'s `operator[](ptrdiff_t) const`.
_READONLY_CARRAY = r"""
template <typename T, int _ndim, bool _c_contiguous, bool _use_32bit_indexing>
class CArray<const T, _ndim, _c_contiguous, _use_32bit_indexing> {
public:
  static const int ndim = _ndim;
  static const bool c_contiguous = _c_contiguous;
  typedef typename cupy::type_traits::conditional<_use_32bit_indexing, int, ptrdiff_t>::type index_t;

private:
  const T* data_;
  ptrdiff_t size_;
  ptrdiff_t shape_[ndim];
  ptrdiff_t strides_[ndim];

public:
  __device__ ptrdiff_t size() const { return size_; }
  __device__ const ptrdiff_t* shape() const { return shape_; }
  __device__ const ptrdiff_t* strides() const { return strides_; }

  __device__ T operator[](ptrdiff_t idx) const {
    if (c_contiguous) {
      return __ldg(data_ + idx);
    } else {
      index_t diff = 0;
      index_t i = static_cast<index_t>(idx);
      for (int dim = ndim; --dim > 0; ) {
          index_t shape_dim = static_cast<index_t>(shape_[dim]);
          diff += static_cast<index_t>(strides_[dim]) * (i % shape_dim);
          i /= shape_dim;
      }
      diff += static_cast<index_t>(strides_[0]) * i;
      const char* ptr = reinterpret_cast<const char*>(data_);
      return __ldg(reinterpret_cast<const T*>(ptr + diff));
    }
  }
};
"""


class _ReadOnlyCArray(_cuda_types.CArray):
    """A ``CArray`` parameter typed ``CArray<const T, ...>`` in generated code.

    ``cupyx.jit`` compares and hashes array types by their C++ spelling, so
    device functions that receive one are instantiated separately.
    """

    def __str__(self):
        return super().__str__().replace("CArray<", "CArray<const ", 1)


_INCLUDE = re.compile(r"^#include .*\n", re.MULTILINE)
_KERNEL_HEAD = re.compile(r'extern "C" __global__ void (\w+)\(')


def build_source(code, build):
    """Return ``code`` rewritten for ``build`` (unchanged for the default build)."""
    max_threads, min_blocks = build.launch_bounds
    if max_threads:
        bounds = f"__launch_bounds__({max_threads}, {min_blocks}) "
        code, n = _KERNEL_HEAD.subn(lambda m: f'extern "C" __global__ {bounds}void {m[1]}(', code)
        if n != 1:
            raise ValueError("expected exactly one __global__ kernel in the generated source")
    if build.readonly:
        # After the generated includes; CuPy's own header, which defines
        # `CArray`, precedes the whole source.
        end = max((m.end() for m in _INCLUDE.finditer(code)), default=0)
        code = code[:end] + _READONLY_CARRAY + code[end:]
    return code


def _readonly_types(func, in_types, names):
    """``in_types`` with the array parameters named in ``names`` made read-only."""
    parameters = func.__code__.co_varnames[: func.__code__.co_argcount]
    unknown = set(names) - set(parameters)
    if unknown:
        raise ValueError(f"not parameters of {func.__name__}: {sorted(unknown)}")
    return tuple(
        _ReadOnlyCArray(t.dtype, t._ndim, t._c_contiguous, t._index_32_bits)
        if name in names and isinstance(t, _cuda_types.CArray)
        else t
        for name, t in zip(parameters, in_types, strict=True)
    )


def argument_types(mode, args):
    """The ``cupyx.jit`` type tuple a launch with ``args`` transpiles for."""
    types = []
    for x in args:
        if isinstance(x, cupy.ndarray):
            types.append(_cuda_types.CArray.from_ndarray(x))
        elif np.isscalar(x):
            types.append(_cuda_typerules.get_ctype_from_scalar(mode, x))
        else:
            raise TypeError(f"{type(x)} is not supported for RawKernel")
    return tuple(types)


class RawModuleKernel:
    """``jit_kernel``'s generated source compiled under ``build``.

    Called like a ``cupyx.jit`` kernel: ``kernel(grid, block, args)``. Each
    argument-type tuple compiles once per device.
    """

    def __init__(self, jit_kernel, build=KernelBuild()):
        self.jit_kernel = jit_kernel
        self.build = build
        self._compiled = {}

    def compiled(self, args):
        """``(Function, source, options, in_types)`` for a launch with ``args``."""
        jit_kernel = self.jit_kernel
        in_types = argument_types(jit_kernel._mode, args)
        key = (in_types, cupy.cuda.get_device_id())
        entry = self._compiled.get(key)
        if entry is None:
            # The JIT's own source when the types are the JIT's.
            result = jit_kernel._cached_codes.get(in_types)
            if self.build.readonly or result is None:
                result = _compile.transpile(
                    jit_kernel._func,
                    ['extern "C"', "__global__"],
                    jit_kernel._mode,
                    _readonly_types(jit_kernel._func, in_types, self.build.readonly),
                    _cuda_types.void,
                )
            if result.backend != "nvrtc" or result.enable_cooperative_groups:
                raise NotImplementedError("only plain NVRTC kernels are supported")
            source = build_source(result.code, self.build)
            options = tuple(result.options) + self.build.options()
            module = core.compile_with_cache(
                source=source, options=options, backend=result.backend, jitify=result.jitify
            )
            entry = (module.get_function(result.func_name), source, options, in_types)
            self._compiled[key] = entry
        return entry

    def __call__(self, grid, block, args, shared_mem=0):
        kernel, _, _, in_types = self.compiled(args)
        converted = []
        for a, t in zip(args, in_types, strict=True):
            if isinstance(t, _cuda_types.Scalar):
                a = np.float32(a) if t.dtype.char == "e" else t.dtype.type(a)
            converted.append(a)
        kernel(grid, block, tuple(converted), shared_mem, None, False)
