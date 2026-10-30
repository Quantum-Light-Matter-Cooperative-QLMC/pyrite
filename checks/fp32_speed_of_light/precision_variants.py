"""FP32 speed-of-light and mixed-precision timing variants of the exact CUDA kernel (#831).

Timing experiment only: no production option. The #827 specialized transport
kernel's ``cupyx.jit`` source is rewritten as text and compiled through the
same CuPy cache path ``cupyx.jit`` uses, so the ``exact`` variant is the
production kernel byte for byte (instrumented only by CUDA events).

Variants (see ``transform``):

``exact``
    The generated source unchanged.
``fp32``
    Upper bound. Every ``double`` becomes ``float`` and every floating literal
    a ``float`` literal, so all arithmetic, transcendental math and local state
    is FP32. Read-only tables are passed as float32 copies; the state and row
    output arrays stay float64 in memory (one conversion per load/store), so
    the host pipeline still sees the dtypes it expects. Physics is wrong by
    construction; it measures what FP64 costs, nothing else.
``mixed``
    Targeted: the device functions that look up and interpolate energy-grid
    tables and sample angles (``MIXED_FUNCTIONS``) are FP32 throughout,
    signatures included, and their tables (``MIXED_TABLES``) are float32
    copies. The kernel body (position, direction, energy, path and clock
    accumulation, boundary distances, rotation) stays FP64; values cross the
    function boundary by implicit conversion.
``mixed_math``
    ``mixed`` plus every ``log``/``exp``/``sqrt``/``sin``/``cos``/``pow`` call
    left in the kernel body evaluated in FP32 (argument narrowed, result
    widened): the inline ELSEPA rate ``exp``, free-path ``log``, azimuth
    ``sincos`` and rotation ``sqrt``. Body arithmetic and accumulation stay FP64.

The variant wraps ``_jit_launch.transport_kernel_variant``; the #821 harness
(``pyrite_bench.py``, passed by path) then runs unchanged with
``--kernel specialized``. CUDA events around every launch give kernel-only
time per ``simulate_trajectories`` seed, written to ``--stats``.
"""

import argparse
import json
import os
import re
import runpy
import sys
from pathlib import Path

VARIANTS = ("exact", "fp32", "mixed", "mixed_math")

# Device functions (base names, without the transpiler's ``_<n>`` suffix) that
# compute in FP32 in the ``mixed`` variants: energy-grid lookup and
# interpolation (ELSEPA grid, SBETHE stopping, Mott screening, BremsLib
# radiative moments and photon sampling) and ELSEPA polar-angle inversion.
MIXED_FUNCTIONS = {
    "_log_grid_lower",
    "_log_grid_fraction",
    "_dEds_sbethe",
    "_interp_mott_log_alpha",
    "_elsepa_invert_row",
    "_elsepa_quantile_mu",
    "_rad_transport_moment",
    "_rad_layer_moment",
    "_rad_moment",
    "_rad_row",
    "_rad_fraction",
    "_rad_next_node",
    "_rad_photon",
    "_rad_row_chi",
    "_rad_chi",
    "_rad_cell_rate",
    "_rad_dielectric_cell",
    "_rad_sample_photon_eV",
}

# Read-only kernel tables passed as float32 copies in the ``mixed`` variants:
# every table the functions above read, plus the ELSEPA log-rate the kernel
# body interpolates inline.
MIXED_TABLES = {
    "el_logE",
    "el_log_rate",
    "el_cdf",
    "el_pdf",
    "el_mu",
    "L_sbethe_logE",
    "L_sbethe_logS",
    "mott_logE_flat",
    "mott_logA_flat",
    "il_logE",
    "rad_ncm3",
    "rad_incident",
    "rad_nominal",
    "rad_top",
    "rad_chi",
    "rad_plasma",
}

# Kernel arrays that stay float64 in memory in the ``fp32`` variant: state
# read and written across continuation rounds, and the retained row outputs.
FP64_STORAGE = {
    "clock",
    "pos",
    "dirs",
    "E_keV",
    "stragg_dE",
    "cont_f64",
    "seg_dir",
    "seg_mid",
    "seg_len",
    "seg_E",
    "seg_t0",
    "seg_E_end",
    "seg_t_end",
    "seg_hard_W",
    "seg_hard_dir",
    "seg_rad_k",
}

# Math calls the ``mixed_math`` variant evaluates in FP32 in the kernel body.
BODY_MATH = ("log10", "log", "exp", "sqrt", "sin", "cos", "pow")

_FLOAT_LITERAL = re.compile(
    r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])"
)
_DEVICE_FUNCTION = re.compile(r"^__device__ [\w:<>, ]+? (\w+)\(", re.M)


def _to_float(text):
    """``double`` to ``float`` and every floating literal to a float literal."""
    text = re.sub(r"\bdouble\b", "float", text)
    return _FLOAT_LITERAL.sub(lambda m: m.group(1) + "f", text)


def _matching(code, start, open_, close):
    """Index one past the bracket that closes ``code[start]``."""
    depth = 0
    for i in range(start, len(code)):
        if code[i] == open_:
            depth += 1
        elif code[i] == close:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced source")


def _split_top(text):
    """Split at commas outside ``<>``/``()``."""
    parts, depth, last = [], 0, 0
    for i, c in enumerate(text):
        depth += c in "<(" and 1 or (c in ">)" and -1) or 0
        if c == "," and depth == 0:
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return parts


def _device_functions(code):
    """``{base name: (start, end)}`` of every non-template device function."""
    out = {}
    for m in _DEVICE_FUNCTION.finditer(code):
        if code[max(0, m.start() - 30) : m.start()].rstrip().endswith(">"):
            continue  # template helper
        end = _matching(code, code.index("{", _matching(code, m.end() - 1, "(", ")")), "{", "}")
        out[re.sub(r"_\d+$", "", m.group(1))] = (m.start(), end)
    return out


def _retype_params(code, kernel_name, to_float):
    """Kernel source with the parameters named in ``to_float`` made ``float``."""
    start = code.index("(", code.index(f"void {kernel_name}("))
    end = _matching(code, start, "(", ")")
    decls = _split_top(code[start + 1 : end - 1])
    out = []
    for decl in decls:
        name = decl.split()[-1]
        out.append(re.sub(r"\bdouble\b", "float", decl) if name in to_float else decl)
    return code[: start + 1] + ",".join(out) + code[end - 1 :], start


def _float_math(text):
    """Rewrite ``fn(args)`` for ``BODY_MATH`` to ``((double)fnf((float)(args)))``."""
    pattern = re.compile(r"(?<![\w.])(" + "|".join(BODY_MATH) + r")\(")
    out, pos = [], 0
    while (m := pattern.search(text, pos)) is not None:
        close = _matching(text, m.end() - 1, "(", ")")
        args = _split_top(text[m.end() : close - 1])
        inner = ", ".join(f"(float)({_float_math(a)})" for a in args)
        out += [text[pos : m.start()], f"((double){m.group(1)}f({inner}))"]
        pos = close
    return "".join(out) + text[pos:]


def transform(code, variant, kernel_name, param_names):
    """The generated kernel source rewritten for ``variant`` (see the module doc)."""
    if variant == "exact":
        return code
    if variant == "fp32":
        storage_float = set(param_names) - FP64_STORAGE
        code, start = _retype_params(code, kernel_name, storage_float)
        body_start = _matching(code, start, "(", ")")
        head, params, body = code[:start], code[start:body_start], code[body_start:]
        # Compound assignments bind a typed reference to the array element;
        # ``auto`` keeps it valid for the arrays that stay double.
        body = re.sub(r"\bdouble &(_tmp_\d+)", r"auto &\1", body)
        return _to_float(head) + params + _to_float(body)
    if variant not in ("mixed", "mixed_math"):
        raise ValueError(variant)
    code, start = _retype_params(code, kernel_name, MIXED_TABLES)
    spans = sorted(
        (span for name, span in _device_functions(code[:start]).items() if name in MIXED_FUNCTIONS),
        reverse=True,
    )
    for lo, hi in spans:
        code = code[:lo] + _to_float(code[lo:hi]) + code[hi:]
    if variant == "mixed_math":
        start = code.index("(", code.index(f"void {kernel_name}("))
        body_start = _matching(code, start, "(", ")")
        code = code[:body_start] + _float_math(code[body_start:])
    return code


def float32_params(variant, param_names):
    """Kernel parameters the launch passes as float32 copies for ``variant``."""
    if variant == "fp32":
        return set(param_names) - FP64_STORAGE
    if variant in ("mixed", "mixed_math"):
        return set(MIXED_TABLES)
    return set()


class InstrumentedKernel:
    """Compile the (transformed) generated source and time each launch."""

    def __init__(self, jit_kernel, variant, stats, dump_dir):
        self.jit_kernel = jit_kernel
        self.variant = variant
        self.stats = stats
        self.dump_dir = dump_dir
        self.param_names = jit_kernel._func.__code__.co_varnames[
            : jit_kernel._func.__code__.co_argcount
        ]
        self.compiled = {}

    def _compile(self, in_types):
        import cupy
        from cupy._core import core
        from cupy_backends.cuda.api import driver
        from cupyx.jit import _compile, _cuda_types

        result = self.jit_kernel._cached_codes.get(in_types)
        if result is None:
            result = _compile.transpile(
                self.jit_kernel._func,
                ['extern "C"', "__global__"],
                self.jit_kernel._mode,
                in_types,
                _cuda_types.void,
            )
            self.jit_kernel._cached_codes[in_types] = result
        code = transform(result.code, self.variant, result.func_name, self.param_names)
        if self.dump_dir:
            d = Path(self.dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"exact-{result.func_name}.cu").write_text(result.code)
            (d / f"{self.variant}-{result.func_name}.cu").write_text(code)
            (d / f"options-{result.func_name}.json").write_text(
                json.dumps([list(result.options), result.backend, bool(result.jitify)])
            )
        options = result.options
        if result.backend == "nvcc":
            options += ("-DCUPY_JIT_NVCC",)
        module = core.compile_with_cache(
            source=code, options=options, backend=result.backend, jitify=result.jitify
        )
        kern = module.get_function(result.func_name)
        attrs = {
            name: driver.funcGetAttribute(
                getattr(driver, f"CU_FUNC_ATTRIBUTE_{name.upper()}"), kern.ptr
            )
            for name in ("num_regs", "local_size_bytes", "const_size_bytes", "binary_version")
        }
        attrs["func_name"] = result.func_name
        attrs["blocks_per_sm_at_128"] = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            kern.ptr, 128, 0
        )
        self.stats["kernels"][result.func_name] = attrs
        names = float32_params(self.variant, self.param_names)
        f32 = [
            i
            for i, (t, name) in enumerate(zip(in_types, self.param_names, strict=True))
            if name in names and getattr(t, "dtype", None) == cupy.float64
        ]
        return kern, result.enable_cooperative_groups, f32

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        import cupy
        import numpy
        from cupyx.jit import _cuda_typerules, _cuda_types
        from cupyx.jit._cuda_types import Scalar

        in_types = []
        for x in args:
            if isinstance(x, cupy.ndarray):
                in_types.append(_cuda_types.CArray.from_ndarray(x))
            else:
                in_types.append(_cuda_typerules.get_ctype_from_scalar(self.jit_kernel._mode, x))
        in_types = tuple(in_types)
        if in_types not in self.compiled:
            self.compiled[in_types] = self._compile(in_types)
        kern, enable_cg, f32 = self.compiled[in_types]
        new_args = []
        for i, (a, t) in enumerate(zip(args, in_types, strict=True)):
            if isinstance(t, Scalar):
                a = numpy.float32(a) if (i in f32 or t.dtype.char == "e") else t.dtype.type(a)
            elif i in f32:
                key = (a.data.ptr, a.size)
                cache = self.stats["_f32_cache"]
                if key not in cache:
                    cache[key] = a.astype(cupy.float32)
                a = cache[key]
            new_args.append(a)
        start, stop = cupy.cuda.Event(), cupy.cuda.Event()
        start.record(stream)
        kern(grid, block, tuple(new_args), shared_mem, stream, enable_cg)
        stop.record(stream)
        self.stats["_events"].setdefault(self.stats["_seed"], []).append((start, stop))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--bench", type=Path, required=True, help="#821 pyrite_bench.py")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--dump-dir", default=os.environ.get("PV_DUMP_DIR"))
    args, bench_args = parser.parse_known_args()

    import cupy

    import pyrite.montecarlo.transport as transport
    from pyrite.montecarlo.transport import _jit_launch

    stats = {"variant": args.variant, "kernels": {}, "_events": {}, "_seed": None, "_f32_cache": {}}
    original_variant = _jit_launch.transport_kernel_variant
    wrapped = {}

    def variant_kernel(switches):
        if switches not in wrapped:
            wrapped[switches] = InstrumentedKernel(
                original_variant(switches), args.variant, stats, args.dump_dir
            )
            stats.setdefault("switches", []).append(switches._asdict())
        return wrapped[switches]

    _jit_launch.transport_kernel_variant = variant_kernel
    simulate = transport.simulate_trajectories

    def tagged_simulate(*a, **kw):
        stats["_seed"] = int(kw["seed"])
        stats["_f32_cache"].clear()
        return simulate(*a, **kw)

    transport.simulate_trajectories = tagged_simulate
    sys.argv = [str(args.bench), *bench_args, "--kernel", "specialized"]
    runpy.run_path(str(args.bench), run_name="__main__")
    cupy.cuda.Device().synchronize()
    per_seed = {
        str(seed): {
            "launches": len(ev),
            "kernel_s": sum(cupy.cuda.get_elapsed_time(s, e) for s, e in ev) / 1e3,
        }
        for seed, ev in stats["_events"].items()
    }
    out = {k: v for k, v in stats.items() if not k.startswith("_")}
    out["per_seed"] = per_seed
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
