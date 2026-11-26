"""Device fast-math arms for the direct Fourier ``rectangles`` kernel (#1463).

Usage:
  gpu_fast_math_1463.py arms FLUSH_DIR OUT_JSON [--repeats N] [--only A,B] [--modes M]

``FLUSH_DIR`` holds ``flush-*.npz`` saved by
``checks/fourier_reduction_phases_1296.py capture`` (production line arrays
of one case). Each flush is planned once by the production
``cuda_fourier_bin_mean_reduce`` (default ``direct_precision="fp32"``), whose
direct-kernel launch arguments are intercepted, so every arm runs the same
tiles. Arms recompile the production ``_fourier_cuda._DIRECT_SOURCE`` under
one compile-option set (NVRTC or hipRTC spelling) or with the hand-placed
``__sincosf`` intrinsic, each as device symbol ``rectangles_<arm>`` so profiler
rows name their arm. Per arm and flush, recorded:

- kernel time (CUDA events, median/min over ``--repeats`` launches into a
  buffer zeroed before the start event; arm order rotates every repeat);
- determinism: whether every repeat is bitwise equal to the first;
- coefficient error against the ``fp64-strict`` arm (FP64 ``sincos``,
  contraction off), normalized per mode by ``sum_l |c_l(j)|`` (the
  ``DIRECT_FP32_PHASE_ERROR`` bound's form, with the ``sqrt(2)`` factor);
- the same against a host long-double sum on ``--modes`` sampled modes;
- the line spectrum (full production reduction with the arm's kernel swapped
  in) against the ``fp64-strict`` spectrum: max abs / peak and relative
  total.

``--only`` selects arms (for a short ``ncu`` pass). Diagnostic only:
production code and defaults are unchanged; no physics claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

#: (name, direct precision, NVRTC options, hipRTC options or None, intrinsic sincos).
ARMS = (
    ("fp64-strict", "fp64", ("--fmad=false",), ("-ffp-contract=off",), False),
    ("fp64-prod", "fp64", (), (), False),
    ("fp64-fastmath", "fp64", ("--use_fast_math",), ("-ffast-math",), False),
    ("fp32-strict", "fp32", ("--fmad=false",), ("-ffp-contract=off",), False),
    ("fp32-prod", "fp32", (), (), False),
    ("fp32-ftz", "fp32", ("-ftz=true",), None, False),
    ("fp32-prec-div", "fp32", ("--prec-div=false",), None, False),
    ("fp32-prec-sqrt", "fp32", ("--prec-sqrt=false",), None, False),
    ("fp32-fmad", "fp32", ("--fmad=true",), ("-ffp-contract=fast",), False),
    ("fp32-fastmath", "fp32", ("--use_fast_math",), ("-ffast-math",), False),
    ("fp32-intrinsic", "fp32", (), (), True),
    ("fp32-intrinsic-fmad", "fp32", ("--fmad=true",), ("-ffp-contract=fast",), True),
)
REFERENCE = "fp64-strict"


def _cp():
    import cupy

    return cupy


def _is_hip(cp):
    return bool(getattr(cp.cuda.runtime, "is_hip", False))


def _source(symbol, intrinsic, abs_sum=False):
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc

    src = fc._DIRECT_SOURCE
    assert src.count("void rectangles(") == 1
    src = src.replace("void rectangles(", f"void {symbol}(")
    if intrinsic:
        assert src.count("sincosf(") == 1
        src = src.replace("sincosf(", "__sincosf(")
    if abs_sum:
        for old, new in (("re += w * c;", "re += fabs(w);"), ("im += w * s;", "im += 0.0 * s;")):
            assert src.count(old) == 1
            src = src.replace(old, new)
    return src


def _kernel(cp, arm, precision, options, intrinsic, abs_sum=False):
    flag = "1" if precision == "fp32" else "0"
    symbol = "rectangles_" + arm.replace("-", "_")
    return cp.RawKernel(
        _source(symbol, intrinsic, abs_sum),
        symbol,
        options=(f"-DDIRECT_FP32={flag}", *options),
    )


def _arms(cp, only):
    hip = _is_hip(cp)
    out = []
    for name, precision, nv, amd, intrinsic in ARMS:
        options = amd if hip else nv
        if options is None or (only and name not in only and name != REFERENCE):
            continue
        kernel = _kernel(cp, name, precision, options, intrinsic)
        out.append((name, precision, options, intrinsic, kernel))
    return out


def _intercept(fc, E, a, w, axis, kwargs):
    """Production planning of one flush; returns the direct launch's arguments."""
    seen = []

    def stub(precision="fp64"):
        def launch(grid, block, args):
            phase, constant, linear, tiles, dt, out = args
            seen.append(
                dict(
                    precision=precision,
                    grid=grid,
                    block=block,
                    phase=phase.copy(),
                    constant=constant.copy(),
                    linear=linear.copy(),
                    tiles=tiles.copy(),
                    dt=dt,
                    n_out=int(out.size),
                )
            )

        return launch

    real = fc._direct_kernel
    fc._direct_kernel = stub
    try:
        fc.cuda_fourier_bin_mean_reduce(E, a, w, axis, **kwargs)
    finally:
        fc._direct_kernel = real
    return seen


def _spectrum(fc, kernel, E, a, w, axis, kwargs):
    real = fc._direct_kernel
    fc._direct_kernel = lambda precision="fp64": kernel
    try:
        return _cp().asnumpy(fc.cuda_fourier_bin_mean_reduce(E, a, w, axis, **kwargs))
    finally:
        fc._direct_kernel = real


def _launch(kernel, launch, buf):
    """One kernel launch, accumulating into ``buf`` (the caller zeroes it)."""
    kernel(
        launch["grid"],
        launch["block"],
        (
            launch["phase"],
            launch["constant"],
            launch["linear"],
            launch["tiles"],
            np.float64(launch["dt"]),
            buf,
        ),
    )


def _host_reference(launch, modes):
    """Long-double direct sums at sampled modes (same tiles, same terms)."""
    cp = _cp()
    tiles = cp.asnumpy(launch["tiles"])
    phase = cp.asnumpy(launch["phase"]).astype(np.longdouble)
    constant = cp.asnumpy(launch["constant"]).astype(np.longdouble)
    linear = cp.asnumpy(launch["linear"]).astype(np.longdouble)
    dt = np.longdouble(launch["dt"])
    out = {}
    for j in modes:
        rows = tiles[(tiles[:, 0] <= j) & (j < tiles[:, 1])]
        re = im = np.longdouble(0)
        jd = np.longdouble(j)
        for _, _, lo, hi in rows.tolist():
            th = -jd * phase[lo:hi]
            wl = constant[lo:hi] - (dt * jd) * linear[lo:hi]
            re += np.sum(wl * np.cos(th))
            im += np.sum(wl * np.sin(th))
        out[int(j)] = (re, im)
    return out


def _coefficient_error(values, ref, scale):
    d = np.abs(values.view(np.complex128) - ref.view(np.complex128))
    mask = scale > 0
    return float(np.max(d[mask] / scale[mask])) if mask.any() else 0.0


def _spectrum_error(x, ref):
    peak = float(np.max(np.abs(ref))) or 1.0
    total = float(np.sum(ref)) or 1.0
    return dict(
        max_abs_over_peak=float(np.max(np.abs(x - ref)) / peak),
        rel_total=float(abs(np.sum(x) - np.sum(ref)) / abs(total)),
        bitwise_equal=bool(np.array_equal(x, ref)),
    )


def _device(cp):
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return dict(
        name=name.decode() if isinstance(name, bytes) else str(name),
        hip=_is_hip(cp),
        cupy=cp.__version__,
        runtime=cp.cuda.runtime.runtimeGetVersion(),
        driver=cp.cuda.runtime.driverGetVersion(),
    )


def _cupy_default_options(cp):
    """CuPy's own fixed compiler options (FTZ and friends), as text evidence."""
    import inspect

    from cupy.cuda import compiler

    text = inspect.getsource(compiler)
    return sorted({ln.strip() for ln in text.splitlines() if "ftz" in ln or "flush" in ln})[:12]


def arms(flush_dir, out_json, repeats=5, only=(), n_modes=16):
    paths = sorted(Path(flush_dir).glob("flush-*.npz"))
    if not paths:
        raise SystemExit(f"no flush-*.npz in {flush_dir}")
    cp = _cp()
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc

    arm_list = _arms(cp, set(only))
    strict = ("-ffp-contract=off",) if _is_hip(cp) else ("--fmad=false",)
    abs_kernel = _kernel(cp, "abs-sum", "fp64", strict, False, abs_sum=True)
    flushes = []
    for path in paths:
        data = np.load(path)
        E, a, w = (cp.asarray(data[k]) for k in ("E_res", "a_width", "weight"))
        axis = tuple(data["axis"].tolist())
        axis = (axis[0], axis[1], int(axis[2]))
        kwargs = json.loads(str(data["kwargs"])) if "kwargs" in data else {}
        launches = _intercept(fc, E, a, w, axis, kwargs)
        if not launches:
            flushes.append(dict(flush=path.name, lines=int(E.size), direct_launch=False))
            continue
        launch = launches[0]
        buf = cp.zeros(2 * launch["n_out"], dtype=cp.float64)
        _launch(abs_kernel, launch, buf)
        scale = cp.asnumpy(buf[0::2]).copy()
        first, same, times = {}, {}, {n: [] for n, *_ in arm_list}
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        for _name, _p, _o, _i, kernel in arm_list:  # warm-up/compile
            _launch(kernel, launch, buf)
        cp.cuda.Device().synchronize()
        for r in range(repeats):
            k = r % len(arm_list)
            for name, _p, _o, _i, kernel in arm_list[k:] + arm_list[:k]:
                buf.fill(0)
                start.record()
                _launch(kernel, launch, buf)
                stop.record()
                stop.synchronize()
                times[name].append(cp.cuda.get_elapsed_time(start, stop))
                host = cp.asnumpy(buf)
                if name not in first:
                    first[name], same[name] = host, True
                elif not np.array_equal(host, first[name]):
                    same[name] = False
        ref = first[REFERENCE]
        nonzero = np.flatnonzero(scale > 0)
        picks = nonzero[np.linspace(0, nonzero.size - 1, min(n_modes, nonzero.size)).astype(int)]
        host_ref = _host_reference(launch, picks.tolist())
        spectra = {}
        for name, _p, _o, _i, kernel in arm_list:
            spectra[name] = _spectrum(fc, kernel, E, a, w, axis, kwargs)
        rows = []
        for name, precision, options, intrinsic, _k in arm_list:
            g = first[name].view(np.complex128)
            ld = max(
                float(abs(complex(g[j].real - float(re), g[j].imag - float(im))) / scale[j])
                for j, (re, im) in host_ref.items()
            )
            rows.append(
                dict(
                    arm=name,
                    precision=precision,
                    options=list(options),
                    intrinsic=intrinsic,
                    ms_median=float(np.median(times[name])),
                    ms_min=float(np.min(times[name])),
                    ms_all=times[name],
                    deterministic=same[name],
                    bitwise_equal_reference=bool(np.array_equal(first[name], ref)),
                    coeff_err_vs_fp64_strict=_coefficient_error(first[name], ref, scale),
                    coeff_err_vs_longdouble=ld,
                    spectrum_vs_fp64_strict=_spectrum_error(spectra[name], spectra[REFERENCE]),
                )
            )
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "ms_all"}), flush=True)
        flushes.append(
            dict(
                flush=path.name,
                lines=int(E.size),
                direct_lines=int(launch["phase"].size),
                direct_tiles=int(launch["tiles"].shape[0]),
                direct_pairs=int(
                    np.sum(
                        np.diff(cp.asnumpy(launch["tiles"])[:, :2], axis=1)
                        * np.diff(cp.asnumpy(launch["tiles"])[:, 2:], axis=1)
                    )
                ),
                kwargs=kwargs,
                arms=rows,
            )
        )
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import DIRECT_FP32_PHASE_ERROR

    record = dict(
        issue=1463,
        code_sha=os.environ.get("PROBE_SHA"),
        host=platform.node(),
        device=_device(cp),
        cupy_ftz_lines=_cupy_default_options(cp),
        repeats=repeats,
        bound_fp32=float(np.sqrt(2.0) * DIRECT_FP32_PHASE_ERROR),
        flushes=flushes,
        when=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    Path(out_json).write_text(json.dumps(record, indent=1, default=str))
    if not any("arms" in f for f in flushes):
        raise SystemExit(f"no flush reached the direct kernel; nothing measured ({out_json})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arms")
    p.add_argument("flush_dir")
    p.add_argument("out_json")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--only", default="")
    p.add_argument("--modes", type=int, default=16)
    args = parser.parse_args(argv)
    only = tuple(x for x in args.only.split(",") if x)
    arms(args.flush_dir, args.out_json, args.repeats, only, args.modes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
