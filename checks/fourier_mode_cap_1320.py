"""Device Fourier bin-mean reduction: mode-count census and capped-route replay (#1320).

Usage:
  fourier_mode_cap_1320.py capture MATERIAL E0_KEV TILT_DEG HISTORIES OUT_DIR
  fourier_mode_cap_1320.py replay OUT_DIR

``capture`` resolves one case of the ``PROFILE`` catalog profile (default
``thick_hiE_1295_45deg``; profile, beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) as
``pyrite run`` does, fixes its population at ``HISTORIES`` and runs it once
through ``run_cases``. Every ``cuda_fourier_bin_mean_reduce`` call is first
censused on the host plan: lines, axis, broad threshold and broad count, the
Fourier lines' first-zero width range ``W = pi / a``, period ``P`` and FFT
length, the per-line mode counts ``floor(P / W)`` (max, sum, quantiles, lines
over 2^20 / 2^22) and whether the widest or the narrowest lines drive them.
``CENSUS_ONLY=1`` (default) skips the Fourier route for flushes over
``2^22`` modes: they are saved as ``flush-<i>.npz`` (``E_res``, ``a_width``,
``weight``, ``axis``; up to ``CAPTURE``, default 3, largest kept) and reduced
by the near/far kernel so the run finishes. ``CENSUS_ONLY=0`` runs the
production function on every flush (the #1320 acceptance run); ``COMPARE=1``
also reduces each such flush by near/far (``auto``) on the same lines and
records peak, max abs difference, totals and their difference
(``COMPARE_EVERY=k``: every k-th flush only). Record:
``capture.json``; numeric arrays of the case result: ``spectrum.npz``.

``replay`` reduces every saved flush through the production device route and
the near/far ``auto`` kernel, records wall, CuPy pool peak and the difference
(max abs / peak, relative total). Record: ``replay.json``.

Remote only (GPU). Engineering evidence for ``cxr-line-fourier-reduction``.
"""

import dataclasses
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROFILE = os.environ.get("PROFILE", "thick_hiE_1295_45deg")
CAPTURE = int(os.environ.get("CAPTURE", "3"))
CENSUS_ONLY = os.environ.get("CENSUS_ONLY", "1") == "1"
#: ``COMPARE=1``: also reduce every Fourier-route flush by near/far and record the difference.
COMPARE = os.environ.get("COMPARE", "0") == "1"
#: Compare only every ``COMPARE_EVERY``-th flush (the near/far reference dominates wall).
COMPARE_EVERY = int(os.environ.get("COMPARE_EVERY", "1"))
os.environ["PROFILE"] = PROFILE
#: Flushes above this many modes skip the Fourier route in census mode.
_SKIP_MODES = 1 << 22


def _cp():
    import cupy

    return cupy


def _sync():
    _cp().cuda.Device().synchronize()


def _write(path, record):
    Path(path).write_text(json.dumps(record, indent=1, default=str))


def census(E_res, a_width, weight, axis):
    """Host replica of the device route's split, period and per-line mode counts."""
    from pyrite.montecarlo.spectrum.cxr._bin_broad import broad_width_for_axis
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL, fourier_period

    cp = _cp()
    c0, h, n_bins = axis
    E, a, w = (
        cp.asnumpy(cp.asarray(x, dtype=cp.float64)).ravel() for x in (E_res, a_width, weight)
    )
    keep = w != 0.0
    E, a, w = E[keep], a[keep], w[keep]
    rec = dict(lines=int(E.size), c0=float(c0), h=float(h), n_bins=int(n_bins))
    if E.size == 0:
        return rec
    W0 = broad_width_for_axis(h, (n_bins - 1) * h, FOURIER_RESIDUAL_TOL, 1000.0)
    broad = a <= np.pi / W0
    E, a, w = E[~broad], a[~broad], w[~broad]
    rec.update(broad_width_eV=W0, broad_lines=int(broad.sum()), fourier_lines=int(E.size))
    if E.size == 0:
        return rec
    n_fft, P, U = fourier_period(E[[E.argmin(), E.argmax()]], a.min(keepdims=True), c0, h, n_bins)
    W = np.pi / a
    counts = np.floor(2.0 * a * P / (2.0 * np.pi)).astype(np.int64)
    q = np.quantile(counts, [0.5, 0.9, 0.99, 0.999])
    mass = np.abs(w) * W
    rec.update(
        P=P,
        span=U,
        n_fft=n_fft,
        E_range=[float(E.min()), float(E.max())],
        W_min=float(W.min()),
        W_max=float(W.max()),
        W_quantiles=[float(x) for x in np.quantile(W, [0.001, 0.01, 0.5, 0.99, 0.999])],
        P_over_3U=P / (3.0 * (U + 0.5 * h)),
        count_max=int(counts.max()),
        count_sum=int(counts.sum()),
        count_quantiles=[int(x) for x in q],
        over_2_20=int((counts > 1 << 20).sum()),
        over_2_22=int((counts > 1 << 22).sum()),
        mass_over_2_20=float(mass[counts > 1 << 20].sum() / mass.sum()),
        narrow_W_hist=np.histogram(np.log10(W), bins=np.arange(-6, 6.5, 0.5))[0].tolist(),
    )
    return rec


def _near_far(E_res, a_width, weight, axis):
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import run_bin_mean_reduction_kernel

    cp = _cp()
    c0, h, n_bins = axis
    edges = c0 + (cp.arange(int(n_bins) + 1, dtype=cp.float64) - 0.5) * h
    inv = cp.full(int(n_bins), 1.0 / h)
    out = cp.zeros(int(n_bins), dtype=cp.float64)
    return run_bin_mean_reduction_kernel(
        cp.asarray(E_res), cp.asarray(a_width), cp.asarray(weight), edges, inv, out=out
    )


def capture(material, e0_kev, tilt_deg, histories, out_dir):
    from pyrite.montecarlo.runner.scheduling import run_cases
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.runs import scan

    helper = runpy.run_path(str(Path(__file__).with_name("adaptive_block_size_1252.py")))
    n = int(histories)
    resolved = scan._resolved_run

    def fixed(args, mat):
        settings, sweep, identity, stem = resolved(args, mat)
        precision = dataclasses.replace(
            settings.precision, min_histories=n, max_histories=n, pilot_histories=None
        )
        return dataclasses.replace(settings, precision=precision), sweep, identity, stem

    scan._resolved_run = fixed
    case, _, _ = helper["profile_case"](material, float(e0_kev), float(tilt_deg), "auto")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calls, saved = [], []
    real = fc.cuda_fourier_bin_mean_reduce
    cp = _cp()
    pool = cp.get_default_memory_pool()

    def probed(E_res, a_width, weight, axis, **kwargs):
        rec = census(E_res, a_width, weight, axis)
        skip = CENSUS_ONLY and rec.get("count_max", 0) > _SKIP_MODES
        _sync()
        t0 = time.perf_counter()
        res = (_near_far if skip else real)(E_res, a_width, weight, axis, **kwargs)
        _sync()
        rec.update(route="near-far" if skip else "fourier", wall_s=time.perf_counter() - t0)
        if COMPARE and not skip and len(calls) % COMPARE_EVERY == 0:
            ref = cp.asnumpy(_near_far(E_res, a_width, weight, axis))
            got = cp.asnumpy(res)
            rec.update(
                ref_peak=float(np.abs(ref).max()),
                max_abs_diff=float(np.abs(got - ref).max()),
                ref_total=float(ref.sum()),
                total_diff=float(got.sum() - ref.sum()),
            )
        rec["pool_total_GiB"] = pool.total_bytes() / 2**30
        calls.append(rec)
        if skip and len(saved) < CAPTURE:
            path = out / f"flush-{len(saved)}.npz"
            np.savez(
                path,
                E_res=cp.asnumpy(cp.asarray(E_res, dtype=cp.float64)),
                a_width=cp.asnumpy(cp.asarray(a_width, dtype=cp.float64)),
                weight=cp.asnumpy(cp.asarray(weight, dtype=cp.float64)),
                axis=np.array(axis, dtype=np.float64),
            )
            saved.append(str(path))
        print(json.dumps({k: v for k, v in rec.items() if k != "narrow_W_hist"}), flush=True)
        return res

    fc.cuda_fourier_bin_mean_reduce = probed
    timings = []
    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)[0]
    except Exception as exc:  # record failure as evidence
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    record = dict(
        issue=1320,
        code_sha=os.environ.get("PROBE_SHA"),
        profile=PROFILE,
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        census_only=CENSUS_ONLY,
        error=error,
        wall_s=time.perf_counter() - t0,
        calls=calls,
        saved=saved,
        timings=timings,
    )
    if result is not None:
        np.savez(out / "spectrum.npz", **_spectrum_arrays(result))
    _write(out / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("calls", "timings")}))


def _spectrum_arrays(result):
    """Numeric array entries of a case result dict, on the host."""
    arrays = {}
    for key, value in dict(result).items():
        if hasattr(value, "__cuda_array_interface__"):
            value = _cp().asnumpy(value)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiuc":
            arrays[key] = value
    return arrays


def replay(out_dir):
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cuda_fourier_bin_mean_reduce

    cp = _cp()
    pool = cp.get_default_memory_pool()
    rows = []
    for path in sorted(Path(out_dir).glob("flush-*.npz")):
        data = np.load(path)
        args = (data["E_res"], data["a_width"], data["weight"], tuple(data["axis"].tolist()))
        args = (*args[:3], (args[3][0], args[3][1], int(args[3][2])))
        row = dict(flush=path.name, census=census(*args))
        for name, fn in (("near_far", _near_far), ("fourier", cuda_fourier_bin_mean_reduce)):
            pool.free_all_blocks()
            _sync()
            t0 = time.perf_counter()
            try:
                res = fn(*args)
                _sync()
                row[name] = dict(
                    wall_s=time.perf_counter() - t0, peak_GiB=pool.total_bytes() / 2**30
                )
                row[name + "_values"] = cp.asnumpy(res)
            except Exception as exc:  # noqa: BLE001 - record failure as evidence
                row[name] = dict(error=f"{type(exc).__name__}: {exc}")
        if "near_far_values" in row and "fourier_values" in row:
            ref, got = row.pop("near_far_values"), row.pop("fourier_values")
            peak = float(np.abs(ref).max())
            row["max_abs_over_peak"] = float(np.abs(got - ref).max() / peak)
            row["rel_total"] = float(abs(got.sum() - ref.sum()) / abs(ref.sum()))
        row.pop("near_far_values", None)
        row.pop("fourier_values", None)
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write(
        Path(out_dir) / "replay.json",
        dict(issue=1320, code_sha=os.environ.get("PROBE_SHA"), rows=rows),
    )


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"capture": capture, "replay": replay}[cmd](*rest)
