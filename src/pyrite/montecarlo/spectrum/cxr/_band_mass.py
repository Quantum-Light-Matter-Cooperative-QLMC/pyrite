"""Closed-form in-band mass of finite-time ``sinc^2`` lines (#1326).

A line ``w sinc^2(a (E - E_res) / pi)`` holds the energy mass
``w (pi / a) [F(x_hi) - F(x_lo)]`` inside ``[E_lo, E_hi]``, with
``x = a (E - E_res) / pi`` and ``F`` the antiderivative of ``sinc^2``
(:mod:`._bin_quadrature`). Through the tail complement ``R = F - s / 2`` this
is ``w (pi / a) [R(x_hi) - R(x_lo) + 1{x_lo < 0 <= x_hi}]``, with no
cancellation between numbers near ``+-1/2``.

Bin means that tile ``[E_lo, E_hi]`` sum to the same number: the bins'
integrals telescope, and the only difference is the far-envelope
replacement of the production route (at most ``1.2e-5`` of a line's mass,
``sinc-bin-far-envelope``). The reflection selector therefore reads a shell's
in-band yield off its line arrays instead of reducing them onto bins.

Validation: sinc-bin-integration
"""

import numpy as np

from ...._backend import REAL, _to_cpu, exact_rawkernel_options, xp
from ._bin_quadrature import SINCSQ_BIN_PREAMBLE, sincsq_tail

__all__ = ["sincsq_band_mass"]

#: Host block size in lines, bounding the float64 temporaries.
_HOST_BLOCK_LINES = 1 << 20

_BAND_MASS_PREAMBLE = r"""
static __device__ double pyrite_band_line_mass(
    double e_r, double w, double a, double lo, double hi
) {
    const double whole = w * (PYRITE_PI / a);
    if (!(isfinite(whole) && whole > 0.0)) {
        return 0.0;
    }
    const double x_lo = a * (lo - e_r) / PYRITE_PI;
    const double x_hi = a * (hi - e_r) / PYRITE_PI;
    double frac = pyrite_sincsq_tail(x_hi) - pyrite_sincsq_tail(x_lo);
    if (x_lo < 0.0 && x_hi >= 0.0) {
        frac += 1.0;
    }
    return whole * frac;
}
"""

_KERNEL: dict = {}


def _host_band_mass(a_width, E_res, weight, start_eV, stop_eV) -> float:
    total = 0.0
    for i in range(0, a_width.size, _HOST_BLOCK_LINES):
        block = slice(i, i + _HOST_BLOCK_LINES)
        a = np.asarray(_to_cpu(a_width[block]), dtype=np.float64)
        e = np.asarray(_to_cpu(E_res[block]), dtype=np.float64)
        w = np.asarray(_to_cpu(weight[block]), dtype=np.float64)
        whole = w * (np.pi / a)
        ok = np.isfinite(whole) & (whole > 0.0)
        if not ok.any():
            continue
        a, e, whole = a[ok], e[ok], whole[ok]
        x_lo = a * (start_eV - e) / np.pi
        x_hi = a * (stop_eV - e) / np.pi
        frac = sincsq_tail(x_hi) - sincsq_tail(x_lo) + ((x_lo < 0.0) & (x_hi >= 0.0))
        total += float(np.sum(whole * frac))
    return total


def _device_kernel(cp):
    # Elementwise per-line masses then the backend ``sum``: a scalar-output
    # ``ReductionKernel`` runs one thread block, which made the band audit
    # several times the line generation it measures (#1326 qlmc timers).
    kernel = _KERNEL.get(id(cp))
    if kernel is None:
        kernel = _KERNEL[id(cp)] = cp.ElementwiseKernel(
            "T e_r, T w, T a, float64 lo, float64 hi",
            "float64 mass",
            "mass = pyrite_band_line_mass((double)e_r, (double)w, (double)a, lo, hi)",
            "pyrite_band_line_mass_kernel",
            preamble=SINCSQ_BIN_PREAMBLE + _BAND_MASS_PREAMBLE,
            options=exact_rawkernel_options(cp),
        )
    return kernel


def sincsq_band_mass(a_width, E_res, weight, start_eV: float, stop_eV: float):
    """Sum of the lines' ``sinc^2`` energy mass inside ``[start_eV, stop_eV]``.

    ``weight * (pi / a_width) [F(x_hi) - F(x_lo)]`` per line, summed in FP64.
    Lines whose whole mass ``weight * pi / a_width`` is not finite and
    positive add nothing. On a device array the sum stays on the device (a
    0-d FP64 array, the backend's deterministic ``sum``); on the host a ``float``.

    Validation: sinc-bin-integration
    """
    if not 0.0 <= float(start_eV) < float(stop_eV):
        raise ValueError("band must satisfy 0 <= start < stop")
    if getattr(xp, "__name__", "") == "cupy" and isinstance(a_width, xp.ndarray):
        if a_width.size == 0:
            return xp.zeros((), dtype=xp.float64)
        e, w, a = (xp.asarray(x, dtype=REAL) for x in (E_res, weight, a_width))
        masses = _device_kernel(xp)(e, w, a, float(start_eV), float(stop_eV))
        return masses.sum(dtype=xp.float64)
    a, e, w = (np.asarray(_to_cpu(x)).reshape(-1) for x in (a_width, E_res, weight))
    return _host_band_mass(a, e, w, float(start_eV), float(stop_eV))
