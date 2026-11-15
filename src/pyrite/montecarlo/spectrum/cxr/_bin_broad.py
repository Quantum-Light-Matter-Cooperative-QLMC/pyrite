"""Experimental scalar broad-line interpolation for the host reference.

Validation: cxr-line-broad-interpolation
"""

import math

import numpy as np

_BROAD_NODES = 64
_BROAD_BLOCK = 1024


def broad_bin_mean_error_bound(width_min, h, centre_span):
    """Absolute truncation bound per sum of absolute line weights, excluding FP64.

    Uniform bins of width ``h`` [eV], centre span [eV], first-zero widths at
    least ``width_min`` [eV]. Includes interpolation and amplified four-point
    Gauss error. The claim is unverified; see the derivation record.

    Validation: cxr-line-broad-interpolation
    """
    phase = np.pi * centre_span / (2.0 * width_min)
    log_bound = math.log(2.0) + _BROAD_NODES * math.log(phase) - math.lgamma(_BROAD_NODES + 1)
    interpolation = math.exp(log_bound) if log_bound < 709.0 else math.inf
    log_gauss = math.log(2.0) + 8 * math.log(np.pi * h / width_min) - math.lgamma(9)
    gauss = math.exp(log_gauss) if log_gauss < 709.0 else math.inf
    return interpolation + (2 * _BROAD_NODES - 1) * gauss


def broad_width_for_axis(h, centre_span, tolerance, floor_eV=1000.0):
    """Smallest broad-route threshold ``>= floor_eV`` [eV] meeting ``tolerance``.

    :func:`broad_bin_mean_error_bound` falls as the threshold ``W0`` grows:
    the Gauss term as ``(h / W0)**8``, the interpolation term as
    ``(span / W0)**64``. A fixed ``W0`` refuses coarse bins (500 eV bins at
    1 keV give ~0.2) and long axes (a 31 keV span at 1 keV gives > 1), so the
    threshold scales up in steps of 2**(1/8) until the bound holds. Lines
    narrower than it stay on the exact Fourier route. Returns ``floor_eV``
    whenever that already meets the bound, so such axes are unchanged.

    Validation: cxr-line-broad-interpolation
    """
    width = float(floor_eV)
    for _ in range(512):
        if broad_bin_mean_error_bound(width, h, centre_span) <= tolerance:
            return width
        width *= 2.0**0.125
    raise ValueError("no broad-route threshold meets the tolerance on this axis")


def broad_bin_mean_reduce(
    E_res, a_width, weight, c0, h, n_bins, *, tolerance, xp=np, block=_BROAD_BLOCK
):
    """Sum broad scalar lines on uniform bins, using bounded temporary storage.

    Inputs have been validated by the host reference. Signed weights are
    allowed; the absolute bound scales with ``sum(abs(weight))``. Refuses
    widths insufficient for the fixed 64-node approximation. No line is
    discarded. Line weights are per-line scalars, as in the production
    reduction contract. Line arrays live on ``xp`` (NumPy or CuPy) in blocks
    of ``block`` lines; the 64-node fit and the result are host FP64.

    Validation: cxr-line-broad-interpolation
    """
    span = (n_bins - 1) * h
    width = np.pi / a_width
    bound = broad_bin_mean_error_bound(float(width.min()), h, span)
    if not bound <= tolerance:
        raise ValueError(
            "broad-line interpolation bound exceeds residual_tol; increase broad_width_eV"
        )
    xn = np.cos(np.pi * (np.arange(_BROAD_NODES) + 0.5) / _BROAD_NODES)
    centres = xp.asarray(c0 + 0.5 * span * (1.0 + xn))
    gx, gw = np.polynomial.legendre.leggauss(4)
    values = xp.zeros(_BROAD_NODES)
    for start in range(0, E_res.size, block):
        sl = slice(start, start + block)
        for x, q in zip(gx, gw, strict=True):
            phase = (centres[None, :] + 0.5 * h * float(x) - E_res[sl, None]) / width[sl, None]
            values += 0.5 * float(q) * (weight[sl] @ xp.sinc(phase) ** 2)
    host_values = np.asarray(getattr(values, "get", lambda: values)())
    coefficients = np.polynomial.chebyshev.chebfit(xn, host_values, _BROAD_NODES - 1)
    targets = 2.0 * np.arange(n_bins) / (n_bins - 1) - 1.0
    return np.polynomial.chebyshev.chebval(targets, coefficients)
