"""Shared-mode Fourier bin means on piecewise-uniform axes (#1392).

A ``resonance-local`` axis is a chain of uniform runs (#1390). Every run of
one flush sees the same lines, and their periods differ only through the
small band term (``fourier_period`` is set by ``W_max / sqrt(6 tol)``). So one
exact-cutoff type-1 transform at a common period serves every run: per run
the modes take the run's bin multiplier ``sinc(t h_j / 2)`` and are evaluated
at the run's bin centres by an ES type-2 transform (the adjoint of the
spreader that builds the modes), instead of ``j mod N`` folding and an
inverse FFT on a grid tied to ``h_j``. The per-run route
(``host_fourier_bin_mean_reduce`` / ``cuda_fourier_bin_mean_reduce``) repeats
the line transform for every run.

Each run keeps its own bounds: the shared broad threshold is the largest
run threshold, and the broad bound falls as the threshold grows; the shared
period is the largest run period, and the image bounds fall as ``P - U``
grows. The type-2 evaluation adds its kernel tolerance ``eps``.

Validation: cxr-line-fourier-reduction
"""

import numpy as np

from ._bin_broad import broad_width_for_axis
from ._bin_fourier import FOURIER_MIN_RESIDUAL_TOL, _envelope_image_correction, fourier_period


def shared_run_period(extrema, axes, residual_tol):
    """Largest :func:`fourier_period` over ``axes``: every run's image bound holds.

    Validation: cxr-line-fourier-reduction
    """
    return max(
        fourier_period(extrema[:2], extrema[2:], c0, h, n, residual_tol=residual_tol)[1]
        for c0, h, n in axes
    )


def shared_broad_width(axes, residual_tol, floor_eV):
    """Largest :func:`broad_width_for_axis` over ``axes``: every run's broad bound holds.

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    return max(broad_width_for_axis(h, (n - 1) * h, residual_tol, floor_eV) for _, h, n in axes)


def _width_groups(axes):
    """Run indices grouped by identical bin width, in first-appearance order."""
    groups: dict[float, list[int]] = {}
    for index, (_, h, _) in enumerate(axes):
        groups.setdefault(float(h), []).append(index)
    return groups.items()


def fourier_runs_reduce(
    E,
    a,
    w,
    axes,
    *,
    xp,
    to_host,
    positive_modes,
    evaluate,
    broad_reduce,
    near_far,
    node_sums,
    residual_tol,
    broad_width_eV,
    max_line_modes,
):
    """``[sum_l w_l mean_k(line l) on run j]`` for uniform ``axes = [(c0, h, n), ...]``.

    Backend-neutral body of the host and device routes. ``E, a, w`` are
    validated, nonzero-weight FP64 line arrays on ``xp``. Callables:
    ``positive_modes(E, a, w, c0, P)`` (exact-cutoff coefficients),
    ``evaluate(coeffs, x)`` (``sum_m coeffs_m exp(i m x)``, ES type-2),
    ``broad_reduce(E, a, w, c0, h, n)``, ``near_far(E, a, w, axis)`` and
    ``node_sums`` (envelope-image node sums). Returns FP64 arrays on ``xp``.

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    if not residual_tol >= FOURIER_MIN_RESIDUAL_TOL:
        raise ValueError(f"residual_tol must be >= {FOURIER_MIN_RESIDUAL_TOL:g}")
    outs = [xp.zeros(n, dtype=xp.float64) for _, _, n in axes]
    if E.size == 0:
        return outs
    if broad_width_eV is not None:
        broad = a <= np.pi / shared_broad_width(axes, residual_tol, broad_width_eV)
        if bool(broad.any()):
            sel = xp.flatnonzero(broad)
            for out, (c0, h, n) in zip(outs, axes, strict=True):
                out += xp.asarray(broad_reduce(E[sel], a[sel], w[sel], c0, h, n))
            rest = xp.flatnonzero(~broad)
            E, a, w = E[rest], a[rest], w[rest]
            if E.size == 0:
                return outs
    extrema = to_host(xp.stack((E.min(), E.max(), a.min())))
    P = shared_run_period(extrema, axes, residual_tol)
    dt = 2.0 * np.pi / P
    # Removing lines only shrinks the span, so P keeps its image bounds.
    limit = np.inf if max_line_modes is None else max_line_modes
    over = xp.floor(2.0 * a / dt) > limit
    if bool(over.any()):
        sel = xp.flatnonzero(over)
        for out, axis in zip(outs, axes, strict=True):
            out += near_far(E[sel], a[sel], w[sel], axis)
        rest = xp.flatnonzero(~over)
        E, a, w = E[rest], a[rest], w[rest]
        if E.size == 0:
            return outs
    origin = float(axes[0][0])
    positive = positive_modes(E, a, w, origin, P)
    t = dt * xp.arange(positive.size, dtype=xp.float64)
    g0 = float(xp.sum(w * np.pi / a))
    for h, members in _width_groups(axes):
        coeffs = positive * xp.sinc(t * (h / (2.0 * np.pi)))
        coeffs[0] = 0.0
        x = xp.concatenate(
            [
                ((axes[j][0] - origin) + h * xp.arange(axes[j][2], dtype=xp.float64)) * dt
                for j in members
            ]
        )
        values = (g0 + 2.0 * evaluate(coeffs, x).real) / P
        start = 0
        for j in members:
            c0, _, n = axes[j]
            outs[j] += values[start : start + n]
            outs[j] -= xp.asarray(
                _envelope_image_correction(E, a, w, c0, h, n, P, xp, node_sums=node_sums)
            )
            start += n
    return outs
