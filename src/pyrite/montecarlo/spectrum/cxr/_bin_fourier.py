"""Fourier-domain bin-mean reduction of incoherent ``sinc^2`` lines (#1192).

Host FP64 reference. Not wired into production routes yet.

A line ``l`` with resonance ``E_l``, inverse width ``a_l`` [1/eV] and weight
``w_l`` has density ``w_l sinc^2(a_l (E - E_l) / pi)`` (``np.sinc``
convention), i.e. ``w_l sin^2(a_l u) / (a_l u)^2`` with ``u = E - E_l``. With
``rho_hat(t) = integral rho(E) exp(-i E t) dE`` its transform is the triangle

    ``w_l (pi / a_l) max(0, 1 - |t| / (2 a_l)) exp(-i E_l t)``,

supported on ``|t| < 2 a_l``. On uniform bins of width ``h`` centred at
``c_k = c_0 + k h`` the bin mean multiplies the transform by
``sinc(t h / 2 pi)``. Sampling ``t_j = j dt``, ``dt = 2 pi / P``, ``P = N h``,
is exact for the period-``P`` sum of bin means by Poisson summation:

    ``sum_n m(c_k + n P) = (1/P) sum_j G_j exp(2 pi i j k / N)``,
    ``G_j = sum_l S_l(t_j) sinc(t_j h / 2 pi)``.

Modes with ``|j| >= N`` fold onto ``j mod N``, so lines narrower than a bin
are exact. The only approximation is the subtraction of the images
``n != 0``. Their ``sin^2``-averaged envelope part,
``(w_l / 2 a_l^2) sum_{n != 0} 1/(u + n P)^2``, is removed in closed form
(``pi^2 / (P^2 sin^2(pi u / P)) - 1/u^2``) and interpolated over the band on
Chebyshev nodes. The oscillatory remainder ``-cos(2 a v)/(2 a^2 v^2)`` at
image distances ``|v| >= P - U`` is left; :func:`image_residual_bounds`
states its pointwise and yield bounds, and :func:`fourier_period` chooses
``P`` so the pointwise bound meets ``residual_tol``.

Validation: cxr-line-fourier-reduction
"""

import numpy as np
import scipy.fft as sf

#: Minimum image clearance ``P - U`` as a multiple of the span ``U`` of band
#: plus resonances. It keeps the Chebyshev ellipse parameter >= 13.9.
FOURIER_SPAN_PAD = 3.0
#: Default pointwise image residual, relative to each line's peak density.
FOURIER_RESIDUAL_TOL = 1.0e-6
#: Smallest accepted ``residual_tol``: below it FP64 rounding of the mode sum
#: dominates and the bound no longer describes the error.
FOURIER_MIN_RESIDUAL_TOL = 1.0e-12
#: Chebyshev nodes for the smooth envelope-image correction over the band.
_CHEB_NODES = 24
#: Host block size in (line, mode) elements.
_HOST_BLOCK_MODES = 1 << 22
#: Relative tolerance for accepting an axis as uniform.
_UNIFORM_RTOL = 1.0e-9


def uniform_bin_axis(E_grid_eV: object) -> tuple[float, float, int]:
    """``(c0, h, n)`` of a uniform node axis whose bins are ``c_k +- h/2``.

    Refuses nonuniform axes and an axis whose first bin edge would be clamped
    at 0 eV by the production bin convention (``bin_axis``).

    Validation: cxr-line-fourier-reduction
    """
    E = np.asarray(E_grid_eV, dtype=np.float64)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("Fourier line reduction needs a 1-D axis with at least two nodes")
    steps = np.diff(E)
    h = float(steps.mean())
    if not h > 0.0 or np.max(np.abs(steps - h)) > _UNIFORM_RTOL * h:
        raise ValueError("Fourier line reduction needs a uniform energy axis")
    if E[0] - 0.5 * h <= 0.0:
        raise ValueError("Fourier line reduction refuses an axis whose low bin edge is clamped")
    return float(E[0]), h, int(E.size)


def fourier_period(
    E_res: object,
    a_width: object,
    c0: float,
    h: float,
    n_bins: int,
    *,
    residual_tol: float = FOURIER_RESIDUAL_TOL,
) -> tuple[int, float, float]:
    """``(N, P, U)``: FFT length, period [eV] and band-plus-resonance span [eV].

    ``P - U' >= max(FOURIER_SPAN_PAD U', W_max / sqrt(6 residual_tol), h)``
    with ``U' = U + h/2`` and ``W_max = pi / min(a)``. It puts the
    :func:`image_residual_bounds` pointwise bound at or below
    ``2 residual_tol`` (``residual_tol`` once ``D >> h``).

    Validation: cxr-line-fourier-reduction
    """
    if not residual_tol >= FOURIER_MIN_RESIDUAL_TOL:
        raise ValueError(f"residual_tol must be >= {FOURIER_MIN_RESIDUAL_TOL:g}")
    E = np.asarray(E_res, dtype=np.float64)
    a = np.asarray(a_width, dtype=np.float64)
    lo = min(c0 - 0.5 * h, float(E.min()))
    hi = max(c0 + (n_bins - 0.5) * h, float(E.max()))
    span = hi - lo
    u_eff = span + 0.5 * h
    w_max = np.pi / float(a.min())
    clearance = max(FOURIER_SPAN_PAD * u_eff, w_max / np.sqrt(6.0 * residual_tol), h)
    n_fft = sf.next_fast_len(int(np.ceil((u_eff + clearance) / h)))
    return n_fft, n_fft * h, span


def image_residual_bounds(a_width: object, P: float, U: float, h: float, band: float):
    """Per-line ``(pointwise, yield)`` bounds on the uncorrected image remainder.

    ``pointwise`` is relative to the line's peak density ``w_l`` and bounds
    every bin; ``yield`` is relative to the whole-line mass ``pi w_l / a_l``
    and bounds the integral over a band of length ``band`` [eV]. With
    ``W = pi / a`` and ``D = P - U - h/2`` (needs ``D >= h``):
    ``W^2 (1 + h^2/D^2) / (6 D^2)`` and
    ``W^2 / (4 pi D^2) + pi^2 W h^2 band / (270 D^4)``. The ``h^2`` terms
    cover the bin-centre form of the envelope correction.

    Validation: cxr-line-fourier-reduction
    """
    a = np.asarray(a_width, dtype=np.float64)
    d = P - U - 0.5 * h
    if not d >= h:
        raise ValueError("image bounds need P - U - h/2 >= h")
    width = np.pi / a
    width_sq = width * width
    pointwise = width_sq * (1.0 + (h / d) ** 2) / (6.0 * d * d)
    centre = np.pi**2 * width * h * h * band / (270.0 * d**4)
    return pointwise, width_sq / (4.0 * np.pi * d * d) + centre


def _inv_sin_sq_minus_inv_sq(x, xp=np):
    """``1/sin^2 x - 1/x^2`` for ``|x| < pi``, without cancellation near 0.

    ``xp`` is the array module (NumPy, or CuPy for the device route).
    """
    x = xp.asarray(x, dtype=xp.float64)
    small = xp.abs(x) < 0.2
    y = x * x
    # Laurent series of csc^2 through x^12; the first omitted term
    # (3617 x^14 / 10854718875) is < 6e-17 at |x| = 0.2.
    tail = 2 / 10395 + y * (1382 / 58046625 + y * (4 / 1403325))
    series = 1 / 3 + y * (1 / 15 + y * (2 / 189 + y * (1 / 675 + y * tail)))
    xb = xp.where(small, 1.0, x)
    return xp.where(small, series, 1.0 / xp.sin(xb) ** 2 - 1.0 / (xb * xb))


def _folded_modes(E_res, a_width, w, c0, h, n_fft):
    """``(G_0, F)``: zero mode and folded positive modes ``j >= 1`` (``j mod N``)."""
    P = n_fft * h
    dt = 2.0 * np.pi / P
    j_max = np.floor(2.0 * a_width / dt).astype(np.int64)
    g0 = float(np.sum(w * np.pi / a_width))
    folded_re = np.zeros(n_fft)
    folded_im = np.zeros(n_fft)
    counts = np.maximum(j_max, 0)
    start = 0
    while start < counts.size:
        stop = start + 1
        total = int(counts[start])
        while stop < counts.size and total + counts[stop] <= _HOST_BLOCK_MODES:
            total += int(counts[stop])
            stop += 1
        block = slice(start, stop)
        line = np.repeat(np.arange(stop - start), counts[block])
        offsets = np.cumsum(counts[block]) - counts[block]
        j = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts[block]) + 1
        a = a_width[block][line]
        t = j * dt
        amp = w[block][line] * (np.pi / a) * (1.0 - t / (2.0 * a)) * np.sinc(t * h / (2.0 * np.pi))
        # |phase| <= 2 a U; its FP64 rounding stays far below the residual bound.
        phase = -(E_res[block][line] - c0) * t
        idx = j % n_fft
        folded_re += np.bincount(idx, amp * np.cos(phase), minlength=n_fft)
        folded_im += np.bincount(idx, amp * np.sin(phase), minlength=n_fft)
        start = stop
    return g0, folded_re + 1j * folded_im


def _envelope_image_correction(E_res, a_width, w, c0, h, n_bins, P, xp=np):
    """Band values of ``sum_l (w_l / 2 a_l^2) sum_{n != 0} 1/(c - E_l + n P)^2``.

    Line arrays live on ``xp``; the 24-node fit and the result are host FP64.
    """
    q = w / (2.0 * a_width * a_width)
    lo, hi = c0, c0 + (n_bins - 1) * h
    k = np.arange(_CHEB_NODES)
    xn = np.cos(np.pi * (k + 0.5) / _CHEB_NODES)
    cn = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xn
    vals = np.empty(_CHEB_NODES)
    scale = (np.pi / P) ** 2
    for i, c in enumerate(cn):
        image = _inv_sin_sq_minus_inv_sq(np.pi * (float(c) - E_res) / P, xp)
        vals[i] = scale * float(xp.dot(q, image))
    coef = np.polynomial.chebyshev.chebfit(xn, vals, _CHEB_NODES - 1)
    c = c0 + h * np.arange(n_bins)
    xc = (2.0 * c - (lo + hi)) / (hi - lo)
    return np.polynomial.chebyshev.chebval(xc, coef)


def fourier_bin_mean_reduce(
    E_res: object,
    a_width: object,
    weight: object,
    E_grid_eV: object,
    *,
    residual_tol: float = FOURIER_RESIDUAL_TOL,
    broad_width_eV: float | None = None,
) -> np.ndarray:
    """``sum_l w_l mean_k(line l)`` on a uniform axis, FP64, shape ``(n_bins,)``.

    The Fourier-domain counterpart of summing ``w[:, None] * _host_bin_mean(...,
    exact_widths=None)`` over lines: no far-field envelope, every bin exact up
    to the bounded image remainder (:func:`image_residual_bounds`). Bins are
    ``c_k +- h/2`` (the ``bin_axis`` convention on a uniform axis).

    ``broad_width_eV`` opts into a separate 64-node interpolation for lines
    whose first-zero width is at least this threshold [eV]. This experimental
    split has its own unverified bound, plus FP64 error; it avoids letting
    those lines enlarge the Fourier period. ``None`` preserves the unsplit
    reference. Both routes retain all lines and require uniform bins.

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    c0, h, n_bins = uniform_bin_axis(E_grid_eV)
    if not residual_tol >= FOURIER_MIN_RESIDUAL_TOL:
        raise ValueError(f"residual_tol must be >= {FOURIER_MIN_RESIDUAL_TOL:g}")
    if broad_width_eV is not None and not (np.isfinite(broad_width_eV) and broad_width_eV > 0):
        raise ValueError("broad_width_eV must be finite and positive")
    E = np.atleast_1d(np.asarray(E_res, dtype=np.float64))
    a = np.atleast_1d(np.asarray(a_width, dtype=np.float64))
    w = np.atleast_1d(np.asarray(weight, dtype=np.float64))
    if not (E.shape == a.shape == w.shape):
        raise ValueError("E_res, a_width and weight must have one shape")
    keep = w != 0.0
    E, a, w = E[keep], a[keep], w[keep]
    if E.size == 0:
        return np.zeros(n_bins)
    finite = np.all(np.isfinite(E)) and np.all(np.isfinite(w)) and np.all(np.isfinite(a))
    if not (finite and np.all(a > 0.0)):
        raise ValueError("lines need finite E_res and weight and positive finite a_width")
    broad = np.zeros(n_bins)
    if broad_width_eV is not None:
        from ._bin_broad import broad_bin_mean_reduce

        select = a <= np.pi / broad_width_eV
        if np.any(select):
            broad = broad_bin_mean_reduce(
                E[select], a[select], w[select], c0, h, n_bins, tolerance=residual_tol
            )
            E, a, w = E[~select], a[~select], w[~select]
        if E.size == 0:
            return broad
    n_fft, P, _ = fourier_period(E, a, c0, h, n_bins, residual_tol=residual_tol)
    g0, folded = _folded_modes(E, a, w, c0, h, n_fft)
    periodic = (g0 + 2.0 * (n_fft * sf.ifft(folded)).real[:n_bins]) / P
    return broad + periodic - _envelope_image_correction(E, a, w, c0, h, n_bins, P)
