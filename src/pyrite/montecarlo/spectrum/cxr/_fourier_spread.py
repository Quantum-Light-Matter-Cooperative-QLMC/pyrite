"""Portable 1D type-1 NUFFT for the device Fourier line reduction (#1192).

ROCm has no cuFINUFFT, so this module supplies the same two-row type-1
transform with CuPy only: an exponential-of-semicircle (ES) spreading
kernel (Barnett, Magland and af Klinteberg, SIAM J. Sci. Comput. 41 (2019)
C479), a ``cupy.fft`` (cuFFT or hipFFT) of the 2x upsampled grid, and
division by the kernel's Fourier transform. It runs on CUDA too, which is
how it is cross-checked against cuFINUFFT.

For sources ``x_k`` and strengths ``c_k`` it returns
``f_j = sum_k c_k exp(-i j x_k)`` for ``j0 <= j < j1``, the convention of
``_fourier_cuda.cufinufft_mode_transform``.

Validation: cxr-line-fourier-reduction
"""

from functools import lru_cache
from importlib import import_module
from math import ceil, log10

import numpy as np

#: Upsampling factor of the fine grid.
_SIGMA = 2
#: ES shape parameter per kernel width, FINUFFT's choice for sigma = 2.
_BETA_PER_WIDTH = 2.30
#: Gauss-Legendre nodes for the kernel transform on [-1, 1]; even, so no node
#: sits at 0 and the rule folds onto its positive half.
_QUADRATURE_NODES = 128


def es_width(eps):
    """Kernel width in fine-grid points for requested relative tolerance ``eps``."""
    if not 1e-15 <= eps < 1.0:
        raise ValueError("eps must lie in [1e-15, 1)")
    return min(16, max(2, ceil(-log10(eps) - 1e-9) + 1))


def _fine_size(modes, width):
    return max(_SIGMA * modes, 2 * width)


@lru_cache(maxsize=1)
def _half_rule():
    """Positive Gauss-Legendre nodes and doubled weights (the rule is symmetric)."""
    z, q = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    half = z > 0.0
    return z[half], 2.0 * q[half]


def _deconvolution(cp, modes, n_fine, width):
    """``2 / (width * I(m))`` for centred modes ``m``, ``I`` the ES transform.

    ``I`` is even in ``m`` and the Gauss-Legendre rule is symmetric, so only
    ``m >= 0`` and the positive nodes are evaluated (4x fewer cosines, the
    dominant cold-start cost); the result is mirrored onto the centred modes.
    """
    beta = _BETA_PER_WIDTH * width
    z, q = _half_rule()
    phi = q * np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
    scale = np.pi * width / n_fine
    m = cp.arange(modes // 2 + 1, dtype=cp.float64)  # 0 .. max |centred m|
    kernel_ft = cp.cos(cp.outer(m * scale, cp.asarray(z))) @ cp.asarray(phi)
    centred = cp.abs(cp.arange(-(modes // 2), modes - modes // 2))
    return 2.0 / (width * kernel_ft[centred])


@lru_cache(maxsize=1)
def _spread_kernel():
    """One thread per source; FP64 atomic adds into a (2, n_fine) complex grid."""
    cp = import_module("cupy")
    return cp.RawKernel(
        r"""
extern "C" __global__ void es_spread(
    const double* x, const double* src, const long long n,
    const long long n_fine, const int width, const double beta, double* grid) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk / h;
    const double half = 0.5 * double(width);
    const long long l0 = (long long)ceil(u - half);
    const double a_re = src[4*k], a_im = src[4*k+1];
    const double b_re = src[4*k+2], b_im = src[4*k+3];
    for (int p = 0; p < width; ++p) {
        const long long l = l0 + p;
        const double z = (double(l) - u) / half;
        const double arg = 1.0 - z * z;
        if (arg <= 0.0) continue;
        const double phi = exp(beta * (sqrt(arg) - 1.0));
        long long idx = l % n_fine;
        if (idx < 0) idx += n_fine;
        atomicAdd(grid + 2*idx, phi * a_re);
        atomicAdd(grid + 2*idx + 1, phi * a_im);
        atomicAdd(grid + 2*(n_fine + idx), phi * b_re);
        atomicAdd(grid + 2*(n_fine + idx) + 1, phi * b_im);
    }
}
""",
        "es_spread",
    )


def es_mode_transform(phase, strengths, j0, j1, *, eps=1e-12, plans=None):
    """Two-row type-1 transform of device arrays into modes ``j0..j1-1``.

    Drop-in for ``cufinufft_mode_transform``: ``strengths`` has shape
    ``(2, n)``; the result is ``(2, j1 - j0)`` complex128 on the current
    stream. ``plans`` caches the deconvolution factors by mode count.
    Accuracy follows the ES kernel width from ``eps`` (FINUFFT's rule);
    it is not a final spectrum error guarantee.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    modes = j1 - j0
    width = es_width(eps)
    n_fine = _fine_size(modes, width)
    shift = j0 + modes // 2
    phase = cp.ascontiguousarray(phase, dtype=cp.float64)
    sources = strengths * cp.exp(-1j * shift * phase)
    packed = cp.ascontiguousarray(sources.T, dtype=cp.complex128)
    deconv = None if plans is None else plans.get(modes)
    if deconv is None:
        deconv = _deconvolution(cp, modes, n_fine, width)
        if plans is not None:
            plans[modes] = deconv
    grid = cp.zeros((2, n_fine), dtype=cp.complex128)
    n = phase.size
    if n:
        _spread_kernel()(
            ((n + 255) // 256,),
            (256,),
            (
                phase,
                packed,
                np.int64(n),
                np.int64(n_fine),
                np.int32(width),
                np.float64(_BETA_PER_WIDTH * width),
                grid,
            ),
        )
    spectrum = cp.fft.fft(grid, axis=1)
    centred = cp.arange(-(modes // 2), modes - modes // 2) % n_fine
    return spectrum[:, centred] * deconv
