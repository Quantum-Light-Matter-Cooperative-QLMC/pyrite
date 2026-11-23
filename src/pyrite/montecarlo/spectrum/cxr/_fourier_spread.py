"""Portable 1D type-1 NUFFT for the device Fourier line reduction (#1192).

ROCm has no cuFINUFFT, so this module supplies the same two-row type-1
transform with CuPy only: an exponential-of-semicircle (ES) spreading
kernel (Barnett, Magland and af Klinteberg, SIAM J. Sci. Comput. 41 (2019)
C479), a ``cupy.fft`` (cuFFT or hipFFT) of the 2x upsampled grid, and
division by the kernel's Fourier transform. It runs on CUDA too, which is
how it is cross-checked against cuFINUFFT.

Two spreaders add the same ES terms (:data:`SPREADERS`): one thread per
source with FP64 atomics, and (default, #1297) a sorted warp gather over
batches of rectangles. On an RTX 5080 the gather spreads about 5x faster
and whole device flushes run 1.3-1.5x faster.

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
#: Modes per block of the kernel-transform evaluation: a 32 MiB
#: (block, 64) FP64 cosine workspace whatever the mode count (#1320).
_DECONV_BLOCK = 1 << 16


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
    phi = cp.asarray(q * np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0)))
    z = cp.asarray(z)
    scale = np.pi * width / n_fine
    n_pos = modes // 2 + 1  # 0 .. max |centred m|
    kernel_ft = cp.empty(n_pos, dtype=cp.float64)
    # Mode blocks bound the (block, nodes) cosine workspace (#1320).
    for lo in range(0, n_pos, _DECONV_BLOCK):
        m = cp.arange(lo, min(lo + _DECONV_BLOCK, n_pos), dtype=cp.float64)
        kernel_ft[lo : lo + m.size] = cp.cos(cp.outer(m * scale, z)) @ phi
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


#: Spreaders: ``"gather"`` (default since ``fourier-auto-v7``, #1297) sorts the
#: sources of a batch of rectangles by first fine point and sums each point's
#: window by warps in registers; ``"atomic"`` is the one-thread-per-source
#: FP64 atomic kernel above. Same (source, point, phi) terms; only the FP64
#: summation order differs.
SPREADERS = ("gather", "atomic")
#: Source copies per gather batch: bounds its ~80 B/source device workspace.
_BATCH_SOURCES = 1 << 23
#: Fine-grid points per gather batch: bounds its shared 32 B/point grid (#1320).
_BATCH_FINE = 1 << 24
#: Sorted sources per warp work item of the gather.
_ITEM_SOURCES = 256

_GATHER_SOURCE = r"""
// Rectangle r owns fine points off[r] .. off[r+1] (n_fine of them, two complex
// rows at complex offset 2 off[r]) and concatenated source copies
// first[r] .. first[r+1], the lines lo[r] .. of the cutoff partition.
__device__ __forceinline__ int find_rect(const long long* edges, const int n, const long long v) {
    int a = 0, b = n;
    while (b - a > 1) { const int m = (a + b) >> 1; if (edges[m] <= v) a = m; else b = m; }
    return a;
}

// Same fine coordinate, first point and wrap as es_spread.
__device__ __forceinline__ void fine_point(
    const double x, const long long n_fine, const int width, double* u, long long* l0) {
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    double xk = fmod(x, two_pi);
    if (xk < 0.0) xk += two_pi;
    *u = xk / h;
    *l0 = (long long)ceil(*u - 0.5 * double(width));
}

__device__ __forceinline__ long long wrap(const long long l, const long long n) {
    long long w = l % n;
    return w < 0 ? w + n : w;
}

extern "C" __global__ void es_batch_keys(
    const double* phase, const long long* first, const long long* lo, const long long* off,
    const int nrect, const long long n, const int width, long long* key) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int r = find_rect(first, nrect, t);
    const long long n_fine = off[r + 1] - off[r];
    double u; long long l0;
    fine_point(phase[lo[r] + t - first[r]], n_fine, width, &u, &l0);
    key[t] = off[r] + wrap(l0, n_fine);
}

extern "C" __global__ void es_batch_sources(
    const double* phase, const double* c0, const double* c1, const long long* order,
    const long long* first, const long long* lo, const long long* off, const double* shift,
    const int nrect, const long long n, const int width,
    double* u_out, long long* l0_out, double* src) {
    // Sorted copy s: fine coordinate and the modulated strengths
    // c * exp(-i shift x) of production, in key order.
    const long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const long long t = order[s];
    const int r = find_rect(first, nrect, t);
    const long long i = lo[r] + t - first[r];
    const double x = phase[i];
    double u; long long l0;
    fine_point(x, off[r + 1] - off[r], width, &u, &l0);
    double sn, cs;
    sincos(-shift[r] * x, &sn, &cs);
    u_out[s] = u; l0_out[s] = l0;
    src[4*s] = c0[i] * cs; src[4*s+1] = c0[i] * sn;
    src[4*s+2] = c1[i] * cs; src[4*s+3] = c1[i] * sn;
}

#define GATHER_THREADS 128
extern "C" __global__ void es_gather(
    const double* u, const long long* l0, const double* src, const long long* item,
    const long long n_items, const long long* off, const int nrect,
    const int width, const double beta, double* grid) {
    // One 32-thread group per (fine point g, sorted copies s_lo .. s_hi) item;
    // every copy of an item has its first point in g's window g-w+1 .. g, so
    // it adds phi(l0 + p - u) at p = g - l0 (mod n_fine), as es_spread does.
    // Groups reduce through shared memory (no warp shuffles: portable to
    // wave64 and to HIP's 64-bit-mask *_sync builtins).
    __shared__ double red[4][GATHER_THREADS];
    const int tid = threadIdx.x, lane = tid & 31;
    const long long it = ((long long)blockIdx.x * GATHER_THREADS + tid) >> 5;
    double ar = 0.0, ai = 0.0, br = 0.0, bi = 0.0;
    long long base = 0, n_fine = 1, l = 0;
    if (it < n_items) {
        const long long g = item[3*it], s_lo = item[3*it+1], s_hi = item[3*it+2];
        const int r = find_rect(off, nrect, g);
        base = off[r]; n_fine = off[r + 1] - base; l = g - base;
        const double half = 0.5 * double(width);
        for (long long s = s_lo + lane; s < s_hi; s += 32) {
            const long long p = wrap(l - l0[s], n_fine);
            const double z = (double(l0[s] + p) - u[s]) / half;
            const double arg = 1.0 - z * z;
            if (arg <= 0.0) continue;
            const double phi = exp(beta * (sqrt(arg) - 1.0));
            ar += phi * src[4*s]; ai += phi * src[4*s+1];
            br += phi * src[4*s+2]; bi += phi * src[4*s+3];
        }
    }
    red[0][tid] = ar; red[1][tid] = ai; red[2][tid] = br; red[3][tid] = bi;
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) {
        if (lane < o)
            for (int c = 0; c < 4; ++c) red[c][tid] += red[c][tid + o];
        __syncthreads();
    }
    if (lane == 0 && it < n_items) {
        double* g0 = grid + 2 * (2 * base + l);
        double* g1 = grid + 2 * (2 * base + n_fine + l);
        atomicAdd(g0, red[0][tid]); atomicAdd(g0 + 1, red[1][tid]);
        atomicAdd(g1, red[2][tid]); atomicAdd(g1 + 1, red[3][tid]);
    }
}
"""


@lru_cache(maxsize=3)
def _gather_kernel(name):
    cp = import_module("cupy")
    return cp.RawKernel(_GATHER_SOURCE, name)


def _gather_items(cp, sorted_key, off, width):
    """``(g, s_lo, s_hi)`` warp items covering every fine point's source window.

    Point ``l`` of a rectangle takes the copies keyed ``l-w+1 .. l``: one
    contiguous key range, plus the wrapped range ``n_fine+l-w+1 .. n_fine-1``
    when ``l < w-1`` (``n_fine >= 2w``, so the two never overlap). Each range
    is split into chunks of at most ``_ITEM_SOURCES`` copies.
    """
    total = int(off[-1])
    d_off = cp.asarray(off)
    g = cp.arange(total, dtype=cp.int64)
    r = cp.searchsorted(d_off, g, side="right") - 1
    base, end = d_off[r], d_off[r + 1]
    pt = g - base
    keys = cp.concatenate((base + cp.maximum(pt - width + 1, 0), g + 1))
    wrapped = pt < width - 1
    keys = cp.concatenate((keys, cp.where(wrapped, end + pt - width + 1, end), end))
    bounds = cp.searchsorted(sorted_key, keys).reshape(4, total)
    seg_lo = cp.concatenate((bounds[0], bounds[2]))
    seg_hi = cp.concatenate((bounds[1], bounds[3]))
    seg_g = cp.concatenate((g, g))
    chunks = (seg_hi - seg_lo + _ITEM_SOURCES - 1) // _ITEM_SOURCES
    ends = cp.cumsum(chunks)
    n_items = int(ends[-1]) if ends.size else 0
    t = cp.arange(n_items, dtype=cp.int64)
    seg = cp.searchsorted(ends, t, side="right")
    s_lo = seg_lo[seg] + (t - (ends - chunks)[seg]) * _ITEM_SOURCES
    s_hi = cp.minimum(s_lo + _ITEM_SOURCES, seg_hi[seg])
    return cp.ascontiguousarray(cp.stack((seg_g[seg], s_lo, s_hi), axis=1))


def _batches(lens, fine=None):
    """Consecutive rectangle groups (or single rectangles) within the batch caps.

    A group holds at most ``_BATCH_SOURCES`` source copies and, when per-rectangle
    fine-grid sizes ``fine`` are given, at most ``_BATCH_FINE`` fine points.
    """
    fine = [0] * len(lens) if fine is None else fine
    start, total, points = 0, 0, 0
    for k, (n, f) in enumerate(zip(lens, fine, strict=True)):
        if k > start and (total + n > _BATCH_SOURCES or points + f > _BATCH_FINE):
            yield start, k
            start, total, points = k, 0, 0
        total += n
        points += f
    if start < len(lens):
        yield start, len(lens)


def es_gather_moments(phase, rows, rects, *, eps=1e-12, plans=None):
    """Two-row type-1 transforms of many rectangles; yields ``(j0, j1, moments)``.

    Rectangle ``(j0, j1, lo, hi)`` transforms the lines ``lo:hi`` of
    ``phase`` with strength rows ``rows[0][lo:hi]``, ``rows[1][lo:hi]`` into
    modes ``j0..j1-1``, as ``es_mode_transform(..., spread="atomic")`` would:
    the same ES terms, summed by sorted warp gathers instead of atomics. The
    rectangles of a batch share one key sort and one gather launch; each
    keeps its own fine grid and FFT. ``moments`` is ``(2, j1 - j0)``
    complex128 on the current stream; ``plans`` caches deconvolution factors.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    width = es_width(eps)
    beta = np.float64(_BETA_PER_WIDTH * width)
    plans = {} if plans is None else plans
    phase = cp.ascontiguousarray(phase, dtype=cp.float64)
    c0, c1 = (cp.ascontiguousarray(row, dtype=cp.float64) for row in rows)
    rects = np.asarray(rects, dtype=np.int64).reshape(-1, 4)
    lens = rects[:, 3] - rects[:, 2]
    sizes = [_fine_size(int(m), width) for m in (rects[:, 1] - rects[:, 0]).tolist()]
    for a, b in _batches(lens.tolist(), sizes):
        group = rects[a:b]
        modes = group[:, 1] - group[:, 0]
        n_fine = np.array(sizes[a:b], dtype=np.int64)
        off = np.concatenate(([0], np.cumsum(n_fine)))
        first = np.concatenate(([0], np.cumsum(lens[a:b])))
        n, nrect = int(first[-1]), np.int32(b - a)
        d_first, d_lo, d_off = (cp.asarray(v) for v in (first, group[:, 2], off))
        grid = cp.zeros(2 * int(off[-1]), dtype=cp.complex128)
        if n:
            blocks = ((n + 255) // 256,)
            key = cp.empty(n, dtype=cp.int64)
            _gather_kernel("es_batch_keys")(
                blocks,
                (256,),
                (phase, d_first, d_lo, d_off, nrect, np.int64(n), np.int32(width), key),
            )
            order = cp.argsort(key)
            key = key[order]
            u = cp.empty(n, dtype=cp.float64)
            l0 = cp.empty(n, dtype=cp.int64)
            src = cp.empty(4 * n, dtype=cp.float64)
            shift = cp.asarray((group[:, 0] + modes // 2).astype(np.float64))
            _gather_kernel("es_batch_sources")(
                blocks,
                (256,),
                (
                    phase,
                    c0,
                    c1,
                    order,
                    d_first,
                    d_lo,
                    d_off,
                    shift,
                    nrect,
                    np.int64(n),
                    np.int32(width),
                    u,
                    l0,
                    src,
                ),
            )
            del order
            item = _gather_items(cp, key, off, width)
            del key
            n_items = item.shape[0]
            _gather_kernel("es_gather")(
                ((n_items + 3) // 4,),
                (128,),
                (u, l0, src, item, np.int64(n_items), d_off, nrect, np.int32(width), beta, grid),
            )
            del u, l0, src, item
        for k, (j0, j1) in enumerate(group[:, :2].tolist()):
            m, nf = j1 - j0, int(n_fine[k])
            deconv = plans.get(m)
            if deconv is None:
                deconv = plans[m] = _deconvolution(cp, m, nf, width)
            fine = grid[2 * int(off[k]) : 2 * int(off[k + 1])].reshape(2, nf)
            spectrum = cp.fft.fft(fine, axis=1)
            centred = cp.arange(-(m // 2), m - m // 2) % nf
            yield j0, j1, spectrum[:, centred] * deconv


def es_mode_transform(phase, strengths, j0, j1, *, eps=1e-12, plans=None, spread="atomic"):
    """Two-row type-1 transform of device arrays into modes ``j0..j1-1``.

    Drop-in for ``cufinufft_mode_transform``: ``strengths`` has shape
    ``(2, n)``; the result is ``(2, j1 - j0)`` complex128 on the current
    stream. ``plans`` caches the deconvolution factors by mode count.
    Accuracy follows the ES kernel width from ``eps`` (FINUFFT's rule);
    it is not a final spectrum error guarantee. ``spread`` picks one of
    :data:`SPREADERS`; ``"gather"`` is :func:`es_gather_moments` on one
    rectangle.

    Validation: cxr-line-fourier-reduction
    """
    if spread not in SPREADERS:
        raise ValueError(f"spread must be one of {SPREADERS}")
    if spread == "gather":
        rect = [(j0, j1, 0, phase.size)]
        return next(es_gather_moments(phase, strengths, rect, eps=eps, plans=plans))[2]
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


@lru_cache(maxsize=1)
def _interp_kernel():
    """One thread per point: ``out[k] = sum_l phi(l step - x_k) grid[l]`` (fixed order)."""
    cp = import_module("cupy")
    return cp.RawKernel(
        r"""
#include <cupy/complex.cuh>
extern "C" __global__ void es_interp(
    const double* x, const complex<double>* grid, const long long n,
    const long long n_fine, const int width, const double beta, complex<double>* out) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk / h;
    const double half = 0.5 * double(width);
    const long long l0 = (long long)ceil(u - half);
    complex<double> acc(0.0, 0.0);
    for (int p = 0; p < width; ++p) {
        const long long l = l0 + p;
        const double z = (double(l) - u) / half;
        const double arg = 1.0 - z * z;
        if (arg <= 0.0) continue;
        long long idx = l % n_fine;
        if (idx < 0) idx += n_fine;
        acc += exp(beta * (sqrt(arg) - 1.0)) * grid[idx];
    }
    out[k] = acc;
}
""",
        "es_interp",
    )


def es_mode_evaluate(coeffs, j0, x, *, eps=1e-12, plans=None):
    """Device type-2 transform ``sum_j coeffs[j - j0] exp(i j x_k)`` (#1392).

    The adjoint of the type-1 spreaders with the same kernel width, shape
    parameter, upsampling and (real) deconvolution factors; mirrors
    ``_fourier_host.host_es_mode_evaluate``. One thread per point sums its
    window in a fixed order, so the result does not depend on launch order.
    ``plans`` caches the deconvolution factors by mode count.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    coeffs = cp.asarray(coeffs, dtype=cp.complex128)
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    modes = coeffs.size
    width = es_width(eps)
    n_fine = _fine_size(modes, width)
    shift = j0 + modes // 2
    deconv = None if plans is None else plans.get(modes)
    if deconv is None:
        deconv = _deconvolution(cp, modes, n_fine, width)
        if plans is not None:
            plans[modes] = deconv
    grid = cp.zeros(n_fine, dtype=cp.complex128)
    grid[cp.arange(-(modes // 2), modes - modes // 2) % n_fine] = coeffs * deconv
    grid = n_fine * cp.fft.ifft(grid)
    out = cp.empty(x.size, dtype=cp.complex128)
    if x.size:
        _interp_kernel()(
            ((x.size + 255) // 256,),
            (256,),
            (
                x,
                grid,
                np.int64(x.size),
                np.int64(n_fine),
                np.int32(width),
                np.float64(_BETA_PER_WIDTH * width),
                out,
            ),
        )
    return out * cp.exp(1j * shift * x)
