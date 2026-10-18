"""CUDA RawKernel: fused per-history transition-radiation field sums (#103).

The CuPy array route of
:func:`~pyrite.montecarlo.spectrum.transition_radiation_crossings.crossing_transition_radiation`
materializes the ``(B, K)`` particular fields, phases and contracted
amplitudes of every energy/direction pair ``B`` and crossing ``K`` in chunks.
This kernel evaluates the same crossing amplitude in registers instead. One
thread walks whole histories: it sums the crossing amplitudes of a history,
adds ``|S_e|^2`` to the power and ``S_e`` to the bunch field, and a block
reduces its threads in a fixed tree order. Blocks are ``(pair, history range)``;
their partials are summed on the device in a fixed order. No atomics, so a
launch geometry gives the same bits on every run. The crossing-independent
stack solve (:func:`~pyrite.montecarlo.spectrum.transition_radiation.interface_response`)
stays one batched CuPy solve per call.

The crossing amplitude is the arithmetic of
:func:`~pyrite.montecarlo.spectrum.transition_radiation.interface_transition_radiation_amplitude`
in complex128. ``_SOURCE`` also compiles as host C++ (``PYRITE_TR_HOST``) with
a sequential entry point, so the device arithmetic is checked against NumPy on
machines without a GPU.

Validation: tr-crossing-superposition, tr-coherent-bunch
"""

from __future__ import annotations

import numpy as np

from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG

__all__ = ["DEFAULT_THREADS", "SOURCE", "history_sums"]

#: Threads per block; a power of two for the tree reduction.
DEFAULT_THREADS = 128
#: Blocks the history ranges are spread over, across all pairs.
_TARGET_BLOCKS = 2048
_SUMS = 7  # power, then (re, im) of the three field components
_COUPLING = 4.0 * np.pi * np.sqrt(ALPHA_FS)

SOURCE = r"""
#ifdef PYRITE_TR_HOST
#include <cmath>
#include <complex>
typedef std::complex<double> cdouble;
#define TR_DEVICE static inline
#else
#include <cupy/complex.cuh>
typedef complex<double> cdouble;
#define TR_DEVICE __device__ __forceinline__
#endif

struct TrArgs {
    int n_pairs, n_interfaces;
    const double *omega, *k_perp;            /* (B,), (B, 2) */
    const cdouble *eps, *response, *lam_top; /* (n_media, B), (B, n_if, 4, 2), (B,) */
    const double *velocity, *time, *position, *charge; /* (K, 3), (K,), (K, 2), (K,) */
    const int *face;                                    /* (K,) */
    double coupling, hbarc;
};

/* Vacuum amplitude of crossing c at pair b: interface_transition_radiation_amplitude. */
TR_DEVICE void tr_amplitude(const TrArgs &a, int b, long long c, cdouble out[3]) {
    const cdouble I(0.0, 1.0);
    const double w = a.omega[b], kx = a.k_perp[2 * b], ky = a.k_perp[2 * b + 1];
    const double vx = a.velocity[3 * c], vy = a.velocity[3 * c + 1], vz = a.velocity[3 * c + 2];
    const double q = (w - (kx * vx + ky * vy)) / vz;
    const double k2 = kx * kx + ky * ky + q * q;
    const int face = a.face[c];
    cdouble src[4] = {0.0, 0.0, 0.0, 0.0};
    for (int side = 0; side < 2; ++side) {
        /* -(P_{i+1} - P_i): the medium above the face, then the one below. */
        const double sign = side == 0 ? -1.0 : 1.0;
        const cdouble e = a.eps[(long long)(face + 1 - side) * a.n_pairs + b];
        const cdouble phi = a.coupling / (fabs(vz) * e * (k2 - e * (w * w)));
        const cdouble ie = I * phi, ih = I * (e * phi);
        src[0] += sign * (ie * (e * w * vx - kx));
        src[1] += sign * (ie * (e * w * vy - ky));
        src[2] += sign * (ih * (ky * vz - q * vy));
        src[3] += sign * (ih * (q * vx - kx * vz));
    }
    const double arg =
        (w * a.time[c] - (kx * a.position[2 * c] + ky * a.position[2 * c + 1])) / a.hbarc;
    const cdouble phase(cos(arg), sin(arg));
    const cdouble *g = a.response + ((long long)b * a.n_interfaces + face) * 8;
    cdouble et0 = 0.0, et1 = 0.0;
    for (int r = 0; r < 4; ++r) {
        const cdouble s = src[r] * phase;
        et0 += g[2 * r] * s;
        et1 += g[2 * r + 1] * s;
    }
    const cdouble ez = -(kx * et0 + ky * et1) / a.lam_top[b];
    const double qc = a.charge[c];
    out[0] = qc * et0;
    out[1] = qc * et1;
    out[2] = qc * ez;
}

/* One history [lo, hi): add |S_e|^2 to sums[0] and S_e to sums[1..6]. */
TR_DEVICE void tr_history(const TrArgs &a, int b, long long lo, long long hi, double sums[7]) {
    cdouble s[3] = {0.0, 0.0, 0.0};
    cdouble amp[3];
    for (long long c = lo; c < hi; ++c) {
        tr_amplitude(a, b, c, amp);
        s[0] += amp[0];
        s[1] += amp[1];
        s[2] += amp[2];
    }
    for (int k = 0; k < 3; ++k) {
        sums[0] += s[k].real() * s[k].real() + s[k].imag() * s[k].imag();
        sums[1 + 2 * k] += s[k].real();
        sums[2 + 2 * k] += s[k].imag();
    }
}

#ifdef PYRITE_TR_HOST
extern "C" void pyrite_tr_host_sums(
    int n_pairs, int n_interfaces, int n_hist, const double *omega, const double *k_perp,
    const double *eps, const double *response, const double *lam_top, const double *velocity,
    const double *time, const double *position, const double *charge, const int *face,
    const long long *starts, double coupling, double hbarc, double *out /* (B, 7) */) {
    TrArgs a = {n_pairs, n_interfaces, omega, k_perp, (const cdouble *)eps,
                (const cdouble *)response, (const cdouble *)lam_top, velocity, time, position,
                charge, face, coupling, hbarc};
    for (int b = 0; b < n_pairs; ++b)
        for (int h = 0; h < n_hist; ++h) tr_history(a, b, starts[h], starts[h + 1], out + 7 * b);
}
#else
extern "C" __global__ void pyrite_tr_history_sums(
    int n_pairs, int n_interfaces, int n_hist, int hist_blocks, const double *omega,
    const double *k_perp, const cdouble *eps, const cdouble *response, const cdouble *lam_top,
    const double *velocity, const double *time, const double *position, const double *charge,
    const int *face, const long long *starts, double coupling, double hbarc,
    double *partial /* (B, hist_blocks, 7) */) {
    extern __shared__ double shared[]; /* (7, blockDim.x) */
    const int b = blockIdx.x / hist_blocks;
    const int hb = blockIdx.x - b * hist_blocks;
    const int t = threadIdx.x, nt = blockDim.x;
    TrArgs a = {n_pairs, n_interfaces, omega, k_perp, eps, response, lam_top, velocity, time,
                position, charge, face, coupling, hbarc};
    double sums[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (b < n_pairs) {
        for (long long h = (long long)hb * nt + t; h < n_hist; h += (long long)hist_blocks * nt)
            tr_history(a, b, starts[h], starts[h + 1], sums);
    }
    for (int k = 0; k < 7; ++k) shared[k * nt + t] = sums[k];
    __syncthreads();
    for (int stride = nt / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            for (int k = 0; k < 7; ++k) shared[k * nt + t] += shared[k * nt + t + stride];
        __syncthreads();
    }
    if (t == 0 && b < n_pairs)
        for (int k = 0; k < 7; ++k) partial[((long long)b * hist_blocks + hb) * 7 + k] = shared[k * nt];
}
#endif
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy

        _KERNEL = cupy.RawKernel(SOURCE, "pyrite_tr_history_sums", options=("-std=c++17",))
    return _KERNEL


def history_sums(
    stack,
    velocity,
    time_ang,
    position_ang,
    face,
    charge,
    starts,
    *,
    threads: int = DEFAULT_THREADS,
    target_blocks: int = _TARGET_BLOCKS,
):
    """``(sum_e |S_e|^2, sum_e S_e)`` per energy/direction pair, on the device.

    ``stack`` is a CuPy :class:`~pyrite.montecarlo.spectrum.transition_radiation.InterfaceResponse`;
    the ``K`` crossings (emitting face on top, charge applied in the kernel)
    are host arrays sorted by history, with history ``h`` the crossings
    ``starts[h]:starts[h + 1]``. Returns CuPy arrays ``(B,)`` and ``(B, 3)``
    in the frame of ``stack``.

    Validation: tr-crossing-superposition, tr-coherent-bunch
    """
    import cupy

    if threads <= 0 or threads & (threads - 1):
        raise ValueError("threads must be a positive power of two")
    pairs = int(stack.omega.size)
    n_hist = int(np.size(starts)) - 1
    hist_blocks = int(min(-(-n_hist // threads), max(1, -(-int(target_blocks) // pairs))))
    hist_blocks = max(hist_blocks, 1)
    if pairs * hist_blocks >= 1 << 31:
        raise ValueError("too many energy/direction pairs for one launch")
    partial = cupy.zeros((pairs, hist_blocks, _SUMS), dtype=cupy.float64)

    def real(value, shape=None):
        array = cupy.ascontiguousarray(cupy.asarray(value, dtype=cupy.float64))
        return array if shape is None else array.reshape(shape)

    def cplx(value):
        return cupy.ascontiguousarray(cupy.asarray(value, dtype=cupy.complex128))

    args = (
        np.int32(pairs),
        np.int32(stack.response.shape[1]),
        np.int32(n_hist),
        np.int32(hist_blocks),
        real(stack.omega),
        real(stack.k_perp),
        cplx(stack.eps),
        cplx(stack.response),
        cplx(stack.lam_top),
        real(velocity, (-1, 3)),
        real(time_ang),
        real(np.asarray(position_ang, dtype=float)[:, :2]),
        real(charge),
        cupy.asarray(np.asarray(face, dtype=np.int32)),
        cupy.asarray(np.asarray(starts, dtype=np.int64)),
        np.float64(_COUPLING),
        np.float64(HBARC_EV_ANG),
        partial,
    )
    _kernel()(
        (pairs * hist_blocks,),
        (int(threads),),
        args,
        shared_mem=_SUMS * int(threads) * 8,
    )
    sums = cupy.sum(partial, axis=1)
    return sums[:, 0], sums[:, 1::2] + 1j * sums[:, 2::2]
