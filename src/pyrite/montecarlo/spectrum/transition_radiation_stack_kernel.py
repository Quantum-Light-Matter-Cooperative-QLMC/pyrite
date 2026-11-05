"""CUDA RawKernels of the O(N) planar stack solve (#962).

One thread owns one energy/direction pair and walks the interfaces, so a stack
costs ``O(B N)`` scalar steps with no per-layer host traffic or array
materialization. The arithmetic is :mod:`.transition_radiation_stack` in
complex128. ``SOURCE`` also compiles as host C++ (``PYRITE_STACK_HOST``) with
sequential entry points, so the device arithmetic is checked against NumPy on
machines without a GPU.

Validation: optical-stack-fast-solver
"""

from __future__ import annotations

import numpy as np

from ...materials.crystal import ALPHA_FS

__all__ = ["SOURCE", "device_stack_amplitude", "device_stack_response"]

#: Threads per block.
_THREADS = 64
#: Interface steps of one launch; bounds a launch's wall time under display drivers.
_MAX_LAUNCH_STEPS = 1 << 27
_COUPLING = 4.0 * np.pi * np.sqrt(ALPHA_FS)

SOURCE = r"""
#ifdef PYRITE_STACK_HOST
#include <cmath>
#include <complex>
using std::exp;
using std::abs;
using std::cos;
using std::fabs;
using std::hypot;
using std::sin;
using std::sqrt;
typedef std::complex<double> cdouble;
#define STACK_DEVICE static inline
#else
#include <cupy/complex.cuh>
typedef complex<double> cdouble;
#define STACK_DEVICE __device__ __forceinline__
#endif

struct Pair {
    double w, kx, ky, k2, khx, khy; /* omega, k_perp, k_perp^2, k_hat; s_hat = (-khy, khx) */
};

STACK_DEVICE Pair stack_pair(const double *omega, const double *k_perp, long long b) {
    Pair p;
    p.w = omega[b];
    p.kx = k_perp[2 * b];
    p.ky = k_perp[2 * b + 1];
    p.k2 = p.kx * p.kx + p.ky * p.ky;
    const double k = hypot(p.kx, p.ky);
    p.khx = k > 0.0 ? p.kx / k : 1.0;
    p.khy = k > 0.0 ? p.ky / k : 0.0;
    return p;
}

STACK_DEVICE cdouble stack_lambda(cdouble eps, const Pair &p) {
    const cdouble root = sqrt(eps * (p.w * p.w) - p.k2);
    const bool flip = root.imag() < 0.0 || (root.imag() == 0.0 && root.real() < 0.0);
    return flip ? -root : root;
}

/* Medium i to i + 1 at one polarization: alpha, beta, tau and the next rho. */
STACK_DEVICE void stack_step(cdouble below, cdouble above, cdouble phase2, cdouble &rho,
                             cdouble &alpha, cdouble &beta, cdouble &tau, double &inverse_d) {
    const cdouble plus = 1.0 + rho, minus = 1.0 - rho;
    const cdouble denominator = plus * above + minus * below;
    inverse_d = 1.0 / abs(denominator);
    alpha = minus * below / denominator;
    beta = plus / denominator;
    tau = 2.0 * below / denominator;
    rho = phase2 * (plus * above - minus * below) / denominator;
}

/* Fill (n_if, 4, 2) for pair b. Bottom-up, slots 0-3 of an interface hold TE/TM alpha, beta and
   slot 4 the phase of the medium above it, and slots 5-6 hold TE/TM 1 / |D| as reals; the top-down
   pass expands them and forms the rounding reach of 1 + rho, sum |C / D| / max |C beta|. trans is (n_if, 2). */
STACK_DEVICE void stack_response_pair(const double *omega, const double *k_perp, const cdouble *eps,
                                      const double *z, int n_media, long long pairs, long long b,
                                      cdouble *response, cdouble *lam_top, cdouble *trans,
                                      double *cancellation) {
    const Pair p = stack_pair(omega, k_perp, b);
    const int n_if = n_media - 1;
    const cdouble I(0.0, 1.0);
    cdouble rho_te = 0.0, rho_tm = 0.0;
    cdouble e_below = eps[b];
    cdouble lam_below = stack_lambda(e_below, p);
    for (int i = 0; i < n_if; ++i) {
        const cdouble e_above = eps[(long long)(i + 1) * pairs + b];
        const cdouble lam_above = stack_lambda(e_above, p);
        cdouble phase = 1.0;
        if (i < n_if - 1) phase = exp(I * lam_above * (z[i + 1] - z[i]));
        const cdouble phase2 = phase * phase;
        cdouble *g = response + (long long)i * 8;
        cdouble tau_te, tau_tm;
        double inverse_te, inverse_tm;
        stack_step(-lam_below / p.w, -lam_above / p.w, phase2, rho_te, g[0], g[1], tau_te, inverse_te);
        stack_step(e_below * p.w / lam_below, e_above * p.w / lam_above, phase2, rho_tm, g[2], g[3],
                   tau_tm, inverse_tm);
        g[5] = inverse_te;
        g[6] = inverse_tm;
        trans[2 * i] = tau_te;
        trans[2 * i + 1] = tau_tm;
        g[4] = phase;
        e_below = e_above;
        lam_below = lam_above;
    }
    lam_top[0] = lam_below;
    const double s[2] = {-p.khy, p.khx}, k[2] = {p.khx, p.khy};
    cdouble weight_te = 1.0, weight_tm = 1.0;
    double reach_te = 0.0, reach_tm = 0.0, peak_te = 0.0, peak_tm = 0.0;
    for (int i = n_if - 1; i >= 0; --i) {
        cdouble *g = response + (long long)i * 8;
        const cdouble ge_te = weight_te * g[0], gh_te = weight_te * g[1];
        const cdouble ge_tm = weight_tm * g[2], gh_tm = weight_tm * g[3];
        reach_te += abs(weight_te) * g[5].real();
        reach_tm += abs(weight_tm) * g[6].real();
        if (abs(gh_te) > peak_te) peak_te = abs(gh_te);
        if (abs(gh_tm) > peak_tm) peak_tm = abs(gh_tm);
        for (int out = 0; out < 2; ++out) {
            for (int r = 0; r < 2; ++r) {
                g[2 * r + out] = ge_te * (s[out] * s[r]) + ge_tm * (k[out] * k[r]);
            }
        }
        for (int out = 0; out < 2; ++out) {
            for (int r = 0; r < 2; ++r) {
                g[4 + 2 * r + out] = gh_te * (s[out] * k[r]) + gh_tm * (k[out] * s[r]);
            }
        }
        if (i > 0) {
            const cdouble phase = g[-4]; /* slot 4 of interface i - 1: medium i */
            weight_te = weight_te * trans[2 * i] * phase;
            weight_tm = weight_tm * trans[2 * i + 1] * phase;
        }
    }
    const double ratio_te = reach_te / peak_te, ratio_tm = reach_tm / peak_tm;
    cancellation[0] = ratio_te > ratio_tm ? ratio_te : ratio_tm;
}

/* Own field tangential (E_x, E_y, H_x, H_y) of medium eps; charge +sqrt(alpha). */
STACK_DEVICE void stack_own_field(cdouble eps, const Pair &p, double q, const double *v,
                                  double coupling, cdouble out[4]) {
    const cdouble I(0.0, 1.0);
    const cdouble denom = (p.k2 + q * q) - eps * (p.w * p.w);
    const cdouble phi = coupling / (fabs(v[2]) * (eps * denom));
    const cdouble ie = I * phi, ih = I * (eps * phi);
    out[0] = ie * (eps * (p.w * v[0]) - p.kx);
    out[1] = ie * (eps * (p.w * v[1]) - p.ky);
    out[2] = ih * (p.ky * v[2] - q * v[1]);
    out[3] = ih * (q * v[0] - p.kx * v[2]);
}

STACK_DEVICE void stack_amplitude_pair(const double *omega, const double *k_perp,
                                       const double *velocity, const cdouble *eps, const double *z,
                                       int n_media, long long pairs, long long b, double coupling,
                                       cdouble *e_t, cdouble *lam_top, double *cancellation) {
    const Pair p = stack_pair(omega, k_perp, b);
    const double *v = velocity + 3 * b;
    const double q = (p.w - (p.kx * v[0] + p.ky * v[1])) / v[2];
    const int n_if = n_media - 1;
    const cdouble I(0.0, 1.0);
    cdouble rho_te = 0.0, rho_tm = 0.0, up_te = 0.0, up_tm = 0.0;
    /* Rounding of 1 + rho carried to the top, and the size of the terms it is compared with. */
    double reach_te = 0.0, reach_tm = 0.0, size_te = 0.0, size_tm = 0.0;
    cdouble e_below = eps[b];
    cdouble lam_below = stack_lambda(e_below, p);
    cdouble own_below[4], own_above[4];
    stack_own_field(e_below, p, q, v, coupling, own_below);
    for (int i = 0; i < n_if; ++i) {
        const cdouble e_above = eps[(long long)(i + 1) * pairs + b];
        const cdouble lam_above = stack_lambda(e_above, p);
        stack_own_field(e_above, p, q, v, coupling, own_above);
        const double arg = q * z[i];
        const cdouble crossing(cos(arg), sin(arg));
        const cdouble sx = (own_below[0] - own_above[0]) * crossing;
        const cdouble sy = (own_below[1] - own_above[1]) * crossing;
        const cdouble tx = (own_below[2] - own_above[2]) * crossing;
        const cdouble ty = (own_below[3] - own_above[3]) * crossing;
        cdouble phase = 1.0;
        if (i < n_if - 1) phase = exp(I * lam_above * (z[i + 1] - z[i]));
        const cdouble phase2 = phase * phase;
        cdouble alpha, beta, tau;
        double inverse;
        const double decay = abs(phase);
        stack_step(-lam_below / p.w, -lam_above / p.w, phase2, rho_te, alpha, beta, tau, inverse);
        const cdouble te_e = sy * p.khx - sx * p.khy, te_h = tx * p.khx + ty * p.khy;
        up_te = phase * (tau * up_te + alpha * te_e + beta * te_h);
        reach_te = decay * (abs(tau) * reach_te + inverse * abs(te_h));
        size_te = decay * (abs(tau) * size_te + abs(alpha * te_e) + abs(beta * te_h));
        stack_step(e_below * p.w / lam_below, e_above * p.w / lam_above, phase2, rho_tm, alpha, beta,
                   tau, inverse);
        const cdouble tm_e = sx * p.khx + sy * p.khy, tm_h = ty * p.khx - tx * p.khy;
        up_tm = phase * (tau * up_tm + alpha * tm_e + beta * tm_h);
        reach_tm = decay * (abs(tau) * reach_tm + inverse * abs(tm_h));
        size_tm = decay * (abs(tau) * size_tm + abs(alpha * tm_e) + abs(beta * tm_h));
        e_below = e_above;
        lam_below = lam_above;
        for (int c = 0; c < 4; ++c) own_below[c] = own_above[c];
    }
    lam_top[0] = lam_below;
    const double ratio_te = size_te > 0.0 ? reach_te / size_te : 0.0;
    const double ratio_tm = size_tm > 0.0 ? reach_tm / size_tm : 0.0;
    cancellation[0] = ratio_te > ratio_tm ? ratio_te : ratio_tm;
    e_t[0] = up_tm * p.khx - up_te * p.khy;
    e_t[1] = up_tm * p.khy + up_te * p.khx;
}

#ifdef PYRITE_STACK_HOST
extern "C" void pyrite_stack_host_response(int pairs, int n_media, const double *omega,
                                           const double *k_perp, const double *eps, const double *z,
                                           double *response, double *lam_top, double *trans,
                                           double *cancellation) {
    const long long n_if = n_media - 1;
    for (long long b = 0; b < pairs; ++b)
        stack_response_pair(omega, k_perp, (const cdouble *)eps, z, n_media, pairs, b,
                            (cdouble *)response + b * n_if * 8, (cdouble *)lam_top + b,
                            (cdouble *)trans + b * n_if * 2, cancellation + b);
}
extern "C" void pyrite_stack_host_amplitude(int pairs, int n_media, const double *omega,
                                            const double *k_perp, const double *velocity,
                                            const double *eps, const double *z, double coupling,
                                            double *e_t, double *lam_top, double *cancellation) {
    for (long long b = 0; b < pairs; ++b)
        stack_amplitude_pair(omega, k_perp, velocity, (const cdouble *)eps, z, n_media, pairs, b,
                             coupling, (cdouble *)e_t + 2 * b, (cdouble *)lam_top + b, cancellation + b);
}
#else
extern "C" __global__ void pyrite_stack_response(long long first, long long count, long long pairs,
                                                 int n_media, const double *omega,
                                                 const double *k_perp, const cdouble *eps,
                                                 const double *z, cdouble *response,
                                                 cdouble *lam_top, cdouble *trans,
                                                 double *cancellation) {
    const long long offset = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) return;
    const long long b = first + offset;
    const long long n_if = n_media - 1;
    stack_response_pair(omega, k_perp, eps, z, n_media, pairs, b, response + b * n_if * 8,
                        lam_top + b, trans + b * n_if * 2, cancellation + b);
}
extern "C" __global__ void pyrite_stack_amplitude(long long first, long long count, long long pairs,
                                                  int n_media, const double *omega,
                                                  const double *k_perp, const double *velocity,
                                                  const cdouble *eps, const double *z,
                                                  double coupling, cdouble *e_t, cdouble *lam_top,
                                                  double *cancellation) {
    const long long offset = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) return;
    const long long b = first + offset;
    stack_amplitude_pair(omega, k_perp, velocity, eps, z, n_media, pairs, b, coupling, e_t + 2 * b,
                         lam_top + b, cancellation + b);
}
#endif
"""

_KERNELS: dict[str, object] = {}


def _kernel(name):
    if name not in _KERNELS:
        import cupy

        _KERNELS[name] = cupy.RawKernel(SOURCE, name, options=("-std=c++17",))
    return _KERNELS[name]


def _launches(pairs, n_if):
    """``(first, count, blocks)`` launches of whole pairs within the step budget."""
    per_launch = max(_THREADS, (_MAX_LAUNCH_STEPS // max(n_if, 1)) // _THREADS * _THREADS)
    for first in range(0, pairs, per_launch):
        count = min(per_launch, pairs - first)
        yield first, count, -(-count // _THREADS)


def device_stack_response(omega, k_perp, media, z_if):
    """Device :func:`~.transition_radiation_stack.stack_response`.

    Validation: optical-stack-fast-solver
    """
    import cupy

    pairs, n_media = int(omega.size), int(media.shape[0])
    n_if = n_media - 1
    response = cupy.zeros((pairs, n_if, 4, 2), dtype=cupy.complex128)
    lam_top = cupy.zeros(pairs, dtype=cupy.complex128)
    trans = cupy.zeros((pairs, n_if, 2), dtype=cupy.complex128)
    cancellation = cupy.zeros(pairs, dtype=cupy.float64)
    z = cupy.asarray(np.asarray(z_if, dtype=float))
    arrays = (
        cupy.ascontiguousarray(omega.astype(cupy.float64)),
        cupy.ascontiguousarray(k_perp.astype(cupy.float64)),
        cupy.ascontiguousarray(media.astype(cupy.complex128)),
        z,
        response,
        lam_top,
        trans,
        cancellation,
    )
    for first, count, blocks in _launches(pairs, n_if):
        args = (np.int64(first), np.int64(count), np.int64(pairs), np.int32(n_media), *arrays)
        _kernel("pyrite_stack_response")((blocks,), (_THREADS,), args)
    return response, lam_top, cancellation


def device_stack_amplitude(omega, k_perp, velocity, media, z_if):
    """Top-wave tangential ``E`` of a straight charge ``(B, 2)`` and vacuum ``lambda``.

    Validation: optical-stack-fast-solver
    """
    import cupy

    pairs, n_media = int(omega.size), int(media.shape[0])
    e_t = cupy.zeros((pairs, 2), dtype=cupy.complex128)
    lam_top = cupy.zeros(pairs, dtype=cupy.complex128)
    cancellation = cupy.zeros(pairs, dtype=cupy.float64)
    arrays = (
        cupy.ascontiguousarray(omega.astype(cupy.float64)),
        cupy.ascontiguousarray(k_perp.astype(cupy.float64)),
        cupy.ascontiguousarray(velocity.astype(cupy.float64)),
        cupy.ascontiguousarray(media.astype(cupy.complex128)),
        cupy.asarray(np.asarray(z_if, dtype=float)),
        np.float64(_COUPLING),
        e_t,
        lam_top,
        cancellation,
    )
    for first, count, blocks in _launches(pairs, n_media - 1):
        args = (np.int64(first), np.int64(count), np.int64(pairs), np.int32(n_media), *arrays)
        _kernel("pyrite_stack_amplitude")((blocks,), (_THREADS,), args)
    return e_t, lam_top, cancellation
