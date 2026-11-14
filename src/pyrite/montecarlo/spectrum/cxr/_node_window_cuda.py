"""CuPy support-window kernels; host plans, device exact evaluation and FFT.

Each block owns one line/flight; lanes traverse its nodes contiguously. Only
support nodes evaluate transcendentals. Float64 corrections/atomics and FFTs
protect cancellation of the near-pole inverse-power tail; fields use float32.
Validation: node-support-window-tail, coherent-formation-absorption
"""

from functools import cache

import numpy as np

from ._node_window import NodeTail, grouped_window_plan, row_window_plan

_SOURCE = r"""
__device__ double tail_value(double m2, double m3, double m4, double x) {
    double inv = 1.0 / x;
    return (m2 + (m3 + m4 * inv) * inv) * inv * inv;
}
extern "C" __global__ void window_rows(
    const float* er, const float* a, const double* weights, const float* energy,
    const long long* node, const long long* lo, const long long* hi,
    const long long* mode, const double* mom, double step,
    int nr, int ne, int nt, double* out) {
    int j = blockIdx.x;
    if (mode[j] == 2) return;
    int begin = mode[j] == 1 ? 0 : lo[j];
    int end = mode[j] == 1 ? ne - 1 : hi[j];
    for (int e = begin + threadIdx.x; e <= end; e += blockDim.x) {
        bool exact = e >= lo[j] && e <= hi[j];
        double v = 0.0;
        if (exact) {
            float x = a[j] * (energy[e] - er[j]);
            float s = x == 0.0f ? 1.0f : sinf(x) / x;
            v = (double)s * s;
        }
        if (e != node[j]) {
            double k = tail_value(mom[j], mom[nr+j], mom[2*nr+j], (e-node[j])*step);
            if (mode[j] == 0) v -= k;
            else if (!exact) v = k;
        }
        for (int t = 0; t < nt; ++t) atomicAdd(out+t*ne+e, weights[t*nr+j]*v);
    }
}
extern "C" __global__ void window_flights(
    const float* rows, const float* coef, const long long* starts,
    const float* grids, const long long* node, const long long* lo,
    const long long* hi, const long long* mode, const double* mom,
    double step, float series_w2, int nr, int nf, int ne, int npol, double* out) {
    int f = blockIdx.x;
    if (mode[f] == 2) return;
    int begin = mode[f] == 1 ? 0 : lo[f];
    int end = mode[f] == 1 ? ne - 1 : hi[f];
    for (int e = begin + threadIdx.x; e <= end; e += blockDim.x) {
        bool exact = e >= lo[f] && e <= hi[f];
        double value = 0.0;
        if (exact) {
            float sr = 0, si = 0, pr = 0, pi = 0;
            float energy = grids[e], om = grids[ne+e], dom = grids[2*ne+e];
            for (int j = starts[f]; j < starts[f+1]; ++j) {
                float v = rows[nr+j]*(energy-rows[j]) - rows[2*nr+j]*dom;
                float apb = rows[3*nr+j], bma = rows[4*nr+j], q = rows[5*nr+j];
                float w2 = q*q + v*v;
                float re, im;
                if (w2 < series_w2) {
                    float h = 0.5f*apb;
                    re = h*(1.0f-q*q/3.0f-v*v/6.0f);
                    im = -h*q*v/3.0f;
                } else {
                    float sv = sinf(v), cv = cosf(v), inv = 0.5f/w2;
                    re = (v*apb*sv-q*bma*cv)*inv;
                    im = -(v*bma*cv+q*apb*sv)*inv;
                }
                float arg = rows[6*nr+j]*om-rows[7*nr+j]-rows[8*nr+j]*dom;
                float ca = cosf(arg), sa = sinf(arg);
                float xr = re*ca-im*sa, xi = re*sa+im*ca;
                sr += coef[j]*xr-coef[nr+j]*xi;
                si += coef[j]*xi+coef[nr+j]*xr;
                if (npol > 1) {
                    pr += coef[2*nr+j]*xr-coef[3*nr+j]*xi;
                    pi += coef[2*nr+j]*xi+coef[3*nr+j]*xr;
                }
            }
            value = (double)sr*sr+(double)si*si+(double)pr*pr+(double)pi*pi;
        }
        if (e != node[f]) {
            double k = tail_value(mom[f], mom[nf+f], mom[2*nf+f], (e-node[f])*step);
            if (mode[f] == 0) value -= k;
            else if (!exact) value = k;
        }
        atomicAdd(out+e, value);
    }
}
"""


@cache
def _kernel(name):
    import cupy as cp

    return cp.RawKernel(_SOURCE, name, options=("--std=c++11", "--fmad=false"))


def _host(a):
    import cupy as cp

    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def _device(a, dtype):
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(a), dtype=dtype)


class DeviceNodeTail(NodeTail):
    """Host moment deposits with float64 device convolution and final summation."""

    def __init__(self, E_grid, n_targets, x_c):
        import cupy as cp

        super().__init__(E_grid, n_targets, x_c)
        self.correction = cp.zeros((n_targets, self.n), dtype=cp.float64)

    def apply(self, targets):
        import cupy as cp
        from scipy.fft import next_fast_len

        L = self.length
        nfft = next_fast_len(3 * L - 2, real=True)
        j = cp.arange(-(L - 1), L, dtype=cp.float64)
        inv = cp.where(j == 0, 0.0, 1.0 / (cp.where(j == 0, 1.0, j) * self.step))
        kernels = [cp.fft.rfft(inv**k, nfft) for k in (2, 3, 4)]
        for t, tgt in enumerate(targets):
            total = self.correction[t] + cp.asarray(self._direct[t])
            if np.any(self._dep[t]):
                dep = cp.asarray(self._dep[t])
                full = cp.fft.irfft(
                    sum(cp.fft.rfft(dep[k], nfft) * kernels[k] for k in range(3)), nfft
                )
                start = self.pad + L - 1
                total += full[start : start + self.n]
            if self._far[t]:
                total += cp.asarray(self._far_field(self._far[t]))
            tgt += total.astype(tgt.dtype)


def _plan_arrays(plan):
    node, lo, hi, mode, *mom = plan
    return tuple(_device(a, np.int64) for a in (node, lo, hi, mode)) + (
        _device(np.stack(mom), np.float64),
    )


def accumulate_rows(E_r, a_width, weights, E_grid, targets, tail, target_ids):
    """Windowed device sinc² reduction, retaining corrections until tail apply."""
    import cupy as cp

    if E_r.size == 0 or E_grid.size == 0:
        return
    er, a = _host(E_r), _host(a_width)
    w = np.stack([_host(v) for v in weights]).astype(np.float64)
    plan = row_window_plan(tail, er, a)
    out = cp.zeros((len(weights), tail.n), dtype=cp.float64)
    _kernel("window_rows")(
        (er.size,),
        (128,),
        (
            _device(E_r, np.float32),
            _device(a_width, np.float32),
            _device(w, np.float64),
            _device(E_grid, np.float32),
            *_plan_arrays(plan),
            np.float64(tail.step),
            np.int32(er.size),
            np.int32(tail.n),
            np.int32(len(weights)),
            out,
        ),
    )
    node, _, _, mode, *mom = plan
    keep = mode != 1
    for t, slot in enumerate(target_ids):
        tail.correction[slot] += out[t]
        tail.deposit(slot, node[keep], *(m[keep] * w[t, keep] for m in mom))


def accumulate_flights(lines, d, g_phase, L_esc, coefs, starts, grids, out, weight, tail):
    """Float32 device formation sum on each flight's planned node support."""
    import cupy as cp

    from ._formation import FORMATION_SERIES_W2

    if grids[0].size == 0 or starts.size < 2:
        return
    rows = [_host(a) for a in (*lines, d, g_phase, L_esc)]
    cs = [_host(c) for c in coefs]
    bounds = _host(starts)
    plan = grouped_window_plan(tail, rows[:6], *rows[6:], cs, bounds, [_host(g) for g in grids])
    acc = cp.zeros(tail.n, dtype=cp.float64)
    nf, nr = bounds.size - 1, rows[0].size
    coef = np.stack([v for c in cs for v in (c.real, c.imag)])
    _kernel("window_flights")(
        (nf,),
        (128,),
        (
            _device(np.stack(rows), np.float32),
            _device(coef, np.float32),
            _device(bounds, np.int64),
            _device(cp.stack(grids), np.float32),
            *_plan_arrays(plan),
            np.float64(tail.step),
            np.float32(FORMATION_SERIES_W2),
            np.int32(nr),
            np.int32(nf),
            np.int32(tail.n),
            np.int32(len(cs)),
            acc,
        ),
    )
    tail.correction[0] += acc * weight
    node, _, _, mode, *mom = plan
    keep = mode != 1
    tail.deposit(0, node[keep], *(m[keep] * weight for m in mom))
