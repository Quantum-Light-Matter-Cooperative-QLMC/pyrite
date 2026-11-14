"""Support-windowed node reduction with an analytic far tail (issue #1191).

The default node reduction evaluates every kept line, or every flight's
coherent substep sum, on every grid node. A line's intensity is concentrated
near its pole, and far from it the profile is a smooth inverse power. This
module evaluates each line or flight exactly only on a window of nodes around
its pole. Everywhere else it uses the cycle-averaged far-field expansion of the
same field, and that expansion is summed over all lines by one FFT
convolution per call.

Far field of a flight
---------------------
A flight's rows ``j`` (linear escape pieces of contiguous substeps) carry the
formation field (``coherent-formation-absorption``)

    S(E) = sum_j c_j e^{i phi_j} [b_j e^{i v_j} - a_j e^{-i v_j}] / (2 w_j),
    w_j = i v_j - q_j = i a_j (E - p_j),   p_j = E_k,j - i q_j / a_j,

with ``E_k,j = E_vac,j + (dL_j/2) delta omega(E_vac,j)/a_j`` the
refraction-shifted centre. Each term is an *endpoint* amplitude ``gamma_k
e^{i psi_k(E)}/(E - p_k)`` with

- end:   ``gamma = c b/(2 i a)``,  ``psi = phi + v``;
- start: ``gamma = -c a/(2 i a)``, ``psi = phi - v``.

``psi_k`` is the emission phase at that endpoint. Its energy slope is the
endpoint's observer arrival time. A junction shared by contiguous rows has
equal phases and slopes on both sides, so its two terms add coherently and
cancel to first order (exactly for an unchirped, unabsorbed straight flight).
That cancellation is what makes the substep sum equal the unsplit flight.
Endpoints with different arrival times beat at ``|delta slope|``. Averaged over
that beat their cross terms vanish, so the cycle-averaged far intensity is
the incoherent sum over *clusters* of equal-slope endpoints:

    <|S|^2> = sum_clusters |sum_k gamma_k e^{i psi_k(E_n)} / (x - e_k)|^2,
    x = E - E_n,   e_k = p_k - E_n,

expanded to second order in ``e_k/x`` about the flight's node ``E_n``:

    <|S|^2> = |A0|^2/x^2 + 2 Re(A0 A1*)/x^3 + (|A1|^2 + 2 Re(A0 A2*))/x^4,
    A_m = sum_k gamma_k e^{i psi_k(E_n)} e_k^m.

Absorption enters through the complex pole and chirp (energy loss along the
flight) through the spread of ``e_k``. The undamped single line recovers the
cycle average ``1/(2 a^2 x^2)`` of ``sinc^2``. For the incoherent per-row
``sinc^2`` route each row is its own flight with ``A0 = 1``, ``A_m =
eps^m``, ``eps = E_r - E_n``: ``m2 = w/(2 a^2)``, ``m3 = 2 eps m2``, ``m4 = 3
eps^2 m2``.

Window and error
----------------
A flight is exact on nodes with ``|E - E_n| <= spread + W``, where ``W = x_c
max(1/a_F, max_j |q_j|/a_j)`` and ``a_F = sum_j a_j`` is the flight's own sinc
slope. The window scales with the *flight*, not with its substeps: the
expansion needs only that every non-cancelling endpoint pair beats at least
``x_c`` times across it. For a straight flight those endpoints are its two
ends, at ``2 a_F``. Truncating the profile would drop ``1/(pi x_c)`` of each
line's mass. The tail keeps it, leaving the oscillating remainder
``O(1/x_c^2)`` plus the expansion's ``O((spread/W)^3)``. The prototype on
substep flights (1-64 rows, chirp, absorption) gives yield ``<= 3e-5`` and
centroid ``<= 2e-3`` eV at ``x_c = 100``, with the substep count irrelevant.

Exactness bookkeeping
---------------------
The tail kernel ``K(x) = m2/x^2 + m3/x^3 + m4/x^4`` (with ``K(0) = 0``) is added
to *every* node by convolution. Inside its window each flight adds ``exact -
K``, so outside the window the node holds ``K`` and inside it holds the exact
value. Poles in a band-wide margin beyond the grid go through the same
convolution. Farther poles are aggregated into geometric bins evaluated
directly, or, when their window still reaches the band, evaluated directly
per flight. The grid must be uniform; callers keep the full-node reduction
otherwise.

Validation: node-support-window-tail
"""

import numpy as np
from numba import get_num_threads, njit, prange

from ....materials.crystal import HBARC_EV_ANG

#: Default window parameter: exact within ``|v| <= x_c`` of each flight.
DEFAULT_NODE_SUPPORT_CUTOFF = 100.0
# Relative node-spacing tolerance below which a grid counts as uniform.
_UNIFORM_RTOL = 1.0e-9
# Endpoint slopes closer than this fraction of the flight's own end-to-end
# beat (2 a_F) are one coherent cluster.
_CLUSTER_TOL = 1.0e-3
# Geometric far-bin ratio: a far pole is re-centred by at most this fraction
# of its distance to the band.
_FAR_BIN_RATIO = 1.0 / 32.0
# Clusters below this fraction of a flight's largest cluster intensity do not
# constrain its window (cancelled junctions).
_SIGNIFICANT_M2 = 1.0e-4

_MODE_CONV, _MODE_DIRECT, _MODE_FAR = 0, 1, 2


def uniform_step(E_grid):
    """``(E0, step)`` of a uniform ascending grid, else None. Validation: node-support-window-tail"""
    raw = np.asarray(E_grid)
    E = raw.astype(np.float64)
    if E.size < 3:
        return None
    step = (E[-1] - E[0]) / (E.size - 1)
    if not step > 0:
        return None
    # A float32 grid carries its cast's rounding (~1e-4 eV at keV energies).
    eps = np.finfo(raw.dtype).eps if raw.dtype.kind == "f" else 0.0
    tol = max(_UNIFORM_RTOL * step, 4.0 * eps * float(np.max(np.abs(E))))
    if np.max(np.abs(E - (E[0] + step * np.arange(E.size)))) > tol:
        return None
    return float(E[0]), float(step)


# ---------------------------------------------------------------------------
# Tail accumulator.
# ---------------------------------------------------------------------------
class NodeTail:
    """Far-tail moments of every windowed line, applied once per call.

    ``deposit`` takes per-pole integer nodes (relative to ``E_grid[0]``,
    possibly outside it) and their ``(m2, m3, m4)`` already scaled by the
    caller's weight. ``apply`` adds ``sum_pole K_pole(E)`` to each target
    spectrum. Validation: node-support-window-tail
    """

    def __init__(self, E_grid, n_targets, x_c):
        spec = uniform_step(E_grid)
        if spec is None:
            raise ValueError("NodeTail needs a uniform energy grid")
        self.E0, self.step = spec
        self.n = int(np.asarray(E_grid).size)
        self.pad = self.n
        self.x_c = float(x_c)
        L = self.n + 2 * self.pad
        self._dep = np.zeros((n_targets, 3, L))
        self._far = [[] for _ in range(n_targets)]
        self._direct = np.zeros((n_targets, self.n))

    @property
    def length(self):
        return self._dep.shape[2]

    def in_conv(self, node):
        ext = node + self.pad
        return (ext >= 0) & (ext < self.length)

    def deposit(self, target, node, m2, m3, m4):
        node = np.asarray(node, dtype=np.int64)
        conv = self.in_conv(node)
        ext = node[conv] + self.pad
        for k, m in enumerate((m2, m3, m4)):
            np.add.at(self._dep[target, k], ext, np.asarray(m)[conv])
        far = ~conv
        if far.any():
            self._far[target].append(
                (node[far], np.asarray(m2)[far], np.asarray(m3)[far], np.asarray(m4)[far])
            )

    def add_direct(self, target, values):
        self._direct[target] += values

    def apply(self, targets):
        """Add the accumulated tails into ``targets`` (NumPy 1-D arrays)."""
        kern_hat = None
        for t, tgt in enumerate(targets):
            total = self._direct[t].copy()
            if np.any(self._dep[t]):
                if kern_hat is None:
                    nfft, kern_hat = self._kernel_spectra()
                total += self._convolve(self._dep[t], nfft, kern_hat)
            if self._far[t]:
                total += self._far_field(self._far[t])
            tgt += total.astype(tgt.dtype, copy=False)

    def _kernel_spectra(self):
        from scipy.fft import next_fast_len, rfft

        L = self.length
        j = np.arange(-(L - 1), L, dtype=np.float64)
        x = j * self.step
        with np.errstate(divide="ignore"):
            inv = np.where(j == 0, 0.0, 1.0 / x)
        nfft = next_fast_len(3 * L - 2, real=True)
        return nfft, [rfft(inv**k, nfft, workers=-1) for k in (2, 3, 4)]

    def _convolve(self, dep, nfft, kern_hat):
        from scipy.fft import irfft, rfft

        L = self.length
        acc = None
        for k in range(3):
            if not np.any(dep[k]):
                continue
            term = rfft(dep[k], nfft, workers=-1) * kern_hat[k]
            acc = term if acc is None else acc + term
        full = irfft(acc, nfft, workers=-1)
        start = self.pad + L - 1
        return full[start : start + self.n]

    def _far_field(self, chunks):
        node = np.concatenate([c[0] for c in chunks])
        m2, m3, m4 = (np.concatenate([c[i] for c in chunks]) for i in (1, 2, 3))
        n = self.n
        dist = np.where(node < 0, -node, node - (n - 1)).astype(np.float64)
        side = node < 0
        key = np.floor(np.log(dist) / np.log1p(_FAR_BIN_RATIO)).astype(np.int64)
        key = np.where(side, -key - 1, key)
        uniq, inv = np.unique(key, return_inverse=True)
        # Bin centre: the m2-free geometric midpoint of the bin's distance range.
        k_abs = np.where(uniq < 0, -uniq - 1, uniq)
        mid = np.sqrt(
            np.exp(k_abs * np.log1p(_FAR_BIN_RATIO))
            * np.exp((k_abs + 1) * np.log1p(_FAR_BIN_RATIO))
        )
        centre = np.where(uniq < 0, -mid, (n - 1) + mid)  # node units
        # Re-centre each pole's moments from its node to its bin centre:
        # x = y - s, s = (node - centre) step.
        s = (node - centre[inv]) * self.step
        b2 = np.bincount(inv, m2, uniq.size)
        b3 = np.bincount(inv, m3 + 2.0 * s * m2, uniq.size)
        b4 = np.bincount(inv, m4 + 3.0 * s * m3 + 3.0 * s * s * m2, uniq.size)
        y = (np.arange(n)[None, :] - centre[:, None]) * self.step
        inv_y = 1.0 / y
        return ((b2[:, None] + (b3[:, None] + b4[:, None] * inv_y) * inv_y) * inv_y * inv_y).sum(0)


# ---------------------------------------------------------------------------
# Window plans.
# ---------------------------------------------------------------------------
def _window_layout(tail, node, R):
    """Clipped exact range ``[lo, hi]`` and mode per flight/line."""
    n = tail.n
    lo = np.maximum(node - R, 0)
    hi = np.minimum(node + R, n - 1)
    conv = tail.in_conv(node)
    reaches = (node - R <= n - 1) & (node + R >= 0)
    mode = np.where(conv, _MODE_CONV, np.where(reaches, _MODE_DIRECT, _MODE_FAR))
    return lo.astype(np.int64), hi.astype(np.int64), mode.astype(np.int64)


def _chunk_bounds(work, n_chunks):
    """Contiguous item ranges of about equal ``work``."""
    if work.size == 0:
        return np.zeros(2, dtype=np.int64)
    cum = np.cumsum(work, dtype=np.float64)
    targets = cum[-1] * np.arange(1, n_chunks) / n_chunks
    cuts = np.searchsorted(cum, targets, side="left")
    return np.unique(np.concatenate(([0], cuts, [work.size]))).astype(np.int64)


def _radius(tail, reach):
    # The cap only guards int64 overflow; it must never shrink a real window.
    return np.minimum(np.ceil(reach / tail.step), 2.0**52).astype(np.int64)


def row_window_plan(tail, E_r, a_width):
    """Per-line ``(node, lo, hi, mode, mu2, mu3, mu4)`` of the ``sinc^2`` route.

    ``mu*`` are the tail moments per unit weight. Validation: node-support-window-tail
    """
    E_r = np.asarray(E_r, dtype=np.float64)
    a = np.asarray(a_width, dtype=np.float64)
    node = np.rint((E_r - tail.E0) / tail.step).astype(np.int64)
    eps = E_r - (tail.E0 + node * tail.step)
    R = _radius(tail, np.abs(eps) + tail.x_c / a)
    lo, hi, mode = _window_layout(tail, node, R)
    mu2 = 0.5 / (a * a)
    return node, lo, hi, mode, mu2, 2.0 * eps * mu2, 3.0 * eps * eps * mu2


def grouped_window_plan(tail, lines, d, g_phase, L_esc, coefs, starts, grids):
    """Per-flight ``(node, lo, hi, mode, m2, m3, m4)`` of the grouped route.

    Inputs are the sorted, flight-referenced rows of
    :func:`._cpu_reduce.accumulate_grouped_formation`. Moments are summed over
    polarizations and endpoint clusters (module docstring).
    Validation: node-support-window-tail
    """
    E_vac, a_vac, half_dL, apb, bma, q = (np.asarray(x, dtype=np.float64) for x in lines)
    d, g_phase, L_esc = (np.asarray(x, dtype=np.float64) for x in (d, g_phase, L_esc))
    E_grid, _, dom_grid = (np.asarray(g, dtype=np.float64) for g in grids)
    starts = np.asarray(starts, dtype=np.int64)
    n_f = starts.size - 1
    heads = starts[:-1]
    fl = np.repeat(np.arange(n_f), np.diff(starts))

    def dom(E):
        return np.interp(E, E_grid, dom_grid)

    Ek = E_vac + half_dL * dom(E_vac) / a_vac
    pole = Ek - 1j * q / a_vac
    re_min = np.minimum.reduceat(Ek, heads)
    re_max = np.maximum.reduceat(Ek, heads)
    node = np.rint((0.5 * (re_min + re_max) - tail.E0) / tail.step).astype(np.int64)
    E_n = tail.E0 + node * tail.step
    a_F = np.add.reduceat(a_vac, heads)
    q_rate = np.maximum.reduceat(np.abs(q) / a_vac, heads)
    W = tail.x_c * np.maximum(1.0 / a_F, q_rate)
    spread = np.maximum(np.abs(re_min - E_n), np.abs(re_max - E_n))

    # Endpoint phases at E_n and their slopes across the provisional reach.
    def psi(E, sign):
        om = E / HBARC_EV_ANG
        dw = dom(E)
        v = a_vac * (E - E_vac) - half_dL * dw
        return d * om - g_phase - L_esc * dw + sign * v

    def both(E):
        return np.concatenate((psi(E, 1.0), psi(E, -1.0)))

    En_r, H_r = E_n[fl], (spread + W)[fl]
    slope = (both(En_r + H_r) - both(En_r - H_r)) / (2.0 * np.concatenate((H_r, H_r)))
    inv = 1.0 / (2j * a_vac)
    gammas = [
        np.concatenate((c * (0.5 * (apb + bma)) * inv, -c * (0.5 * (apb - bma)) * inv))
        for c in (np.asarray(c, dtype=np.complex128) for c in coefs)
    ]
    fl2 = np.concatenate((fl, fl))
    clusters = _endpoint_clusters(
        fl2, slope, both(En_r), np.concatenate((pole, pole)) - E_n[fl2], gammas, a_F
    )
    cl_flight, cl_slope, cl_m = clusters
    m2, m3, m4 = (np.bincount(cl_flight, m, n_f) for m in cl_m)
    W = np.maximum(W, 2.0 * tail.x_c / _significant_gap(cl_flight, cl_slope, cl_m[0], n_f))
    R = _radius(tail, spread + W)
    lo, hi, mode = _window_layout(tail, node, R)
    return node, lo, hi, mode, m2, m3, m4


def _endpoint_clusters(fl2, slope, psi_n, e, gammas, a_F):
    """Equal-slope endpoint clusters and their intensity moments.

    Endpoints sort by ``(flight, slope)``; a slope step above ``_CLUSTER_TOL``
    of the flight's end-to-end beat ``2 a_F`` starts a new cluster. Returns
    ``(flight, slope, (m2, m3, m4))`` per cluster, moments summed over
    polarizations. Validation: node-support-window-tail
    """
    order = np.lexsort((slope, fl2))
    fl2, slope, e = fl2[order], slope[order], e[order]
    new = np.ones(fl2.size, dtype=bool)
    new[1:] = (fl2[1:] != fl2[:-1]) | (np.diff(slope) > _CLUSTER_TOL * 2.0 * a_F[fl2[1:]])
    heads = np.flatnonzero(new)
    phase = np.exp(1j * psi_n[order])
    m2 = m3 = m4 = 0.0
    for gam in gammas:
        g = gam[order] * phase
        A0 = np.add.reduceat(g, heads)
        A1 = np.add.reduceat(g * e, heads)
        A2 = np.add.reduceat(g * e * e, heads)
        m2 = m2 + np.abs(A0) ** 2
        m3 = m3 + 2.0 * (A0 * A1.conj()).real
        m4 = m4 + np.abs(A1) ** 2 + 2.0 * (A0 * A2.conj()).real
    return fl2[heads], slope[heads], (m2, m3, m4)


def _significant_gap(cl_flight, cl_slope, cl_m2, n_f):
    """Smallest slope gap between a flight's significant clusters (else inf).

    Clusters holding at least ``_SIGNIFICANT_M2`` of the flight's largest
    cluster intensity must beat ``x_c`` times across the window: the cross
    term of two that beat slower is not averaged out. Cancelled substep
    junctions fall below it. Validation: node-support-window-tail
    """
    peak = np.zeros(n_f)
    np.maximum.at(peak, cl_flight, cl_m2)
    keep = cl_m2 >= _SIGNIFICANT_M2 * peak[cl_flight]
    f, s = cl_flight[keep], cl_slope[keep]
    gap = np.full(n_f, np.inf)
    same = f[1:] == f[:-1]
    np.minimum.at(gap, f[1:][same], np.diff(s)[same])
    return gap


# ---------------------------------------------------------------------------
# Exact window kernels.
# ---------------------------------------------------------------------------
@njit(inline="always", cache=True)
def _tail_value(m2, m3, m4, x):  # pragma: no cover - numba
    inv = 1.0 / x
    return (m2 + (m3 + m4 * inv) * inv) * inv * inv


@njit(inline="always", cache=True)
def _tail_correction(e, node, step, mode, m2, m3, m4, n_E):  # pragma: no cover - numba
    """Adjustment at window node ``e``: ``-K`` under convolution, else 0."""
    if mode != 0 or e == node:
        return 0.0
    return -_tail_value(m2, m3, m4, (e - node) * step)


@njit(parallel=True, cache=True)
def _sincsq_window_reduce(  # pragma: no cover - numba
    E_r, a_width, weights, E_grid, node, lo, hi, mode, mu, step, bounds, out
):
    n_l = E_r.shape[0]
    n_E = E_grid.shape[0]
    for c in prange(bounds.shape[0] - 1):
        acc = out[c]
        for j in range(bounds[c], bounds[c + 1]):
            if j >= n_l or mode[j] == 2:
                continue
            for e in range(lo[j], hi[j] + 1):
                x = a_width[j] * (E_grid[e] - E_r[j]) / np.pi
                if x == 0.0:
                    x = 1.0e-20
                y = np.pi * x
                s = np.sin(y) / y
                v = s * s + _tail_correction(
                    e, node[j], step, mode[j], mu[0, j], mu[1, j], mu[2, j], n_E
                )
                for t in range(weights.shape[0]):
                    acc[t, e] += weights[t, j] * v
            if mode[j] == 1:
                for e in range(n_E):
                    if e < lo[j] or e > hi[j]:
                        v = _tail_value(mu[0, j], mu[1, j], mu[2, j], (e - node[j]) * step)
                        for t in range(weights.shape[0]):
                            acc[t, e] += weights[t, j] * v


def accumulate_sincsq_windowed(E_r, a_width, weights, E_grid, targets, tail, target_ids):
    """Windowed :func:`._cpu_reduce.accumulate_sincsq` plus tail deposits.

    ``target_ids[t]`` is ``targets[t]``'s slot in ``tail``. Validation: node-support-window-tail
    """
    if E_r.size == 0 or E_grid.size == 0:
        return
    E_r = np.ascontiguousarray(E_r, dtype=np.float64)
    a_width = np.ascontiguousarray(a_width, dtype=np.float64)
    w = np.ascontiguousarray(np.stack([np.asarray(r, dtype=np.float64) for r in weights]))
    node, lo, hi, mode, mu2, mu3, mu4 = row_window_plan(tail, E_r, a_width)
    work = np.maximum(hi - lo + 1, 0) + np.where(mode == _MODE_DIRECT, tail.n, 0)
    bounds = _chunk_bounds(work.astype(np.float64), get_num_threads())
    out = np.zeros((bounds.size - 1, w.shape[0], tail.n))
    E_grid = np.ascontiguousarray(E_grid, dtype=np.float64)
    mu = np.ascontiguousarray(np.stack((mu2, mu3, mu4)))
    _sincsq_window_reduce(E_r, a_width, w, E_grid, node, lo, hi, mode, mu, tail.step, bounds, out)
    total = out.sum(0)
    keep = mode != _MODE_DIRECT
    for t, (tgt, slot) in enumerate(zip(targets, target_ids, strict=True)):
        tgt += total[t]
        tail.deposit(slot, node[keep], *(m[keep] * w[t, keep] for m in (mu2, mu3, mu4)))


@njit(parallel=True, cache=True)
def _grouped_window_reduce(  # pragma: no cover - numba
    E_vac, a_vac, half_dL, apb, bma, q, d, g_phase, L_esc, coefs, starts,
    E_grid, omega_grid, delta_omega_grid, series_w2,
    node, lo, hi, mode, mom, step, bounds, out,
):  # fmt: skip
    n_pol = coefs.shape[0]
    n_E = E_grid.shape[0]
    for c in prange(bounds.shape[0] - 1):
        acc = out[c]
        for f in range(bounds[c], bounds[c + 1]):
            if mode[f] == 2:
                continue
            for e in range(lo[f], hi[f] + 1):
                E = E_grid[e]
                om = omega_grid[e]
                dom = delta_omega_grid[e]
                s_re = 0.0
                s_im = 0.0
                p_re = 0.0
                p_im = 0.0
                for j in range(starts[f], starts[f + 1]):
                    v = a_vac[j] * (E - E_vac[j]) - half_dL[j] * dom
                    qj = q[j]
                    w2 = qj * qj + v * v
                    if w2 < series_w2:
                        h = 0.5 * apb[j]
                        f_re = h * (1.0 - qj * qj / 3.0 - v * v / 6.0)
                        f_im = -h * qj * v / 3.0
                    else:
                        sv = np.sin(v)
                        cv = np.cos(v)
                        inv = 1.0 / (2.0 * w2)
                        f_re = (v * apb[j] * sv - qj * bma[j] * cv) * inv
                        f_im = -(v * bma[j] * cv + qj * apb[j] * sv) * inv
                    arg = d[j] * om - g_phase[j] - L_esc[j] * dom
                    ca = np.cos(arg)
                    sa = np.sin(arg)
                    x_re = f_re * ca - f_im * sa
                    x_im = f_re * sa + f_im * ca
                    cc = coefs[0, j]
                    s_re += cc.real * x_re - cc.imag * x_im
                    s_im += cc.real * x_im + cc.imag * x_re
                    if n_pol > 1:
                        cc = coefs[1, j]
                        p_re += cc.real * x_re - cc.imag * x_im
                        p_im += cc.real * x_im + cc.imag * x_re
                acc[e] += (s_re * s_re + s_im * s_im + p_re * p_re + p_im * p_im) + (
                    _tail_correction(
                        e, node[f], step, mode[f], mom[0, f], mom[1, f], mom[2, f], n_E
                    )
                )
            if mode[f] == 1:
                for e in range(n_E):
                    if e < lo[f] or e > hi[f]:
                        acc[e] += _tail_value(mom[0, f], mom[1, f], mom[2, f], (e - node[f]) * step)


def accumulate_grouped_windowed(lines, d, g_phase, L_esc, coefs, starts, grids, out, weight, tail):
    """Windowed :func:`._cpu_reduce.accumulate_grouped_formation` (target slot 0).

    Validation: node-support-window-tail, coherent-formation-absorption
    """
    from ._formation import FORMATION_SERIES_W2

    E_grid = grids[0]
    if E_grid.size == 0 or starts.size < 2:
        return
    f64 = [np.ascontiguousarray(a, dtype=np.float64) for a in (*lines, d, g_phase, L_esc)]
    grid64 = [np.ascontiguousarray(g, dtype=np.float64) for g in grids]
    cstack = np.ascontiguousarray(np.stack(coefs), dtype=np.complex128)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    node, lo, hi, mode, m2, m3, m4 = grouped_window_plan(
        tail, f64[:6], f64[6], f64[7], f64[8], list(cstack), starts, grid64
    )
    rows = np.diff(starts)
    work = rows * (np.maximum(hi - lo + 1, 0) + np.where(mode == _MODE_DIRECT, tail.n, 0))
    bounds = _chunk_bounds(work.astype(np.float64), get_num_threads())
    acc = np.zeros((bounds.size - 1, tail.n))
    mom = np.ascontiguousarray(np.stack((m2, m3, m4)))
    _grouped_window_reduce(
        *f64, cstack, starts, *grid64, float(FORMATION_SERIES_W2),
        node, lo, hi, mode, mom, tail.step, bounds, acc,
    )  # fmt: skip
    out += acc.sum(0) * weight
    keep = mode != _MODE_DIRECT
    tail.deposit(0, node[keep], m2[keep] * weight, m3[keep] * weight, m4[keep] * weight)
