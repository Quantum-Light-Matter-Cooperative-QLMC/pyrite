"""Per-layer bounds that let the CPU midpoint core skip a provably idle cutoff solve.

The unstraggled midpoint row computes ``cutoff_rate`` at ``(E + E_cut)/2`` and
truncates when ``(E_cut - E)/cutoff_rate < step`` or ties it without a
geometry event (#841). The skip test ``delta > 0 and delta / bound[L] > step``
with ``delta = fl(E - E_cut)`` decides that neither can happen, given that the
evaluated ``cutoff_rate`` is negative with ``|cutoff_rate| <= bound[L]``:

* ``fl(E_cut - E) = -delta`` exactly, so the core's distance is
  ``fl(delta / |cutoff_rate|)``; ``delta / |cutoff_rate| >= delta / bound`` and
  rounding is monotone, hence the distance is at least ``fl(delta / bound)``,
  which is strictly greater than ``step``. No ``<`` and no tie: the truncation
  branch is not taken and the row is bit-identical to evaluating it.
* ``delta > 0`` keeps the inequality direction; ``E > E_cut`` also places the
  skipped radiative query inside the coverage the step-start query already
  admitted (``[min E_cut, E]``), so skipping cannot suppress a coverage error.

Each bound is a node maximum of the very table the core interpolates, widened
by ``CUTOFF_BOUND_MARGIN``. The margin covers every float64 rounding of the
interpolant; the worst case is SBETHE, whose log value carries an absolute
error below ``3u * 2 * 745`` (``|log| <= 745`` for any positive finite double,
``u = 2**-53``), i.e. a relative error below ``1e-12`` after ``exp``, against a
margin of ``1e-6``. Branches without a node-maximum proof (the analytic
Joy-Luo/Berger-Seltzer splice and the exact radiative quadrature) return
``False`` and the core keeps evaluating the cutoff on every row.
"""

import numpy as np
from numba import njit

CUTOFF_BOUND_MARGIN = 1e-6
# A LUT row whose node magnitudes span more than this ratio could change sign
# under interpolation rounding (error <= 5u * max|node|); such rows are refused.
_LUT_DYNAMIC_RANGE = 1e6
# exp(-700) is a normal double: SBETHE stopping cannot underflow to zero.
_SBETHE_MIN_LOG = -700.0


@njit(cache=True, inline="always")
def cutoff_solve_idle(lazy, bound, E_keV, E_cut_keV, step_ang):
    """True when the midpoint cutoff solve provably cannot truncate this row."""
    delta = E_keV - E_cut_keV
    return lazy and delta > 0.0 and delta / bound > step_ang


@njit(cache=True)
def sbethe_cutoff_bound(bound, enabled, counts, log_stopping):
    """Fill ``bound`` with ``max |_dEds_sbethe_packed_scalar|`` per layer, or refuse.

    The value is ``-exp(s)`` with ``s`` a convex log-log combination of two
    adjacent nodes (fraction in ``[0, 1]`` by monotone rounding of the bracket)
    or a clamped endpoint node, so ``s <= max(node)`` up to rounding and the
    rate is strictly negative when every node exceeds ``_SBETHE_MIN_LOG``.
    """
    if not enabled:
        return False
    for layer in range(bound.size):
        if counts[layer] < 1:
            return False
        high = -np.inf
        for i in range(counts[layer]):
            value = log_stopping[layer, i]
            if not (np.isfinite(value) and value > _SBETHE_MIN_LOG):
                return False
            high = max(high, value)
        bound[layer] = np.exp(high)
    return True


@njit(cache=True)
def lut_cutoff_bound(bound, lut_dEds):
    """Fill ``bound`` with ``max |_lut_lerp_2d(lut_dEds, L, i, f)|``, or refuse.

    ``a + f * (b - a)`` with ``f`` in ``[0, 1]`` and same-sign nodes lies within
    ``5u * max|node|`` of a convex combination, so the bound is the row maximum
    and the sign stays negative when the row's dynamic range is limited.
    """
    for layer in range(bound.size):
        high, low = 0.0, np.inf
        for i in range(lut_dEds.shape[1]):
            value = lut_dEds[layer, i]
            if not (np.isfinite(value) and value < 0.0):
                return False
            high = max(high, -value)
            low = min(low, -value)
        if not high <= _LUT_DYNAMIC_RANGE * low:
            return False
        bound[layer] = high
    return True


@njit(cache=True)
def add_soft_moment_bound(ok, bound, nel, plasma, cutoff_keV):
    """Add the table soft radiative stopping [keV/Angstrom] bound, or refuse.

    Table soft stopping is ``(1 - f) * v_lo + f * v_hi`` (or a node) with
    ``f`` clamped to ``[0, 1]`` and nonnegative nodes: at most
    ``max(node) * (1 + 4u)`` and nonnegative, so subtracting it keeps the
    collision rate's sign. Exact-quadrature buffers have no node maximum.
    ``cutoff_keV`` holds every electron's cutoff; coverage must reach the least.
    """
    n_layers = nel.size
    if not ok or plasma.size == n_layers or cutoff_keV.size == 0:
        return False
    total = (plasma.size - 3 * n_layers) // 5
    energy_base = 3 * n_layers
    value_base = energy_base + 2 * total
    least_eV = cutoff_keV.min() * 1e3
    for layer in range(n_layers):
        start = np.int64(plasma[n_layers + layer])
        count = np.int64(plasma[2 * n_layers + layer])
        if count < 2 or not least_eV >= plasma[energy_base + start]:
            return False
        high = 0.0
        for i in range(start, start + count):
            value = plasma[value_base + 3 * i]
            if not (np.isfinite(value) and value >= 0.0):
                return False
            high = max(high, value)
        bound[layer] += high * 1e-3
    return True


@njit(cache=True)
def widen_cutoff_bound(ok, bound):
    """Apply the rounding margin; refuse a nonfinite or nonpositive bound."""
    if not ok:
        return False
    for layer in range(bound.size):
        if not (np.isfinite(bound[layer]) and bound[layer] > 0.0):
            return False
        bound[layer] *= 1.0 + CUTOFF_BOUND_MARGIN
    return True
