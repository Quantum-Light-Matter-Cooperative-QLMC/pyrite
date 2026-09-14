"""Precomputed energy/depth lookup table for the LUT transport cores."""

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit

from .stopping import _dEds_spliced_compound

_LN10 = np.log(10.0)

# Adjacent grid nodes must stay distinct after any narrowing a backend applies
# to the energy coordinate. float32 is the narrowest real type the compute
# backends use, so require the node ratio exp(dlogE) to clear one float32 ulp
# with margin; below that the grid would collapse and the interpolation
# fraction would divide by a vanishing interval.
_MIN_DLOG_E = 8.0 * float(np.finfo(np.float32).eps)


class TransportLUTToleranceWarning(UserWarning):
    """The built transport LUT does not meet its requested interpolation tolerance."""


@dataclass(frozen=True)
class TransportLUTConfig:
    """Energy-grid policy for the ungrooved transport hot loops.

    The LUT is uniform in ``log`` kinetic energy, so an event needs one
    logarithm, one multiply, one integer conversion and linear interpolation --
    no binary search. A uniform *linear* grid cannot serve a slowing electron:
    over 0.1--100,000 keV a 16,384-point linear grid is 6.1 keV coarse at the
    cutoff, exactly where the transport coefficients vary fastest. Uniform log
    spacing puts constant *relative* resolution everywhere.

    ``intervals_per_decade`` is the target resolution; ``min_points`` and
    ``max_points`` bound memory. ``rel_tol`` is the requested maximum relative
    interpolation error. The builder measures the achieved error inside every
    interval and warns with :class:`TransportLUTToleranceWarning` when the point
    cap, or a model join, leaves it unmet -- the grid is never silently
    coarsened past the request without saying so.

    The default 1,024 intervals per decade holds the smooth tables (total rate,
    1/beta, screening) to <1e-6 relative and costs 6,145 nodes and ~0.3 MB of
    tables over the full 0.1--100,000 keV span. The default ``rel_tol`` of 1e-4
    is set by the *joins*, not by the smooth error: the Joy-Luo/Berger-Seltzer
    stopping crossover and the NIST Mott screening nodes are continuous but
    kinked, so the one interval straddling each is O(h) rather than O(h^2) and
    measures ~1.3e-5 for real carbon. Halving h halves that term, so chasing it
    with resolution alone is a losing game; 1e-4 is the honest contract, and
    ``TransportEnergyLUT.max_rel_interp_error`` always reports what was achieved.
    """

    enabled: bool = True
    intervals_per_decade: int = 1024
    min_points: int = 256
    max_points: int = 16384
    rel_tol: float = 1e-4


DEFAULT_TRANSPORT_LUT_CONFIG = TransportLUTConfig()


@dataclass(frozen=True)
class TransportEnergyLUT:
    """Per-run transport tables shared by lockstep CPU, per-electron CPU and CUDA.

    ``log_E_min`` and ``inv_dlogE`` are the only grid fields a hot loop needs:
    ``x = (log(E) - log_E_min) * inv_dlogE`` is the grid coordinate. The energy
    bounds and the audit fields are diagnostics and never reach a kernel.
    """

    log_E_min: float
    inv_dlogE: float
    n_energy: int
    n_el: np.ndarray
    total_rate: np.ndarray
    dEds: np.ndarray
    inv_beta: np.ndarray
    cdf: np.ndarray
    alpha: np.ndarray
    E_min_keV: float
    E_max_keV: float
    intervals_per_decade: float
    max_rel_interp_error: float
    worst_energy_keV: float
    tolerance_met: bool


@njit(cache=True, inline="always")
def _lut_index_frac_scalar(E_keV, log_E_min, inv_dlogE, n_energy):
    """Return lower LUT index and interpolation fraction, clamped to the grid.

    One logarithm, one multiply, one integer conversion. The branch order tests
    the upper clamp first so a non-finite coordinate (``E_keV == 0`` gives
    ``-inf``, a negative energy gives NaN) fails both comparisons and lands on
    the lower clamp instead of reaching ``int(x)``. ``_lut_lerp_at`` and the
    inline forms in ``_jit_kernel`` must keep this exact branch order or the CPU
    and CUDA cores would clamp differently at the endpoints.
    """
    x = (np.log(E_keV) - log_E_min) * inv_dlogE
    last = n_energy - 1
    if x >= last:
        return last - 1, 1.0
    if x > 0.0:
        i = int(x)
        return i, x - i
    return 0, 0.0


@njit(cache=True, inline="always")
def _lut_lerp_1d(table, i, f):
    return table[i] + f * (table[i + 1] - table[i])


@njit(cache=True, inline="always")
def _lut_lerp_2d(table, row, i, f):
    return table[row, i] + f * (table[row, i + 1] - table[row, i])


@njit(cache=True, inline="always")
def _lut_lerp_3d(table, row, col, i, f):
    return table[row, col, i] + f * (table[row, col, i + 1] - table[row, col, i])


def _evaluate_transport_nodes(
    E_grid,
    elastic_model_code,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    mott_tables,
    n_el,
    max_el,
):
    """Evaluate every tabulated transport quantity exactly on ``E_grid``.

    Shared by the grid build and by the midpoint audit, so the audit compares
    the interpolant against the same physics the nodes carry rather than a
    second transcription of it.
    """
    n_layers = len(L_Zs)
    n_energy = E_grid.size

    total_rate = np.empty((n_layers, n_energy), dtype=np.float64)
    dEds = np.empty((n_layers, n_energy), dtype=np.float64)
    cdf = np.ones((n_layers, max_el, n_energy), dtype=np.float64)
    alpha = np.zeros((n_layers, max_el, n_energy), dtype=np.float64)

    g = 1.0 + E_grid / 510.99895
    beta = np.sqrt(1.0 - 1.0 / (g * g))
    inv_beta = 1.0 / beta
    sqrt_E = np.sqrt(E_grid)
    E_511_over_1024 = (E_grid + 511.0) / (E_grid + 1024.0)
    rel2 = E_511_over_1024 * E_511_over_1024

    for L in range(n_layers):
        n = int(n_el[L])
        rates = np.empty((n, n_energy), dtype=np.float64)

        # The LUT bakes dE/ds, so it must carry the same spliced model the direct
        # cores evaluate or the LUT paths would silently keep the old physics.
        dEds[L] = _dEds_spliced_compound(
            np.asarray(L_Js[L], dtype=np.float64),
            np.asarray(L_ks[L], dtype=np.float64),
            np.asarray(L_coeffs[L], dtype=np.float64),
            np.asarray(L_E_cross[L], dtype=np.float64),
            0.0,
            E_grid.copy(),
        )

        for i_el in range(n):
            sr_joy = float(L_sr_joy_numer[L][i_el])
            if elastic_model_code == 1:
                rates[i_el] = float(L_mott_numer[L][i_el]) / (
                    E_grid
                    + float(L_mott_denom1[L][i_el]) * sqrt_E
                    + float(L_mott_denom2[L][i_el]) / sqrt_E
                )
            else:
                a = sr_joy / E_grid
                rates[i_el] = (
                    float(L_sr_rate_numer[L][i_el]) / (E_grid * E_grid) / (a * (1.0 + a)) * rel2
                )

            table = mott_tables[L][i_el]
            if elastic_model_code == 1 and table is not None:
                logE, logA = table
                alpha[L, i_el] = 10.0 ** np.interp(
                    np.log10(E_grid * 1e3),
                    np.asarray(logE, dtype=np.float64),
                    np.asarray(logA, dtype=np.float64),
                )
            else:
                alpha[L, i_el] = sr_joy / E_grid

        layer_total = rates.sum(axis=0)
        if np.any(~np.isfinite(layer_total)) or np.any(layer_total <= 0.0):
            raise ValueError("transport LUT produced a non-positive elastic rate")
        total_rate[L] = layer_total
        layer_cdf = np.cumsum(rates, axis=0) / layer_total[None, :]
        layer_cdf[-1, :] = 1.0
        cdf[L, :n, :] = layer_cdf

    return total_rate, dEds, inv_beta, cdf, alpha


def _scan_interp_error(node_table, sample_table, weight, worst, worst_index):
    """Fold one table's in-interval interpolation error into the running maximum."""
    predicted = (1.0 - weight) * node_table[..., :-1] + weight * node_table[..., 1:]
    err = np.abs(predicted - sample_table) / np.abs(sample_table)
    err = np.where(np.isfinite(err), err, 0.0)
    while err.ndim > 1:
        err = err.max(axis=0)
    if err.size == 0:
        return worst, worst_index
    local = float(err.max())
    if local > worst:
        return local, int(np.argmax(err))
    return worst, worst_index


_AUDIT_WEIGHTS = (0.25, 0.5, 0.75)


def _interp_audit(nodes, samples, n_el):
    """Max relative error of the linear interpolant inside each grid interval.

    For a smooth table the midpoint of the interval is the extremum of the
    linear-interpolation error, so a midpoint sample bounds it to leading order.
    A model join moves the extremum off centre, so the quarter points are
    sampled too. Joins are where the surviving error lives -- the
    Joy-Luo/Berger-Seltzer stopping crossover and the NIST Mott screening nodes
    are continuous but kinked, so the straddling interval drops from O(h^2) to
    O(h) and shows up here instead of passing unnoticed.

    ``cdf`` is deliberately excluded from the relative measure: it is a
    probability, judged by the absolute error, positivity and monotonicity its
    own tests assert, and it is never log-transformed.
    """
    total_rate_n, dEds_n, inv_beta_n, _cdf_n, alpha_n = nodes

    worst = 0.0
    worst_index = 0
    worst_weight = _AUDIT_WEIGHTS[0]
    for weight, sample in zip(_AUDIT_WEIGHTS, samples, strict=True):
        total_rate_s, dEds_s, inv_beta_s, _cdf_s, alpha_s = sample
        before = worst
        worst, worst_index = _scan_interp_error(
            total_rate_n, total_rate_s, weight, worst, worst_index
        )
        worst, worst_index = _scan_interp_error(dEds_n, dEds_s, weight, worst, worst_index)
        worst, worst_index = _scan_interp_error(
            inv_beta_n[None, :], inv_beta_s[None, :], weight, worst, worst_index
        )
        for L, n in enumerate(n_el):
            worst, worst_index = _scan_interp_error(
                alpha_n[L, : int(n)], alpha_s[L, : int(n)], weight, worst, worst_index
            )
        if worst > before:
            worst_weight = weight
    return worst, worst_index, worst_weight


def build_transport_energy_lut(
    E_min_keV,
    E_max_keV,
    elastic_model_code,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    mott_tables,
    config=DEFAULT_TRANSPORT_LUT_CONFIG,
):
    """Precompute all energy-dependent scalar transport physics for one run.

    Tables are material/layer specific. Runtime transport then needs only the
    closed-form log-grid index plus linear interpolation for the total elastic
    rate, stopping power, 1/beta, collision-element CDF and scattering screening
    parameter.

    Values are tabulated and interpolated directly; only the *coordinate* is
    logarithmic. Interpolating ``log`` values instead was measured and rejected:
    it costs an exponential per table read in the innermost loop on both
    backends, and at the resolution needed anyway the residual error is set by
    model joins, where a log transform buys nothing.
    """
    E_min_keV = float(E_min_keV)
    E_max_keV = float(E_max_keV)
    if not np.isfinite(E_min_keV) or not np.isfinite(E_max_keV):
        raise ValueError("transport LUT energy bounds must be finite")
    if E_min_keV <= 0.0:
        raise ValueError("transport LUT requires positive kinetic energies")
    if E_max_keV <= E_min_keV:
        E_max_keV = np.nextafter(E_min_keV, np.inf)

    intervals_per_decade = float(config.intervals_per_decade)
    if not np.isfinite(intervals_per_decade) or intervals_per_decade <= 0.0:
        raise ValueError("transport LUT intervals_per_decade must be finite and positive")
    rel_tol = float(config.rel_tol)
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("transport LUT rel_tol must be finite and non-negative")
    min_points = max(2, int(config.min_points))
    max_points = max(min_points, int(config.max_points))

    log_E_min = float(np.log(E_min_keV))
    log_E_max = float(np.log(E_max_keV))
    log_span = log_E_max - log_E_min
    decades = log_span / _LN10
    requested = int(np.ceil(decades * intervals_per_decade)) + 1
    n_energy = min(max_points, max(min_points, requested))

    if log_span / (n_energy - 1) < _MIN_DLOG_E:
        # A degenerate span asked for more nodes than the energy coordinate can
        # separate. Coarsen to the finest grid whose adjacent nodes stay
        # distinct rather than emit a table with repeated energies.
        n_energy = max(2, int(log_span / _MIN_DLOG_E) + 1)
    inv_dlogE = np.float64((n_energy - 1) / log_span)

    logE_grid = np.linspace(log_E_min, log_E_max, n_energy, dtype=np.float64)
    E_grid = np.exp(logE_grid)
    # Pin the endpoints so the clamp regime reads the exact requested bounds
    # rather than exp(log(E)) round-trip drift.
    E_grid[0] = E_min_keV
    E_grid[-1] = E_max_keV

    n_el = np.asarray([arr.size for arr in L_Zs], dtype=np.int32)
    max_el = max(arr.size for arr in L_Zs)

    physics = (
        elastic_model_code,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_E_cross,
        L_sr_rate_numer,
        L_mott_numer,
        L_mott_denom1,
        L_mott_denom2,
        L_sr_joy_numer,
        mott_tables,
        n_el,
        max_el,
    )
    nodes = _evaluate_transport_nodes(E_grid, *physics)
    total_rate, dEds, inv_beta, cdf, alpha = nodes

    achieved_ipd = (n_energy - 1) / decades if decades > 0.0 else float("inf")
    max_rel_interp_error = 0.0
    worst_energy_keV = E_min_keV
    tolerance_met = True
    if rel_tol > 0.0:
        sample_energies = [
            np.exp((1.0 - w) * logE_grid[:-1] + w * logE_grid[1:]) for w in _AUDIT_WEIGHTS
        ]
        samples = [_evaluate_transport_nodes(E, *physics) for E in sample_energies]
        max_rel_interp_error, worst_index, worst_weight = _interp_audit(nodes, samples, n_el)
        worst_energy_keV = float(sample_energies[_AUDIT_WEIGHTS.index(worst_weight)][worst_index])
        tolerance_met = max_rel_interp_error <= rel_tol
        if not tolerance_met:
            capped = n_energy < requested
            warnings.warn(
                "transport LUT tolerance unmet: measured max relative interpolation "
                f"error {max_rel_interp_error:.3e} > rel_tol {rel_tol:.3e} at "
                f"{worst_energy_keV:.6g} keV with {n_energy} points "
                f"({achieved_ipd:.1f} intervals/decade over "
                f"{E_min_keV:.6g}-{E_max_keV:.6g} keV)"
                + (
                    f"; max_points={max_points} capped the requested {requested} points"
                    if capped
                    else "; raise intervals_per_decade, or accept the O(h) error a "
                    "model join leaves in the interval that straddles it"
                ),
                TransportLUTToleranceWarning,
                stacklevel=2,
            )

    return TransportEnergyLUT(
        log_E_min=np.float64(log_E_min),
        inv_dlogE=inv_dlogE,
        n_energy=n_energy,
        n_el=n_el,
        total_rate=total_rate,
        dEds=dEds,
        inv_beta=inv_beta,
        cdf=cdf,
        alpha=alpha,
        E_min_keV=np.float64(E_min_keV),
        E_max_keV=np.float64(E_max_keV),
        intervals_per_decade=achieved_ipd,
        max_rel_interp_error=max_rel_interp_error,
        worst_energy_keV=worst_energy_keV,
        tolerance_met=tolerance_met,
    )
