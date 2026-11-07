"""Opt-in mixed (class II) elastic scattering with a random hinge (#894).

``transport_mode="mixed"`` keeps the detailed ELSEPA angular law and splits it
at a per-element cutoff quantile: events above the cutoff stay individual hard
collisions, and the rest are aggregated into one artificial deflection at a
uniformly placed hinge. The model and its evidence are in
``docs/validation/beam-transport/mixed-elastic-transport.md``.

The cutoff is frozen across each energy panel, so at every energy the soft and
hard parts partition exactly the distribution the detailed sampler draws from.
With ``C1 = 0`` nothing is soft, no hinge draws are consumed, and the core
reproduces detailed transport bit for bit.
"""

import numpy as np
from numba import njit

from ..._mixed_elastic_policy import (
    MIXED_URBAN_REFUSAL as MIXED_URBAN_REFUSAL,
)
from ..._mixed_elastic_policy import (
    MixedElasticParameters as MixedElasticParameters,
)
from ..._mixed_elastic_policy import (
    validate_mixed_parameters as validate_mixed_parameters,
)
from .core_geometry import _rotate_direction_scalar
from .hard_inelastic import _log_grid_frac
from .kinematics import _SM64_ONE, _stream_uniform_scalar
from .scattering import _elsepa_bracket

#: PENELOPE's limit on a step, in hard mean free paths (Sec. 4.4).
MIXED_MAX_STEP_HARD_MFP = 4.0
#: Default step limit as a fraction of the layer thickness. PENELOPE's tenth of
#: the track length (Sec. 4.4.1) shifts 1-3 MeV angular and backscatter tallies
#: by up to 10% under continuous stopping; a hundredth removes the shift (#894).
MIXED_MAX_STEP_LAYER_FRACTION = 0.01
#: Energy window of a step's hazard bounds, as a multiple of ``C2``. A step loses
#: ``C2 E`` at the start's stopping power; the stopping rises as the energy falls.
#: With ``S ~ 1/E`` the loss is ``E (1 - sqrt(1 - 2 C2))``, 0.225 E at ``C2 = 0.2``.
MIXED_STEP_LOSS_MARGIN = 1.5
#: What a mixed step ends in: a limit, or a proposal from one hard process.
STEP_END_LIMIT = 0
STEP_END_ELASTIC = 1
STEP_END_INELASTIC = 2
STEP_END_RADIATIVE = 3
#: Doubles per history in the CUDA kernel's ``mix_state``: remaining step, distance
#: to the hinge, full length, three hazard bounds and what ends the step.
MIX_STATE_WORDS = 7
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(6)


def layer_step_limits_ang(L_top, L_bot, max_step_um=None):
    """Per-layer limit on a mixed step [Angstrom].

    Validation: mixed-elastic-hinge
    """
    limits = MIXED_MAX_STEP_LAYER_FRACTION * (np.asarray(L_bot) - np.asarray(L_top))
    if max_step_um is not None:
        limits = np.minimum(limits, float(max_step_um) * 1e4)
    return np.asarray(limits, dtype=np.float64)


def _invert_cdf_row(cdf, pdf, mu, xi):
    """Vectorized :func:`.scattering._elsepa_invert_row` without the guide."""
    panel = np.clip(np.searchsorted(cdf, xi, side="right") - 1, 0, mu.size - 2)
    width = mu[panel + 1] - mu[panel]
    lower = pdf[panel]
    slope = pdf[panel + 1] - lower
    remainder = xi - cdf[panel]
    base = width * lower
    root = base + np.sqrt(np.maximum(base * base + 2.0 * width * slope * remainder, 0.0))
    fraction = np.divide(2.0 * remainder, root, out=np.zeros_like(root), where=root > 0.0)
    return mu[panel] + np.clip(fraction, 0.0, 1.0) * width


def _panel_moments(cdf_lo, pdf_lo, cdf_hi, pdf_hi, mu, xi_cut):
    """``(Q1_lo, Q1_hi, Q2_lo, Q_cross, Q2_hi)`` quantile integrals to ``xi_cut``.

    ``Q1 = int_0^xi_cut mu(xi) dxi`` on each bracketing node, the second moments
    likewise, and ``Q_cross`` the node-product integral the quantile-interpolated
    second moment needs. Gauss-Legendre on the union of both rows' CDF knots,
    where each inverse is smooth.

    Validation: mixed-elastic-soft-moments
    """
    if xi_cut <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    knots = np.concatenate([cdf_lo[cdf_lo < xi_cut], cdf_hi[cdf_hi < xi_cut], [0.0, xi_cut]])
    knots = np.unique(knots)
    left, width = knots[:-1], np.diff(knots)
    xi = left[:, None] + 0.5 * width[:, None] * (_GAUSS_NODES + 1.0)
    weight = 0.5 * width[:, None] * _GAUSS_WEIGHTS
    mu_lo = _invert_cdf_row(cdf_lo, pdf_lo, mu, xi.ravel()).reshape(xi.shape)
    mu_hi = _invert_cdf_row(cdf_hi, pdf_hi, mu, xi.ravel()).reshape(xi.shape)
    return (
        float(np.sum(weight * mu_lo)),
        float(np.sum(weight * mu_hi)),
        float(np.sum(weight * mu_lo * mu_lo)),
        float(np.sum(weight * mu_lo * mu_hi)),
        float(np.sum(weight * mu_hi * mu_hi)),
    )


def mean_angular_deflection(pdf, mu):
    """Exact ``<mu>`` of each row's piecewise-linear density."""
    width = np.diff(mu)
    return (
        np.sum(
            width
            * (pdf[:, :-1] * (2.0 * mu[:-1] + mu[1:]) + pdf[:, 1:] * (mu[:-1] + 2.0 * mu[1:])),
            axis=1,
        )
        / 6.0
    )


def mixed_elastic_tables(elsepa_group, c1):
    """Per-panel split tables for one packed ELSEPA group.

    Returns ``(hard_fraction, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi)``, indexed
    like the group's flat node rows; row ``k`` describes the panel from node
    ``k`` to ``k + 1`` of its element. The hard fraction comes from the lower
    node, whose first moment is usually the larger one (heavy elements below
    about 1 keV are the exception), so the mean deflection per hard step stays
    near or below ``c1`` across the panel:
    ``hard_fraction_k = min(1, 2 <mu>_k / c1)``.

    Validation: mixed-elastic-split
    Validation: mixed-elastic-soft-moments
    """
    has, start, length, _, _, cdf, pdf, mu, _ = elsepa_group
    n_rows = cdf.shape[0]
    hard_fraction = np.ones(n_rows, dtype=np.float64)
    moments = np.zeros((5, n_rows), dtype=np.float64)
    if c1 <= 0.0:
        return (hard_fraction, *moments)
    hard_fraction = np.minimum(1.0, 2.0 * mean_angular_deflection(pdf, mu) / c1)
    for begin, count in zip(start[has], length[has], strict=True):
        for row in range(int(begin), int(begin + count - 1)):
            moments[:, row] = _panel_moments(
                cdf[row], pdf[row], cdf[row + 1], pdf[row + 1], mu, 1.0 - hard_fraction[row]
            )
    return (hard_fraction, *moments)


def mixed_core_args(elsepa_group, parameters, L_top, L_bot):
    """The ``mixed_args`` tuple of the mixed CPU core."""
    return (
        float(parameters.c2),
        layer_step_limits_ang(L_top, L_bot, parameters.max_step_um),
        mixed_elastic_tables(elsepa_group, float(parameters.c1)),
    )


def mixed_device_args(core_args, xp, Ne):
    """``core_args`` as the CUDA kernel's device tables plus per-history state."""
    c2, limits, tables = core_args
    state = xp.full((int(Ne), MIX_STATE_WORDS), -1.0, dtype=xp.float64)
    return c2, xp.asarray(limits), tuple(xp.asarray(table) for table in tables), state


@njit(cache=True)
def _mixed_hard_rate_scalar(E_keV, logE_flat, log_rate_flat, hard_fraction, start, length):
    """Macroscopic hard elastic cross section [1/cm] of one element.

    Validation: mixed-elastic-split
    """
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    rate = np.exp(log_rate_flat[row] + f * (log_rate_flat[row + 1] - log_rate_flat[row]))
    return rate * hard_fraction[row]


@njit(cache=True)
def _hard_rate_bound_scalar(E_low, E_high, logE_flat, log_rate_flat, hard_fraction, start, length):
    """Upper bound [1/cm] of one element's hard rate on ``[E_low, E_high]``.

    The rate is log-linear inside a panel, where the hard fraction is constant,
    so its maximum lies at a window end or on either side of an interior node.

    Validation: mixed-elastic-split
    """
    bound = max(
        _mixed_hard_rate_scalar(E_low, logE_flat, log_rate_flat, hard_fraction, start, length),
        _mixed_hard_rate_scalar(E_high, logE_flat, log_rate_flat, hard_fraction, start, length),
    )
    row_low, _ = _elsepa_bracket(np.log(E_low * 1e3), logE_flat, start, length)
    row_high, _ = _elsepa_bracket(np.log(E_high * 1e3), logE_flat, start, length)
    for node in range(row_low + 1, row_high + 1):
        rate = np.exp(log_rate_flat[node]) * max(hard_fraction[node - 1], hard_fraction[node])
        if rate > bound:
            bound = rate
    return bound


@njit(cache=True)
def _layer_hard_rate_bound(
    E_low, E_high, L, n_el, logE_flat, log_rate_flat, hard_fraction, start, length
):
    """Upper bound [1/cm] of a layer's hard elastic rate; element bounds add.

    Validation: mixed-elastic-split
    """
    bound = 0.0
    for i_el in range(n_el):
        bound += _hard_rate_bound_scalar(
            E_low, E_high, logE_flat, log_rate_flat, hard_fraction,
            start[L, i_el], length[L, i_el],
        )  # fmt: skip
    return bound


@njit(cache=True)
def _grid_rate_bound(rate, log_grid, n, log_low, log_high):
    """Upper bound of a rate linear between the knots of ``log_grid``.

    Validation: mixed-elastic-hinge
    """
    low, frac_low = _log_grid_frac(log_grid, n, log_low)
    high, frac_high = _log_grid_frac(log_grid, n, log_high)
    bound = max(
        rate[low] + frac_low * (rate[low + 1] - rate[low]),
        rate[high] + frac_high * (rate[high + 1] - rate[high]),
    )
    for knot in range(low + 1, high + 1):
        if rate[knot] > bound:
            bound = rate[knot]
    return bound


@njit(cache=True)
def _mixed_hard_element(
    target, E_keV, L, n_el, logE_flat, log_rate_flat, hard_fraction, start, length
):
    """Element whose true hard rate at ``E_keV`` holds ``target``; -1 rejects.

    ``target`` is uniform on the step's elastic bound [1/cm], so an element is
    chosen with probability ``rate_i / bound`` (PENELOPE Eqs. 4.96-4.98).

    Validation: mixed-elastic-split
    """
    cumulative = 0.0
    for i_el in range(n_el):
        cumulative += _mixed_hard_rate_scalar(
            E_keV, logE_flat, log_rate_flat, hard_fraction, start[L, i_el], length[L, i_el]
        )
        if target < cumulative:
            return i_el
    return -1


@njit(cache=True)
def _plan_mixed_step(elastic, inelastic, radiative, bounds, limit):
    """Step length [Angstrom] and what ends it, from each clock's proposal distance.

    ``bounds`` holds the elastic [1/cm], inelastic and radiative [1/Angstrom]
    rate bounds; ``limit`` is the caller's layer and ``C2`` limit. A step is at
    most ``4 lambda_T^(h)`` (Sec. 4.4), and a limit wins only when strictly shorter.

    Validation: mixed-elastic-hinge
    """
    length, end = elastic, STEP_END_ELASTIC
    if inelastic < length:
        length, end = inelastic, STEP_END_INELASTIC
    if radiative < length:
        length, end = radiative, STEP_END_RADIATIVE
    hard_total = bounds[0] + (bounds[1] + bounds[2]) * 1e8
    cap = min(limit, MIXED_MAX_STEP_HARD_MFP * 1e8 / hard_total)
    if cap < length:
        return cap, STEP_END_LIMIT
    return length, end


@njit(cache=True)
def _mixed_hard_quantile_scalar(u, E_keV, logE_flat, hard_fraction, start, length):
    """Map a uniform onto the hard part ``(xi_c, 1)`` of the angular quantile.

    Validation: mixed-elastic-split
    """
    row, _ = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    return 1.0 - hard_fraction[row] * (1.0 - u)


@njit(cache=True)
def _mixed_soft_rates_scalar(
    E_keV, logE_flat, log_rate_flat, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi, start, length
):
    """Inverse soft transport mean free paths ``(1/lam1, 1/lam2)`` [1/cm].

    PENELOPE-2024 Eq. (4.38), with the moments of the detailed sampler's
    quantile-interpolated law at ``E``: ``mu(xi) = (1-f) mu_k(xi) + f mu_k+1(xi)``
    gives a first moment linear in ``f`` and a second moment quadratic in ``f``.

    Validation: mixed-elastic-soft-moments
    """
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    rate = np.exp(log_rate_flat[row] + f * (log_rate_flat[row + 1] - log_rate_flat[row]))
    g = 1.0 - f
    t1 = g * t1_lo[row] + f * t1_hi[row]
    t2 = g * g * t2_lo[row] + 2.0 * f * g * t2_cross[row] + f * f * t2_hi[row]
    return 2.0 * rate * t1, 6.0 * rate * (t1 - t2)


@njit(cache=True)
def _hinge_moments(step_cm, soft1_per_cm, soft2_per_cm):
    """``(<mu>, <mu^2>)`` of the soft deflection over a step, Eqs. (4.28)-(4.29).

    Validation: mixed-elastic-hinge
    """
    first = -0.5 * np.expm1(-step_cm * soft1_per_cm)
    second = first + np.expm1(-step_cm * soft2_per_cm) / 6.0
    return first, second


@njit(cache=True)
def _sample_hinge_mu(first, second, u):
    """Two-box artificial soft distribution, PENELOPE-2024 Eqs. (4.30)-(4.32).

    Validation: mixed-elastic-hinge
    """
    if first <= 0.0:
        return 0.0
    if second < 0.0:
        second = 0.0
    mu_0 = (2.0 * first - 3.0 * second) / (1.0 - 2.0 * first)
    if mu_0 < 0.0:
        mu_0 = 0.0
    a = 1.0 - 2.0 * first + mu_0
    if a > 1.0:
        a = 1.0
    if u < a:
        return mu_0 * u / a
    return mu_0 + (1.0 - mu_0) * (u - a) / (1.0 - a)


@njit(cache=True)
def _layer_soft_rates(E_keV, L, n_el, logE_flat, log_rate_flat, start, length, tables):
    """Soft inverse transport paths of a layer [1/cm]; element rates add.

    Validation: mixed-elastic-soft-moments
    """
    _, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi = tables
    soft_1 = soft_2 = 0.0
    for i_el in range(n_el):
        rate_1, rate_2 = _mixed_soft_rates_scalar(
            E_keV, logE_flat, log_rate_flat, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi,
            start[L, i_el], length[L, i_el],
        )  # fmt: skip
        soft_1 += rate_1
        soft_2 += rate_2
    return soft_1, soft_2


@njit(cache=True)
def _hinge_direction(
    dx, dy, dz, step_ang, E_keV, L, n_el, logE_flat, log_rate_flat, start, length, tables, key, draw
):
    """Direction after the hinge, from two draws at ``draw`` on the stream ``key``.

    Validation: mixed-elastic-hinge
    """
    soft_1, soft_2 = _layer_soft_rates(
        E_keV, L, n_el, logE_flat, log_rate_flat, start, length, tables
    )
    first, second = _hinge_moments(step_ang * 1e-8, soft_1, soft_2)
    mu = _sample_hinge_mu(first, second, _stream_uniform_scalar(key, draw))
    phi = 2.0 * np.pi * _stream_uniform_scalar(key, draw + _SM64_ONE)
    return _rotate_direction_scalar(dx, dy, dz, 1.0 - 2.0 * mu, phi)
