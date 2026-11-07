"""Device functions of mixed elastic transport in the exact CUDA kernel (#894).

Ports of the :mod:`.mixed_elastic` host functions. Device functions return one
value, so the host's tuples split here: soft rates come one moment at a time,
and the kernel keeps a step's fields in its per-history ``mix_state`` row.

The lookups are FP32 twins in ``transport_precision="mixed"``
(:data:`._jit_precision.MIXED_HELPERS`), like ``_rad_layer_moment``. The rest
stay FP64 on purpose. The soft moment tables are not narrowed, because the
hinge's second moment is a difference of the two exponentials, which cancels to
``s rate T2``, and ``T2 / T1`` is small enough at high energy for FP32 rounding
of ``T1`` to reach it.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__``.

Validation: mixed-elastic-split
Validation: mixed-elastic-soft-moments
Validation: mixed-elastic-hinge
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import F64_HALF, F64_ONE, F64_TWO, F64_ZERO, I32_ONE, I32_ZERO
from ._jit_shell_device import _log_grid_fraction, _log_grid_lower
from .mixed_elastic import MIX_STATE_WORDS as MIX_STATE_WORDS
from .mixed_elastic import (
    MIXED_MAX_STEP_HARD_MFP,
    MIXED_STEP_LOSS_MARGIN,
    STEP_END_ELASTIC,
    STEP_END_INELASTIC,
    STEP_END_LIMIT,
    STEP_END_RADIATIVE,
)

# Fields of one history's ``mix_state`` row. Global like position and energy, so
# a continuation yield stores nothing extra for them.
I32_MIX_STATE_WORDS = np.int32(MIX_STATE_WORDS)
I32_MIX_LEFT = np.int32(0)
I32_MIX_HINGE = np.int32(1)
I32_MIX_LENGTH = np.int32(2)
I32_MIX_BOUND_ELASTIC = np.int32(3)
I32_MIX_BOUND_INELASTIC = np.int32(4)
I32_MIX_BOUND_RADIATIVE = np.int32(5)
I32_MIX_END = np.int32(6)
I32_END_LIMIT = np.int32(STEP_END_LIMIT)
I32_END_ELASTIC = np.int32(STEP_END_ELASTIC)
I32_END_INELASTIC = np.int32(STEP_END_INELASTIC)
I32_END_RADIATIVE = np.int32(STEP_END_RADIATIVE)
I32_NO_ELEMENT = np.int32(-1)
F64_MIX_LOSS_MARGIN = np.float64(MIXED_STEP_LOSS_MARGIN)
F64_ANG_PER_CM = np.float64(1e8)
F64_CM_PER_ANG = np.float64(1e-8)
F64_MAX_STEP_MFP = np.float64(MIXED_MAX_STEP_HARD_MFP)
F64_THREE = np.float64(3.0)
F64_SIX = np.float64(6.0)


@jit.rawkernel(device=True)
def _mixed_hard_rate(log_E_eV, el_logE, el_log_rate, mix_hf, row, count):
    """One element's hard rate [1/cm]; host ``_mixed_hard_rate_scalar``.

    Validation: mixed-elastic-split
    """
    lower = _log_grid_lower(el_logE, row, count, log_E_eV)
    fraction = _log_grid_fraction(el_logE, row, count, log_E_eV, lower)
    log_rate = el_log_rate[row + lower]
    return (
        xp.exp(log_rate + fraction * (el_log_rate[row + lower + I32_ONE] - log_rate))
        * (mix_hf[row + lower])
    )


@jit.rawkernel(device=True)
def _mixed_layer_hard_bound(
    log_E_low, log_E_high, el_logE, el_log_rate, mix_hf, el_start, el_len, row_offset, n_elements
):
    """Upper bound [1/cm] of a layer's hard rate; host ``_layer_hard_rate_bound``.

    Each element's rate is log-linear inside a panel, where the hard fraction is
    constant, so its maximum is at a window end or on either side of a node.

    Validation: mixed-elastic-split
    """
    bound = F64_ZERO
    element = I32_ZERO
    while element < n_elements:
        row = el_start[row_offset + element]
        count = el_len[row_offset + element]
        best = xp.maximum(
            _mixed_hard_rate(log_E_low, el_logE, el_log_rate, mix_hf, row, count),
            _mixed_hard_rate(log_E_high, el_logE, el_log_rate, mix_hf, row, count),
        )
        node = row + _log_grid_lower(el_logE, row, count, log_E_low) + I32_ONE
        last = row + _log_grid_lower(el_logE, row, count, log_E_high)
        while node <= last:
            fraction = (
                mix_hf[node] if mix_hf[node] > mix_hf[node - I32_ONE] else mix_hf[node - I32_ONE]
            )
            rate = xp.exp(el_log_rate[node]) * fraction
            if rate > best:
                best = rate
            node += I32_ONE
        bound += best
        element += I32_ONE
    return bound


@jit.rawkernel(device=True)
def _mixed_hard_element(
    target, log_E_eV, el_logE, el_log_rate, mix_hf, el_start, el_len, row_offset, n_elements
):
    """Element whose true hard rate holds ``target``, or -1 to reject the proposal.

    Validation: mixed-elastic-split
    """
    cumulative = F64_ZERO
    element = I32_ZERO
    picked = I32_NO_ELEMENT
    while element < n_elements and picked < I32_ZERO:
        cumulative += _mixed_hard_rate(
            log_E_eV,
            el_logE,
            el_log_rate,
            mix_hf,
            el_start[row_offset + element],
            el_len[row_offset + element],
        )
        if target < cumulative:
            picked = element
        element += I32_ONE
    return picked


@jit.rawkernel(device=True)
def _mixed_has_soft_part(log_E_eV, el_logE, mix_hf, el_start, el_len, row_offset, n_elements):
    """1 when any element has a soft part at this energy, else 0.

    The host tests the soft first moment. A panel's soft moments are zero
    exactly when its hard fraction is one, so the two tests agree, and this one
    needs no rate or moment lookup.

    Validation: mixed-elastic-split
    """
    found = I32_ZERO
    element = I32_ZERO
    while element < n_elements and found == I32_ZERO:
        row = el_start[row_offset + element]
        lower = _log_grid_lower(el_logE, row, el_len[row_offset + element], log_E_eV)
        if mix_hf[row + lower] < F64_ONE:
            found = I32_ONE
        element += I32_ONE
    return found


@jit.rawkernel(device=True)
def _mixed_grid_rate_bound(rate, row, low, fraction_low, high, fraction_high):
    """Upper bound of a rate linear between knots; host ``_grid_rate_bound``.

    Validation: mixed-elastic-hinge
    """
    bound = xp.maximum(
        rate[row + low] + fraction_low * (rate[row + low + I32_ONE] - rate[row + low]),
        rate[row + high] + fraction_high * (rate[row + high + I32_ONE] - rate[row + high]),
    )
    knot = low + I32_ONE
    while knot <= high:
        if rate[row + knot] > bound:
            bound = rate[row + knot]
        knot += I32_ONE
    return bound


@jit.rawkernel(device=True)
def _mixed_step_cap(limit_ang, bound_elastic, bound_inelastic, bound_radiative):
    """Longest step [Angstrom]: the limit, or ``4 lambda_T^(h)`` (Sec. 4.4).

    Validation: mixed-elastic-hinge
    """
    hard_total = bound_elastic + (bound_inelastic + bound_radiative) * F64_ANG_PER_CM
    return xp.minimum(limit_ang, F64_MAX_STEP_MFP * F64_ANG_PER_CM / hard_total)


@jit.rawkernel(device=True)
def _mixed_step_length(elastic, inelastic, radiative, cap):
    """Step length [Angstrom]; host ``_plan_mixed_step``.

    Validation: mixed-elastic-hinge
    """
    return xp.minimum(xp.minimum(xp.minimum(elastic, inelastic), radiative), cap)


@jit.rawkernel(device=True)
def _mixed_step_end(elastic, inelastic, radiative, cap):
    """What ends the step, as stored in ``mix_state``; a limit wins only if shorter.

    Validation: mixed-elastic-hinge
    """
    length = elastic
    end = I32_END_ELASTIC
    if inelastic < length:
        length = inelastic
        end = I32_END_INELASTIC
    if radiative < length:
        length = radiative
        end = I32_END_RADIATIVE
    if cap < length:
        end = I32_END_LIMIT
    return np.float64(end)


@jit.rawkernel(device=True)
def _mixed_hinge_mu(
    step_ang, log_E_eV, u, el_logE, el_log_rate, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi,
    el_start, el_len, row_offset, n_elements,
):  # fmt: skip
    """Hinge ``mu`` for a step of ``step_ang``, at the hinge's energy ``log_E_eV``.

    Soft rates (PENELOPE-2024 Eq. 4.38) at the energy reached at the hinge, then
    the moments (4.28)-(4.29) and the two-box law (4.30)-(4.32). Host
    ``_layer_soft_rates``, ``_hinge_moments`` and ``_sample_hinge_mu``. All FP64,
    reading FP32 knots in the mixed build; see the module docstring.

    Validation: mixed-elastic-soft-moments
    Validation: mixed-elastic-hinge
    """
    soft_first = F64_ZERO
    soft_second = F64_ZERO
    element = I32_ZERO
    while element < n_elements:
        row = el_start[row_offset + element]
        count = el_len[row_offset + element]
        panel = row + _log_grid_lower(el_logE, row, count, log_E_eV)
        f = F64_ZERO
        if log_E_eV >= el_logE[row + count - I32_ONE]:
            f = F64_ONE
        elif log_E_eV > el_logE[row]:
            f = (log_E_eV - el_logE[panel]) / np.float64(el_logE[panel + I32_ONE] - el_logE[panel])
        g = F64_ONE - f
        rate = xp.exp(
            np.float64(el_log_rate[panel]) + f * (el_log_rate[panel + I32_ONE] - el_log_rate[panel])
        )
        t1 = g * t1_lo[panel] + f * t1_hi[panel]
        t2 = g * g * t2_lo[panel] + F64_TWO * f * g * t2_cross[panel] + f * f * t2_hi[panel]
        soft_first += F64_TWO * rate * t1
        soft_second += F64_SIX * rate * (t1 - t2)
        element += I32_ONE
    step_cm = step_ang * F64_CM_PER_ANG
    first = -F64_HALF * xp.expm1(-step_cm * soft_first)
    second = first + xp.expm1(-step_cm * soft_second) / F64_SIX
    if first <= F64_ZERO:
        return F64_ZERO
    if second < F64_ZERO:
        second = F64_ZERO
    mu_0 = (F64_TWO * first - F64_THREE * second) / (F64_ONE - F64_TWO * first)
    if mu_0 < F64_ZERO:
        mu_0 = F64_ZERO
    a = F64_ONE - F64_TWO * first + mu_0
    if a > F64_ONE:
        a = F64_ONE
    if u < a:
        return mu_0 * u / a
    return mu_0 + (F64_ONE - mu_0) * (u - a) / (F64_ONE - a)
