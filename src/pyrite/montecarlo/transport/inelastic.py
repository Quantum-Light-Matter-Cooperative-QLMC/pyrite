"""SBETHE optical-response soft/hard inelastic cross sections.

This host-side model builds one positive transfer distribution per material,
incident energy and transfer cutoff. It does not itself alter trajectories.
``asymptotic.dat`` contains uncorrected free-atom moments; the production
``stp.dat`` collision stopping cross section is the mean-loss target.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, e, m_e, physical_constants

_MC2_EV = m_e * c * c / e
_RE_CM = 100.0 * physical_constants["classical electron radius"][0]
_PREF_CM2_EV = 2.0 * np.pi * _RE_CM**2 * _MC2_EV

_DISTANT_LONGITUDINAL = np.int8(0)
_DISTANT_TRANSVERSE = np.int8(1)
_CLOSE = np.int8(2)


@dataclass(frozen=True)
class GOSPartition:
    """One calibrated GOS transfer distribution, in per-molecule units."""

    energy_ev: float
    threshold_ev: float
    soft_stopping_ev_cm2: float
    hard_stopping_ev_cm2: float
    hard_sigma_cm2: float
    total_sigma_cm2: float
    raw_sigma0_cm2: float
    raw_sigma1_ev_cm2: float
    raw_sigma2_ev2_cm2: float
    calibration: float
    kind: np.ndarray
    lower_ev: np.ndarray
    upper_ev: np.ndarray
    density_lower_cm2_per_ev: np.ndarray
    density_upper_cm2_per_ev: np.ndarray
    cumulative_hard_cm2: np.ndarray
    core_edge_ev: float | None = None


@dataclass(frozen=True)
class HardCollision:
    """Sampled primary loss and recoil; secondary energy is an outer-shell proxy."""

    kind: str
    transfer_ev: float
    cos_primary: float
    secondary_energy_ev: float | None
    cos_secondary: float | None


def _oscillator_quadrature(arrays):
    """Integrate a piecewise-linear OOS density into resonances.

    Each interval's strength is its trapezoidal area and its resonance is the
    first-moment centroid. Zero-width shell-edge intervals contribute nothing.
    Source: SBETHE ``OOS.dat`` column definitions and linear quadrature.
    Assumption: an optical interval is represented by one resonance.
    Limit: a constant density on an interval has its resonance at the midpoint.
    Validation: gos-optical-quadrature
    """
    w = np.asarray(arrays["oos_energy_eV"], dtype=np.float64)
    f = np.asarray(arrays["oos_per_eV"], dtype=np.float64)
    if w.ndim != 1 or w.size < 2 or f.shape != w.shape:
        raise ValueError("SBETHE OOS energy and strength must be matching vectors")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(f)):
        raise ValueError("SBETHE OOS contains non-finite values")
    if w[0] <= 0.0 or np.any(np.diff(w) < 0.0) or np.any(f < 0.0):
        raise ValueError("SBETHE OOS energies and strengths must be non-negative and ordered")
    dw = np.diff(w)
    strength = 0.5 * (f[:-1] + f[1:]) * dw
    moment = dw * ((2.0 * w[:-1] + w[1:]) * f[:-1] + (w[:-1] + 2.0 * w[1:]) * f[1:]) / 6.0
    keep = strength > 0.0
    if not np.any(keep):
        raise ValueError("SBETHE OOS has no positive oscillator strength")
    return moment[keep] / strength[keep], strength[keep]


def _qmin_ev(energy_ev, transfer_ev):
    """Stable minimum recoil energy from PENELOPE longitudinal kinematics.

    Source: Geant4 Physics Reference Manual, Penelope ionisation, Eq. 127.
    Assumption: free relativistic projectile electron, 0 < W <= E.
    Limit: Qmin tends to zero quadratically as W tends to zero.
    Validation: gos-distant-response
    """
    p0 = np.sqrt(energy_ev * (energy_ev + 2.0 * _MC2_EV))
    remaining = energy_ev - transfer_ev
    p1 = np.sqrt(remaining * (remaining + 2.0 * _MC2_EV))
    dp = transfer_ev * (2.0 * (energy_ev + _MC2_EV) - transfer_ev) / (p0 + p1)
    return dp * dp / (np.sqrt(dp * dp + _MC2_EV**2) + _MC2_EV)


def _moller_factor(energy_ev, transfer_ev):
    """Møller close-collision factor for an electron projectile.

    Source: Geant4 Physics Reference Manual, Penelope ionisation, Eqs. 129, 133.
    Assumption: indistinguishable outgoing electrons, 0 < W <= E/2.
    Limit: tends to one as W/E tends to zero.
    Validation: gos-moller-close
    """
    ratio = transfer_ev / (energy_ev - transfer_ev)
    rel = energy_ev / (energy_ev + _MC2_EV)
    return 1.0 + ratio * ratio - ratio + rel * rel * (ratio + (transfer_ev / energy_ev) ** 2)


def _linear_moments(lower, upper, y0, y1):
    """Zeroth, first and second moments of a linear nonnegative density.

    Source: exact polynomial integration on each transfer bin.
    Assumption: the differential cross section is linear within the bin.
    Limit: constant density gives m0 = density times bin width.
    Validation: gos-soft-hard-partition
    """
    width = upper - lower
    dy = y1 - y0
    m0 = width * (y0 + 0.5 * dy)
    m1 = width * (lower * (y0 + 0.5 * dy) + width * (0.5 * y0 + dy / 3.0))
    m2 = width * (
        lower**2 * (y0 + 0.5 * dy)
        + 2.0 * lower * width * (0.5 * y0 + dy / 3.0)
        + width**2 * (y0 / 3.0 + dy / 4.0)
    )
    return m0, m1, m2


def build_gos_partition(
    arrays,
    energy_ev: float,
    threshold_ev: float,
    *,
    core_edge_ev: float | None = None,
) -> GOSPartition:
    """Partition an OOS-derived GOS spectrum at one transfer threshold.

    Distant longitudinal/transverse resonances and the close Møller continuum
    follow Geant4's simplified PENELOPE-like GOS model (Eqs. 127–133). This is
    not the PENELOPE-2024 shell model: there, bound-shell close collisions begin
    at binding energy U, while resonances are W >= U (Secs. 3.2.1–3.2.2).
    Optical intervals stand in for shell oscillators; binding energies and a
    finite-q material response are unavailable. The transverse density
    correction is omitted in the raw shape. One common positive multiplier
    matches the corrected ``stp.dat`` first moment for both soft and hard
    pieces. ``asymptotic.dat`` moments are uncorrected comparison data only.

    ``core_edge_ev`` selects only oscillators at or above a duplicated OOS
    shell edge and leaves their GOS spectrum uncalibrated. This raw core-only
    diagnostic must not be used as a complete material loss spectrum.
    Validation: gos-core-edge

    Units: E, W in eV; cross sections per molecule in cm² and eV cm².
    Limit: Wc >= E makes every interaction soft and the hard rate zero.
    Validation: gos-soft-hard-partition

    Validation: gos-distant-response
    Validation: gos-moller-close
    """
    e0 = np.asarray(arrays["stopping_energy_eV"], dtype=np.float64)
    s0 = np.asarray(arrays["stopping_cs_eV_cm2"], dtype=np.float64)
    if e0.ndim != 1 or s0.shape != e0.shape or e0.size < 2:
        raise ValueError("SBETHE stopping table must have matching vectors")
    if (
        not np.all(np.isfinite(e0))
        or not np.all(np.isfinite(s0))
        or np.any(np.diff(e0) <= 0.0)
        or np.any(e0 <= 0.0)
        or np.any(s0 <= 0.0)
    ):
        raise ValueError("SBETHE stopping table must be ordered and positive")
    if not np.isfinite(energy_ev) or energy_ev < e0[0] or energy_ev > e0[-1]:
        raise ValueError("incident energy is outside the SBETHE stopping table")
    if not np.isfinite(threshold_ev) or threshold_ev <= 0.0:
        raise ValueError("inelastic transfer threshold must be finite and positive")
    target = float(np.exp(np.interp(np.log(energy_ev), np.log(e0), np.log(s0))))
    all_w, all_f = _oscillator_quadrature(arrays)
    if core_edge_ev is None:
        osc_w, osc_f = all_w, all_f
    else:
        native_w = np.asarray(arrays["oos_energy_eV"], dtype=np.float64)
        native_f = np.asarray(arrays["oos_per_eV"], dtype=np.float64)
        if not np.isfinite(core_edge_ev) or not np.any(
            (native_w[:-1] == core_edge_ev)
            & (native_w[1:] == core_edge_ev)
            & (native_f[1:] > native_f[:-1])
        ):
            raise ValueError("core edge must be a positive OOS shell-edge jump")
        core = all_w >= core_edge_ev
        osc_w, osc_f = all_w[core], all_f[core]
        if osc_w.size == 0:
            raise ValueError("OOS core edge has no positive oscillator strength")
    beta2 = energy_ev * (energy_ev + 2.0 * _MC2_EV) / (energy_ev + _MC2_EV) ** 2
    pref = _PREF_CM2_EV / beta2

    accessible = osc_w <= energy_ev
    w = osc_w[accessible]
    f = osc_f[accessible]
    if w.size == 0:
        raise ValueError("no SBETHE oscillator is accessible at this energy")
    qmin = _qmin_ev(energy_ev, w)
    longitudinal = np.maximum(
        np.log((w / qmin) * ((qmin + 2.0 * _MC2_EV) / (w + 2.0 * _MC2_EV))), 0.0
    )
    transverse = np.log1p(energy_ev * (energy_ev + 2.0 * _MC2_EV) / _MC2_EV**2) - beta2
    distant_l = pref * f / w * longitudinal
    distant_t = pref * f / w * max(transverse, 0.0)

    close_max = energy_ev / 2.0
    close_lower = np.empty(0)
    close_upper = np.empty(0)
    close_y0 = np.empty(0)
    close_y1 = np.empty(0)
    close_m0 = close_m1 = close_m2 = np.empty(0)
    if close_max > all_w[0]:
        grid = np.concatenate(
            (
                np.geomspace(all_w[0], close_max, 1200),
                all_w[(all_w >= all_w[0]) & (all_w <= close_max)],
                [close_max],
            )
        )
        grid = np.unique(grid)
        close_lower, close_upper = grid[:-1], grid[1:]
        midpoint = 0.5 * (close_lower + close_upper)
        active_count = np.searchsorted(osc_w, midpoint, side="right")
        cumulative = np.concatenate(([0.0], np.cumsum(osc_f)))
        active = cumulative[active_count]
        close_y0 = pref * active * _moller_factor(energy_ev, close_lower) / close_lower**2
        close_y1 = pref * active * _moller_factor(energy_ev, close_upper) / close_upper**2
        close_m0, close_m1, close_m2 = _linear_moments(close_lower, close_upper, close_y0, close_y1)

    distant = distant_l + distant_t
    raw0 = float(np.sum(distant) + np.sum(close_m0))
    raw1 = float(np.dot(distant, w) + np.sum(close_m1))
    raw2 = float(np.dot(distant, w * w) + np.sum(close_m2))
    if not np.isfinite(raw1) or raw1 <= 0.0:
        raise ValueError("GOS spectrum has no finite positive stopping moment")
    scale = target / raw1 if core_edge_ev is None else 1.0
    if core_edge_ev is not None:
        target = raw1
    # A resonance exactly on the cutoff belongs to the soft side. This also
    # makes Wc=E a zero-hard-rate limit even when a resonance sits at E.
    hard_distant = w > threshold_ev
    # Build the raw spectrum once per energy. Clipping a linear bin at Wc
    # preserves that spectrum; inserting Wc into the quadrature grid would
    # change its interpolant and therefore its total rate and calibration.
    hard_close = close_upper > threshold_ev
    hard_lower = np.maximum(close_lower[hard_close], threshold_ev)
    hard_upper = close_upper[hard_close]
    hard_y0 = close_y0[hard_close] + (
        (close_y1[hard_close] - close_y0[hard_close])
        * (hard_lower - close_lower[hard_close])
        / (close_upper[hard_close] - close_lower[hard_close])
    )
    hard_y1 = close_y1[hard_close]
    hard_m0, hard_m1, _ = _linear_moments(hard_lower, hard_upper, hard_y0, hard_y1)
    hard_stop = scale * float(np.dot(distant[hard_distant], w[hard_distant]) + np.sum(hard_m1))
    soft_stop = target - hard_stop

    kinds = np.concatenate(
        (
            np.full(np.count_nonzero(hard_distant), _DISTANT_LONGITUDINAL, dtype=np.int8),
            np.full(np.count_nonzero(hard_distant), _DISTANT_TRANSVERSE, dtype=np.int8),
            np.full(np.count_nonzero(hard_close), _CLOSE, dtype=np.int8),
        )
    )
    lower = np.concatenate((w[hard_distant], w[hard_distant], hard_lower))
    upper = np.concatenate((w[hard_distant], w[hard_distant], hard_upper))
    y0 = np.concatenate((np.zeros(2 * np.count_nonzero(hard_distant)), hard_y0 * scale))
    y1 = np.concatenate((np.zeros(2 * np.count_nonzero(hard_distant)), hard_y1 * scale))
    weights = scale * np.concatenate((distant_l[hard_distant], distant_t[hard_distant], hard_m0))
    positive = weights > 0.0
    cumulative_hard = np.cumsum(weights[positive])
    return GOSPartition(
        energy_ev=float(energy_ev),
        threshold_ev=float(threshold_ev),
        soft_stopping_ev_cm2=float(soft_stop),
        hard_stopping_ev_cm2=float(hard_stop),
        hard_sigma_cm2=float(cumulative_hard[-1]) if cumulative_hard.size else 0.0,
        total_sigma_cm2=float(scale * raw0),
        raw_sigma0_cm2=raw0,
        raw_sigma1_ev_cm2=raw1,
        raw_sigma2_ev2_cm2=raw2,
        calibration=float(scale),
        kind=kinds[positive],
        lower_ev=lower[positive],
        upper_ev=upper[positive],
        density_lower_cm2_per_ev=y0[positive],
        density_upper_cm2_per_ev=y1[positive],
        cumulative_hard_cm2=cumulative_hard,
        core_edge_ev=core_edge_ev,
    )


def sample_hard_collision(
    partition: GOSPartition,
    u_event: float,
    u_transfer: float,
    u_recoil: float,
    *,
    production_threshold_ev: float = 0.0,
) -> HardCollision:
    """Sample a hard GOS branch, transferred energy, and primary polar recoil.

    Source: Geant4 Penelope ionisation manual, Eqs. 54–65 on that page.
    Assumption: the full-OOS close branch treats the optical oscillator as an
    outer shell; its secondary kinetic energy equals W and carries no vacancy
    metadata. Core-only diagnostics suppress that secondary proxy.
    Limit: transverse distant interactions leave direction unchanged.
    Validation: gos-hard-recoil
    """
    if partition.hard_sigma_cm2 <= 0.0:
        raise ValueError("partition has no hard collisions")
    if not np.isfinite(production_threshold_ev) or production_threshold_ev < 0.0:
        raise ValueError("secondary production threshold must be finite and non-negative")
    if not all(np.isfinite(u) and 0.0 <= u < 1.0 for u in (u_event, u_transfer, u_recoil)):
        raise ValueError("sample uniforms must be finite and in [0, 1)")
    index = min(
        int(np.searchsorted(partition.cumulative_hard_cm2, u_event * partition.hard_sigma_cm2)),
        partition.kind.size - 1,
    )
    kind = partition.kind[index]
    lower = partition.lower_ev[index]
    upper = partition.upper_ev[index]
    if kind == _CLOSE:
        y0 = partition.density_lower_cm2_per_ev[index]
        y1 = partition.density_upper_cm2_per_ev[index]
        width = upper - lower
        area = width * 0.5 * (y0 + y1)
        c = u_transfer * area / width
        discriminant = max(y0 * y0 + 2.0 * (y1 - y0) * c, 0.0)
        denominator = y0 + np.sqrt(discriminant)
        t = 2.0 * c / denominator if denominator > 0.0 else 0.0
        transfer = float(lower + width * min(max(t, 0.0), 1.0))
        e = partition.energy_ev
        cos_theta = np.sqrt(
            (e - transfer) / e * (e + 2.0 * _MC2_EV) / (e - transfer + 2.0 * _MC2_EV)
        )
        # The core-only diagnostic has no binding/subshell state. Treating W
        # as free-secondary kinetic energy would create an unphysical product.
        if partition.core_edge_ev is None and transfer >= production_threshold_ev:
            beta2 = e * (e + 2.0 * _MC2_EV) / (e + _MC2_EV) ** 2
            cos_secondary_sq = (
                transfer
                / (beta2 * (transfer + 2.0 * _MC2_EV))
                * ((e + 2.0 * _MC2_EV) / (e + _MC2_EV)) ** 2
            )
            secondary_energy = transfer
            cos_secondary = float(np.sqrt(np.clip(cos_secondary_sq, 0.0, 1.0)))
        else:
            secondary_energy = None
            cos_secondary = None
        return HardCollision("close", transfer, float(cos_theta), secondary_energy, cos_secondary)
    if kind == _DISTANT_TRANSVERSE:
        return HardCollision("distant_transverse", float(lower), 1.0, None, None)
    transfer = float(lower)
    qmin = float(_qmin_ev(partition.energy_ev, transfer))
    log_min = np.log(qmin / (qmin + 2.0 * _MC2_EV))
    log_max = np.log(transfer / (transfer + 2.0 * _MC2_EV))
    ratio = np.exp((1.0 - u_recoil) * log_min + u_recoil * log_max)
    q = 2.0 * _MC2_EV * ratio / (1.0 - ratio)
    e = partition.energy_ev
    p0_sq = e * (e + 2.0 * _MC2_EV)
    p1_sq = (e - transfer) * (e - transfer + 2.0 * _MC2_EV)
    cos_theta = (p0_sq + p1_sq - q * (q + 2.0 * _MC2_EV)) / (2.0 * np.sqrt(p0_sq * p1_sq))
    return HardCollision(
        "distant_longitudinal", transfer, float(np.clip(cos_theta, -1.0, 1.0)), None, None
    )


def hard_transfer_cdf(partition: GOSPartition, transfers_ev) -> np.ndarray:
    """CDF of the hard-event transfer distribution at requested energies.

    Delta resonances contribute a step at their transfer energy. Each close
    bin contributes the exact integral of its stored linear density. This
    gives a deterministic spectrum comparator independent of the random
    branch and inverse-CDF path in ``sample_hard_collision``.

    Source: exact integral of the GOS partition's delta and linear terms.
    Assumption: ``partition`` represents one fixed incident energy and Wc.
    Limit: CDF is zero below Wc and one at or above the incident energy.
    Validation: gos-soft-hard-partition
    """
    if partition.hard_sigma_cm2 <= 0.0:
        raise ValueError("partition has no hard collisions")
    transfer = np.asarray(transfers_ev, dtype=np.float64)
    if not np.all(np.isfinite(transfer)):
        raise ValueError("transfer evaluation energies must be finite")
    flat = np.atleast_1d(transfer).reshape(-1)
    mass = np.diff(np.concatenate(([0.0], partition.cumulative_hard_cm2)))
    delta = partition.lower_ev == partition.upper_ev
    cumulative = np.zeros(flat.size, dtype=np.float64)
    for i in np.flatnonzero(delta):
        cumulative += mass[i] * (flat >= partition.lower_ev[i])
    for i in np.flatnonzero(~delta):
        lower = partition.lower_ev[i]
        upper = partition.upper_ev[i]
        width = upper - lower
        distance = np.clip(flat - lower, 0.0, width)
        y0 = partition.density_lower_cm2_per_ev[i]
        dy = partition.density_upper_cm2_per_ev[i] - y0
        cumulative += y0 * distance + 0.5 * dy * distance * distance / width
    return np.clip(cumulative / partition.hard_sigma_cm2, 0.0, 1.0).reshape(transfer.shape)
