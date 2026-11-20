"""CPU unpolarized Compton/Rayleigh final states (#341, #1231).

Integrated EPDL rates remain the owner of channel/atom competition. These
conditional samplers own no RNG or tracking; the opt-in photon history queue
(``photon_histories``) is their production caller.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ...materials.crystal import ALPHA_FS, HC_EV_ANG, M_E_EV
from ...materials.photon_differential import (
    coherent_form_factor_table,
    incoherent_scattering_function_table,
)
from .compton_profiles import ComptonShells, compton_shells
from .photon_sampling import (
    MAX_REJECTIONS,
    UniformDraws,
    clipped_factor_table,
    cumulative_weights,
    incident_ray,
    scatter_direction,
    select_interval,
)

#: PENELOPE analytical-profile constants d1 = 1/sqrt(2), d2 = sqrt(2).
_D1, _D2 = math.sqrt(0.5), math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class PhotonScatter:
    """One outgoing photon and optional recoil electron, eV and unit vectors.

    ``one_minus_cosine`` retains small-angle precision independently of dot
    products. A Compton electron moves along the momentum transfer; with zero
    transfer it keeps the incident direction by convention. Rayleigh has no
    electron direction. ``draws_used`` includes rejected proposals. A Compton
    event leaves one vacancy in EADL subshell ``subshell_designator`` whose
    binding energy ``binding_energy_eV`` closes the energy balance; Rayleigh
    leaves none (designator 0).
    """

    photon_energy_eV: float
    photon_direction: np.ndarray
    electron_energy_eV: float
    electron_direction: np.ndarray | None
    one_minus_cosine: float
    draws_used: int
    binding_energy_eV: float = 0.0
    subshell_designator: int = 0


def sample_rayleigh(
    element: str, energy_eV: float, direction: object, uniforms: Iterable[float]
) -> PhotonScatter:
    """Sample p(mu) proportional to (1+mu²) F(x,Z)², x=E/(hc) sqrt((1-mu)/2).

    Source: Geant4 Physics Reference Manual 11.4, Livermore Rayleigh model,
    final-state angular law (Hubbell 1979; Cullen 1995); EPDL2025 MF=27 F.
    Exact rejection for the stored lin-lin F: within each knot interval use
    max(F_left,F_right)² as an envelope in x², then accept F²/envelope times
    (1+mu²)/2. Interval weights include their x² widths (the angular Jacobian).
    Proposals consume interval, within-interval, acceptance draws; the accepted
    event then consumes one uniform azimuth. Rejection-cap exhaustion raises.
    Assumes an unpolarized isolated atom, no anomalous amplitudes or crystal
    coherence. Infinite-mass atomic recoil: photon energy is unchanged. With
    constant F the distribution is Thomson; with F=0 no conditional event exists.
    Validation: photon-rayleigh-scattering
    """
    ray = incident_ray(energy_eV, direction)
    draws = UniformDraws(uniforms)
    limit = energy_eV / HC_EV_ANG
    grid, factor = clipped_factor_table(coherent_form_factor_table(element), limit)
    low, high = grid[:-1], grid[1:]
    square_width = (high - low) * (high + low)
    ceiling = np.maximum(factor[:-1], factor[1:]) ** 2
    cumulative = cumulative_weights(square_width * ceiling)
    for _ in range(MAX_REJECTIONS):
        slot = select_interval(cumulative, next(draws))
        x = np.sqrt(low[slot] ** 2 + next(draws) * square_width[slot])
        delta = float(np.clip(2 * (x / limit) ** 2, 0, 2))
        form = np.interp(x, grid, factor)
        acceptance = (form**2 / ceiling[slot]) * (1 + (1 - delta) ** 2) / 2
        if next(draws) < acceptance:
            outgoing, _ = scatter_direction(ray, delta, 2 * np.pi * next(draws))
            return PhotonScatter(energy_eV, outgoing, 0.0, None, delta, draws.count)
    raise RuntimeError("Rayleigh rejection budget exhausted")


def _compton_proposal(alpha: float, draws: UniformDraws) -> tuple[float, float]:
    """Butcher–Messel energy-ratio mixture; return epsilon and 1-cos(theta).

    Source: Geant4 11.4 standard Compton final state: proposal 1/epsilon+epsilon,
    epsilon in [1/(1+2alpha),1]. log1p/expm1 avoid cancellation at low alpha.
    Validation: photon-compton-scattering
    """
    logarithm = np.log1p(2 * alpha)
    epsilon0 = 1 / (1 + 2 * alpha)
    width = -np.expm1(-2 * logarithm)
    branch, u = next(draws), next(draws)
    if branch < logarithm / (logarithm + width / 2):
        exponent = logarithm * u
        epsilon = np.exp(-exponent)
        delta = np.expm1(exponent) / alpha
    else:
        epsilon = np.sqrt(epsilon0**2 + width * u)
        delta = width * (1 - u) / ((1 + epsilon) * epsilon * alpha)
    return float(epsilon), float(np.clip(delta, 0, 2))


def _profile_cdf(j0: float, momentum: float) -> float:
    """Cumulative analytical one-electron Compton profile n_i(p_z), p_z [a.u.].

    Source: PENELOPE-2024 (NEA/MBDAV/R(2024)1) §2.3.1, Eqs. 2.54–2.55 (Brusa
    et al. 1996) with n=2: J_i(p)=J0 d2 (d1+d2 J0|p|) exp[d1²-(d1+d2 J0|p|)²],
    d1=1/sqrt(2), d2=sqrt(2), normalized to one electron with J_i(0)=J0.
    Limits: n_i(0)=1/2, n_i(-inf)=0, n_i(+inf)=1.
    Validation: photon-compton-scattering
    """
    if momentum < 0:
        a = _D1 - _D2 * j0 * momentum
        return 0.5 * math.exp(0.5 - a * a)
    a = _D1 + _D2 * j0 * momentum
    return 1 - 0.5 * math.exp(0.5 - a * a)


def _profile_inverse(j0: float, level: float) -> float:
    """Inverse of :func:`_profile_cdf` [a.u.] (PENELOPE-2024 Eq. 2.56).

    Level 0 gives -inf and level 1 gives +inf; callers clip to their interval.
    Validation: photon-compton-scattering
    """
    if level < 0.5:
        if level <= 0:
            return -math.inf
        return (_D1 - math.sqrt(0.5 - math.log(2 * level))) / (_D2 * j0)
    tail = 1 - level
    if tail <= 0:
        return math.inf
    return (math.sqrt(0.5 - math.log(2 * tail)) - _D1) / (_D2 * j0)


def _projected_momentum(tau: float, tau_c: float, delta: float) -> float:
    """p_z/(m_e c) of the struck electron for E'/E=tau at 1-cos(theta)=delta.

    PENELOPE-2024 Eq. 2.31: p_z/(m_e c) = E(E'-E_C)/(E_C cq), i.e.
    (tau - tau_c)/(tau_c sqrt((1-tau)² + 2 tau delta)), tau_c=E_C/E;
    increasing in tau on [0, 1], with p_z(0) = -m_e c.
    Validation: photon-compton-scattering
    """
    rest = 1 - tau
    return (tau - tau_c) / (tau_c * math.sqrt(rest * rest + 2 * tau * delta))


def _doppler_ratio(momentum: float, tau_c: float, delta: float, tau_max: float) -> float:
    """Solve p_z(tau)=momentum [m_e c] for tau=E'/E in [0, tau_max].

    PENELOPE-2024 Eq. 2.70: squaring Eq. 2.31 gives A tau² - 2B tau + C = 0,
    A=1-t tau_c², B=tau_c(1-t tau_c cos(theta)), C=tau_c²(1-t), t=p_z².
    Both roots, in the cancellation-free forms C/q and q/A with
    q=B+sign(B)sqrt(B²-AC), are clipped to the physical interval and the one
    reproducing the signed momentum is kept (the other solves -p_z). This
    equals Eq. 2.70 where its sign(p_z) root is real and also covers
    t tau_c² >= 1. Limit: momentum 0 gives tau_c.
    Validation: photon-compton-scattering
    """
    t = momentum * momentum
    a = 1 - t * tau_c * tau_c
    b = tau_c * (1 - t * tau_c * (1 - delta))
    c = tau_c * tau_c * (1 - t)
    q = b + math.copysign(math.sqrt(max(b * b - a * c, 0.0)), b)
    best, miss = min(tau_c, tau_max), math.inf
    for root in (c / q if q != 0 else math.nan, q / a if a != 0 else math.nan):
        if math.isfinite(root):
            root = min(max(root, 0.0), tau_max)
            error = abs(_projected_momentum(root, tau_c, delta) - momentum)
            if error < miss:
                best, miss = root, error
    return best


def _shell_support(
    energy_eV: float, alpha: float, delta: float, shells: ComptonShells
) -> tuple[list[float], np.ndarray] | None:
    """Per-shell p_i,max [a.u.] and cumulative proposal weights, or None.

    Source: PENELOPE-2024 Eqs. 2.43 and 2.47: shell i is accessible when
    E > U_i, with weight f_i [n_i(p_i,max) - n_i(-m_e c)]; p_i,max = p_z at
    E'=E-U_i (binding cutoff) and p_z = -m_e c at E'=0. None means the
    impulse approximation has no final state at this angle (e.g. theta=0, or
    weights below double-precision range).
    Validation: photon-compton-scattering
    """
    tau_c = 1 / (1 + alpha * delta)
    lowest = -1 / ALPHA_FS
    limits, weights = [], []
    for electrons, binding, j0 in zip(
        shells.electrons, shells.binding_energy_eV, shells.profile_j0_au, strict=True
    ):
        fraction = float(binding) / energy_eV
        scale = tau_c * math.sqrt(fraction * fraction + 2 * (1 - fraction) * delta)
        highest = (alpha * delta * tau_c - fraction) / scale / ALPHA_FS if scale > 0 else lowest
        limits.append(highest)
        weights.append(
            float(electrons) * (_profile_cdf(j0, highest) - _profile_cdf(j0, lowest))
            if fraction < 1 and highest > lowest
            else 0.0
        )
    cumulative = np.cumsum(weights)
    return (limits, cumulative) if cumulative[-1] > 0 else None


def _doppler_energy(
    energy_eV: float,
    alpha: float,
    delta: float,
    shells: ComptonShells,
    support: tuple[list[float], np.ndarray],
    draws: UniformDraws,
) -> tuple[float, int]:
    """Impulse-approximation scattered energy [eV] and vacancy shell index.

    Source: PENELOPE-2024 §2.3, Eq. 2.40 and the §2.3.2 final-state algorithm
    (Brusa et al. 1996). Given theta, the DCS in p_z is proportional to
    F(p_z) sum_i f_i Theta(E-E'-U_i) J_i(p_z). Shell i is proposed with the
    :func:`_shell_support` weight and p_z from the truncated n_i (Eq. 2.56),
    which equals PENELOPE's rejection of p_z < -m_e c. The pair is accepted
    when xi F_max < F(p_z), Eq. 2.38: F = 1 + (cq_C/E)[1 + E_C(E_C-E
    cos(theta))/(cq_C)²] p_z/(m_e c), with p_z clamped to ±0.2 m_e c and
    F >= 0; otherwise shell and p_z are redrawn. E' follows from
    :func:`_doppler_ratio`. Draws per trial: shell, momentum, acceptance.
    Assumes independent electrons, one vacancy and theta already sampled;
    PENELOPE's E > 5 MeV switch to the unbroadened Compton line is not applied.
    Limit: J0 -> inf and U -> 0 give E'=E/(1+alpha*delta).
    Rejection exhaustion raises RuntimeError.
    Validation: photon-compton-scattering
    """
    limits, cumulative = support
    tau_c = 1 / (1 + alpha * delta)
    lowest = -1 / ALPHA_FS
    transfer = math.sqrt((alpha * delta * tau_c) * (alpha * delta * tau_c) + 2 * tau_c * delta)
    slope = transfer + tau_c * delta * (1 - alpha * tau_c) / transfer
    ceiling = 1 + 0.2 * abs(slope)
    for _ in range(MAX_REJECTIONS):
        shell = select_interval(cumulative, next(draws))
        j0, highest = float(shells.profile_j0_au[shell]), limits[shell]
        low, high = _profile_cdf(j0, lowest), _profile_cdf(j0, highest)
        momentum = _profile_inverse(j0, low + next(draws) * (high - low))
        momentum = min(max(momentum, lowest), highest) * ALPHA_FS
        factor = max(1 + slope * min(max(momentum, -0.2), 0.2), 0.0)
        if next(draws) * ceiling < factor:
            tau_max = 1 - float(shells.binding_energy_eV[shell]) / energy_eV
            return energy_eV * _doppler_ratio(momentum, tau_c, delta, tau_max), shell
    raise RuntimeError("Compton Doppler rejection budget exhausted")


def sample_compton(
    element: str, energy_eV: float, direction: object, uniforms: Iterable[float]
) -> PhotonScatter:
    """Sample Klein–Nishina times EPDL S(x,Z) angles with Doppler-broadened energy.

    Angles: Geant4 Physics Reference Manual 11.4, standard/Livermore Compton
    final states (Butcher–Messel; Cullen 1995): p(epsilon) proportional to
    (1/epsilon+epsilon-sin²theta) S(x,Z), epsilon=1/(1+alpha(1-mu)),
    alpha=E/(m_e c²), x=E/(hc) sqrt((1-mu)/2). S_max is the largest
    knot/endpoint value on the accessible x interval, so the rejection bound
    does not assume monotonicity. Each proposal consumes mixture, energy,
    acceptance draws. An accepted angle at which no subshell can be ionized
    (:func:`_shell_support` is None) is rejected as well, so the angular law
    is KN x S(x,Z) restricted to the impulse-approximation support. Then one
    uniform azimuth follows. Exhaustion raises; no forced acceptance or
    fallback occurs. If even backscatter (largest momentum transfer) has no
    support -- E below every EADL binding, or weights that underflow at tens
    of eV -- ValueError is raised before any draw.

    Energy: impulse approximation (PENELOPE-2024 §2.3, Eqs. 2.40–2.56) on
    occupied EADL2025 subshells with Biggs et al. (1975) J_i(0) analytical
    profiles; see ``_doppler_energy`` (three draws per trial, after the
    azimuth). The vacancy's binding U_i is returned, T=E-E'-U_i, and the
    electron moves along the momentum transfer q=E n-E' n' (PENELOPE-2024
    Eq. 2.72). The atom absorbs the momentum imbalance. The angular marginal
    keeps the EPDL Waller–Hartree S(x,Z), not the impulse-approximation
    S(E,theta) of Eq. 2.47; no polarization or multiple vacancies.
    Limit: J_i(0) -> inf, U_i -> 0 recovers free Compton kinematics,
    E'=E/(1+alpha*delta), T=E-E'.
    Validation: photon-compton-scattering
    """
    ray = incident_ray(energy_eV, direction)
    draws = UniformDraws(uniforms)
    alpha, limit = energy_eV / M_E_EV, energy_eV / HC_EV_ANG
    grid, factor = clipped_factor_table(incoherent_scattering_function_table(element), limit)
    shells = compton_shells(element)
    maximum = float(np.max(factor))
    if maximum <= 0:
        raise ValueError("incoherent scattering function has no positive support")
    if _shell_support(energy_eV, alpha, 2.0, shells) is None:
        raise ValueError("no Compton subshell is accessible at this energy")
    for _ in range(MAX_REJECTIONS):
        epsilon, delta = _compton_proposal(alpha, draws)
        x = limit * np.sqrt(delta / 2)
        acceptance = 1 - epsilon * delta * (2 - delta) / (1 + epsilon**2)
        acceptance *= np.interp(x, grid, factor) / maximum
        if next(draws) >= acceptance:
            continue
        support = _shell_support(energy_eV, alpha, delta, shells)
        if support is not None:
            outgoing, transverse = scatter_direction(ray, delta, 2 * np.pi * next(draws))
            photon_energy, shell = _doppler_energy(energy_eV, alpha, delta, shells, support, draws)
            binding = float(shells.binding_energy_eV[shell])
            kinetic = max(energy_eV - photon_energy - binding, 0.0)
            momentum = (energy_eV - photon_energy * (1 - delta)) * ray
            momentum -= photon_energy * np.sqrt(delta * (2 - delta)) * transverse
            norm = np.linalg.norm(momentum)
            electron = momentum / norm if norm > 0 else ray.copy()
            electron.flags.writeable = False
            return PhotonScatter(
                photon_energy,
                outgoing,
                kinetic,
                electron,
                delta,
                draws.count,
                binding,
                int(shells.designators[shell]),
            )
    raise RuntimeError("Compton rejection budget exhausted")
