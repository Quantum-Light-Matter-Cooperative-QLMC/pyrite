"""Standalone CPU unpolarized Compton/Rayleigh final states (#341).

Integrated EPDL rates remain the owner of channel/atom competition. These
conditional samplers have no production callers and own no RNG or tracking.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ...materials.crystal import HC_EV_ANG, M_E_EV
from ...materials.photon_differential import (
    coherent_form_factor_table,
    incoherent_scattering_function_table,
)
from .photon_sampling import (
    MAX_REJECTIONS,
    UniformDraws,
    clipped_factor_table,
    cumulative_weights,
    incident_ray,
    scatter_direction,
    select_interval,
)


@dataclass(frozen=True, slots=True)
class PhotonScatter:
    """One outgoing photon and optional recoil electron, eV and unit vectors.

    ``one_minus_cosine`` retains small-angle precision independently of dot
    products. Zero-energy Compton recoil has the incident direction by convention;
    Rayleigh has no electron direction. ``draws_used`` includes rejected proposals.
    """

    photon_energy_eV: float
    photon_direction: np.ndarray
    electron_energy_eV: float
    electron_direction: np.ndarray | None
    one_minus_cosine: float
    draws_used: int


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


def sample_compton(
    element: str, energy_eV: float, direction: object, uniforms: Iterable[float]
) -> PhotonScatter:
    """Sample Klein–Nishina times EPDL S(x,Z), with free-electron recoil.

    Source: Geant4 Physics Reference Manual 11.4, standard/Livermore Compton
    final states (Butcher–Messel; Cullen 1995): p(epsilon) proportional to
    (1/epsilon+epsilon-sin²theta) S(x,Z), epsilon=1/(1+alpha(1-mu)),
    alpha=E/(m_e c²), x=E/(hc) sqrt((1-mu)/2). Binding and Doppler broadening
    are omitted; no atomic vacancy is produced in this approximation.
    S_max is the largest knot/endpoint value on the accessible x interval, so
    the rejection bound does not assume monotonicity. Each proposal consumes
    mixture, energy, acceptance draws, followed by one uniform azimuth on
    acceptance. Exhaustion raises; no forced acceptance or fallback occurs.
    E'=E/(1+alpha*delta); T=E alpha*delta/(1+alpha*delta); recoil momentum is
    incident minus outgoing photon momentum. S constant recovers free KN.
    Validation: photon-compton-scattering
    """
    ray = incident_ray(energy_eV, direction)
    draws = UniformDraws(uniforms)
    alpha, limit = energy_eV / M_E_EV, energy_eV / HC_EV_ANG
    grid, factor = clipped_factor_table(incoherent_scattering_function_table(element), limit)
    maximum = float(np.max(factor))
    if maximum <= 0:
        raise ValueError("incoherent scattering function has no positive support")
    for _ in range(MAX_REJECTIONS):
        epsilon, delta = _compton_proposal(alpha, draws)
        x = limit * np.sqrt(delta / 2)
        acceptance = 1 - epsilon * delta * (2 - delta) / (1 + epsilon**2)
        acceptance *= np.interp(x, grid, factor) / maximum
        if next(draws) < acceptance:
            outgoing, transverse = scatter_direction(ray, delta, 2 * np.pi * next(draws))
            denominator = 1 + alpha * delta
            photon_energy = energy_eV / denominator
            kinetic = energy_eV * (alpha * delta / denominator)
            momentum = (kinetic + photon_energy * delta) * ray
            momentum -= photon_energy * np.sqrt(delta * (2 - delta)) * transverse
            norm = np.linalg.norm(momentum)
            electron = momentum / norm if norm > 0 else ray.copy()
            electron.flags.writeable = False
            return PhotonScatter(photon_energy, outgoing, kinetic, electron, delta, draws.count)
    raise RuntimeError("Compton rejection budget exhausted")
