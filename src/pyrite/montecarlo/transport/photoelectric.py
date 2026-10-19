"""Standalone CPU EPDL photoelectric events and EADL relaxation (#341)."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ...materials.crystal import M_E_EV
from ...materials.photon_cross_sections import _atomic_number
from ...materials.photon_differential import (
    photoionization_subshells,
    subshell_photoionization_cross_sections_ang2,
)
from ..eadl_relaxation import EADLRelaxation
from .photon_sampling import (
    MAX_REJECTIONS,
    UniformDraws,
    cumulative_weights,
    incident_ray,
    scatter_direction,
    select_interval,
)
from .relaxation import SampledRelaxationCascade, sample_relaxation_cascade


@dataclass(frozen=True, slots=True)
class PhotoelectricProducts:
    """Photoelectron and relaxed primary hole; energies [eV], direction unitless.

    Incident energy equals photoelectron kinetic energy plus the relaxation
    budget's primary binding. The latter includes terminal binding and signed
    transition defects. Relaxation particles still need directions/queue handoff.
    """

    subshell_designator: int
    electron_energy_eV: float
    electron_direction: np.ndarray
    relaxation: SampledRelaxationCascade
    draws_used: int


def _sauter_delta(kinetic_eV: float, draws: UniformDraws) -> float:
    """Sauter–Gavrila photoelectron polar law, returned as 1-cos(theta).

    Source: Geant4 11.4 photoelectric final state (Gavrila 1959):
    p(mu) proportional to (1-mu²)/(1-beta*mu)^4 *
    [1+gamma*(gamma-1)*(gamma-2)*(1-beta*mu)/2].
    Let A=1/beta-1, C=beta*gamma*(gamma-1)*(gamma-2)/2 and
    w=delta/(A+delta). The transformed density is proportional to
    w*(2-delta)*(1-w+C*A); propose w=w_max*sqrt(u), w_max=2/(A+2),
    and accept (2-delta)*(1-w/(1+C*A))/2. A is computed without
    subtracting beta from one. At zero kinetic energy direction is immaterial
    and delta=0 consumes no draw; the positive-energy nonrelativistic limit
    is sin²theta. Unpolarized K-shell approximation applied to all shells;
    no Coulomb, multipole, or multiple-vacancy angular corrections.
    Validation: photon-photoelectric-sampling
    """
    if kinetic_eV == 0:
        return 0.0
    reduced = kinetic_eV / M_E_EV
    gamma = 1 + reduced
    beta = np.sqrt(reduced * (reduced + 2)) / gamma
    a = 1 / (beta * gamma**2 * (1 + beta))
    c = beta * gamma * reduced * (reduced - 1) / 2
    maximum = 2 / (a + 2)
    for _ in range(MAX_REJECTIONS):
        w = maximum * np.sqrt(next(draws))
        delta = float(np.clip(a * w / (1 - w), 0, 2))
        acceptance = (2 - delta) * (1 - w / (1 + c * a)) / 2
        if next(draws) < acceptance:
            return delta
    raise RuntimeError("photoelectron angular rejection budget exhausted")


def sample_photoelectric(
    element: str,
    energy_eV: float,
    direction: object,
    relaxation: EADLRelaxation,
    cutoff_eV: float,
    uniforms: Iterable[float],
) -> PhotoelectricProducts:
    """EPDL subshell absorption, Sauter photoelectron and sampled EADL tree.

    Source: EPDL2025 MF=23 MT534–599; ENDF-102 File 28 vacancy designators;
    Geant4 11.4 photoelectric final state for T=E-B and Sauter–Gavrila angles.
    Subshell i has probability sigma_i/sum(sigma_i); only positive rates at
    E>=B_i contribute. The atom has already been selected by channel competition.
    The supplied EADL element must match and its bindings must be accessible.
    Photoelectron energy plus the full sampled relaxation budget closes to E.
    Infinite-mass atom absorbs photoelectric momentum imbalance without energy
    recoil. No polarization, natural linewidths, transport or detector scoring.
    Draw order: subshell, angular proposals (two each), azimuth, EADL decays.
    At T=0 the photoelectron direction is incident and no angular draws are used.
    Zero total accessible photoionization rate fails instead of inventing a hole.
    Validation: photon-photoelectric-sampling
    """
    ray = incident_ray(energy_eV, direction)
    if _atomic_number(element) != relaxation.atomic_number:
        raise ValueError("photoelectric element does not match the EADL relaxation data")
    if not np.isfinite(cutoff_eV) or cutoff_eV <= 0:
        raise ValueError("relaxation cutoff must be finite and positive")
    designators, rates = subshell_photoionization_cross_sections_ang2(element, energy_eV)
    binding = {s.shell_designator: s.binding_energy_eV for s in relaxation.subshells}
    if any(d not in binding for d in designators):
        raise ValueError("EPDL subshell is absent from the EADL relaxation data")
    if any(
        s.binding_energy_eV != binding[s.designator] for s in photoionization_subshells(element)
    ):
        raise ValueError("EPDL and EADL binding energies must match")
    rates = np.where([energy_eV >= binding[d] for d in designators], rates, 0.0)
    cumulative = cumulative_weights(rates)
    draws = UniformDraws(uniforms)
    primary = designators[select_interval(cumulative, next(draws))]
    kinetic = energy_eV - binding[primary]
    if kinetic > 0:
        delta = _sauter_delta(kinetic, draws)
        electron, _ = scatter_direction(ray, delta, 2 * np.pi * next(draws))
    else:
        electron = ray.copy()
        electron.flags.writeable = False
    cascade = sample_relaxation_cascade(relaxation, primary, cutoff_eV, draws)
    return PhotoelectricProducts(primary, kinetic, electron, cascade, draws.count)
