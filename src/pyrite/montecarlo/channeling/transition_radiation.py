"""Transition-radiation bound for channeling-radiation spectra (ADR-0018, P3).

Single-boundary Ginzburg--Frank transition radiation of the beam crossing the
crystal faces, used only to bound its contribution under the channeling lines
(it is excluded from scoring).

Energies in eV, angles in radians. CPU only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.attenuation import plasma_energy_eV
from ...materials.crystal import ALPHA_FS, M_E_EV
from .dipole import _denominator
from .emission import KIND_LINE, ChannelingSpectrum


def transition_radiation(material, kinetic_energy_eV: float, photon_energy_eV, theta_rad):
    """Return single-boundary forward transition radiation [photons/eV/sr].

    Ginzburg--Frank radiation of a charge crossing a vacuum--medium boundary
    in the X-ray regime (``eps = 1 - (omega_p / omega)^2``), ultrarelativistic
    small-angle form (Jackson, *Classical Electrodynamics* 3rd ed., Eq. 13.91;
    Ter-Mikaelian 1972):

    ``d^2N / dE dOmega = (alpha / (pi^2 E)) theta^2
    [1 / (gamma^-2 + theta^2) - 1 / (gamma^-2 + theta^2 + (omega_p / omega)^2)]^2``.

    Assumptions: ``gamma >> 1``, ``theta << 1``, ``omega >> omega_p``
    (Drude medium from :func:`~pyrite.materials.attenuation.plasma_energy_eV`),
    no absorption. Limits: vanishes as ``omega_p -> 0`` (no boundary contrast)
    and as ``omega >> gamma omega_p`` (as ``(gamma omega_p / omega)^4``).

    Validation: cr-transition-radiation-bound
    """
    plasma = plasma_energy_eV(material)
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    energy = np.asarray(photon_energy_eV, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    vacuum = 1.0 / (gamma**-2 + theta**2)
    medium = 1.0 / (gamma**-2 + theta**2 + (plasma / energy) ** 2)
    return ALPHA_FS / (np.pi**2 * energy) * theta**2 * (vacuum - medium) ** 2


def transition_radiation_per_eV(material, kinetic_energy_eV: float, photon_energy_eV):
    """Return angle-integrated single-boundary transition radiation [photons/eV].

    ``dN/dE = (alpha / (pi E)) [(1 + 2 nu^2) ln(1 + 1/nu^2) - 2]`` with
    ``nu = omega / (gamma omega_p)`` (Jackson Eq. 13.92 per photon), the
    small-angle integral of :func:`transition_radiation`. Its energy integral
    is ``alpha gamma hbar omega_p / 3``.

    Validation: cr-transition-radiation-bound
    """
    plasma = plasma_energy_eV(material)
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    energy = np.asarray(photon_energy_eV, dtype=float)
    nu2 = (energy / (gamma * plasma)) ** 2
    return ALPHA_FS / (np.pi * energy) * ((1.0 + 2.0 * nu2) * np.log1p(1.0 / nu2) - 2.0)


@dataclass(frozen=True)
class TransitionRadiationBound:
    """Transition-radiation background against the channeling lines.

    Attributes
    ----------
    photons
        Two-boundary upper bound in the band [photons/electron]: four times
        one boundary, the maximum of the entrance--exit interference
        ``|1 - e^{i phi}|^2`` with no absorption of the entrance radiation.
    peak_density
        Two-boundary bound on ``d^2N/dE dOmega`` at each bound -> bound
        line's forward energy, maximized over ``0 <= theta <= theta_max``
        [photons/eV/sr].
    line_density
        Peak of the channeling spectrum (lines plus capture) at each bound ->
        bound line [photons/eV/sr].
    ratio
        ``peak_density / line_density`` per bound -> bound line.
    energy_eV
        Single-boundary emitted energy ``alpha gamma hbar omega_p / 3`` [eV].
    """

    photons: float
    peak_density: np.ndarray
    line_density: np.ndarray
    ratio: np.ndarray
    energy_eV: float


def transition_radiation_bound(
    spectrum: ChannelingSpectrum,
    material,
    kinetic_energy_eV: float,
    band_eV: tuple[float, float],
    *,
    theta_max_rad: float | None = None,
    n_band: int = 512,
) -> TransitionRadiationBound:
    """Compare the transition-radiation bound with a channeling spectrum.

    The TR density is maximized over ``0 <= theta <= theta_max`` (default
    ``3 / gamma``) at each line's forward energy, doubled for two
    boundaries and doubled again for constructive interference; the CR
    density is the maximum of ``spectrum.spectrum`` within one half width of
    each forward line energy over the spectrum's directions.

    Validation: cr-transition-radiation-bound
    """
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    plasma = plasma_energy_eV(material)
    theta_max = 3.0 / gamma if theta_max_rad is None else float(theta_max_rad)
    grid = np.geomspace(band_eV[0], band_eV[1], n_band)
    photons = 4.0 * float(
        np.trapezoid(transition_radiation_per_eV(material, kinetic_energy_eV, grid), grid)
    )
    angles = np.linspace(0.0, theta_max, 400)
    is_line = spectrum.lines.kind == KIND_LINE
    forward = spectrum.lines.forward_energy_eV()[is_line]
    peak = np.array(
        [4.0 * transition_radiation(material, kinetic_energy_eV, e, angles).max() for e in forward]
    )
    energies = spectrum.photon_energy_eV
    flat = spectrum.spectrum.reshape(-1, energies.size)
    step = float(np.min(np.diff(energies))) if energies.size > 1 else 0.0
    width = np.maximum(spectrum.lines.hwhm_eV[is_line] / _denominator(gamma, 0.0), step)
    line = np.array(
        [
            flat[:, np.abs(energies - e) <= w].max(initial=0.0)
            for e, w in zip(forward, width, strict=True)
        ]
    )
    with np.errstate(divide="ignore"):
        ratio = peak / line
    return TransitionRadiationBound(
        photons=photons,
        peak_density=peak,
        line_density=line,
        ratio=ratio,
        energy_eV=ALPHA_FS * gamma * plasma / 3.0,
    )
