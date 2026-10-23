"""Ter--Mikaelian dielectric-suppression estimates for bremsstrahlung.

The reference factor and a fixed-trajectory *omission bound* for runs that do
not apply it (it excludes feedback through electron slowing down). Coupled
BremsLib transport applies the same factor through
``transport.dielectric_radiative`` and the BremsLib scorers.

Validation: dielectric-suppression
"""

from collections.abc import Sequence

import numpy as np

from ...materials import MediumSpec
from ...materials.attenuation import plasma_energy_eV

_ELECTRON_REST_EV = 510998.95


def ter_mikaelian_factor(
    electron_kinetic_eV: object, photon_energy_eV: object, plasma_energy_eV: object
) -> np.ndarray | np.float64:
    r"""Return the leading dielectric factor ``k² / (k² + (gamma*Ep)²)``.

    Here ``k`` and ``Ep = hbar*omega_p`` are in eV and
    ``gamma = 1 + T / (m_e*c²)`` uses the *kinetic* electron energy, not beta.
    The inputs broadcast following NumPy rules. Photon energies must be
    positive, kinetic energies nonnegative, and plasma energies nonnegative.
    A zero plasma energy gives unity, the vacuum limit.

    This approximate high-energy factor does not include LPM suppression,
    surface effects, or a full dielectric-dependent angular distribution.
    It must not be multiplied into the coupled tables without recomputing the
    hard-event sampler and the soft/hard stopping moments from the same DCS.

    Validation: dielectric-suppression
    """
    kinetic = np.asarray(electron_kinetic_eV, dtype=float)
    photon = np.asarray(photon_energy_eV, dtype=float)
    plasma = np.asarray(plasma_energy_eV, dtype=float)
    if not np.all(np.isfinite(kinetic)) or np.any(kinetic < 0.0):
        raise ValueError("electron_kinetic_eV must be finite and nonnegative")
    if not np.all(np.isfinite(photon)) or np.any(photon <= 0.0):
        raise ValueError("photon_energy_eV must be finite and positive")
    if not np.all(np.isfinite(plasma)) or np.any(plasma < 0.0):
        raise ValueError("plasma_energy_eV must be finite and nonnegative")
    gamma = 1.0 + kinetic / _ELECTRON_REST_EV
    # hypot avoids intermediate squares overflowing/underflowing. Its ratio
    # is in [0, 1] even for strongly suppressed photons.
    return np.square(photon / np.hypot(photon, gamma * plasma))


def dielectric_omission_bound(
    max_electron_kinetic_eV: float,
    min_photon_energy_eV: float,
    materials: Sequence[str | MediumSpec],
) -> float:
    """Worst-case fractional *unsuppressed* brem yield omitted by the model.

    For photons with ``k >= min_photon_energy_eV`` emitted by electrons with
    ``0 <= T <= max_electron_kinetic_eV``, take the largest plasma energy of
    the homogeneous emitting media. Since the factor rises with ``k`` and
    falls with ``T`` and ``Ep``, ``1 - factor(Tmax, kmin, Epmax)`` is an upper
    bound on the *fractional* overestimate of unsuppressed bremsstrahlung
    yield integrated with arbitrary nonnegative fixed-trajectory weights.

    ``materials`` must enumerate ALL emitting media (film and any substrate),
    not just the entrance crystal. The same bound applies after nonnegative
    escape/detector weighting. It does NOT bound errors to electron tracks or
    other radiative components when dielectric suppression alters transport.
    A bound above an application's error budget means a production correction
    or additional study is needed, not that the physical error attains it.

    Validation: dielectric-suppression
    """
    if not materials:
        raise ValueError("materials must include at least one emitting medium")
    plasmas = [plasma_energy_eV(material) for material in materials]
    factor = ter_mikaelian_factor(max_electron_kinetic_eV, min_photon_energy_eV, max(plasmas))
    return float(1.0 - factor)
