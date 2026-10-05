"""BremsLib photon angular shape against the Schiff formula (Koch-Motz 2BS).

Compares the angular distribution of the BremsLib double-differential
bremsstrahlung cross section with Schiff's small-angle, high-energy result
(Koch and Motz, Rev. Mod. Phys. 31, 920 (1959), Formula 2BS; Schiff, Phys. Rev.
70, 87 (1946)), at fixed incident energy ``T`` and reduced photon energy
``kappa = k/T``:

    dsigma/(dk dOmega) ~ y dy/dOmega * { 16 y^2 E / ((y^2+1)^4 E0)
        - (E0+E)^2 / ((y^2+1)^2 E0^2)
        + [(E0^2+E^2) / ((y^2+1)^2 E0^2) - 4 y^2 E / ((y^2+1)^4 E0)] ln M(y) },
    y = E0 theta,   1/M = (k/(2 E0 E))^2 + (Z^(1/3) / (111 (y^2+1)))^2,

with total energies ``E0 = 1 + T/mc^2`` before and ``E = E0 - k/mc^2`` after
emission in units of ``mc^2``. ``y dy`` is proportional to the solid angle, so
the bracket is the density per steradian. Only the shape is compared: both
densities are integrated over the same ``theta`` window with ``dOmega = 2 pi
sin(theta) dtheta``, and the comparison statistic is the enclosed-fraction
angle ``theta_p`` (the half-flux and 90 % angles).

Assumptions: Born approximation with Thomas-Fermi screening, no
electron-electron term, ``theta << 1``. The window is therefore cut at
``theta <= min(1 rad, 40/E0)``, and the formula is trusted where ``E0 >> 1``.
Limiting case: ``Z -> 0`` in ``M`` gives the unscreened Formula 2BN(a); the
peak angle scales as ``1/E0``.

Validation: bremslib-angular-schiff
"""

import numpy as np

from pyrite.montecarlo.spectrum.brem_bremslib import BremsLibBremsstrahlungTable

ELECTRON_REST_ENERGY_MEV = 0.51099895
QUANTILES = (0.5, 0.9)


def schiff_density(
    atomic_number: int, incident_energy_MeV: float, kappa: float, theta_rad: np.ndarray
) -> np.ndarray:
    """Unnormalized Schiff photon density per steradian (Koch-Motz 2BS).

    Validation: bremslib-angular-schiff
    """
    E0 = 1.0 + incident_energy_MeV / ELECTRON_REST_ENERGY_MEV
    k = kappa * incident_energy_MeV / ELECTRON_REST_ENERGY_MEV
    E = E0 - k
    y2 = (E0 * np.asarray(theta_rad, dtype=float)) ** 2
    d = y2 + 1.0
    inv_M = (k / (2.0 * E0 * E)) ** 2 + (atomic_number ** (1.0 / 3.0) / (111.0 * d)) ** 2
    return (
        16.0 * y2 * E / (d**4 * E0)
        - (E0 + E) ** 2 / (d**2 * E0**2)
        + ((E0**2 + E**2) / (d**2 * E0**2) - 4.0 * y2 * E / (d**4 * E0)) * -np.log(inv_M)
    )


def enclosed_angle(theta_rad: np.ndarray, density: np.ndarray, fraction: float) -> float:
    """Angle enclosing ``fraction`` of the flux, ``dOmega = 2 pi sin(theta) dtheta``.

    Validation: bremslib-angular-schiff
    """
    weight = 2.0 * np.pi * np.sin(theta_rad) * density
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(theta_rad))]
    )
    return float(np.interp(fraction, cumulative / cumulative[-1], theta_rad))


def angular_window_rad(incident_energy_MeV: float) -> float:
    """Upper ``theta`` bound where the small-angle formula is used."""
    return min(1.0, 40.0 / (1.0 + incident_energy_MeV / ELECTRON_REST_ENERGY_MEV))


def compare_angular_shape(
    table: BremsLibBremsstrahlungTable, incident_index: int, kappa_index: int
) -> dict[float, float]:
    """BremsLib over Schiff enclosed-angle ratios at one table node.

    Validation: bremslib-angular-schiff
    """
    energy_MeV = float(table.incident_energy_keV[incident_index]) / 1e3
    theta = table.theta_rad
    keep = theta <= angular_window_rad(energy_MeV)
    theta = theta[keep]
    ddcs = np.asarray(table.scaled_ddcs_mb_sr[incident_index, kappa_index], dtype=float)[keep]
    schiff = schiff_density(
        table.atomic_number, energy_MeV, float(table.nominal_reduced_energy[kappa_index]), theta
    )
    return {p: enclosed_angle(theta, ddcs, p) / enclosed_angle(theta, schiff, p) for p in QUANTILES}
