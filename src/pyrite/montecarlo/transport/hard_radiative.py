"""Host-side BremsLib soft/hard radiative partition and photon sampling.

Validation: bremslib-radiative-partition
"""

from dataclasses import dataclass

import numpy as np

from ..._backend import BACKEND
from ..spectrum.brem_bremslib import (
    BremsLibBremsstrahlungTable,
    bremslib_segment_state,
    evaluate_bremslib,
    stage_bremslib_table,
)

_MB_CM2 = 1.0e-27
_ELECTRON_REST_EV = 510998.95


def _cell_moments(
    left: float, right: float, chi_left: float, chi_right: float
) -> tuple[float, float]:
    """Integrate ``chi(k)/k`` and ``chi(k)`` for a linear chi cell."""
    if right <= left:
        return 0.0, 0.0
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    rate = intercept * np.log(right / left) + slope * (right - left)
    first = 0.5 * (chi_left + chi_right) * (right - left)
    return float(rate), float(first)


@dataclass(frozen=True, slots=True)
class RadiativePartition:
    """One fixed-energy, fixed-cutoff BremsLib partition for one atom.

    Cross sections are per atom. ``soft_stopping_cs_eV_cm2`` is a first
    moment, while ``hard_rate_cs_cm2`` is a zeroth moment. The caller multiplies
    by atomic number density for loss per length or collision frequency.
    """

    table: BremsLibBremsstrahlungTable
    incident_energy_eV: float
    cutoff_eV: float
    photon_grid_eV: np.ndarray
    scaled_sdcs_mb: np.ndarray
    hard_cell_cs_cm2: np.ndarray
    soft_stopping_cs_eV_cm2: float
    hard_rate_cs_cm2: float
    hard_stopping_cs_eV_cm2: float
    total_stopping_cs_eV_cm2: float

    def sample_photon_energy(self, uniform: float) -> float:
        """Sample a hard photon from the exact piecewise-linear SDCS CDF."""
        if self.hard_rate_cs_cm2 <= 0.0:
            raise ValueError("partition has no hard photons")
        if not 0.0 <= uniform < 1.0:
            raise ValueError("uniform must be in [0, 1)")
        masses = self.hard_cell_cs_cm2
        target = uniform * self.hard_rate_cs_cm2
        cumulative = np.cumsum(masses)
        cell = min(int(np.searchsorted(cumulative, target, side="right")), masses.size - 1)
        target -= float(cumulative[cell - 1]) if cell else 0.0
        left, right = self.photon_grid_eV[cell : cell + 2]
        chi_left, chi_right = self.scaled_sdcs_mb[cell : cell + 2]
        scale = _MB_CM2 * self.table.atomic_number**2
        low, high = float(left), float(right)
        for _ in range(52):
            mid = 0.5 * (low + high)
            chi_mid = chi_left + (chi_right - chi_left) * (mid - left) / (right - left)
            partial, _ = _cell_moments(float(left), mid, float(chi_left), float(chi_mid))
            if scale * partial < target:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class HardRadiativePhoton:
    energy_eV: float
    direction: np.ndarray
    electron_energy_eV: float
    electron_direction: np.ndarray
    target_momentum_eV_c: np.ndarray


def build_radiative_partition(
    table: BremsLibBremsstrahlungTable, incident_energy_eV: float, cutoff_eV: float
) -> RadiativePartition:
    """Split one BremsLib SDCS at positive ``kc`` without altering its shape.

    The BremsLib scaled SDCS is linearly interpolated in reduced photon energy.
    Integrating each resulting linear cell analytically avoids a quadrature
    dependence on ``kc``. The zero-energy zeroth moment is divergent, so only
    the soft *first* moment is formed below ``kc``.

    Validation: bremslib-radiative-partition
    """
    energy = float(incident_energy_eV)
    cutoff = float(cutoff_eV)
    if not np.isfinite(energy) or not np.isfinite(cutoff) or energy <= 0.0 or cutoff <= 0.0:
        raise ValueError("incident energy and radiative cutoff must be positive and finite")
    if not table.minimum_incident_energy_keV <= energy / 1.0e3 <= table.maximum_incident_energy_keV:
        raise ValueError("incident energy is outside the BremsLib table")
    if cutoff > energy:
        raise ValueError("radiative cutoff cannot exceed incident energy")

    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(staged, [energy / 1.0e3])
    row = int(BACKEND.to_cpu(state.lower_row)[0])
    tops = np.asarray(table.top_reduced_energy, dtype=float)
    nodes = np.asarray(table.nominal_reduced_energy[:-1], dtype=float)
    reduced = np.unique(np.r_[0.0, nodes, tops[row], tops[row + 1], cutoff / energy, 1.0])
    reduced = reduced[(reduced >= 0.0) & (reduced <= 1.0)]
    photon_grid = energy * reduced
    positive = photon_grid > 0.0
    sdcs = np.zeros_like(photon_grid)
    sdcs[positive] = BACKEND.to_cpu(evaluate_bremslib(staged, state, photon_grid[positive]))[0]
    chi = sdcs * photon_grid / (_MB_CM2 * table.atomic_number**2)
    fraction = float(BACKEND.to_cpu(state.energy_fraction)[0])
    lower = float(BACKEND.to_cpu(state.lower_values)[0, 0])
    upper = float(BACKEND.to_cpu(state.upper_values)[0, 0])
    chi[0] = lower + fraction * (upper - lower)
    if np.any(~np.isfinite(chi)) or np.any(chi < 0.0):
        raise ValueError("BremsLib SDCS is not finite and nonnegative")

    scale = _MB_CM2 * table.atomic_number**2
    soft = hard_first = 0.0
    hard_cells = np.zeros(photon_grid.size - 1, dtype=float)
    for i, (left, right) in enumerate(zip(photon_grid[:-1], photon_grid[1:], strict=True)):
        if left == 0.0:
            first = 0.5 * (chi[i] + chi[i + 1]) * right
            soft += scale * first
            continue
        rate, first = _cell_moments(float(left), float(right), float(chi[i]), float(chi[i + 1]))
        if right <= cutoff:
            soft += scale * first
        else:
            hard_cells[i] = scale * rate
            hard_first += scale * first
    hard_rate = float(hard_cells.sum())
    return RadiativePartition(
        table,
        energy,
        cutoff,
        photon_grid,
        chi,
        hard_cells,
        soft,
        hard_rate,
        hard_first,
        soft + hard_first,
    )


def sample_hard_radiative_photon(
    partition: RadiativePartition,
    electron_direction: np.ndarray,
    energy_uniform: float,
    angle_uniform: float,
    azimuth_uniform: float,
) -> HardRadiativePhoton:
    """Sample photon direction from the parent DDCS and account for momentum.

    The electron retains its direction, as in the existing segment contract;
    residual momentum is assigned to an infinitely heavy target and its recoil
    energy is neglected. BremsLib does not provide a joint photon/electron
    angular distribution. Photon propagation and downstream absorption are
    outside this host-side event contract.

    Validation: bremslib-radiative-partition
    """
    if not 0.0 <= angle_uniform < 1.0 or not 0.0 <= azimuth_uniform < 1.0:
        raise ValueError("angle uniforms must be in [0, 1)")
    incoming = np.asarray(electron_direction, dtype=float)
    if (
        incoming.shape != (3,)
        or not np.all(np.isfinite(incoming))
        or not np.isclose(np.linalg.norm(incoming), 1.0, atol=1e-10)
    ):
        raise ValueError("electron_direction must be a finite unit vector")
    energy = partition.sample_photon_energy(energy_uniform)
    table = partition.table
    theta = np.asarray(table.theta_rad, dtype=float)
    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(
        staged, np.full(theta.size, partition.incident_energy_eV / 1e3), np.cos(theta)
    )
    ddcs = BACKEND.to_cpu(evaluate_bremslib(staged, state, [energy]))[:, 0]
    # Exact integral of each linear-in-theta cell; the same rule normalizes DDCS.
    widths = np.diff(theta)
    a, b = theta[:-1], theta[1:]
    cell_mass = (
        2.0
        * np.pi
        * (
            ddcs[:-1] * (np.cos(a) - np.cos(b))
            + np.diff(ddcs) / widths * (np.sin(b) - np.sin(a) - widths * np.cos(b))
        )
    )
    cumulative = np.cumsum(cell_mass)
    target = angle_uniform * float(cumulative[-1])
    cell = min(int(np.searchsorted(cumulative, target, side="right")), cell_mass.size - 1)
    target -= float(cumulative[cell - 1]) if cell else 0.0
    low, high = float(theta[cell]), float(theta[cell + 1])
    for _ in range(52):
        mid = 0.5 * (low + high)
        partial = (
            2.0
            * np.pi
            * (
                ddcs[cell] * (np.cos(a[cell]) - np.cos(mid))
                + (ddcs[cell + 1] - ddcs[cell])
                / widths[cell]
                * (np.sin(mid) - np.sin(a[cell]) - (mid - a[cell]) * np.cos(mid))
            )
        )
        if partial < target:
            low = mid
        else:
            high = mid
    polar = 0.5 * (low + high)
    reference = np.array([1.0, 0.0, 0.0]) if abs(incoming[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    transverse = np.cross(incoming, reference)
    transverse /= np.linalg.norm(transverse)
    other = np.cross(incoming, transverse)
    phi = 2.0 * np.pi * azimuth_uniform
    photon_direction = np.cos(polar) * incoming + np.sin(polar) * (
        np.cos(phi) * transverse + np.sin(phi) * other
    )
    remaining = partition.incident_energy_eV - energy
    p_in = np.sqrt(
        partition.incident_energy_eV * (partition.incident_energy_eV + 2.0 * _ELECTRON_REST_EV)
    )
    p_out = np.sqrt(remaining * (remaining + 2.0 * _ELECTRON_REST_EV))
    electron_out = incoming.copy()
    target_momentum = p_in * incoming - energy * photon_direction - p_out * electron_out
    return HardRadiativePhoton(energy, photon_direction, remaining, electron_out, target_momentum)
