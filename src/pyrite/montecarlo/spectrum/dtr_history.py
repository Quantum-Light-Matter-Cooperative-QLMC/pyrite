"""Transported-current two-harmonic Maxwell solve (experimental, CPU).

Finite currents are integrated in time, including horizontal segments. A
moving polarization/magnetization supplies the first-order lattice-force
current (CBS) in the same Maxwell convention as the charge current.

Validation: dtr-transported-current
Validation: line-tr-phase-reconciliation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG
from .dtr_two_beam import _boundary_response, _embed, _maxwell, _modes
from .transition_radiation import _particular_field

DTR_MODEL = "two-beam-transported-current-v1"


@dataclass(frozen=True)
class TransportedCurrent:
    """Piecewise straight current; distances/clocks in Å, velocity in c.

    ``history`` and ``layer`` index each segment. Boundary arrays index
    exterior continuations only; ``boundary_side`` is 0 at the substrate
    and 1 at emitting vacuum. Charge is in units of +e.
    """

    position: np.ndarray
    velocity: np.ndarray
    time: np.ndarray
    duration: np.ndarray
    history: np.ndarray
    layer: np.ndarray
    charge: np.ndarray
    boundary_position: np.ndarray
    boundary_velocity: np.ndarray
    boundary_time: np.ndarray
    boundary_history: np.ndarray
    boundary_side: np.ndarray
    boundary_charge: np.ndarray
    histories: int


def _drive(omega, kxy, eps, current, polarization, magnetization):
    """Maxwell source after eliminating Ez/Hz, integrated per unit time.

    D=eps E+4pi P, B=H+4pi M, curl H=-i omega D+4pi J.
    Validation: dtr-transported-current, line-tr-phase-reconciliation
    """
    inverse = np.linalg.inv(eps)
    kx, ky = kxy[:, 0], kxy[:, 1]
    longitudinal = inverse @ (current[:, 2] / omega - 1j * polarization[:, 2])
    return (
        4
        * np.pi
        * np.concatenate(
            [
                kx * longitudinal + 1j * omega * magnetization[:, 1],
                ky * longitudinal - 1j * omega * magnetization[:, 0],
                current[:, 1] - 1j * (kx * magnetization[:, 2] + omega * polarization[:, 1]),
                -current[:, 0] - 1j * ky * magnetization[:, 2] + 1j * omega * polarization[:, 0],
            ]
        )
    )


def _time_integral(slope, duration, start_phase, end_phase):
    """Integral from bounded endpoint phases, including coincident exponents."""
    delta = 1j * slope * duration
    use_end = delta.real > 0
    small = np.abs(delta) < 1e-8
    safe = np.where(small, 1, delta)
    bounded = np.where(use_end, -delta, delta)
    ratio = np.expm1(bounded) / np.where(use_end, -safe, safe)
    ratio = np.where(small, 1 + delta / 2 + delta**2 / 6, ratio)
    # The small-delta series is referenced to the start endpoint.
    anchor = np.where(use_end & ~small, end_phase, start_phase)
    return duration * anchor * ratio


def _segment_faces(
    modes, drive, omega, kxy, harmonic, gz, position, velocity, time, duration, zlow, zhigh
):
    """Exact Green integral of one current on a straight finite segment."""
    wave, vectors, up = modes
    projected = np.linalg.solve(vectors, drive)
    q = omega - kxy @ velocity[:2] - harmonic * gz * velocity[2]
    phase = omega * time - kxy @ position[:2] - harmonic * gz * position[2]
    zface = np.where(up, zhigh, zlow)
    start = np.exp(1j * (phase + wave * (zface - position[2])))
    end = np.exp(
        1j * (phase + q * duration + wave * (zface - position[2] - velocity[2] * duration))
    )
    integral = _time_integral(q - wave * velocity[2], duration, start, end)
    high = vectors[:, up] @ (projected[up] * integral[up])
    low = -vectors[:, ~up] @ (projected[~up] * integral[~up])
    return low, high


def _cbs_dipole(velocity, g, potential):
    """First-order lattice displacement in inverse eV; source charge excluded.

    For the electron's potential energy -U_g, delta v=+U_g/(gamma m v.g)
    [g-v(v.g)], delta r=i delta v/(v.g). Its resonant Fourier current is
    minus charge times the relativistic Feranchuk CBS amplitude, giving the
    same outgoing-field sign as Maxwell PXR (not the opposite sign).
    Validation: line-tr-phase-reconciliation
    """
    vdg = float(velocity @ g)
    if abs(vdg) < 1e-12 * np.linalg.norm(g):
        raise ValueError("two-beam CBS requires nonzero v.g (no channeling resonance)")
    gamma = 1 / np.sqrt(1 - velocity @ velocity)
    delta_v = potential / (gamma * vdg) * (g - velocity * vdg)
    return 1j * delta_v / vdg


def _segment_rhs(rhs, modes, media, heights, current, omega, kxy, g, potential, origin):
    """Accumulate transported-current and CBS sources in their harmonic rows.

    Validation: line-tr-phase-reconciliation
    """
    zero = np.zeros((2, 3), complex)
    for i, position in enumerate(current.position):
        layer, history = int(current.layer[i]), int(current.history[i])
        velocity = current.velocity[i]
        charge = np.sqrt(ALPHA_FS) * current.charge[i]
        for harmonic in (0, 1):
            j = zero.copy()
            j[harmonic] = charge * velocity
            drive = _drive(omega, kxy, media[layer + 1], j, zero, zero)
            sources = [(drive, harmonic)]
            if potential is not None and layer == potential[0]:
                sign = 1 if harmonic == 0 else -1
                dipole = charge * _cbs_dipole(velocity, sign * g, potential[1 + harmonic])
                p = zero.copy()
                p[harmonic] = dipole * np.exp(1j * sign * origin)
                drive = _drive(omega, kxy, media[layer + 1], zero, p, np.cross(p, velocity))
                sources.append((drive, 1 - harmonic))
            for drive, source_harmonic in sources:
                # CBS in row 0 has the phase of charge harmonic 1; in row 1
                # it has the phase of charge harmonic 0 after the g_z gauge.
                low, high = _segment_faces(
                    modes[layer + 1],
                    drive,
                    omega,
                    kxy[source_harmonic],
                    source_harmonic,
                    g[2],
                    position / HBARC_EV_ANG,
                    velocity,
                    current.time[i] / HBARC_EV_ANG,
                    current.duration[i] / HBARC_EV_ANG,
                    heights[layer],
                    heights[layer + 1],
                )
                rhs[8 * layer : 8 * layer + 8, history] -= low
                rhs[8 * (layer + 1) : 8 * (layer + 1) + 8, history] += high


def _exterior_rhs(rhs, current, omega, kxy, g, heights):
    for i, position in enumerate(current.boundary_position):
        face = int(current.boundary_side[i])
        interface = 0 if face == 0 else len(heights) - 1
        velocity = current.boundary_velocity[i]
        if velocity[2] == 0:
            raise ValueError("an exterior continuation must cross a planar face")
        for harmonic in (0, 1):
            q = (omega - kxy[harmonic] @ velocity[:2]) / velocity[2]
            e, h = _particular_field(
                np.ones(1), np.array([omega]), kxy[harmonic][None], np.array([q]), velocity, np
            )
            phase = np.exp(
                1j
                * (
                    omega * current.boundary_time[i]
                    - kxy[harmonic] @ position[:2]
                    - harmonic * g[2] * position[2]
                )
                / HBARC_EV_ANG
            )
            own = _embed(np.r_[e[0, :2], h[0, :2]], harmonic)
            rhs[8 * interface : 8 * interface + 8, current.boundary_history[i]] += (
                (1 if face == 0 else -1) * current.boundary_charge[i] * own * phase
            )


def _solve_current(omega, n, eps, heights, layer, g, chi, potential, origin, current):
    kxy = np.stack([omega * n[:2], omega * n[:2] + g[:2]])
    media, modes = [], []
    for j, diagonal in enumerate(eps):
        tensor = np.eye(2, dtype=complex) * diagonal
        if j == layer + 1:
            tensor[0, 1] = chi[0] * np.exp(1j * origin)
            tensor[1, 0] = chi[1] * np.exp(-1j * origin)
        if np.linalg.eigvalsh((tensor - tensor.conj().T) / (2j)).min() < -1e-14:
            raise ValueError("two-beam permittivity must be passive")
        media.append(tensor)
        modes.append(_modes(_maxwell(omega, kxy, tensor, g[2]), omega))
    matrix, (top, chosen) = _boundary_response(modes, heights)
    rhs = np.zeros((matrix.shape[0], current.histories), complex)
    _segment_rhs(rhs, modes, media, heights, current, omega, kxy, g, potential, origin)
    _exterior_rhs(rhs, current, omega, kxy, g, heights)
    coefficients = np.linalg.solve(matrix, rhs)
    fields = modes[-1][1][:, chosen] @ coefficients[top : top + 4]
    tangential = fields[[0, 2]]
    normal = -(n[:2] @ tangential) / n[2]
    return np.concatenate([tangential, normal[None]], axis=0).T


def two_beam_current_fields(
    photon_energy_eV: Any,
    direction: Any,
    layers: Any,
    current: TransportedCurrent,
    *,
    crystal_layer: int,
    reciprocal_vector_ang: Any,
    chi_transfer: Any,
    chi_reverse: Any,
    potential_transfer_over_m: complex = 0,
    potential_reverse_over_m: complex = 0,
    lattice_origin_ang: Any = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return coupled and uncoupled charge fields [energy, history, Cartesian].

    Vacuum surrounds the finite stack; n_z>0, lengths in Å, potentials in
    units of electron rest energy. The coupled field includes CBS when either
    potential is nonzero; the uncoupled field never includes CBS. Both fields
    have the existing TR normalization [1/eV]. Subtract the uncoupled charge
    field to remove ordinary acceleration radiation before adding the existing
    per-crossing TR field. No inter-electron reduction is performed here.

    Validation: dtr-transported-current, line-tr-phase-reconciliation
    """
    energy = np.atleast_1d(np.asarray(photon_energy_eV, float))
    n, g = np.asarray(direction, float), np.asarray(reciprocal_vector_ang, float) * HBARC_EV_ANG
    if n.shape != (3,) or not np.all(np.isfinite(n)) or n[2] <= 0 or not np.isclose(n @ n, 1):
        raise ValueError("two-beam observation must be a unit vector with n_z > 0")
    if energy.ndim != 1 or np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("photon energies must be a finite positive vector")
    eps = [np.broadcast_to(np.asarray(e, complex), energy.shape) for e, _ in layers]
    if not all(np.all(np.isfinite(e)) for e in eps):
        raise ValueError("layer permittivities must be finite")
    heights = np.r_[0, np.cumsum([d for _, d in layers])] / HBARC_EV_ANG
    transfer, reverse = [
        np.broadcast_to(np.asarray(c, complex), energy.shape) for c in (chi_transfer, chi_reverse)
    ]
    origin = float(np.asarray(reciprocal_vector_ang) @ np.asarray(lattice_origin_ang))
    potential = None
    if potential_transfer_over_m != 0 or potential_reverse_over_m != 0:
        potential = (crystal_layer, potential_transfer_over_m, potential_reverse_over_m)
    coupled, bare = [], []
    for i, omega in enumerate(energy):
        media = [1.0, *(e[i] for e in eps), 1.0]
        args = (omega, n, media, heights, crystal_layer, g)
        coupled.append(_solve_current(*args, (transfer[i], reverse[i]), potential, origin, current))
        bare.append(_solve_current(*args, (0, 0), None, origin, current))
    return np.asarray(coupled), np.asarray(bare)
