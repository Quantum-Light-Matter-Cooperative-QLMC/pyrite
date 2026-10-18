"""Driven two-beam Maxwell boundary diagnostic for one planar crystal layer.

The two transverse harmonics are k_parallel and k_parallel + g_parallel.
Both polarizations, both propagation senses, absorption, and repeated
diffraction are retained. This module is not yet a production selector.

Source: Maxwell's curl equations projected onto two spatial harmonics;
Caticha, Phys. Rev. A 40, 4322 (1989), two-wave boundary problem. The precise
first-order system and driven integral are derived in the owning physics page.

Validation: dtr-two-beam-boundary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG
from .dtr import _integral, _stack
from .transition_radiation import _cross, _free_system, _particular_field


@dataclass(frozen=True)
class TwoBeamField:
    """Vacuum Cartesian fields [1/eV], with the TR solver's reference plane.

    ``transition`` is the response to the charge's observed harmonic,
    including its diffraction extinction. ``line`` is the response to the
    second charge harmonic; it contains PXR and DTR in one amplitude.
    ``total`` retains their cross term when squared. These are complex fields,
    not additive intensities. Source charge is +sqrt(alpha).

    Validation: dtr-two-beam-boundary
    """

    transition: np.ndarray
    line: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.transition + self.line


def _maxwell(omega, kxy, eps, gz):
    """Tangential [Ex(2), Ey(2), Hx(2), Hy(2)] generator, dY/dz=i M Y.

    Eliminate Ez using eps Ez=(Ky Hx-Kx Hy)/omega, then Hz from Faraday.
    The harmonic-1 gauge E1=exp(i gz z) Y1 makes the lattice coupling
    constant; subtract gz from that harmonic's four diagonal entries.
    Zero coupling leaves the two independent isotropic Maxwell systems.

    Validation: dtr-two-beam-boundary
    """
    kx, ky = np.diag(kxy[:, 0]), np.diag(kxy[:, 1])
    inv, unit = np.linalg.inv(eps), np.eye(2)
    matrix = np.zeros((8, 8), complex)
    matrix[0:2, 4:6] = kx @ inv @ ky / omega
    matrix[0:2, 6:8] = omega * unit - kx @ inv @ kx / omega
    matrix[2:4, 4:6] = ky @ inv @ ky / omega - omega * unit
    matrix[2:4, 6:8] = -ky @ inv @ kx / omega
    matrix[4:6, 0:2] = -kx @ ky / omega
    matrix[4:6, 2:4] = kx @ kx / omega - omega * eps
    matrix[6:8, 0:2] = omega * eps - ky @ ky / omega
    matrix[6:8, 2:4] = ky @ kx / omega
    matrix[np.arange(1, 8, 2), np.arange(1, 8, 2)] -= gz
    return matrix


def _modes(matrix, omega):
    """Classify passive modes by decay sense, or by Poynting flux when real.

    Every finite layer references each mode at the face it leaves. No
    propagator grows through a passive layer, including thick Bragg slabs.

    Validation: dtr-two-beam-boundary
    """
    wave, vectors = np.linalg.eig(matrix)
    flux = np.real(
        np.sum(vectors[0:2] * vectors[6:8].conj() - vectors[2:4] * vectors[4:6].conj(), axis=0)
    )
    sense = np.where(np.abs(wave.imag) > 1e-10 * omega, wave.imag, flux)
    up = sense > 0
    if np.count_nonzero(up) != 4:
        raise ValueError("two-beam modes require four outgoing waves per face")
    return wave, vectors, up


def _embed(tangential, harmonic):
    field = np.zeros(8, complex)
    field[harmonic::2] = tangential
    return field


def _background(omega, kxy, media, heights, velocity, time, position, layer, harmonic, gz):
    """Homogeneous straight-charge own/free fields inside the crystal.

    Three exponentials (own, upward, downward) reproduce its total field.
    Each source is anchored at a face so its phase remains bounded.

    Validation: dtr-two-beam-boundary
    """
    kxy = kxy[None]
    matrix, top, lam = _free_system(np.array([omega]), kxy, media, heights, np)
    q = (omega - kxy[0] @ velocity[:2]) / velocity[2]
    phase = np.exp(1j * (omega * time - kxy[0] @ position[:2] - q * position[2]) / HBARC_EV_ANG)
    own = []
    for eps in media:
        e, h = _particular_field(eps, np.array([omega]), kxy, np.array([q]), velocity, np)
        own.append(np.r_[e[0, :2], h[0, :2]])
    rhs = np.concatenate(
        [(own[i] - own[i + 1]) * np.exp(1j * q * z) * phase for i, z in enumerate(heights)]
    )
    coefficients = np.linalg.solve(matrix[0], rhs)
    j = layer + 1
    zlo, zhi = heights[layer : layer + 2]
    terms = [
        (
            _embed(own[j], harmonic) * phase * np.exp(1j * (q - harmonic * gz) * zlo),
            q - harmonic * gz,
            False,
        )
    ]
    column = 2 + 4 * layer
    for wave, sign, anchor in ((0, 1, zlo), (1, -1, zhi)):
        et = coefficients[column + 2 * wave : column + 2 * wave + 2]
        kz = sign * lam[j][0]
        e = np.r_[et, -(kxy[0] @ et) / kz]
        h = _cross(np.r_[kxy[0], kz], e, np) / omega
        field = _embed(np.r_[e[:2], h[:2]], harmonic) * np.exp(-1j * harmonic * gz * anchor)
        terms.append((field, kz - harmonic * gz, wave == 1))
    return terms, coefficients[top : top + 2]


def _source_faces(wave, vectors, up, terms, thickness):
    """Outgoing particular field at the two faces of a driven finite layer.

    Variation of constants integrates each exponential exactly, including
    equal exponents. Downward modes integrate toward decreasing z and carry
    a minus sign. All exponential endpoint factors decay in passive media.

    Validation: dtr-two-beam-boundary
    """
    low, high = np.zeros(8, complex), np.zeros(8, complex)
    for source, exponent, at_high in terms:
        projected = np.linalg.solve(vectors, source)
        if at_high:
            upper = _integral(0.0, wave[up] - exponent, thickness, np)
            lower = _integral(-wave[~up], -exponent, thickness, np)
        else:
            upper = _integral(exponent, wave[up], thickness, np)
            lower = _integral(exponent - wave[~up], 0.0, thickness, np)
        # Evaluate only the outward modes. The other half of either integral
        # can grow and is neither needed nor a physical propagator.
        high += vectors[:, up] @ (projected[up] * upper)
        low -= vectors[:, ~up] @ (projected[~up] * lower)
    return low, high


def _boundary_response(modes, heights):
    """Tangential continuity for the two harmonics across every interface."""
    size = 8 * len(heights)
    matrix = np.zeros((size, size), complex)
    columns, cursor = [], 0
    for j, (_wave, _vectors, up) in enumerate(modes):
        keep = ~up if j == 0 else up if j == len(modes) - 1 else np.ones(8, bool)
        chosen = np.flatnonzero(keep)
        columns.append((cursor, chosen))
        cursor += len(chosen)
    for i, z in enumerate(heights):
        for j, sign in ((i, -1), (i + 1, 1)):
            wave, vectors, up = modes[j]
            col, chosen = columns[j]
            if j == 0 or j == len(modes) - 1:
                phase = np.ones(len(chosen), complex)
            else:
                anchors = np.where(up[chosen], heights[j - 1], heights[j])
                phase = np.exp(1j * wave[chosen] * (z - anchors))
            matrix[8 * i : 8 * i + 8, col : col + len(chosen)] = sign * vectors[:, chosen] * phase
    return matrix, columns[-1]


def _one_energy(omega, n, media, heights, velocity, time, position, layer, g, chi, origin):
    kxy = np.stack([omega * n[:2], omega * n[:2] + g[:2]])
    matrices, modes = [], []
    for j, eps in enumerate(media):
        permittivity = np.eye(2, dtype=complex) * eps[0]
        if j == layer + 1:
            permittivity[0, 1] = chi[0] * np.exp(1j * origin)
            permittivity[1, 0] = chi[1] * np.exp(-1j * origin)
        loss = (permittivity - permittivity.conj().T) / (2j)
        if np.linalg.eigvalsh(loss).min() < -1e-14:
            raise ValueError("two-beam permittivity must be passive")
        operator = _maxwell(omega, kxy, permittivity, g[2])
        matrices.append(operator)
        modes.append(_modes(operator, omega))
    boundary, (top, chosen) = _boundary_response(modes, heights)
    uncoupled = _maxwell(omega, kxy, np.eye(2) * media[layer + 1][0], g[2])
    drive = matrices[layer + 1] - uncoupled
    permittivity = np.eye(2, dtype=complex) * media[layer + 1][0]
    permittivity[0, 1] = chi[0] * np.exp(1j * origin)
    permittivity[1, 0] = chi[1] * np.exp(-1j * origin)
    inverse_jump = np.linalg.inv(permittivity) - np.eye(2) / media[layer + 1][0]
    rhs = np.zeros((boundary.shape[0], 2), complex)
    transition = None
    for harmonic in (0, 1):
        terms, bare = _background(
            omega, kxy[harmonic], media, heights, velocity, time, position, layer, harmonic, g[2]
        )
        if harmonic == 0:
            transition = bare
        sources = [(1j * drive @ field, exponent, at_high) for field, exponent, at_high in terms]
        # Eliminating Ez also changes the longitudinal external-current term:
        # Ez = eps^-1[(Ky Hx-Kx Hy)/omega - 4 pi i Jz/omega].
        # The own field contains Jz; treating it as a source-free wave would
        # omit this contact term and fail even the first-Born limit.
        q = (omega - kxy[harmonic] @ velocity[:2]) / velocity[2]
        phase = np.exp(
            1j * (omega * time - kxy[harmonic] @ position[:2] - q * position[2]) / HBARC_EV_ANG
            + 1j * (q - harmonic * g[2]) * heights[layer]
        )
        current = 4 * np.pi * np.sqrt(ALPHA_FS) * np.sign(velocity[2]) * phase / omega
        contact = np.zeros(8, complex)
        contact[:2] = kxy[:, 0] * inverse_jump[:, harmonic] * current
        contact[2:4] = kxy[:, 1] * inverse_jump[:, harmonic] * current
        sources.append((contact, q - harmonic * g[2], False))
        low, high = _source_faces(*modes[layer + 1], sources, heights[layer + 1] - heights[layer])
        rhs[8 * layer : 8 * layer + 8, harmonic] = -low
        rhs[8 * (layer + 1) : 8 * (layer + 1) + 8, harmonic] = high
    coefficients = np.linalg.solve(boundary, rhs)
    fields = modes[-1][1][:, chosen] @ coefficients[top : top + 4]
    tangential = fields[[0, 2]]
    tangential[:, 0] += transition
    normal = -(n[:2] @ tangential) / n[2]
    return np.concatenate([tangential, normal[None]], axis=0).T


def two_beam_straight_field(
    photon_energy_eV: Any,
    direction: Any,
    layers: Any,
    substrate_eps: Any,
    velocity: Any,
    time_ang: float,
    position_ang: Any,
    *,
    crystal_layer: int,
    reciprocal_vector_ang: Any,
    chi_transfer: Any,
    chi_reverse: Any,
    lattice_origin_ang: Any = (0.0, 0.0, 0.0),
) -> TwoBeamField:
    """Full two-harmonic field of a globally straight +e passage [1/eV].

    Coordinates and normalization match ``dtr.pxr_straight_amplitude``;
    finite layers run from substrate toward emitting vacuum, n_z > 0.
    ``chi_transfer`` multiplies exp[-i g.(r-origin)]; ``chi_reverse`` is the
    opposite harmonic and is not generally its complex conjugate in an
    absorbing crystal. Local, isotropic, passive permittivity is required.
    Time/position fix one point on the globally straight line, in Angstrom
    with c=1. No recoil, CBS, mosaicity or transported-track approximation is
    included here. Both propagation senses and polarizations are solved.

    This is an experimental CPU diagnostic. It does not replace production
    Feranchuk PXR/CBS. Source: projected Maxwell curl equations (Caticha 1989
    two-wave boundary formulation); see the module's owning derivation.
    Limits: zero harmonics recover homogeneous TR; the first derivative in
    chi_transfer is the first-Born PXR+DTR field; opaque passive slabs remain
    bounded. Field intensity is E n_z^2 |a|^2/(16 pi^4).

    Validation: dtr-two-beam-boundary
    """
    s = _stack(
        photon_energy_eV,
        direction,
        layers,
        substrate_eps,
        crystal_layer,
        reciprocal_vector_ang,
        chi_transfer,
        lattice_origin_ang,
        np,
    )
    v, r = np.asarray(velocity, float), np.asarray(position_ang, float)
    if v.shape != (3,) or r.shape != (3,) or not np.isfinite(time_ang):
        raise ValueError("one finite straight velocity/position and time are required")
    if not np.all(np.isfinite([v, r])) or not 0 < np.linalg.norm(v) < 1 or v[2] == 0:
        raise ValueError("straight velocity must be subluminal and cross the faces")
    n = np.broadcast_to(np.asarray(direction, float), (*s.shape, 3)).reshape(-1, 3)
    reverse = np.broadcast_to(np.asarray(chi_reverse, complex), s.shape).reshape(-1)
    transfer = np.broadcast_to(np.asarray(chi_transfer, complex), s.shape).reshape(-1)
    if not np.all(np.isfinite([transfer, reverse])):
        raise ValueError("crystal harmonics must be finite")
    phase_origin = float(np.asarray(reciprocal_vector_ang) @ np.asarray(lattice_origin_ang))
    result = []
    for i, omega in enumerate(s.omega):
        media = [eps[i : i + 1] for eps in s.eps]
        result.append(
            _one_energy(
                omega,
                n[i],
                media,
                s.heights,
                v,
                time_ang,
                r,
                crystal_layer,
                s.g,
                (transfer[i], reverse[i]),
                phase_origin,
            )
        )
    fields = np.asarray(result).reshape((*s.shape, 2, 3))
    return TwoBeamField(fields[..., 0, :], fields[..., 1, :])
