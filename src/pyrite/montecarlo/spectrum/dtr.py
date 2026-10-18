"""Experimental first-Born diffraction of a planar stack's boundary field.

This is a field-level diagnostic for #581, not a production emission selector.
It does not reconcile the absolute phase of the Feranchuk PXR/CBS amplitudes.
The source is the free field of the homogeneous stack, scattered once by one
crystal harmonic. Extinction and rescattering of the diffracted wave are absent.

Source: Maxwell's driven wave equation and tangential E/H continuity, derived
in docs/physics/radiation-physics/diffracted-transition-radiation.md. The split
into diffraction of the own and free fields follows Artru & Rullhusen,
Nucl. Instrum. Methods B 145 (1998), 1–7, DOI 10.1016/S0168-583X(98)00241-9.

Validation: dtr-kinematic-boundary
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ...materials.crystal import HBARC_EV_ANG
from .transition_radiation import (
    _checked_kinematics,
    _cross,
    _free_system,
    _interface_heights,
    _particular_field,
    _stack_media,
)


@dataclass(frozen=True)
class _Stack:
    shape: tuple[int, ...]
    omega: Any
    k_in: Any
    k_out: Any
    g: Any
    eps: list[Any]
    heights: np.ndarray
    incoming: Any
    response: Any
    lam_in: Any
    lam_out: Any
    lam_top: Any
    layer: int
    coupling: Any


def _stack(energy_eV, direction, layers, substrate_eps, layer, g_ang, chi, origin, xp):
    energy, n = np.asarray(energy_eV, float), np.asarray(direction, float)
    if n.shape[-1:] != (3,) or not np.all(np.isfinite(n)):
        raise ValueError("direction must contain finite unit vectors")
    if not np.allclose(np.linalg.norm(n, axis=-1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("direction must contain unit vectors")
    if not np.all(np.isfinite(energy)):
        raise ValueError("photon energies must be finite")
    _checked_kinematics(energy, n, np.array([[0.0, 0.0, 0.5]]))
    if not isinstance(layer, (int, np.integer)) or not 0 <= layer < len(layers):
        raise ValueError("crystal_layer must index a finite layer")
    thicknesses = np.asarray([d for _, d in layers], float)
    if np.any(~np.isfinite(thicknesses)) or np.any(thicknesses <= 0):
        raise ValueError("layer thicknesses must be finite and positive")
    g_host, origin_host = np.asarray(g_ang, float), np.asarray(origin, float)
    if g_host.shape != (3,) or origin_host.shape != (3,):
        raise ValueError("reciprocal_vector_ang and lattice_origin_ang must have shape (3,)")
    if not np.all(np.isfinite([g_host, origin_host])) or np.linalg.norm(g_host) == 0:
        raise ValueError("lattice origin must be finite and reciprocal vector finite and nonzero")
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
    omega = xp.asarray(np.broadcast_to(energy, shape).ravel())
    k_out = omega[:, None] * xp.asarray(np.broadcast_to(n, (*shape, 3)).reshape(-1, 3))[:, :2]
    g = xp.asarray(g_host * HBARC_EV_ANG)
    k_in = k_out + g[:2]
    eps = _stack_media(substrate_eps, layers, shape, xp)
    heights = _interface_heights(layers)
    incoming, _, lam_in = _free_system(omega, k_in, eps, heights, xp)
    outgoing, top, lam_out = _free_system(omega, k_out, eps, heights, xp)
    select = xp.zeros((omega.size, outgoing.shape[1], 2), dtype=complex)
    select[:, top, 0] = select[:, top + 1, 1] = 1.0
    response = xp.linalg.solve(xp.swapaxes(outgoing, 1, 2), select)
    coupling = xp.broadcast_to(xp.asarray(chi, dtype=complex), shape).reshape(-1)
    coupling = coupling * xp.exp(1j * (float(g_host @ origin_host) - g[2] * heights[layer]))
    return _Stack(
        shape,
        omega,
        k_in,
        k_out,
        g,
        eps,
        heights,
        incoming,
        response,
        lam_in[layer + 1],
        lam_out[layer + 1],
        lam_out[-1],
        int(layer),
        coupling,
    )


def _crossings(stack, velocity, time_ang, position_ang, interface, xp):
    v, r = (
        np.atleast_2d(np.asarray(velocity, float)),
        np.atleast_2d(np.asarray(position_ang, float)),
    )
    t, face = np.asarray(time_ang, float).reshape(-1), np.asarray(interface)
    if v.shape[1:] != (3,) or r.shape != v.shape or t.shape != (len(v),):
        raise ValueError("crossings require velocity/position (K, 3) and time/interface (K,)")
    if face.shape != t.shape or not np.issubdtype(face.dtype, np.integer):
        raise ValueError("interface must contain K integer indices")
    if not all(np.all(np.isfinite(x)) for x in (v, r, t)):
        raise ValueError("crossing states must be finite")
    _checked_kinematics(np.array(1.0), np.array([0.0, 0.0, 1.0]), v)
    if np.any((face < 0) | (face >= len(stack.heights))):
        raise ValueError("interface index outside the stack")
    return xp.asarray(v), xp.asarray(t), xp.asarray(r), xp.asarray(face)


def _integral(a, b, thickness, xp):
    """Integral of exp(i a x + i b (d-x)); bounded at phase matching.

    Both endpoint exponentials decay in passive media. The small-difference
    series avoids cancellation without exponentials that grow inside a layer.

    Validation: dtr-kinematic-boundary
    """
    delta = (a - b) * thickness
    small = xp.abs(delta) < 1e-4
    denominator = xp.where(small, 1.0, 1j * (a - b))
    divided = (xp.exp(1j * a * thickness) - xp.exp(1j * b * thickness)) / denominator
    series = thickness * xp.exp(0.5j * (a + b) * thickness) * (1 - delta**2 / 24 + delta**4 / 1920)
    return xp.where(small, series, divided)


def _wave_vector(k_perp, kz, xp):
    return xp.concatenate([k_perp, kz[:, None]], axis=-1)


def _radiate(stack, up_integral, down_integral, xp):
    """Propagate a layer polarization source to vacuum via tangential jumps.

    For K^2 = eps omega^2 a sheet polarization has outgoing field
    i omega^2/(2 lambda) [I - K K^T/(eps omega^2)] P. Units: the integrated
    polarization is eV^-2; the outgoing E Fourier coefficient is eV^-1.

    Validation: dtr-kinematic-boundary
    """
    omega, lam, j = stack.omega, stack.lam_out, stack.layer
    rhs = xp.zeros((*up_integral.shape[:2], stack.incoming.shape[-1]), dtype=complex)
    for sign, integral, face, side in ((1, up_integral, j + 1, 1), (-1, down_integral, j, -1)):
        k = _wave_vector(stack.k_out, sign * lam, xp)
        projection = (
            integral
            - k[:, None]
            * (xp.sum(k[:, None] * integral, axis=-1) / (stack.eps[j + 1] * omega**2)[:, None])[
                ..., None
            ]
        )
        e_vec = (0.5j * omega**2 / lam)[:, None, None] * projection
        h_vec = _cross(k[:, None], e_vec, xp) / omega[:, None, None]
        rhs[..., 4 * face : 4 * face + 4] += side * xp.concatenate(
            [e_vec[..., :2], h_vec[..., :2]], axis=-1
        )
    e_t = xp.einsum("brc,bkr->bkc", stack.response, rhs)
    e_z = -xp.sum(stack.k_out[:, None] * e_t, axis=-1) / stack.lam_top[:, None]
    return xp.concatenate([e_t, e_z[..., None]], axis=-1)


def dtr_crossing_amplitudes(
    photon_energy_eV: ArrayLike,
    direction: ArrayLike,
    layers: Sequence[tuple[Any, float]],
    substrate_eps: Any,
    velocity: ArrayLike,
    time_ang: ArrayLike,
    position_ang: ArrayLike,
    interface: ArrayLike,
    *,
    crystal_layer: int,
    reciprocal_vector_ang: ArrayLike,
    chi_transfer: Any,
    lattice_origin_ang: ArrayLike = (0.0, 0.0, 0.0),
    xp: Any = np,
) -> Any:
    """First-Born DTR vacuum vector per crossing, shape ``(*s, K, 3)`` [1/eV].

    ``layers`` and the emission-side ``n_z > 0`` convention match
    ``interface_transition_radiation_amplitude``: finite layers run from the
    substrate (z=0) toward vacuum. ``crystal_layer`` indexes that list, starting
    at zero. Crossings share its coordinates; time/position are in Angstrom,
    velocity is v/c. The charge is +e; multiply each vector by its charge sign.

    ``reciprocal_vector_ang`` is g [1/Angstrom], with virtual transverse
    momentum k_in = omega n_parallel + g_parallel. ``chi_transfer`` is the
    coefficient of exp[-i g.(r-origin)] in relative permittivity, in the
    Maxwell exp(-i omega t) convention (not an unreconciled line-table value).
    It broadcasts against energies/directions. The returned origin and yield
    normalization match the TR solver: E n_z^2 |sum a|^2/(16 pi^4).

    One harmonic scatters only the homogeneous stack's *free* field. All
    ordinary Fresnel reflections and absorption are retained on both paths;
    diffraction extinction, CBS and mosaic averaging are absent. Use only
    where omega^2 |chi| d/(2 sqrt(|lambda_in lambda_out|)) << 1, with d in
    1/eV. This function deliberately does not expose a production selector.
    Limits: chi -> 0 and globally matched dielectric media give zero DTR;
    the finite-volume integral stays finite at exact diffraction matching.

    Source: the driven Maxwell sheet solution, equations (1)-(5) of the linked
    DTR derivation page; Artru & Rullhusen (1998), own/free-field split.
    Validation: dtr-kinematic-boundary
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
        xp,
    )
    v, t, r, face = _crossings(s, velocity, time_ang, position_ang, interface, xp)
    q = (s.omega[:, None] - xp.sum(s.k_in[:, None] * v[None, :, :2], axis=-1)) / v[None, :, 2]
    rhs = xp.zeros((s.omega.size, s.incoming.shape[-1], len(v)), dtype=complex)
    phase = xp.exp(
        1j
        * (s.omega[:, None] * t - xp.sum(s.k_in[:, None] * r[None, :, :2], axis=-1))
        / HBARC_EV_ANG
    )
    for i in range(len(s.heights)):
        jump = xp.zeros((s.omega.size, len(v), 4), dtype=complex)
        for offset, sign in ((0, 1), (1, -1)):
            e_vec, h_vec = _particular_field(
                s.eps[i + offset][:, None], s.omega[:, None], s.k_in[:, None], q, v[None], xp
            )
            jump += sign * xp.concatenate([e_vec[..., :2], h_vec[..., :2]], axis=-1)
        rhs[:, 4 * i : 4 * i + 4] = xp.swapaxes(
            jump * phase[..., None] * (face == i)[None, :, None], 1, 2
        )
    coefficients = xp.swapaxes(xp.linalg.solve(s.incoming, rhs), 1, 2)
    column = 2 + 4 * s.layer  # substrate - wave, then (+,-) in each finite layer
    d = s.heights[s.layer + 1] - s.heights[s.layer]
    up = xp.zeros((*q.shape, 3), dtype=complex)
    down = xp.zeros_like(up)
    integrals = (
        (
            _integral(s.lam_in - s.g[2], s.lam_out, d, xp),
            _integral(s.lam_in + s.lam_out - s.g[2], 0, d, xp),
        ),
        (
            _integral(-s.g[2], s.lam_in + s.lam_out, d, xp),
            _integral(s.lam_out - s.g[2], s.lam_in, d, xp),
        ),
    )
    for wave, (iu, idown) in enumerate(integrals):
        e_t = coefficients[..., column + 2 * wave : column + 2 * wave + 2]
        e_z = -xp.sum(s.k_in[:, None] * e_t, axis=-1) / ((1 - 2 * wave) * s.lam_in[:, None])
        e_vec = xp.concatenate([e_t, e_z[..., None]], axis=-1)
        up += iu[:, None, None] * e_vec
        down += idown[:, None, None] * e_vec
    result = _radiate(s, s.coupling[:, None, None] * up, s.coupling[:, None, None] * down, xp)
    return result.reshape((*s.shape, len(v), 3))


def pxr_straight_amplitude(
    photon_energy_eV: ArrayLike,
    direction: ArrayLike,
    layers: Sequence[tuple[Any, float]],
    substrate_eps: Any,
    velocity: ArrayLike,
    time_ang: ArrayLike,
    position_ang: ArrayLike,
    *,
    crystal_layer: int,
    reciprocal_vector_ang: ArrayLike,
    chi_transfer: Any,
    lattice_origin_ang: ArrayLike = (0.0, 0.0, 0.0),
    xp: Any = np,
) -> Any:
    """Diagnostic first-Born own-field diffraction on one straight trajectory.

    Same arguments/units/charge/normalization as ``dtr_crossing_amplitudes``;
    ``velocity``, ``time_ang`` and ``position_ang`` specify one straight passage
    at any point on its line. Returns ``(*s, 3)``. The trajectory is extrapolated
    through the whole crystal layer at constant velocity. Sum with the DTR
    vectors of that same passage before taking the modulus square. This is
    Maxwell PXR, not the production Feranchuk PXR/CBS field.

    Source: driven Maxwell volume polarization, equations (1)-(5) of the DTR
    derivation. Vanishing harmonic or vanishing crystal thickness gives zero.
    Validation: dtr-kinematic-boundary
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
        xp,
    )
    v, t, r, _ = _crossings(s, velocity, time_ang, position_ang, np.array([0]), xp)
    if len(v) != 1:
        raise ValueError("pxr_straight_amplitude requires one straight trajectory")
    q = (s.omega[:, None] - xp.sum(s.k_in[:, None] * v[None, :, :2], axis=-1)) / v[None, :, 2]
    e_vec, _ = _particular_field(
        s.eps[s.layer + 1][:, None], s.omega[:, None], s.k_in[:, None], q, v[None], xp
    )
    zlo, zhi = s.heights[s.layer : s.layer + 2]
    phase = xp.exp(
        1j
        * (
            s.omega[:, None] * t
            - xp.sum(s.k_in[:, None] * r[None, :, :2], axis=-1)
            + q * (zlo * HBARC_EV_ANG - r[None, :, 2])
        )
        / HBARC_EV_ANG
    )
    source = s.coupling[:, None, None] * e_vec * phase[..., None]
    up = _integral(q - s.g[2], s.lam_out[:, None], zhi - zlo, xp)
    down = _integral(q + s.lam_out[:, None] - s.g[2], 0, zhi - zlo, xp)
    return _radiate(s, source * up[..., None], source * down[..., None], xp)[:, 0].reshape(
        (*s.shape, 3)
    )
