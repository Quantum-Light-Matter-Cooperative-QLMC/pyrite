"""Validity bounds of fixed-trajectory optical TR/CTR scoring (#615).

The crossing amplitude treats each crossing as a locally straight, uniform
passage, and transport does not feed radiated energy back into the
trajectory. Two bounds make that explicit for a supported scene and refuse
use outside it:

- **formation** — the own field of the medium on either side of a crossing
  builds up over ``L = max(1/|q - lambda|, 1/|q + lambda|)`` (``hbar = c = 1``;
  ``q = (omega - k_par . v_par)/v_z``, ``lambda = sqrt(eps omega^2 - k_par^2)``,
  ``Im lambda >= 0``). Where the transported straight run on that side is
  shorter than ``L``, that side's own-field term of the amplitude is
  unreliable. With ``A_s`` the part of the crossing amplitude driven by side
  ``s`` (linear split of the jump) and ``d_s`` the indicator of a short run,
  each history gets ``|dS| <= sum_c 2 sum_s d_s |A_s|`` and the self power a
  relative error at most ``sum_e (2 |dS_e||S_e| + |dS_e|^2) / sum_e |S_e|^2``.
  The factor 2 replaces the term by one of equal magnitude; it is a
  sensitivity bound for a deflected formation zone, not a radiation model
  of the deflection (bremsstrahlung).
- **energy budget** — the radiated energy per physical electron, including
  the coherent pair term, integrated over the scored band and both
  hemispheres, must be a small fraction of the kinetic energy; and the pair
  term must be negligible at both band edges, otherwise coherent energy
  just outside the band is unbounded by this calculation. The budget is
  in-band only: it cannot see coherent emission below a band whose lower edge
  lies above the coherent window (``gaussian_coherent_band_eV``), where
  ``|F|^2`` rises again toward one; such sub-band energy needs supported optics
  there and is reported, not bounded, by the caller.

Both bounds concern the self (single-history) power: the pair term's
relative error is not bounded by the formation check. The factor 2 assumes
the true term of a short run is no larger than the straight-run term; it is a
heuristic sensitivity measure, not a proven bound.

Validation: optical-planar-tr, optical-ctr-population
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...materials.crystal import HBARC_EV_ANG
from .transition_radiation import _outgoing_root, interface_transition_radiation_amplitude
from .transition_radiation_crossings import VACUUM, BoundaryCrossings, _emission_frame

__all__ = [
    "formation_error_bound",
    "hemisphere_directions",
    "kinetic_energy_eV",
    "radiated_energy_fraction",
]


def _slab_layer(solver_index, n_layers, forward):
    """Slab layer of a solver medium (substrate first, vacuum last); -1 is vacuum."""
    layer = solver_index - 1 if forward else n_layers - solver_index
    return np.where((layer < 0) | (layer >= n_layers), VACUUM, layer)


def _side_runs(crossings, side_layer):
    """Straight run on the side of each crossing occupied by ``side_layer``."""
    before = np.asarray(crossings.straight_before_ang, dtype=float)
    after = np.asarray(crossings.straight_after_ang, dtype=float)
    run = np.where(side_layer == crossings.layer_before, before, after)
    return np.where(side_layer == VACUUM, np.inf, run)


def formation_error_bound(
    crossings: BoundaryCrossings, photon_energy_eV, direction, layer_eps
) -> np.ndarray:
    """Relative self-power error bound per energy/direction (module docstring).

    Needs transport-derived crossings (``straight_*_ang``). Host NumPy; a
    validity preflight for bounded scenes, not a production reduction.

    Validation: optical-planar-tr
    """
    if crossings.straight_before_ang is None or crossings.straight_after_ang is None:
        raise ValueError("formation bound needs transport-derived straight runs")
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
    frame = _emission_frame(crossings, n, layer_eps)
    forward = bool(np.reshape(n, (-1, 3))[0, 2] > 0.0)
    n_layers = len(frame.layers)
    below = _slab_layer(frame.face, n_layers, forward)
    above = _slab_layer(frame.face + 1, n_layers, forward)
    omega = np.broadcast_to(energy, shape).reshape(-1)
    n_frame = np.broadcast_to(frame.direction, (*shape, 3)).reshape(-1, 3)
    k_par = omega[:, None] * n_frame[:, :2]
    v = frame.velocity
    q = (omega[:, None] - k_par @ v[:, :2].T) / v[:, 2]  # (B, K)
    eps_slab = [np.broadcast_to(np.asarray(e, dtype=complex), shape).reshape(-1) for e in layer_eps]
    args = (
        energy, frame.direction, frame.layers, 1.0, frame.velocity,
        crossings.time_ang, frame.position, frame.face,
    )  # fmt: skip
    scale = crossings.charge.astype(float)[:, None] * frame.flip
    error = np.zeros((omega.size, len(crossings)))
    for column, side in enumerate((below, above)):
        own = np.zeros((len(crossings), 2))
        own[:, column] = 1.0
        part = interface_transition_radiation_amplitude(*args, own_field_sides=own)
        magnitude = np.linalg.norm((part * scale).reshape(omega.size, len(crossings), 3), axis=-1)
        length = _formation_length(side, eps_slab, omega, k_par, q)
        short = _side_runs(crossings, side)[None, :] < length
        error += 2.0 * np.where(short, magnitude, 0.0)
    total = interface_transition_radiation_amplitude(*args) * scale
    total = total.reshape(omega.size, len(crossings), 3)
    return _history_relative_error(crossings.electron_id, total, error).reshape(shape)


def _formation_length(side, eps_slab, omega, k_par, q):
    """``max(1/|q - lambda|, 1/|q + lambda|)`` [Å] of each crossing's medium; 0 in vacuum."""
    length = np.zeros(q.shape)
    k2 = np.sum(k_par**2, axis=-1)[:, None]
    for layer in np.unique(side[side != VACUUM]):
        mask = side == layer
        lam = _outgoing_root(eps_slab[layer][:, None] * omega[:, None] ** 2 - k2)
        qm = q[:, mask]
        length[:, mask] = HBARC_EV_ANG / np.minimum(np.abs(qm - lam), np.abs(qm + lam))
    return length


def _history_relative_error(electron_id, amplitude, error):
    histories, inverse = np.unique(electron_id, return_inverse=True)
    summed = np.zeros((amplitude.shape[0], histories.size, 3), dtype=complex)
    bound = np.zeros((amplitude.shape[0], histories.size))
    for column, history in enumerate(inverse):  # amplitude (B, K, 3), error (B, K)
        summed[:, history] += amplitude[:, column]
        bound[:, history] += error[:, column]
    norm = np.linalg.norm(summed, axis=-1)
    power = np.sum(norm**2, axis=-1)
    excess = np.sum(2.0 * bound * norm + bound**2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(power > 0, excess / power, np.where(excess > 0, np.inf, 0.0))


def hemisphere_directions(n_cos: int = 12, n_phi: int = 12, *, forward: bool):
    """Gauss–Legendre in ``cos theta`` x uniform ``phi`` directions and weights [sr]."""
    nodes, weights = np.polynomial.legendre.leggauss(int(n_cos))
    mu = 0.5 * (nodes + 1.0)
    phi = (np.arange(int(n_phi)) + 0.5) * 2.0 * np.pi / int(n_phi)
    sin = np.sqrt(1.0 - mu**2)
    sign = 1.0 if forward else -1.0
    direction = np.stack(
        [
            np.outer(sin, np.cos(phi)),
            np.outer(sin, np.sin(phi)),
            sign * np.outer(mu, np.ones_like(phi)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weight = np.outer(0.5 * weights, np.full(phi.size, 2.0 * np.pi / phi.size)).reshape(-1)
    return direction, weight


def radiated_energy_fraction(
    score: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    photon_energy_eV,
    kinetic_energy_eV: float,
    *,
    max_fraction: float = 1e-3,
    edge_tolerance: float = 1e-3,
    n_cos: int = 12,
    n_phi: int = 12,
) -> float:
    """Radiated energy per physical electron over kinetic energy; refuses unbounded use.

    ``score(energy, directions)`` returns ``(yield_per_electron, self_yield,
    pair_weight)`` with shapes ``(E, D)``, ``pair_weight`` being
    ``max(N - 1, 0) F pair_yield``. Integrates ``E * yield`` over both
    hemispheres (Gauss–Legendre/uniform) and the band (trapezoid). Raises
    when the fraction exceeds ``max_fraction`` or the pair-to-self ratio at a
    band edge exceeds ``edge_tolerance``.

    Validation: optical-ctr-population
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size < 2 or kinetic_energy_eV <= 0:
        raise ValueError("energy budget needs a 1-D band and positive kinetic energy")
    radiated = 0.0
    for forward in (True, False):
        direction, weight = hemisphere_directions(n_cos, n_phi, forward=forward)
        total, self_part, pair_part = score(energy[:, None], direction[None])
        edges = np.abs(pair_part[[0, -1]]) / np.maximum(np.abs(self_part[[0, -1]]), 1e-300)
        if np.any(edges > edge_tolerance):
            raise ValueError(
                "coherent pair emission is not negligible at a band edge "
                f"(pair/self up to {float(np.max(edges)):.3g}); radiated energy outside the "
                "scored band is unbounded: extend the band with supported optics"
            )
        radiated += float(np.trapezoid(energy * (total @ weight), energy))
    fraction = radiated / float(kinetic_energy_eV)
    if not np.isfinite(fraction) or fraction > max_fraction:
        raise ValueError(
            f"radiated energy is {fraction:.3g} of the kinetic energy (limit {max_fraction:g}): "
            "the fixed-trajectory approximation is unbounded for this scene"
        )
    return fraction


def kinetic_energy_eV(crossings: BoundaryCrossings) -> float:
    """Smallest entrance kinetic energy [eV] of the primary crossings."""
    entry = (crossings.layer_before == VACUUM) & (crossings.interface == 0)
    if not np.any(entry):
        raise ValueError("no entrance crossings")
    return float(np.min(crossings.kinetic_energy_keV[entry])) * 1e3
