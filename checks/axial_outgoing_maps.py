"""Research builder: material TDS jump maps for the absorbing jump instrument (#623).

Run the bounded CPU example with::

    uv run python checks/axial_outgoing_maps.py

For one supercell-periodic string and a Bloch eigenbasis on a rectangular
window of plane-wave orders (the represented space R), split each TDS Kraus
jump L_j of the transfer covariance into

    A_j = P L_j P              retained recycling,
    B_j = (R - P) L_j P        represented outgoing, in the window's
                               discrete position basis (C order, FFT layout),
    E_O = P (Gamma - sum_j L_j^dagger R L_j) P   unresolved effect,

with P the spectral projector of an energy cut, so [H, P] = 0. Gamma is the
lattice-sum loss effect of the same periodic finite-transfer model, so E_O is
sum_j P L_j^dagger (1 - R) L_j P >= 0 up to the order tail and Kraus rank
truncation; its smallest eigenvalue is reported, and a materially negative
value refuses. These maps feed axial_jump_instrument.absorbing_axial_model.

Scope: TDS only (no electronic jumps), one string, Gamma point of one cell,
periodic lattice transfers (the midpoint rule of the continuum box split),
absorbing no-return model. The outgoing states are represented, not a
physical escape instrument; matching, return flux, stopping and native launch
remain open under cr-axial-zone-handoff-physical.

Source: TDS transfer covariance and Kraus factorization in
docs/research/physics/axial-escape-instrument.md (eq-axial-coherent-transfer-
covariance, eq-axial-escape-effects); Castin, Dalibard & Molmer,
arXiv:0805.4002, Eq. (2). Units: jumps Å^-1/2, effects Å^-1, energies eV.
Zero displacement or string density gives zero maps.

Validation: cr-axial-zone-handoff-accounting
"""

from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STUDY = runpy.run_path(str(Path(__file__).with_name("axial_matching_study.py")))
_kraus = _STUDY["_kraus"]
loss_operator = _STUDY["loss_operator"]

#: Negative unresolved eigenvalues beyond this fraction of ||Gamma|| refuse.
_EFFECT_RTOL = 1e-9


@dataclass(frozen=True)
class TDSOutgoingMaps:
    """Retained/outgoing/unresolved TDS maps on one Bloch eigenbasis.

    ``retained_jumps`` (j, n, n) and ``outgoing_jumps`` (j, m, n) carry Å^-1/2;
    ``outgoing_jumps`` rows are C-order positions of ``outgoing_shape``.
    ``unresolved_effect`` and ``total_effect`` are (n, n) in Å^-1.
    """

    retained_jumps: np.ndarray
    outgoing_jumps: np.ndarray
    unresolved_effect: np.ndarray
    total_effect: np.ndarray
    outgoing_shape: tuple[int, int]
    unresolved_min_eigenvalue: float


def _window_shape(orders: np.ndarray) -> tuple[int, int]:
    low, high = orders.min(0), orders.max(0)
    shape = tuple(int(x) for x in high - low + 1)
    full = {tuple(o) for o in orders}
    if len(full) != len(orders) or len(full) != shape[0] * shape[1]:
        raise ValueError("orders must fill a rectangular window exactly once")
    return shape  # type: ignore[return-value]


def position_basis(orders: np.ndarray) -> np.ndarray:
    """Unitary (m, K) map from window plane-wave orders to discrete positions.

    Order n sits at FFT index n mod N; positions are the orthonormal inverse
    DFT, so a forward FFT of a position state returns the order amplitude.
    Validation: cr-axial-zone-handoff-accounting
    """
    shape = _window_shape(orders)
    u = np.zeros((shape[0] * shape[1], len(orders)), dtype=complex)
    for k, (nx, ny) in enumerate(orders):
        delta = np.zeros(shape, dtype=complex)
        delta[nx % shape[0], ny % shape[1]] = 1.0
        u[:, k] = np.fft.ifft2(delta, norm="ortho").ravel()
    return u


def tds_outgoing_maps(
    orders: np.ndarray,
    vectors: np.ndarray,
    retained: np.ndarray,
    reciprocal_inv_ang: tuple[float, float],
    phase_transform,
    position_ang: tuple[float, float],
    displacement_ang2: float,
    linear_density_inv_ang: float,
    area_ang2: float,
    *,
    tail_orders: int = 300,
    rank_rtol: float = 1e-14,
) -> TDSOutgoingMaps:
    """Split TDS Kraus jumps into retained, outgoing and unresolved parts.

    ``vectors`` (K, K) is the complete orthonormal eigenbasis on the window
    ``orders`` (K, 2); ``retained`` (K,) selects P. Assumptions and source:
    module docstring. Zero density gives zero maps.
    Validation: cr-axial-zone-handoff-accounting
    """
    orders = np.asarray(orders)
    vectors = np.asarray(vectors, dtype=complex)
    retained = np.asarray(retained, dtype=bool)
    if vectors.shape != (len(orders), len(orders)) or retained.shape != (len(orders),):
        raise ValueError("vectors must be a complete basis on the order window")
    if not np.allclose(vectors.conj().T @ vectors, np.eye(len(orders)), atol=1e-10):
        raise ValueError("vectors must be orthonormal")
    shape = _window_shape(orders)
    g = np.asarray(reciprocal_inv_ang, dtype=float)
    r = np.asarray(position_ang, dtype=float)
    a2, lam = float(displacement_ang2), float(linear_density_inv_ang)
    span = orders.max(0) - orders.min(0)
    kraus = _kraus(-span, span, g, (0.0, 0.0), phase_transform, r, a2, lam, area_ang2, rank_rtol)
    diff = orders[:, None, :] - orders[None, :, :] + span
    index = diff[..., 0] * (2 * span[1] + 1) + diff[..., 1]
    v = vectors[:, retained]
    n = v.shape[1]
    images = np.empty((kraus.shape[1], len(orders), n), dtype=complex)
    for start in range(0, kraus.shape[1], 32):
        block = kraus[index, start : start + 32]  # (K_out, K_in, j)
        images[start : start + 32] = np.einsum("pqj,qa->jpa", block, v, optimize=True)
    retained_jumps = np.einsum("pb,jpa->jba", v.conj(), images, optimize=True)
    complement = images - np.einsum("pb,jba->jpa", v, retained_jumps, optimize=True)
    outgoing_jumps = np.einsum("xp,jpa->jxa", position_basis(orders), complement, optimize=True)
    total = (
        v.conj().T
        @ loss_operator(orders, g, phase_transform, r, a2, lam, area_ang2, tail_orders)
        @ v
    )
    total = 0.5 * (total + total.conj().T)
    represented = np.einsum("jpa,jpb->ab", images.conj(), images, optimize=True)
    unresolved = total - 0.5 * (represented + represented.conj().T)
    minimum = float(np.linalg.eigvalsh(unresolved).min()) if n else 0.0
    if minimum < -_EFFECT_RTOL * max(float(np.linalg.norm(total, 2)) if n else 0.0, 1e-300):
        raise ValueError(
            "unresolved effect is not positive: order tail or rank truncation too coarse"
        )
    return TDSOutgoingMaps(retained_jumps, outgoing_jumps, unresolved, total, shape, minimum)


def gap_cut(energies: np.ndarray, target_eV: float, window_eV: float) -> float:
    """Midpoint of the widest level-free interval meeting ``target ± window`` [eV].

    Provisional numerical rule of the escape research note; not a physical
    matching surface. Validation: cr-axial-zone-handoff-accounting
    """
    levels = np.sort(np.asarray(energies, dtype=float))
    lo, hi = target_eV - window_eV, target_eV + window_eV
    below, above = levels[:-1], levels[1:]
    meets = (above > lo) & (below < hi)
    if not meets.any():
        raise ValueError("no spectral gap meets the matching window")
    widths = np.where(meets, above - below, -np.inf)
    i = int(np.argmax(widths))
    return float(0.5 * (below[i] + above[i]))


def report(max_order: int = 4, target_eV: float = 30.0, window_eV: float = 2.0) -> dict:
    """Bounded Si [001], 5 MeV, one-cell Gamma-point example (TDS only).

    Not a converged material prediction: one cell, small window, no electronic
    jumps, absorbing no-return model. The default 30 eV cut (24 retained
    states) keeps the dense (2n^2)^2 Liouvillian near 20 MiB per matrix; the
    provisional ~95 eV rule retains 77 states (about 2 GiB per matrix), which
    needs the trajectory unraveling rather than this dense oracle.
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.materials.channeling.axial import axial_profile, axial_states
    from pyrite.materials.channeling.transverse import longitudinal_kinematics
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_jump_instrument import (
        absorbing_axial_model,
        sample_absorbing_outgoing,
    )
    from pyrite.montecarlo.channeling.axial_phase_space import phase_space_stream
    from pyrite.montecarlo.channeling.operators import _potential

    energy = 5e6
    profile = axial_profile("silicon", (0, 0, 1))
    gamma, _ = longitudinal_kinematics(energy)
    dissipator = axial_dissipator(profile, energy, (1, 1), (8, 8), mechanisms=("tds",))
    (source,) = dissipator.tds_sources
    hbar_v = dissipator.hbar_v_eV_ang

    def phase(q):
        return -_potential(source.element, q) / hbar_v

    states = axial_states(profile, gamma, (0.0, 0.0), max_order)
    cut = gap_cut(states.energies_eV, target_eV, window_eV)
    retained = states.energies_eV < cut
    maps = tds_outgoing_maps(
        states.orders,
        states.coefficients,
        retained,
        profile.g_inv_ang,
        phase,
        source.position_ang,
        source.displacement_ang2,
        source.linear_density,
        profile.area_ang2,
    )
    entrance = states.coefficients[np.all(states.orders == 0, axis=1)][0].conj()[retained]
    model = absorbing_axial_model(
        np.diag(states.energies_eV[retained]),
        np.outer(entrance, entrance.conj()),
        retained_jumps_per_sqrt_ang=tuple(maps.retained_jumps),
        outgoing_jumps_per_sqrt_ang=tuple(maps.outgoing_jumps),
        unresolved_effect_inv_ang=maps.unresolved_effect,
        outgoing_shape=maps.outgoing_shape,
        hbar_v_eV_ang=hbar_v,
        max_working_bytes=1024**3,
    )
    rows = []
    for depth in (1e3, 1e4, 3e4):
        snap = model.at_depth(depth)
        kept = float(np.trace(snap.retained_density).real)
        out = float(np.trace(snap.outgoing_density).real)
        rows.append(
            {
                "depth_ang": depth,
                "retained": kept,
                "represented_outgoing": out,
                "unresolved": snap.unresolved_weight,
                "closure_defect": kept + out + snap.unresolved_weight - model.initial_weight,
            }
        )
    window = np.zeros(maps.outgoing_shape, dtype=complex)
    window[0, 0] = 1.0
    event = sample_absorbing_outgoing(model, window, 1e4, phase_space_stream(0, 0))
    return {
        "status": "research-only; TDS, one string, Gamma point of one cell, absorbing model",
        "crystal": "silicon",
        "axis": [0, 0, 1],
        "energy_eV": energy,
        "max_order": max_order,
        "cut_eV": cut,
        "retained_states": int(retained.sum()),
        "kraus_jumps": int(maps.retained_jumps.shape[0]),
        "entrance_weight": model.initial_weight,
        "unresolved_min_eigenvalue_inv_ang": maps.unresolved_min_eigenvalue,
        "rows": rows,
        "example_event": {
            "depth_ang": event.depth_ang,
            "position_index": list(event.position_index),
            "momentum_index": list(event.momentum_index),
            "outgoing_weight": event.outgoing_weight,
        },
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
