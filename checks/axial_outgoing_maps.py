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


def material_case(max_order: int, target_eV: float, window_eV: float = 2.0) -> dict:
    """Si [001], 5 MeV, one cell, Gamma point: TDS maps and model inputs.

    Returns keyword arguments shared by ``absorbing_axial_model`` and
    ``unravel_absorbing_outgoing`` plus ``cut_eV`` and ``retained_states``.
    The entrance is the zero-tilt plane wave projected on P (subunit trace).
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.materials.channeling.axial import axial_profile, axial_states
    from pyrite.materials.channeling.transverse import longitudinal_kinematics
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
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
    return {
        "model": dict(
            hamiltonian_eV=np.diag(states.energies_eV[retained]),
            initial_density=np.outer(entrance, entrance.conj()),
            retained_jumps_per_sqrt_ang=tuple(maps.retained_jumps),
            outgoing_jumps_per_sqrt_ang=tuple(maps.outgoing_jumps),
            unresolved_effect_inv_ang=maps.unresolved_effect,
            outgoing_shape=maps.outgoing_shape,
            hbar_v_eV_ang=hbar_v,
        ),
        "cut_eV": cut,
        "retained_states": int(retained.sum()),
        "kraus_jumps": int(maps.retained_jumps.shape[0]),
        "unresolved_min_eigenvalue_inv_ang": maps.unresolved_min_eigenvalue,
    }


def report(max_order: int = 4, target_eV: float = 30.0, window_eV: float = 2.0) -> dict:
    """Bounded Si [001], 5 MeV, one-cell Gamma-point example (TDS only).

    Not a converged material prediction: one cell, small window, no electronic
    jumps, absorbing no-return model. The default 30 eV cut (24 retained
    states) keeps the dense (2n^2)^2 Liouvillian near 20 MiB per matrix; the
    provisional ~95 eV rule retains 77 states (about 2 GiB per matrix), which
    needs the trajectory unraveling (:func:`unraveled_report`).
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.montecarlo.channeling.axial_jump_instrument import (
        absorbing_axial_model,
        sample_absorbing_outgoing,
    )
    from pyrite.montecarlo.channeling.axial_phase_space import phase_space_stream

    case = material_case(max_order, target_eV, window_eV)
    model = absorbing_axial_model(**case["model"], max_working_bytes=1024**3)
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
    shape = case["model"]["outgoing_shape"]
    window = np.zeros(shape, dtype=complex)
    window[0, 0] = 1.0
    event = sample_absorbing_outgoing(model, window, 1e4, phase_space_stream(0, 0))
    return {
        "status": "research-only; TDS, one string, Gamma point of one cell, absorbing model",
        "crystal": "silicon",
        "axis": [0, 0, 1],
        "energy_eV": 5e6,
        "max_order": max_order,
        **{k: case[k] for k in ("cut_eV", "retained_states", "kraus_jumps")},
        "unresolved_min_eigenvalue_inv_ang": case["unresolved_min_eigenvalue_inv_ang"],
        "entrance_weight": model.initial_weight,
        "rows": rows,
        "example_event": {
            "depth_ang": event.depth_ang,
            "position_index": list(event.position_index),
            "momentum_index": list(event.momentum_index),
            "outgoing_weight": event.outgoing_weight,
        },
    }


def unraveled_report(
    n_incidents: int = 2000,
    depths_ang=(1e3, 1e4, 3e4),
    *,
    seed: int = 0,
    cases=((4, 30.0, True), (6, 95.0, False)),
) -> dict:
    """Trajectory unraveling of the material maps, with the dense oracle where it fits.

    Each case is ``(max_order, target_eV, dense)``. Errors are binomial
    standard errors of incident-weighted tallies. Research only.
    Validation: cr-axial-zone-handoff-accounting
    """
    import time

    from pyrite.montecarlo.channeling.axial_jump_instrument import absorbing_axial_model
    from pyrite.montecarlo.channeling.axial_jump_unraveling import (
        event_tallies,
        unravel_absorbing_outgoing,
    )

    depths = np.asarray(depths_ang, dtype=float)
    out = []
    for max_order, target, dense in cases:
        case = material_case(max_order, target)
        shape = case["model"]["outgoing_shape"]
        window = np.zeros(shape, dtype=complex)
        window[0, 0] = 1.0
        clock = time.perf_counter()
        events = unravel_absorbing_outgoing(
            **case["model"],
            window=window,
            max_depth_ang=float(depths.max()),
            incident_ids=range(n_incidents),
            seed=seed,
            search_step_ang=200.0,
        )
        elapsed = time.perf_counter() - clock
        tallies = event_tallies(events, depths)
        weight = events[0].weight
        row = {
            "max_order": max_order,
            "cut_eV": case["cut_eV"],
            "retained_states": case["retained_states"],
            "kraus_jumps": case["kraus_jumps"],
            "entrance_weight": weight,
            "incidents": n_incidents,
            "seconds": elapsed,
            "mean_retained_jumps": float(np.mean([e.retained_jumps for e in events])),
            "depths_ang": depths.tolist(),
        }
        for kind, values in tallies.items():
            p = np.clip(values / weight, 0, 1)
            row[kind] = values.tolist()
            row[kind + "_sem"] = (weight * np.sqrt(p * (1 - p) / n_incidents)).tolist()
        if dense:
            model = absorbing_axial_model(**case["model"], max_working_bytes=1024**3)
            snaps = [model.at_depth(z) for z in depths]
            row["dense_outgoing"] = [float(np.trace(s.outgoing_density).real) for s in snaps]
            row["dense_unresolved"] = [s.unresolved_weight for s in snaps]
            row["dense_retained"] = [float(np.trace(s.retained_density).real) for s in snaps]
        out.append(row)
    return {
        "status": "research-only; TDS, one string, Gamma point of one cell, absorbing model",
        "seed": seed,
        "rows": out,
    }


def material_fiber_channels(
    max_order: int | tuple[int, int] = 6,
    target_eV: float = 95.0,
    window_eV: float = 2.0,
    *,
    electronic: bool = True,
    harmonic_max: int = 4,
    grid_points: int = 32,
    outgoing_order_max: int | tuple[int, int] | None = None,
    tds: str = "lattice",
    axis: tuple[int, int, int] = (0, 0, 1),
    tail_theta_max_rad: float | None = None,
):
    """Si, 5 MeV, one-cell fiber channels: TDS + electronic kicks.

    ``max_order`` = M or (M_x, M_y) is the retained order window and
    ``outgoing_order_max`` = B or (B_x, B_y) (default ``max_order``) the
    outgoing box; see :func:`matched_extents` for reach-matched pairs.
    ``tds="lattice"``: one-cell (image-correlated) Kraus columns spanning
    ``B + max_order`` so box outputs are exact; one string per cell only.
    ``tds="strings"``: independent-string continuum jumps, one
    :class:`TdsString` per dissipator source (any number of strings per cell).
    ``tail_theta_max_rad`` classifies string outputs beyond the box with
    ``|K| <= k theta_max`` as ``"tail"`` records (``cr-axial-tds-tail``); None
    keeps them unresolved.

    Kick columns are ``sqrt(c_i) sqrt(rho_{q_i})`` harmonics on the cell
    lattice truncated at ``|n| <= harmonic_max``; the dropped fraction of
    ``int rho_q`` is returned as ``kick_truncation`` (max over nodes). The
    truncated columns define the model's electronic loss consistently.
    Returns ``(channels, entry_orders, hbar_v, info)``.
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.materials.channeling.axial import axial_profile, axial_states
    from pyrite.materials.channeling.transverse import longitudinal_kinematics
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import FiberChannels
    from pyrite.montecarlo.channeling.axial_tds_continuum import TdsString
    from pyrite.montecarlo.channeling.operators import _potential

    if tds not in ("lattice", "strings"):
        raise ValueError("tds must be 'lattice' or 'strings'")
    energy = 5e6
    profile = axial_profile("silicon", axis)
    gamma, beta = longitudinal_kinematics(energy)
    k_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG  # p / hbar
    mechanisms = ("tds", "electronic") if electronic else ("tds",)
    dissipator = axial_dissipator(
        profile, energy, (1, 1), (grid_points, grid_points), mechanisms=mechanisms
    )
    hbar_v = dissipator.hbar_v_eV_ang
    g = np.asarray(profile.g_inv_ang, dtype=float)
    m = np.broadcast_to(np.asarray(max_order, dtype=int), (2,))
    box = m if outgoing_order_max is None else np.broadcast_to(np.asarray(outgoing_order_max), (2,))
    span = box + m
    window = (int(m[0]), int(m[1]))
    zero = axial_states(profile, gamma, (0.0, 0.0), window)

    def phase_of(element):
        return lambda q: -_potential(element, q) / hbar_v

    strings = ()
    if tds == "strings":
        strings = tuple(
            TdsString(
                tuple(src.position_ang),
                src.displacement_ang2,
                src.linear_density,
                phase_of(src.element),
            )
            for src in dissipator.tds_sources
        )
        kraus = np.zeros((int(np.prod(2 * span + 1)), 1))
        lattice_loss = np.zeros((len(zero.orders),) * 2)
    else:
        (source,) = dissipator.tds_sources
        r = np.asarray(source.position_ang, dtype=float)
        args = (phase_of(source.element), r, source.displacement_ang2, source.linear_density)
        kraus = _kraus(-span, span, g, (0.0, 0.0), *args, profile.area_ang2, 1e-14)
        lattice_loss = loss_operator(zero.orders, g, *args, profile.area_ang2, 300)
    h = harmonic_max
    columns, nodes, truncation = [], [], 0.0
    for q, c in zip(dissipator.electronic_q_inv_ang, dissipator.electronic_weights, strict=True):
        rho = dissipator.smoothed_density(float(q))
        harmonics = np.fft.fft2(np.sqrt(rho)) / rho.size
        column = harmonics[
            np.ix_(np.arange(-h, h + 1) % grid_points, np.arange(-h, h + 1) % grid_points)
        ]
        kept = float(np.sum(np.abs(column) ** 2))
        truncation = max(truncation, 1.0 - kept / float(np.sum(np.abs(harmonics) ** 2)))
        columns.append(np.sqrt(c) * column)
        nodes.append(float(q))
    if not columns:
        columns, nodes = [np.zeros((2 * h + 1, 2 * h + 1))], [0.0]

    def spectrum(kappa):
        states = axial_states(profile, gamma, kappa, window)
        return states.energies_eV, states.coefficients

    square = isinstance(max_order, int) and (
        outgoing_order_max is None or isinstance(outgoing_order_max, int)
    )
    box_extents = (int(box[0]), int(box[1]))
    channels = FiberChannels(
        order_max=max_order if square else window,
        reciprocal_inv_ang=(float(g[0]), float(g[1])),
        spectrum=spectrum,
        cut=lambda energies: gap_cut(energies, target_eV, window_eV),
        lattice_columns=kraus.T.reshape(-1, 2 * span[0] + 1, 2 * span[1] + 1),
        lattice_loss=lattice_loss,
        kick_columns=np.array(columns),
        kick_q_inv_ang=np.array(nodes),
        outgoing_order_max=int(box[0]) if square else box_extents,
        tds_strings=strings,
        tail_q_max_inv_ang=None if tail_theta_max_rad is None else k_inv_ang * tail_theta_max_rad,
    )
    entry = (np.all(zero.orders == 0, axis=1)).astype(complex)
    info = {
        "max_order": max_order if square else list(window),
        "window_orders": len(zero.orders),
        "target_eV": target_eV,
        "electronic": electronic,
        "kick_nodes": len(nodes),
        "harmonic_max": h,
        "kick_truncation": truncation,
        "lattice_kraus": int(kraus.shape[1]) if tds == "lattice" else 0,
        "outgoing_order_max": int(box[0]) if square else list(box_extents),
        "box_orders": int(np.prod(2 * box + 1)),
        "tds": tds,
        "axis": list(axis),
        "strings": len(dissipator.tds_sources),
        "k_inv_ang": k_inv_ang,
        "tail_q_max_inv_ang": channels.tail_q_max_inv_ang,
    }
    return channels, entry, hbar_v, info


def matched_extents(
    axis: tuple[int, int, int], reach_orders: int, box_orders: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reach-matched ``(M_x, M_y)`` and ``(B_x, B_y)`` for a Si axis.

    ``reach_orders`` and ``box_orders`` count orders along the smaller-``|g|``
    axis; the other axis gets the fewest orders reaching the same momentum.
    """
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import matched_order_extents

    g = np.asarray(axial_profile("silicon", axis).g_inv_ang, dtype=float)
    g_min = float(np.min(np.abs(g)))
    return (
        matched_order_extents(tuple(g), reach_orders * g_min),
        matched_order_extents(tuple(g), box_orders * g_min),
    )


def balanced_window(width) -> np.ndarray:
    """Unit-norm periodic Gaussian Weyl window with equal position/momentum spread.

    ``width`` is N or a per-axis shape (N_x, N_y). Width ``sqrt(N / 2 pi)``
    grid points on an N-point axis balances the two resolutions in units of
    the grid and the order spacing.
    """
    axes = []
    for n in np.broadcast_to(np.asarray(width, dtype=int), (2,)):
        sigma = np.sqrt(n / (2 * np.pi))
        d = np.minimum(np.arange(n), n - np.arange(n))
        axes.append(np.exp(-(d**2) / (4 * sigma**2)))
    window = np.outer(*axes).astype(complex)
    return window / np.linalg.norm(window)


def fiber_report(
    n_incidents: int = 1000,
    depths_ang=(1e3, 1e4, 3e4),
    *,
    seed: int = 0,
    electronic: bool = True,
    outgoing_order_max: int | tuple[int, int] | None = None,
    tds: str = "lattice",
    axis: tuple[int, int, int] = (0, 0, 1),
    max_order: int | tuple[int, int] = 6,
    target_eV: float = 95.0,
    incident_start: int = 0,
    keep_events: bool = False,
    tail_theta_max_rad: float | None = None,
) -> dict:
    """Fiber unraveling of Si 5 MeV with TDS (+ electronic) jumps. Research only.

    ``beyond_window`` is the fraction of outgoing events whose measured order
    lies outside the retained window ``|n_i| <= M_i``. Extents may be
    per-axis pairs (rectangular window and box). Incidents are
    ``incident_start + range(n_incidents)``; events depend only on their
    identity, so chunks run in separate processes and ``keep_events`` chunks
    combine exactly with :func:`merge_fiber_reports`.

    Validation: cr-axial-zone-handoff-accounting
    """
    import time

    from pyrite.montecarlo.channeling.axial_fiber_unraveling import unravel_fiber_outgoing

    channels, entry, hbar_v, info = material_fiber_channels(
        max_order,
        target_eV,
        electronic=electronic,
        outgoing_order_max=outgoing_order_max,
        tds=tds,
        axis=axis,
        tail_theta_max_rad=tail_theta_max_rad,
    )
    box = np.broadcast_to(np.asarray(info["outgoing_order_max"]), (2,))
    window = balanced_window(tuple(2 * box + 1))
    depths = np.asarray(depths_ang, dtype=float)
    clock = time.perf_counter()
    events = unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=hbar_v,
        window=window,
        max_depth_ang=float(depths.max()),
        incident_ids=range(incident_start, incident_start + n_incidents),
        seed=seed,
        search_step_ang=200.0,
    )
    info["seconds"] = time.perf_counter() - clock
    compact = [
        [
            e.incident_id,
            e.kind,
            e.depth_ang,
            e.lattice_jumps,
            e.kick_jumps,
            None if e.momentum_index is None else list(e.momentum_index),
            None if e.tail_momentum_inv_ang is None else float(np.hypot(*e.tail_momentum_inv_ang)),
        ]
        for e in events
    ]
    info.update(seed=seed, entrance_weight=events[0].weight, depths_ang=depths.tolist())
    info.update(_fiber_tallies(compact, info))
    if keep_events:
        info["events"] = compact
    return {"status": "research-only; one cell, absorbing model, fibred carrier", **info}


def _fiber_tallies(events, info) -> dict:
    """Weighted terminal tallies, binomial SEMs and mean jump counts of compact events."""
    n, weight = len(events), info["entrance_weight"]
    depths = np.asarray(info["depths_ang"], dtype=float)
    box = np.broadcast_to(np.asarray(info["outgoing_order_max"]), (2,))
    m = np.broadcast_to(np.asarray(info["max_order"]), (2,))
    rows = {"incidents": n}
    for kind in ("outgoing", "tail", "unresolved"):
        hits = np.array([sum(e[1] == kind and e[2] <= z for e in events) for z in depths])
        p = hits / n
        rows[kind] = (weight * p).tolist()
        rows[kind + "_sem"] = (weight * np.sqrt(p * (1 - p) / n)).tolist()
    terminal = sum(np.array(rows[k]) for k in ("outgoing", "tail", "unresolved"))
    rows["retained"] = (weight - terminal).tolist()
    tails = [e[6] for e in events if e[1] == "tail"]
    if tails:
        rows["tail_k_quantiles_inv_ang"] = np.quantile(tails, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
    orders = [
        np.where(np.array(e[5]) > box, np.array(e[5]) - 2 * box - 1, e[5])
        for e in events
        if e[1] == "outgoing"
    ]
    rows["beyond_window"] = float(
        np.mean([np.any(np.abs(o) > m) for o in orders]) if orders else 0.0
    )
    rows["mean_lattice_jumps"] = float(np.mean([e[3] for e in events]))
    rows["mean_kick_jumps"] = float(np.mean([e[4] for e in events]))
    return rows


def merge_fiber_reports(parts) -> dict:
    """Combine ``keep_events`` chunks of one configuration (disjoint incidents).

    Refuses chunks whose configuration differs or whose incidents overlap.
    Validation: cr-axial-zone-handoff-accounting
    """
    parts = list(parts)
    variable = {"events", "seconds", "incidents", "outgoing", "outgoing_sem", "unresolved"}
    variable |= {"unresolved_sem", "retained", "beyond_window", "mean_lattice_jumps"}
    variable |= {"mean_kick_jumps", "tail", "tail_sem", "tail_k_quantiles_inv_ang"}
    head = {k: v for k, v in parts[0].items() if k not in variable}
    for part in parts[1:]:
        if {k: v for k, v in part.items() if k not in variable} != head:
            raise ValueError("chunks differ in configuration")
    events = [e for part in parts for e in part["events"]]
    if len({e[0] for e in events}) != len(events):
        raise ValueError("chunks overlap in incident identity")
    events.sort(key=lambda e: e[0])
    merged = {**head, **_fiber_tallies(events, head)}
    merged["chunks"] = len(parts)
    merged["seconds"] = [part["seconds"] for part in parts]
    return merged


if __name__ == "__main__":
    import sys

    mode = sys.argv[1:2]
    if mode == ["unravel"]:
        result = unraveled_report()
    elif mode == ["fiber"]:
        result = fiber_report(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    elif mode == ["fiber-chunk"]:
        # fiber-chunk '<json fiber_report kwargs>' <incident_start> <n> <out.json>
        kwargs = json.loads(sys.argv[2])
        for key in ("max_order", "outgoing_order_max", "axis"):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(kwargs[key])
        result = fiber_report(
            int(sys.argv[4]), incident_start=int(sys.argv[3]), keep_events=True, **kwargs
        )
        Path(sys.argv[5]).write_text(json.dumps(result))
        result = {k: v for k, v in result.items() if k != "events"}
    elif mode == ["fiber-merge"]:
        # fiber-merge <out.json> <chunk.json>...
        result = merge_fiber_reports(json.loads(Path(p).read_text()) for p in sys.argv[3:])
        Path(sys.argv[2]).write_text(json.dumps(result, indent=1))
    else:
        result = report()
    print(json.dumps(result, indent=2))
