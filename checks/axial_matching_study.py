"""Research diagnostic: energy-defined retained projector and return-flux study.

Run the bounded CPU example with::

    uv run python checks/axial_matching_study.py          # Gamma point, ~25 s
    uv run python checks/axial_matching_study.py bloch    # Bloch-k grids, 4-8 min

The retained projector is the spectral projector P(E_m) of the transverse
Hamiltonian, so [H_perp, P] = 0 and the closure of the escape effects in
docs/research/physics/axial-escape-instrument.md holds exactly. TDS jumps of
one supercell-periodic string give eigenstate transition rates W[b, a]; the
populations then follow the secular (Pauli) rate equation. Absorbing all
weight above E_m discards its return flux; comparing the below-barrier
population with a reference that keeps every represented state measures that
omission. Coherences between nondegenerate states are dropped, electronic
jumps are excluded, and Bloch momenta are sampled on a cells x cells grid of
one cell (the Gamma point alone by default).
This is not a dechanneling prediction or a native launch model.

Source: Lindblad gain and jump probabilities of Castin, Dalibard & Molmer,
arXiv:0805.4002, Eqs. (2), (8); TDS covariance in cr-axial-realspace-dissipator;
secular rate equation as in Breuer & Petruccione, The Theory of Open Quantum
Systems (2002), Sec. 3.3.1. Rates per angstrom; zero displacement or string
density gives zero rates and frozen populations.

Validation: cr-axial-zone-handoff-accounting
"""

from __future__ import annotations

import json

import numpy as np


def periodic_tds_rates(
    orders: np.ndarray,
    vectors: np.ndarray,
    reciprocal_inv_ang: tuple[float, float],
    phase_transform,
    position_ang: tuple[float, float],
    displacement_ang2: float,
    linear_density_inv_ang: float,
    area_ang2: float,
    tail_orders: int = 300,
    rank_rtol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(W, total)``: in-basis transition rates and full loss per state.

    ``orders`` (n, 2) integer plane-wave orders on the rectangular reciprocal
    lattice ``reciprocal_inv_ang``; ``vectors`` (n, k) orthonormal columns in
    that basis. ``phase_transform(q)`` is the real, even per-atom phase
    transform [Å²]; the phase repeats in every image of the cell, so transfers
    are lattice orders. ``W[b, a] = λ E_u |<b|J_u|a>|²`` uses every transfer
    that connects represented orders, through a Kraus factorization of the
    transfer covariance truncated at ``rank_rtol``. ``total[a] = <a|Γ|a>``
    sums the variance harmonics over orders ``|n| <= tail_orders`` per axis;
    ``total - W.sum(0)`` is the weight leaving the represented space.

    Source: Gaussian covariance of the TDS jumps (cr-axial-realspace-dissipator)
    and the Lindblad gain/loss of Castin et al., arXiv:0805.4002 Eq. (2).
    Zero displacement or density gives zero. Units 1/Å.
    Validation: cr-axial-zone-handoff-accounting
    """
    orders = np.asarray(orders)
    vectors = np.asarray(vectors, dtype=complex)
    if orders.ndim != 2 or orders.shape[1] != 2 or not np.issubdtype(orders.dtype, np.integer):
        raise ValueError("orders must be an array of integer pairs")
    if vectors.shape[0] != len(orders):
        raise ValueError("vectors must be expressed in the order basis")
    for value in (displacement_ang2, linear_density_inv_ang):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(
                "displacement variance and string density must be finite and nonnegative"
            )
    g = np.asarray(reciprocal_inv_ang, dtype=float)
    r = np.asarray(position_ang, dtype=float)
    a2, lam = float(displacement_ang2), float(linear_density_inv_ang)
    rates = _kraus_rates(
        (orders, vectors),
        (orders, vectors),
        g,
        (0.0, 0.0),
        phase_transform,
        r,
        a2,
        lam,
        area_ang2,
        rank_rtol,
    )
    total = vectors_loss(orders, vectors, g, phase_transform, r, a2, lam, area_ang2, tail_orders)
    return rates, total


def _kraus_rates(
    incoming, outgoing, g, shift, phase_transform, r, a2, lam, area, rank_rtol, cache=None
):
    """``λ E_u |<b|J_u|a>|²`` for transfers ``q = shift + n g`` between two bases.

    ``n`` is the outgoing minus incoming integer order. Kraus factorization of
    the transfer covariance, truncated at relative eigenvalue ``rank_rtol``;
    ``cache`` reuses it for blocks with the same shift and order ranges.
    Validation: cr-axial-zone-handoff-accounting
    """
    (in_orders, in_vectors), (out_orders, out_vectors) = incoming, outgoing
    in_vectors = np.asarray(in_vectors, dtype=complex)
    out_vectors = np.asarray(out_vectors, dtype=complex)
    low = out_orders.min(0) - in_orders.max(0)
    high = out_orders.max(0) - in_orders.min(0)
    key = (tuple(np.round(np.asarray(shift, dtype=float), 12)), tuple(low), tuple(high))
    if cache is not None and key in cache:
        kraus = cache[key]
    else:
        kraus = _kraus(low, high, g, shift, phase_transform, r, a2, lam, area, rank_rtol)
        if cache is not None:
            cache[key] = kraus
    # Transfer index of every (outgoing, incoming) pair of represented orders.
    diff = out_orders[:, None, :] - in_orders[None, :, :] - low
    index = diff[..., 0] * (high[1] - low[1] + 1) + diff[..., 1]
    rates = np.zeros((out_vectors.shape[1], in_vectors.shape[1]))
    for start in range(0, kraus.shape[1], 32):
        block = kraus[index, start : start + 32]  # (n_out, n_in, j)
        amplitude = np.einsum(
            "pb,pqj,qa->jba", out_vectors.conj(), block, in_vectors, optimize=True
        )
        rates += np.sum(np.abs(amplitude) ** 2, axis=0)
    return rates


def _kraus(low, high, g, shift, phase_transform, r, a2, lam, area, rank_rtol):
    """Columns ``B`` with ``B B^dagger`` the TDS transfer covariance on ``shift + n g``."""
    grid = np.stack(
        np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(low, high, strict=True)), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    q = np.asarray(shift, dtype=float) + grid * g
    c = phase_transform(np.hypot(q[:, 0], q[:, 1])) / area * np.exp(-1j * q @ r)
    q2 = np.sum(q * q, axis=1)
    covariance = np.exp(-0.5 * a2 * np.sum((q[:, None] - q[None]) ** 2, axis=2))
    covariance = covariance - np.exp(-0.5 * a2 * (q2[:, None] + q2[None]))
    covariance = lam * c[:, None] * c.conj()[None] * covariance
    weights, modes = np.linalg.eigh(covariance)
    keep = weights > rank_rtol * max(float(weights.max()), 0.0)
    return modes[:, keep] * np.sqrt(weights[keep])


def bloch_tds_rates(
    blocks,
    cells: int,
    reciprocal_inv_ang: tuple[float, float],
    phase_transform,
    position_ang: tuple[float, float],
    displacement_ang2: float,
    linear_density_inv_ang: float,
    area_ang2: float,
    tail_orders: int = 300,
    rank_rtol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Rates and losses over Bloch blocks ``kappa = (i, j) g / cells`` of one cell.

    ``blocks`` is a sequence of ``((i, j), orders, vectors)``: eigenvectors of
    the Bloch Hamiltonian at that ``kappa`` in cell orders. This is the
    ``cells x cells`` supercell with one independent string per image: the
    Hamiltonian is block diagonal, and a jump from ``kappa`` to ``kappa'``
    transfers ``q = kappa' - kappa + n g``. Summing the images' equal
    squared amplitudes over supercell-normalized states leaves the one-cell
    formula with area ``cells * area_ang2``. Loss harmonics are cell vectors,
    summed on the ``g / cells`` lattice. ``cells = 1`` is
    :func:`periodic_tds_rates`. Returns the global ``(W, total)`` in block
    order.

    Source: Bloch decomposition of the supercell form of the TDS jumps in
    cr-axial-realspace-dissipator, with gain and loss from Castin et al.,
    arXiv:0805.4002 Eq. (2). Units 1/Å; zero displacement gives zero.
    Validation: cr-axial-zone-handoff-accounting
    """
    g = np.asarray(reciprocal_inv_ang, dtype=float)
    r = np.asarray(position_ang, dtype=float)
    a2, lam = float(displacement_ang2), float(linear_density_inv_ang)
    area = cells * area_ang2
    sizes = [np.asarray(v).shape[1] for _, _, v in blocks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    rates = np.zeros((offsets[-1], offsets[-1]))
    total = np.zeros(offsets[-1])
    cache: dict = {}
    for a, (kappa_in, orders_in, vectors_in) in enumerate(blocks):
        orders_in = np.asarray(orders_in)
        fine = orders_in * cells + np.asarray(kappa_in)
        total[offsets[a] : offsets[a + 1]] = vectors_loss(
            fine,
            vectors_in,
            g / cells,
            phase_transform,
            r,
            a2,
            lam,
            area,
            tail_orders * cells,
        )
        for b, (kappa_out, orders_out, vectors_out) in enumerate(blocks):
            shift = (np.asarray(kappa_out) - np.asarray(kappa_in)) * g / cells
            rates[offsets[b] : offsets[b + 1], offsets[a] : offsets[a + 1]] = _kraus_rates(
                (orders_in, vectors_in),
                (np.asarray(orders_out), vectors_out),
                g,
                shift,
                phase_transform,
                r,
                a2,
                lam,
                area,
                rank_rtol,
                cache,
            )
    return rates, total


def vectors_loss(orders, vectors, g, phase_transform, r, a2, lam, area, tail_orders):
    """``<a|Γ|a>`` from harmonics ``Γ_G = λ Σ_n c_n c*_{n-G}[e^{-a²G²/2} - χ_n χ_{n-G}]``.

    Correlations use an FFT over orders ``|n| <= tail_orders``; harmonics with
    ``G`` beyond the represented differences are not needed.
    Validation: cr-axial-zone-handoff-accounting
    """
    n = np.arange(-tail_orders, tail_orders + 1)
    qx, qy = n[:, None] * g[0], n[None, :] * g[1]
    c = phase_transform(np.hypot(qx, qy)) / area * np.exp(-1j * (qx * r[0] + qy * r[1]))
    damped = c * np.exp(-0.5 * a2 * (qx**2 + qy**2))
    size = 2 * (2 * tail_orders + 1)

    def correlation(f):
        spectrum = np.fft.fft2(f, s=(size, size))
        return np.fft.ifft2(spectrum * spectrum.conj())  # Σ_n f_n f*_{n-G} at index G

    span = orders.max(0) - orders.min(0)
    gx = np.arange(-span[0], span[0] + 1)
    gy = np.arange(-span[1], span[1] + 1)
    first = correlation(c)[np.ix_(gx % size, gy % size)]
    second = correlation(damped)[np.ix_(gx % size, gy % size)]
    gg = (gx[:, None] * g[0]) ** 2 + (gy[None, :] * g[1]) ** 2
    harmonic = lam * (np.exp(-0.5 * a2 * gg) * first - second)
    diff = orders[:, None, :] - orders[None, :, :] + span
    loss = harmonic[diff[..., 0], diff[..., 1]]
    return np.einsum("pa,pq,qa->a", vectors.conj(), loss, vectors).real


def retained_populations(rates, total, initial, retained, depths_ang) -> dict[str, np.ndarray]:
    """Absorbing secular model on ``retained`` states, with its flux balance.

    ``dn/dz = (W_PP - diag(total_P)) n``: every jump out of the retained set
    is absorbed, into represented states ``F`` (rate ``W_FP``) or out of the
    represented space (rate ``total - W.sum(0)``). Returns populations
    ``(len(depths), n_states)``, zero outside the retained set, and the two
    cumulative absorbed weights. Their sum with the retained population equals
    the initial retained weight. ``exposure`` is ``int_0^z n dz`` [Å], the
    depth-integrated population that weights an emission rate per state.
    Zero rates keep ``n`` constant.

    Source: traced retained generator, eq-axial-retained-generator of
    docs/research/physics/axial-escape-instrument.md, in the secular limit.
    Validation: cr-axial-zone-handoff-accounting
    """
    keep = np.flatnonzero(retained)
    generator = rates[np.ix_(keep, keep)] - np.diag(total[keep])
    values, right = np.linalg.eig(generator)
    coefficients = np.linalg.solve(right, initial[keep])
    to_f = rates[np.ix_(np.flatnonzero(~retained), keep)].sum(0)
    to_o = (total - rates.sum(0))[keep]
    out = np.zeros((len(depths_ang), len(initial)))
    escaped_f = np.zeros(len(depths_ang))
    escaped_o = np.zeros(len(depths_ang))
    exposure = np.zeros_like(out)
    for i, z in enumerate(depths_ang):
        out[i, keep] = (right @ (np.exp(values * z) * coefficients)).real
        small = np.abs(values * z) < 1e-12
        safe = np.where(small, 1.0, values)
        integral = np.where(small, z, np.expm1(values * z) / safe)
        cumulative = (right @ (integral * coefficients)).real
        escaped_f[i] = to_f @ cumulative
        escaped_o[i] = to_o @ cumulative
        exposure[i, keep] = cumulative
    return {
        "populations": out,
        "exposure": exposure,
        "escaped_represented": escaped_f,
        "escaped_unresolved": escaped_o,
    }


def bloch_report(max_order: int = 6, cells_list=(1, 2, 3)) -> dict:
    """Bloch-k (supercell) sensitivity of the return-flux deficit, Si [001] 5 MeV.

    Same entrance and matching energies as :func:`report`, sampled on
    ``cells x cells`` Bloch grids. Bounded grids only.
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
    u_min, u_max = profile.extrema()
    depths = np.array([1e3, 1e4, 3e4, 1e5, 3e5])
    g = np.asarray(profile.g_inv_ang)

    def phase(q):
        return -_potential(source.element, q) / hbar_v

    rows = []
    for cells in cells_list:
        blocks, energies = [], []
        # Bloch momenta folded into (-g/2, g/2], so the truncated window is symmetric.
        folded = [i if 2 * i <= cells else i - cells for i in range(cells)]
        for i in folded:
            for j in folded:
                states = axial_states(
                    profile, gamma, tuple(np.array([i, j]) * g / cells), max_order
                )
                blocks.append(((i, j), states.orders, states.coefficients))
                energies.append(states.energies_eV)
        rates, total = bloch_tds_rates(
            blocks,
            cells,
            profile.g_inv_ang,
            phase,
            source.position_ang,
            source.displacement_ang2,
            source.linear_density,
            profile.area_ang2,
        )
        energies = np.concatenate(energies)
        initial = np.zeros(energies.size)
        origin = blocks[0]
        initial[: origin[2].shape[1]] = np.abs(origin[2][np.all(origin[1] == 0, axis=1)][0]) ** 2
        bound = energies < u_max
        reference = retained_populations(rates, total, initial, np.ones_like(bound), depths)
        ref_x = reference["exposure"][:, bound].sum(1)
        for target in (u_max, u_max + (u_max - u_min), u_max + 2 * (u_max - u_min), np.inf):
            if np.isfinite(target):
                cut = 0.5 * (energies[energies <= target].max() + energies[energies > target].min())
            else:
                cut = np.inf
            model = retained_populations(rates, total, initial, energies <= cut, depths)
            rows.append(
                {
                    "cells": cells,
                    "max_order": max_order,
                    "matching_energy_eV": None if not np.isfinite(cut) else float(cut),
                    "bound_population": [float(x) for x in model["populations"][:, bound].sum(1)],
                    "bound_exposure_ang": [float(x) for x in model["exposure"][:, bound].sum(1)],
                    "reference_bound_exposure_ang": [float(x) for x in ref_x],
                }
            )
    return {"depths_ang": depths.tolist(), "rows": rows}


def report() -> dict:
    """Bound-population sensitivity to the matching energy, Si [001] at 5 MeV.

    Plane wave at zero tilt; bound means transverse energy below the cell
    maximum of U. Bounded grids only: not converged material predictions.
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
    u_min, u_max = profile.extrema()
    depths = np.array([1e3, 1e4, 3e4, 1e5, 3e5])

    def phase(q):
        return -_potential(source.element, q) / hbar_v

    rows = []
    for max_order in (6, 8, 10):
        states = axial_states(profile, gamma, (0.0, 0.0), max_order)
        rates, total = periodic_tds_rates(
            states.orders,
            states.coefficients,
            profile.g_inv_ang,
            phase,
            source.position_ang,
            source.displacement_ang2,
            source.linear_density,
            profile.area_ang2,
        )
        energies = states.energies_eV
        initial = np.abs(states.coefficients[np.all(states.orders == 0, axis=1)][0]) ** 2
        bound = energies < u_max
        # Matching energies in spectral gaps, so no degenerate multiplet is split.
        cuts = []
        for target in (
            u_max,
            u_max + 0.5 * (u_max - u_min),
            u_max + (u_max - u_min),
            u_max + 2 * (u_max - u_min),
            np.inf,
        ):
            above = energies[energies > target]
            cuts.append(
                np.inf
                if above.size == 0 or not np.isfinite(target)
                else 0.5 * (energies[energies <= target].max() + above.min())
            )
        reference = retained_populations(rates, total, initial, np.ones_like(bound), depths)
        reference_bound = reference["populations"][:, bound].sum(1)
        reference_exposure = reference["exposure"][:, bound].sum(1)
        for cut in cuts:
            retained = energies <= cut
            model = retained_populations(rates, total, initial, retained, depths)
            n = model["populations"]
            balance = n.sum(1) + model["escaped_represented"] + model["escaped_unresolved"]
            rows.append(
                {
                    "max_order": max_order,
                    "matching_energy_eV": None if not np.isfinite(cut) else float(cut),
                    "retained_states": int(retained.sum()),
                    "entrance_outside_retained": float(initial[~retained].sum()),
                    "bound_population": [float(x) for x in n[:, bound].sum(1)],
                    "reference_bound_population": [float(x) for x in reference_bound],
                    "bound_exposure_ang": [float(x) for x in model["exposure"][:, bound].sum(1)],
                    "reference_bound_exposure_ang": [float(x) for x in reference_exposure],
                    "escaped_represented": [float(x) for x in model["escaped_represented"]],
                    "escaped_unresolved": [float(x) for x in model["escaped_unresolved"]],
                    "flux_balance_defect": float(np.abs(balance - initial[retained].sum()).max()),
                }
            )
        rows[-1]["leaving_represented_rate_bound_mean_inv_ang"] = float(
            np.average((total - rates.sum(0))[bound], weights=initial[bound])
        )
    return {
        "status": "research-only; Pauli rates, TDS only, Gamma point of one cell",
        "crystal": "silicon",
        "axis": [0, 0, 1],
        "energy_eV": energy,
        "potential_min_max_eV": [u_min, u_max],
        "depths_ang": depths.tolist(),
        "rows": rows,
    }


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["bloch"]:
        result = bloch_report(6, (1, 2, 3))
        result["rows"] += bloch_report(8, (1, 2))["rows"]
    else:
        result = report()
    print(json.dumps(result, indent=2, allow_nan=False))
