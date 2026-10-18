"""Research diagnostic: a TDS diagonal sink is not a momentum escape instrument.

Run the bounded CPU example with::

    uv run python checks/axial_escape_accounting.py

For a plane wave with integer momentum order p, independent TDS sources have
transfer rates lambda * |s_q|**2 * (1 - exp(-a**2 * |q|**2)). Count p + q
outside a declared Fourier state window without wrapping it back into the FFT.
The phase coefficients and source data come from the diagnostic dissipator.
This is finite-transfer bookkeeping, not a dechanneling model or an angular
distribution for native transport. Electronic jumps are outside its scope.
The coherent helper constructs the finite-transfer gain for general incident
density matrices. The continuum helper integrates the independent-atom gain
over a momentum box and closes its total on the full diagonal C(r, r), so the
unresolved weight carries no transfer cutoff. No MCWF histories or transport
are run.

Source: the TDS covariance in cr-axial-realspace-dissipator; jump probabilities
in Castin, Dalibard & Molmer, arXiv:0805.4002, Eqs. (2), (8). The window split
is derived in docs/research/physics/axial-escape-instrument.md. Rates are per Å;
zero jumps give zero retained and escaped rate, and an enclosing window has
zero escape. The transfer cutoff remains explicitly numerical.

Validation: cr-axial-zone-handoff-accounting
"""

from __future__ import annotations

import json

import numpy as np


def plane_wave_escape_rates(
    transfer_rates: np.ndarray,
    orders: tuple[np.ndarray, np.ndarray],
    retained_shape: tuple[int, int],
    incoming_order: tuple[int, int],
) -> dict[str, float]:
    """Partition plane-wave jump rates by a Fourier window, without aliasing.

    ``orders`` are integer transfer orders along each axis. The represented
    state window contains -n/2 through n/2-1 on each axis. Rates are positive
    per-length weights of independent channels, not complex amplitudes.
    Assumes one incoming plane wave, not a general coherent superposition.
    Zero input gives zero rates; an enclosing window retains all transfers.

    Source: orthogonal P/Q decomposition of the Lindblad gain operator, derived
    in docs/research/physics/axial-escape-instrument.md, from Castin et al.
    arXiv:0805.4002 Eq. (8). This is not a physical retained-state selection.
    Validation: cr-axial-zone-handoff-accounting
    """
    rates = np.asarray(transfer_rates, dtype=float)
    if rates.shape != (len(orders[0]), len(orders[1])):
        raise ValueError("transfer rates must match the two order axes")
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("transfer rates must be finite and nonnegative")
    for value in (*retained_shape, *incoming_order):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("window sizes and incident orders must be integers")
    if any(n < 2 or n % 2 for n in retained_shape):
        raise ValueError("window dimensions must be positive even integers")
    for axis in orders:
        if np.asarray(axis).ndim != 1 or not np.issubdtype(np.asarray(axis).dtype, np.integer):
            raise ValueError("transfer order axes must be one-dimensional integer arrays")
        if len(np.unique(axis)) != len(axis):
            raise ValueError("transfer order axes must contain unique orders")
    nx, ny = retained_shape
    px, py = incoming_order
    if not (-nx // 2 <= px < nx // 2 and -ny // 2 <= py < ny // 2):
        raise ValueError("incident order must be inside the state window")
    final_x = orders[0][:, None] + px
    final_y = orders[1][None, :] + py
    retained = (final_x >= -nx // 2) & (final_x < nx // 2)
    retained = retained & (final_y >= -ny // 2) & (final_y < ny // 2)
    return {
        "retained_rate_inv_ang": float(rates[retained].sum()),
        "window_escape_rate_inv_ang": float(rates[~retained].sum()),
        "total_rate_inv_ang": float(rates.sum()),
    }


def _check_density(rho: np.ndarray, size: int) -> None:
    if rho.shape != (size, size) or not np.isfinite(rho).all():
        raise ValueError("density must be a finite incident square matrix")
    if not np.allclose(rho, rho.conj().T, rtol=1e-12, atol=1e-14):
        raise ValueError("density must be Hermitian positive semidefinite")
    if np.linalg.eigvalsh(rho).min() < -1e-12 * max(float(np.linalg.norm(rho)), 1e-30):
        raise ValueError("density must be Hermitian positive semidefinite")


def coherent_tds_gain(
    incoming_orders: np.ndarray,
    density: np.ndarray,
    transfer_orders: np.ndarray,
    transfer_wavevectors: np.ndarray,
    phase_coefficients: np.ndarray,
    displacement_ang2: float,
    linear_density_inv_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite-transfer outgoing density per Å, without FFT wrapping.

    Integer order pairs label orthonormal plane waves. ``density`` is a positive
    incident density matrix, with its incident weight in its trace. Wavevectors
    are Cartesian inverse Å; phase coefficients are dimensionless and include
    the source-position phase and cell-area normalization. One independent
    isotropic Gaussian-displaced string is included; add gains for independent
    strings. Output is (unique outgoing integer orders, gain matrix).

    Source: Gaussian characteristic-function covariance of the TDS jumps,
    derived in docs/research/physics/axial-escape-instrument.md from the
    Lindblad gain in Castin et al., arXiv:0805.4002 Eq. (2). The zero-displacement
    and zero-density limits give zero gain. This finite polynomial is neither
    an infinite-transfer completion nor a physical escape instrument.
    Validation: cr-axial-zone-handoff-accounting
    """
    incoming = np.asarray(incoming_orders)
    transfers = np.asarray(transfer_orders)
    rho = np.asarray(density, dtype=complex)
    q = np.asarray(transfer_wavevectors, dtype=float)
    coefficients = np.asarray(phase_coefficients, dtype=complex)
    for orders in (incoming, transfers):
        if orders.ndim != 2 or orders.shape[1] != 2 or len(orders) == 0:
            raise ValueError("orders must be nonempty arrays of integer pairs")
        if not np.issubdtype(orders.dtype, np.integer):
            raise ValueError("orders must be nonempty arrays of integer pairs")
        if len(np.unique(orders, axis=0)) != len(orders):
            raise ValueError("orders must be unique")
    _check_density(rho, len(incoming))
    if q.shape != transfers.shape or not np.isfinite(q).all():
        raise ValueError("wavevectors must be finite Cartesian pairs")
    if coefficients.shape != (len(transfers),) or not np.isfinite(coefficients).all():
        raise ValueError("phase coefficients must match the transfer orders")
    if not all(np.isfinite(x) and x >= 0 for x in (displacement_ang2, linear_density_inv_ang)):
        raise ValueError("displacement variance and string density must be finite and nonnegative")
    q2 = np.sum(q * q, axis=1)
    difference2 = np.sum((q[:, None, :] - q[None, :, :]) ** 2, axis=2)
    covariance = np.exp(-0.5 * displacement_ang2 * difference2) - np.exp(
        -0.5 * displacement_ang2 * (q2[:, None] + q2[None, :])
    )
    covariance = (
        linear_density_inv_ang * coefficients[:, None] * coefficients.conj()[None, :] * covariance
    )
    outgoing, indices = np.unique(
        (incoming[:, None, :] + transfers[None, :, :]).reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    # Each (incident, transfer) pair maps to its literal sum, never modulo n.
    indices = indices.reshape(len(incoming), len(transfers))
    gain = np.zeros((len(outgoing), len(outgoing)), dtype=complex)
    contributions = rho[:, None, :, None] * covariance[None, :, None, :]
    row = np.broadcast_to(indices[:, :, None, None], contributions.shape)
    column = np.broadcast_to(indices[None, None, :, :], contributions.shape)
    np.add.at(gain, (row.ravel(), column.ravel()), contributions.ravel())
    return outgoing, gain


def continuum_tds_window_split(
    incoming_wavevectors: np.ndarray,
    density: np.ndarray,
    phase_transform,
    diagonal_transform,
    position_ang: tuple[float, float],
    displacement_ang2: float,
    linear_density_inv_ang: float,
    area_ang2: float,
    window_edges: tuple[np.ndarray, np.ndarray],
    nodes: int = 6,
) -> dict[str, float]:
    """Split one independent-atom string's full TDS gain by a momentum box.

    ``incoming_wavevectors`` are Cartesian momenta [1/Å] of plane waves
    normalized on the supercell area ``area_ang2``; ``density`` is their
    positive incident matrix, unnormalized. ``phase_transform(q)`` is the real,
    even 2D transform of the per-atom phase [Å²]. ``diagonal_transform(G)`` is
    ``int d²q t(q-G/2) t(q+G/2) [exp(-a²G²/2) - exp(-a²(q²+G²/4))]`` [Å²],
    the transform behind the full diagonal ``C(r, r)``; it must use
    the same ``a²`` as ``displacement_ang2``. ``window_edges`` are
    increasing panel edges [1/Å]; ``nodes`` Gauss-Legendre points per panel.

    Each atom displaces independently, so outgoing momentum is continuous and
    no transfer cutoff or image correlation enters. The gain density

    ``g(k) = λ/(A(2π)²) Σ_ij ρ_ij e^{i(p_i-p_j)·R} t_i t_j [e^{-a²|p_i-p_j|²/2} - χ_i χ_j]``

    (``t_i = t(|k-p_i|)``, ``χ_i = exp(-a²|k-p_i|²/2)``) is pointwise
    nonnegative. Its total integral is ``λ/(A(2π)²) Σ_ij ρ_ij
    e^{i(p_i-p_j)·R} D(|p_i-p_j|)``, the expectation of the production full
    diagonal; the box integral is the represented weight, and their
    difference is the unresolved weight with no omitted transfer tail.
    Zero displacement or string density gives zero weights. The box is a
    declared numerical window, not a physical retained projector.

    Source: Lindblad gain of Castin et al., arXiv:0805.4002 Eq. (2), applied
    to the TDS jumps of cr-axial-realspace-dissipator; derived in
    docs/research/physics/axial-escape-instrument.md. Rates per Å.
    Validation: cr-axial-zone-handoff-accounting
    """
    p = np.asarray(incoming_wavevectors, dtype=float)
    rho = np.asarray(density, dtype=complex)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("incident wavevectors must be a nonempty array of finite pairs")
    _check_density(rho, len(p))
    for value in (displacement_ang2, linear_density_inv_ang):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(
                "displacement variance and string density must be finite and nonnegative"
            )
    if not (np.isfinite(area_ang2) and area_ang2 > 0):
        raise ValueError("cell area must be finite and positive")
    axes = []
    gl_x, gl_w = np.polynomial.legendre.leggauss(nodes)
    for edges in window_edges:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("window edges must be increasing")
        half = 0.5 * np.diff(edges)
        centre = 0.5 * (edges[1:] + edges[:-1])
        axes.append(
            ((centre[:, None] + half[:, None] * gl_x).ravel(), (half[:, None] * gl_w).ravel())
        )
    (kx, wx), (ky, wy) = axes
    r = np.asarray(position_ang, dtype=float)
    a2 = float(displacement_ang2)
    scale = linear_density_inv_ang / (area_ang2 * (2.0 * np.pi) ** 2)
    difference = p[:, None, :] - p[None, :, :]
    mixed = rho * np.exp(1j * difference @ r)
    dx = kx[None, :, None] - p[:, 0, None, None]
    dy = ky[None, None, :] - p[:, 1, None, None]
    transfer = np.hypot(dx, dy)
    t = phase_transform(transfer)
    damped = t * np.exp(-0.5 * a2 * transfer**2)
    coherence = np.exp(-0.5 * a2 * np.sum(difference**2, axis=2))
    gain = np.einsum("ij,ixy,jxy->xy", mixed * coherence, t, t) - np.einsum(
        "ij,ixy,jxy->xy", mixed, damped, damped
    )
    represented = scale * float(wx @ gain.real @ wy)
    lengths = np.sqrt(np.sum(difference**2, axis=2))
    unique, inverse = np.unique(np.round(lengths, 12), return_inverse=True)
    diagonal = np.asarray(diagonal_transform(unique), dtype=float)[inverse.reshape(lengths.shape)]
    total = scale * float(np.sum(mixed * diagonal).real)
    return {
        "represented_rate_inv_ang": represented,
        "unresolved_rate_inv_ang": total - represented,
        "total_rate_inv_ang": total,
    }


def report() -> dict:
    """Compare equal-density incident waves in one bounded Si TDS example.

    The transfer variance follows Gaussian characteristic functions applied to
    the sourced phase coefficients: E|exp(-i q.u)-E exp(-i q.u)|² equals
    1-exp(-a²q²). Strings and displacement draws add incoherently. Zero
    displacement gives zero transfer weights. This probes the existing
    diagnostic operator, not an independent validation of the atomic model.
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator

    profile = axial_profile("silicon", (0, 0, 1))
    # Deliberately small numerical windows, not convergence/predictive settings.
    shape = (16, 16)
    fine_shape = (32, 32)
    coarse = axial_dissipator(profile, 5e6, (1, 1), shape, mechanisms=("tds",))
    fine = axial_dissipator(profile, 5e6, (1, 1), fine_shape, mechanisms=("tds",))
    k2 = fine.grid.kx[:, None] ** 2 + fine.grid.ky[None, :] ** 2
    transfers = np.zeros(fine_shape)
    for source in fine.tds_sources:
        transfers += (
            source.linear_density
            * np.abs(fine.phase_spectra[source.element]) ** 2
            * -np.expm1(-source.displacement_ang2 * k2)
        )
    orders = tuple(np.rint(np.fft.fftfreq(n) * n).astype(int) for n in fine_shape)
    rows = []
    for incident in ((0, 0), (7, 0), (0, 7), (7, 7)):
        # Every normalized plane wave has the same uniform spatial density.
        rows.append(
            {
                **plane_wave_escape_rates(transfers, orders, shape, incident),
                "incoming_order": incident,
                "current_diagonal_sink_rate_inv_ang": float(coarse.tds_sink_rate.mean()),
            }
        )
    gain_rate = float(fine.tds_grid_rate.mean())
    defect = abs(float(transfers.sum()) - gain_rate)
    if not np.isfinite(gain_rate) or gain_rate <= 0:
        raise RuntimeError("nonpositive/nonfinite TDS variance")
    if defect > 1e-11 * gain_rate:
        raise RuntimeError("Fourier transfer normalization disagrees with grid variance")
    return {
        "status": "research-only; no physical escape boundary selected",
        "crystal": "silicon",
        "axis": [0, 0, 1],
        "energy_eV": 5e6,
        "temperature_K": float(profile.temperature_K),
        "cells": [1, 1],
        "state_window_shape": shape,
        "phase_transfer_shape": fine_shape,
        "transfer_normalization_defect_inv_ang": defect,
        "rows": rows,
        "continuum": continuum_report(),
    }


def continuum_report() -> dict:
    """Cutoff-free window split and its convergence in the same Si example.

    Uses the production phase transform ``-v(q)/hbar v`` and full-diagonal
    transform of the diagnostic dissipator for one independent-atom string.
    Reports the unresolved weight against window size, quadrature order, a
    coherent two-wave interference probe, the production grid expectation of
    the no-jump loss, and the supercell-periodic lattice sum approaching the
    continuum box integral (it is that integral's midpoint rule). Settings are
    bounded diagnostics, not converged predictions.
    Validation: cr-axial-zone-handoff-accounting
    """
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.operators import _potential, _tds_transform

    profile = axial_profile("silicon", (0, 0, 1))
    shape = (16, 16)
    dissipator = axial_dissipator(profile, 5e6, (1, 1), shape, mechanisms=("tds",))
    hbar_v = dissipator.hbar_v_eV_ang
    (source,) = dissipator.tds_sources
    a2 = source.displacement_ang2
    area = dissipator.grid.area_ang2
    spacing = 2.0 * np.pi / np.asarray(dissipator.grid.extent_ang)

    def phase(q):
        return -_potential(source.element, q) / hbar_v

    def diagonal(g):
        return _tds_transform(source.element, a2, g) / hbar_v**2

    def split(orders, rho, n, nodes=6, cells=1):
        edges = [(np.arange(-n // 2, n // 2 + 1) - 0.5) * d / cells for d in spacing]
        k = np.asarray(orders, dtype=float) * spacing / cells
        return continuum_tds_window_split(
            k,
            rho,
            phase,
            diagonal,
            source.position_ang,
            a2,
            source.linear_density,
            area,
            tuple(edges),
            nodes,
        )

    windows = []
    for n in (16, 32, 64, 128):
        for incident in ((0, 0), (7, 0), (7, 7)):
            windows.append(
                {"window": n, "incoming_order": incident, **split([incident], np.eye(1), n)}
            )
    quadrature = [{"nodes": m, **split([(7, 7)], np.eye(1), 16, m)} for m in (4, 6, 10)]
    # Equal mixture and in-/anti-phase superpositions of orders (0,0), (1,0).
    orders = [(0, 0), (1, 0)]
    coherent = []
    grid = dissipator.grid
    x = np.arange(shape[0]) * grid.extent_ang[0] / shape[0]
    for label, rho in (
        ("mixture", np.eye(2) / 2),
        ("in_phase", np.full((2, 2), 0.5)),
        ("antiphase", np.array([[0.5, -0.5], [-0.5, 0.5]])),
    ):
        row = {"state": label, **split(orders, rho, 16)}
        # Production no-jump loss: grid samples of C(r, r) weighted by |psi|^2.
        waves = np.exp(1j * np.multiply.outer(spacing[0] * np.array([0, 1]), x))
        density_r = np.einsum("ij,ix,jx->x", rho, waves, waves.conj()).real
        row["production_grid_loss_inv_ang"] = float(
            np.mean(density_r[:, None] * dissipator.total_rate)
        )
        coherent.append(row)
    lattice = []
    for cells in (1, 2, 4):
        # Image-correlated periodic phase: a midpoint rule of the continuum integrand.
        n = 31 * cells
        orders = np.arange(-(n // 2), n // 2 + 1)
        qx = orders[:, None] * spacing[0] / cells
        qy = orders[None, :] * spacing[1] / cells
        q = np.hypot(qx, qy)
        periodic = (
            source.linear_density
            / area
            / (cells**2 * area)
            * float(np.sum(phase(q) ** 2 * -np.expm1(-a2 * q**2)))
        )
        edges = [(np.arange(-(n // 2), n // 2 + 2) - 0.5) * d / cells for d in spacing]
        box = continuum_tds_window_split(
            np.zeros((1, 2)),
            np.eye(1),
            phase,
            diagonal,
            source.position_ang,
            a2,
            source.linear_density,
            area,
            tuple(edges),
            6,
        )["represented_rate_inv_ang"]
        lattice.append(
            {
                "cells": cells,
                "periodic_lattice_rate_inv_ang": periodic,
                "continuum_box_rate_inv_ang": box,
            }
        )
    return {
        "status": "numerical window only; no physical retained projector",
        "production_mean_loss_inv_ang": float(dissipator.total_rate.mean()),
        "windows": windows,
        "quadrature": quadrature,
        "coherent": coherent,
        "lattice_vs_continuum": lattice,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
