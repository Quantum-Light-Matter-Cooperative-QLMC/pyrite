"""Experimental, non-wrapped momentum instrument for planar scattering.

This diagnostic has no production callers. Build a larger differential kernel
with the existing TDS/electronic builders; a loss diagonal alone cannot specify
outgoing momenta. Source equation: ``J(rho)(x,x') = C(x,x') rho(x,x')`` in
``cr-lindblad-jump-operators``. Fourier multiplication gives the convolution
implemented here, retaining interference within each Bloch block.

Validation: cr-planar-overflow-spectrum
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .operators import ScatteringKernel


@dataclass(frozen=True)
class OverflowSpectrum:
    """Finite outgoing jump rates [1/angstrom], without folding momentum.

    ``outside`` selects momenta outside the incoming plane-wave window. The
    reference solver's overflow also contains transfer-truncation error;
    ``expanded_retained_rate - reference_retained_rate`` exposes that error.
    No incoming trace normalization or unresolved-tail redistribution is made.

    Validation: cr-planar-overflow-spectrum
    """

    wavevectors_inv_ang: np.ndarray
    rates_per_ang: np.ndarray
    outside: np.ndarray
    full_loss_rate: float
    reference_retained_rate: float
    expanded_retained_rate: float

    @property
    def reference_overflow_rate(self) -> float:
        """Full loss minus the reference kernel's in-window gain [1/angstrom]."""
        return self.full_loss_rate - self.reference_retained_rate

    @property
    def unresolved_rate(self) -> float:
        """Signed full-loss/finite-gain closure defect [1/angstrom]."""
        return self.full_loss_rate - float(self.rates_per_ang.sum())

    @property
    def outside_rate(self) -> float:
        """Resolved gain landing outside the incoming window [1/angstrom]."""
        return float(self.rates_per_ang[self.outside].sum())

    def sample_angles(
        self,
        kinetic_energy_eV: float,
        size: int,
        rng: np.random.Generator,
        *,
        relative_tolerance: float = 0.01,
    ) -> np.ndarray:
        """Draw signed paraxial plane angles after closure against reference loss.

        Source: the momentum-diagonal Born rule, ``P(k|outside) ∝ J_kk``;
        ``psi = hbar k / p``, with ``pc = sqrt(E(E + 2 m c²))``. This is a
        finite-grid approximation, conditional on resolved outside flux. Both
        omitted gain and the change to in-window gain must individually be
        within ``relative_tolerance`` of reference overflow. A missing tail or
        zero overflow raises; neither is assigned a band-edge angle. The caller
        owns the RNG stream; no production RNG or transport is changed.

        Validation: cr-planar-overflow-spectrum
        """
        if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0:
            raise ValueError("kinetic_energy_eV must be finite and positive")
        if int(size) != size or size < 0:
            raise ValueError("size must be a nonnegative integer")
        if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
            raise ValueError("relative_tolerance must be between zero and one")
        overflow = self.reference_overflow_rate
        shift = self.expanded_retained_rate - self.reference_retained_rate
        if overflow <= 0 or self.outside_rate <= 0:
            raise ValueError("no positive overflow to sample")
        defects = (self.unresolved_rate, shift, self.outside_rate - overflow)
        if any(abs(value) > relative_tolerance * overflow for value in defects):
            raise ValueError("overflow spectrum is not closed against reference loss")
        weights = self.rates_per_ang[self.outside]
        k = self.wavevectors_inv_ang[self.outside]
        selected = rng.choice(k, size=int(size), p=weights / weights.sum())
        pc = np.sqrt(kinetic_energy_eV * (kinetic_energy_eV + 2 * M_E_EV))
        return HBARC_EV_ANG * selected / pc


def _check_inputs(plane, kernel, reference):
    """Validate the common Bloch grid and positive incoming density."""
    size = reference.orders.size
    if plane.shape != (reference.n_cells, size, size):
        raise ValueError("plane must have the reference kernel's Bloch shape")
    same = (
        kernel.n_cells == reference.n_cells
        and np.isclose(kernel.period_ang, reference.period_ang)
        and np.isclose(kernel.hbar_v_eV_ang, reference.hbar_v_eV_ang)
        and kernel.max_order >= reference.max_order
    )
    if not same:
        raise ValueError("expanded and reference kernels must share grid and energy")
    if not np.isfinite(plane).all():
        raise ValueError("plane must be finite")
    scale = max(float(np.max(np.abs(plane))), np.finfo(float).tiny)
    if not np.allclose(plane, plane.conj().swapaxes(1, 2), rtol=0, atol=1e-12 * scale):
        raise ValueError("plane must be Hermitian")
    if np.linalg.eigvalsh(plane).min() < -1e-12 * scale:
        raise ValueError("plane must be positive semidefinite")


def _outgoing_diagonal(plane, kernel):
    """Noncyclic Fourier convolution; output h = a + i + Bloch carry.

    In ``rho[s,a,b] K[r,i,j]`` the outgoing diagonal requires ``a+i=b+j``.
    Summing matching diagonals by linear convolution retains their coherent
    cross terms. For one kick and a plane wave this reduces to momentum
    translation; a zero kernel gives zero gain.

    Validation: cr-planar-overflow-spectrum
    """
    cells, incoming, _ = plane.shape
    transfer = kernel.orders.size
    width = incoming + transfer
    gain = np.zeros((cells, width), dtype=complex)
    limit = min(incoming, transfer) - 1
    for s in range(cells):
        for r in range(cells):
            residue, carry = (s + r) % cells, (s + r) // cells
            for lag in range(-limit, limit + 1):
                density = np.diagonal(plane[s], offset=lag)
                kick = np.diagonal(kernel.momentum_kernel[r], offset=-lag)
                first = abs(lag) + carry
                values = np.convolve(density, kick)
                gain[residue, first : first + values.size] += values
    scale = max(float(np.abs(gain).sum()), np.finfo(float).tiny)
    if np.max(np.abs(gain.imag)) > 1e-12 * scale or gain.real.min() < -1e-12 * scale:
        raise ValueError("outgoing gain is not a positive real measure")
    orders = np.arange(width) - (incoming // 2 + kernel.max_order)
    return orders, np.maximum(gain.real, 0.0)


def overflow_spectrum(
    plane: np.ndarray,
    kernel: ScatteringKernel,
    *,
    reference_kernel: ScatteringKernel,
) -> OverflowSpectrum:
    """Resolve a finite tail conditioned on the supplied incoming plane density.

    ``kernel`` is a separately built, expanded differential transfer kernel;
    ``reference_kernel`` is the solver kernel whose overflow is being audited.
    Source: Fourier convolution of the position-space jump equation in
    ``cr-lindblad-jump-operators``. The reference overflow is full local loss
    minus reference in-window gain. It equals physical outside-window gain
    only when transfer truncation and quadrature errors vanish. Rates scale
    linearly with the incoming weight; no scattering gives zero rates.

    Validation: cr-planar-overflow-spectrum
    """
    plane = np.asarray(plane, dtype=complex)
    _check_inputs(plane, kernel, reference_kernel)
    orders, gain = _outgoing_diagonal(plane, kernel)
    ref_orders, ref_gain = _outgoing_diagonal(plane, reference_kernel)
    bound = reference_kernel.max_order
    outside = np.broadcast_to(np.abs(orders)[None, :] > bound, gain.shape)
    ref_inside = np.abs(ref_orders) <= bound
    lag = np.arange(plane.shape[1])[None, :] - np.arange(plane.shape[1])[:, None]
    coefficients = np.zeros(plane.shape[1], dtype=complex)
    count = min(coefficients.size, reference_kernel.diagonal_coefficients.size)
    coefficients[:count] = reference_kernel.diagonal_coefficients[:count]
    loss = np.where(lag >= 0, coefficients[np.abs(lag)], coefficients[np.abs(lag)].conj())
    full_loss = float(np.einsum("ab,sab->", loss, plane).real)
    kappa = np.arange(kernel.n_cells) * kernel.g_inv_ang / kernel.n_cells
    return OverflowSpectrum(
        wavevectors_inv_ang=kappa[:, None] + orders[None, :] * kernel.g_inv_ang,
        rates_per_ang=gain,
        outside=outside,
        full_loss_rate=full_loss,
        reference_retained_rate=float(ref_gain[:, ref_inside].sum()),
        expanded_retained_rate=float(gain[~outside].sum()),
    )
