"""Device copy of the anchored axial dissipator; no wavefunction host transfers.

Validation: cr-axial-realspace-dissipator
"""

from __future__ import annotations

import cupy as cp

from .axial_dissipator import _pad_index
from .realspace import RealSpaceGrid


class _DeviceGrid:
    def __init__(self, extent_ang, shape):
        grid = RealSpaceGrid(extent_ang, shape)
        self.extent_ang = grid.extent_ang
        self.shape = grid.shape
        self.cell_area_ang2 = grid.cell_area_ang2
        self.kx = cp.asarray(grid.kx)
        self.ky = cp.asarray(grid.ky)


class DeviceDissipator:
    """Upload immutable jump data once; evaluate the CPU equations on CUDA."""

    def __init__(self, dissipator):
        self.grid = _DeviceGrid(dissipator.grid.extent_ang, dissipator.grid.shape)
        self.fine = _DeviceGrid(
            dissipator.grid.extent_ang, tuple(2 * n for n in dissipator.grid.shape)
        )
        self._envelopes = {}
        self._spectra = {}
        self.tds_sources = dissipator.tds_sources
        self.tds_source_rates = cp.asarray(dissipator.tds_source_rates)
        self.phase_spectra = {k: cp.asarray(v) for k, v in dissipator.phase_spectra.items()}
        self.electronic_q_inv_ang = cp.asarray(dissipator.electronic_q_inv_ang)
        self.electronic_weights = cp.asarray(dissipator.electronic_weights)
        self.electron_density = cp.asarray(dissipator.electron_density)
        self.electronic_damping = cp.asarray(dissipator.electronic_damping)

    def tds_phase(self, source: int, offset_ang) -> cp.ndarray:
        """``phi_u(r) - <phi>(r)`` of one source displaced by ``offset_ang`` (real).

        ``offset_ang`` may be a stack of shape ``(n, 2)``; the result is then
        ``(n, nx, ny)``.
        """
        src = self.tds_sources[source]
        kx = self.grid.kx[:, None]
        ky = self.grid.ky[None, :]
        spectrum = self.phase_spectra[src.element]
        x0, y0 = src.position_ang
        base = spectrum * cp.exp(-1j * (kx * x0 + ky * y0))
        mean = cp.exp(-0.5 * src.displacement_ang2 * (kx**2 + ky**2))
        offsets = cp.asarray(offset_ang, dtype=float)
        single = offsets.ndim == 1
        offsets = cp.atleast_2d(offsets)
        moved = cp.exp(
            -1j * (kx[None] * offsets[:, 0, None, None] + ky[None] * offsets[:, 1, None, None])
        )
        out = cp.fft.ifft2(base[None] * (moved - mean[None]), axes=(1, 2)).real * spectrum.size
        return out[0] if single else out

    def tds_weight_spectrum(self, source: int, density: cp.ndarray) -> cp.ndarray:
        """Coefficients ``w_K`` with ``int |psi|^2 L_u^2 = Re sum_K w_K e^{-iK.u}``.

        ``density`` is ``|psi|^2`` on the grid. ``L_u(r_j)^2`` is a trigonometric
        polynomial in ``u`` of twice the grid band, so the coefficients live on
        a 2x-oversampled momentum grid (FFT layout, ``2n_x x 2n_y``) and the
        expansion is exact for every displacement.
        """
        if source not in self._spectra:
            self._spectra[source] = self._source_spectrum(source)
        spectrum, rows, cols, convolution, mean_field, extension, shift = self._spectra[source]
        area = self.grid.cell_area_ang2
        overlap = cp.conj(cp.fft.fft2(density))
        extended = overlap[extension]
        cross = cp.zeros(self.fine.shape, dtype=complex)
        cross[cp.ix_(rows, cols)] = spectrum * cp.conj(cp.fft.fft2(density * mean_field))
        out = shift * (convolution * extended - 2.0 * cross) * area
        out[0, 0] += cp.sum(density * mean_field**2) * area
        return out

    def _source_spectrum(self, source):
        src = self.tds_sources[source]
        nx, ny = self.grid.shape
        spectrum = self.phase_spectra[src.element]
        kx, ky = self.grid.kx[:, None], self.grid.ky[None, :]
        rows, cols = cp.asarray(_pad_index(nx)), cp.asarray(_pad_index(ny))
        padded = cp.zeros(self.fine.shape, dtype=complex)
        padded[cp.ix_(rows, cols)] = spectrum
        convolution = cp.fft.fft2(cp.fft.ifft2(padded) ** 2) * padded.size
        mean_field = (
            cp.fft.ifft2(
                spectrum
                * cp.exp(-0.5 * src.displacement_ang2 * (kx**2 + ky**2))
                * cp.exp(-1j * (kx * src.position_ang[0] + ky * src.position_ang[1]))
            ).real
            * spectrum.size
        )
        fx = cp.rint(self.fine.kx / (2 * cp.pi / self.grid.extent_ang[0])).astype(int) % nx
        fy = cp.rint(self.fine.ky / (2 * cp.pi / self.grid.extent_ang[1])).astype(int) % ny
        shift = cp.exp(
            -1j
            * (
                self.fine.kx[:, None] * src.position_ang[0]
                + self.fine.ky[None, :] * src.position_ang[1]
            )
        )
        return spectrum, rows, cols, convolution, mean_field, cp.ix_(fx, fy), shift

    def tds_weight(self, coefficients: cp.ndarray, offsets) -> cp.ndarray:
        """Evaluate :meth:`tds_weight_spectrum` coefficients at displacements ``(n, 2)``."""
        fine = self.fine
        u = cp.atleast_2d(cp.asarray(offsets, dtype=float))
        ex = cp.exp(-1j * cp.multiply.outer(u[:, 0], fine.kx))
        ey = cp.exp(-1j * cp.multiply.outer(u[:, 1], fine.ky))
        return cp.sum((ex @ coefficients) * ey, axis=1).real

    def tds_weight_envelope(self, coefficients: cp.ndarray, edges_ang) -> cp.ndarray:
        """Rigorous maximum of the weight on each annulus ``edges[i] <= |u| <= edges[i+1]``.

        The weight is sampled on a square lattice of spacing ``delta =
        edges[-1] / 64`` and each annulus maximum is raised by the Lipschitz
        bound ``sum_K |w_K| |K| delta / sqrt 2`` (every point lies within
        ``delta / sqrt 2`` of a lattice node).
        """
        geometry = self._envelope_geometry(edges_ang)
        return self._envelope_values(coefficients, geometry)

    def _envelope_geometry(self, edges_ang):
        edges = cp.asarray(edges_ang, dtype=float)
        delta = edges[-1] / 64.0
        axis = cp.arange(-65, 66) * delta
        ex = cp.exp(-1j * cp.multiply.outer(axis, self.fine.kx))
        ey = cp.exp(-1j * cp.multiply.outer(axis, self.fine.ky))
        radius = cp.hypot(axis[:, None], axis[None, :])
        reach = delta / cp.sqrt(2.0)
        masks = (radius[None] >= edges[:-1, None, None] - reach) & (
            radius[None] <= edges[1:, None, None] + reach
        )
        k = cp.hypot(self.fine.kx[:, None], self.fine.ky[None, :])
        return ex, ey, masks, k * reach

    def _envelope_values(self, coefficients, geometry):
        ex, ey, masks, slack_weight = geometry
        values = (ex @ coefficients @ ey.T).real
        slack = cp.sum(cp.abs(coefficients) * slack_weight)
        return cp.max(cp.where(masks, values[None], -cp.inf), axis=(1, 2)) + slack

    def source_envelope(self, source, coefficients, edges_ang):
        if source not in self._envelopes:
            self._envelopes[source] = self._envelope_geometry(edges_ang)
        return self._envelope_values(coefficients, self._envelopes[source])

    def smoothed_density(self, q_inv_ang: float) -> cp.ndarray:
        """``rho_q(r)`` on the grid [1/angstrom^3], clipped at zero."""
        k2 = self.grid.kx[:, None] ** 2 + self.grid.ky[None, :] ** 2
        damp = cp.exp(-0.5 * k2 / q_inv_ang**2)
        values = cp.fft.ifft2(self.electron_density * damp).real * self.electron_density.size**2
        return cp.maximum(values, 0.0)

    def electronic_expectations(self, density: cp.ndarray) -> cp.ndarray:
        """``<rho_q>`` of a normalized ``|psi|^2`` sample at every transfer node."""
        spectrum = cp.fft.fft2(density) * self.grid.cell_area_ang2
        live = self.electron_density != 0.0
        overlap = (self.electron_density[live] * cp.conj(spectrum[live])).real * spectrum.size
        return self.electronic_damping @ overlap
