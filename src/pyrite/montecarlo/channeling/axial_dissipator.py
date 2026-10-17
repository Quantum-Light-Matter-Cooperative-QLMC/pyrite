"""Real-space Lindblad dissipator for the axial tier (experimental, #508).

The planar kernels of :mod:`.operators` carried to a 2D periodic grid, where
every jump is a pointwise product (proposal §9).

**Thermal diffuse scattering.** An atom of species ``s`` on a string at
``R`` crossed with in-plane displacement ``u ~ N(0, a^2 I)`` adds the phase
``phi_u(r) = -(1 / hbar v) V_s(r - R - u)``, with ``V_s`` the projected
Mott--Bethe potential. The phase covariance per unit depth,
``C(r, r') = sum_atoms lambda_s Cov_u[phi_u(r), phi_u(r')]`` (``lambda_s``
atoms per angstrom along the string), is unravelled exactly by the jumps
``L_{s,R,u}(r) = phi_u(r) - <phi>(r)`` with rate measure ``lambda_s p(u) d^2u``.
On the grid the phase is band-limited to the grid momenta (Nyquist bin
dropped, so ``phi`` stays real), giving the in-grid kernel ``C_grid``; its
diagonal follows from ``E_u[phi_u^2] = (phi_0^2 * N(0, a^2))`` evaluated on a
2x-oversampled grid (exact, ``phi_0^2`` has twice the band). The full diagonal
``C(r, r)`` keeps every momentum transfer:
``C(r, r) = sum_G n_{s,G} T_s(|G|) e^{iG.r} / (2 pi hbar v)^2`` with the
planar transform ``T_s`` of :func:`.operators._tds_transform` and the static
site density ``n_{s,G}``. The difference ``C(r,r) - C_grid(r,r)`` is the
large-angle transfer the grid cannot hold; it is a sink (the electron leaves
the channeled zone), the analogue of the planar out-of-band flux.

**Electronic excitation.** The local-density jumps of :mod:`.operators`,
``L_q(r) = sqrt(w(q) rho_q(r)) e^{i q.r}``, with ``rho_q`` the atomic
electron density smoothed at ``1/|q|`` and ``w`` the PENELOPE-2024 shell GOS
per electron. ``q`` runs over the planar quadrature nodes
(:func:`.operators._electronic_nodes`) with a uniform azimuth; the kick
``e^{i q.r}`` is added to the electron's carrier tilt, so it is exact for any
``q``. ``C_el(r, r) = sum_G rho_G E(|G|) e^{iG.r}`` with the planar transform.

Averaging ``C(r, r)`` over ``y`` gives the planar diagonals of the zone plane
normal to ``x``; tests pin both mechanisms against
:meth:`.operators.ScatteringKernel.absorptive_potential_eV`.

Assumptions as in :mod:`.operators` (first Born per atom, Einstein
displacements, local-density electronic channels), plus an in-plane
isotropic displacement for the TDS diagonal. A ``c_x x c_y`` supercell repeats
each atom's displacement in its periodic images.

Lengths in angstroms, energies in eV, rates in 1/angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.channeling.axial import AxialProfile
from ..transport.shell_oscillators import MaterialShellOscillators
from ..transport.shell_rates import catalog_shell_oscillators
from .operators import (
    _electronic_nodes,
    _kinematics,
    _potential,
    _tds_transform,
    electronic_channels,
)
from .realspace import RealSpaceGrid

#: Relative in-plane anisotropy of a displacement tensor treated as isotropic.
_ISOTROPY_RTOL = 1e-6
#: Displacement-damped TDS harmonics below this relative weight are dropped.
_TDS_TAIL = 1e-12


@dataclass(frozen=True)
class TDSSource:
    """One atomic string of the supercell for TDS jumps.

    Attributes
    ----------
    element
        Species.
    position_ang
        ``(x, y)`` in the supercell [angstrom].
    linear_density
        Atoms per angstrom along the axis.
    displacement_ang2
        In-plane mean-square displacement ``a^2`` per axis [angstrom^2].
    """

    element: str
    position_ang: tuple[float, float]
    linear_density: float
    displacement_ang2: float


@dataclass(frozen=True)
class AxialDissipator:
    """Grid kernels and jump data of the axial dissipator.

    Attributes
    ----------
    grid
        Supercell grid.
    hbar_v_eV_ang
        ``hbar v`` [eV angstrom].
    tds_grid_rate, tds_sink_rate, electronic_rate
        ``C_grid(r, r)``, ``C(r, r) - C_grid(r, r)`` and ``C_el(r, r)`` [1/angstrom].
    tds_sources
        Strings of the supercell.
    tds_source_rates
        ``lambda_s Var_u phi_band(r)`` per source [1/angstrom], shape
        ``(n_sources, nx, ny)``; sums to ``tds_grid_rate``.
    phase_spectra
        Band-limited ``phi~_0(k)`` of an atom at the origin per element
        (supercell FFT layout, ``1/A`` normalization folded in).
    electronic_q_inv_ang, electronic_weights
        Transfer nodes and ``int d^2q w`` weights [angstrom^2].
    electron_density
        ``rho_G / (n_x n_y)`` [1/angstrom^3] in supercell FFT layout (zero off the
        cell lattice); the ``1/(n_x n_y)`` is the inverse-FFT normalization.
    """

    grid: RealSpaceGrid
    hbar_v_eV_ang: float
    tds_grid_rate: np.ndarray
    tds_sink_rate: np.ndarray
    electronic_rate: np.ndarray
    tds_sources: tuple[TDSSource, ...]
    tds_source_rates: np.ndarray
    phase_spectra: dict[str, np.ndarray]
    electronic_q_inv_ang: np.ndarray
    electronic_weights: np.ndarray
    electron_density: np.ndarray
    electronic_damping: np.ndarray

    @property
    def total_rate(self) -> np.ndarray:
        """``C(r, r)`` of every mechanism [1/angstrom]: the no-jump loss rate."""
        return self.tds_grid_rate + self.tds_sink_rate + self.electronic_rate

    @property
    def absorptive_eV(self) -> np.ndarray:
        """``W(r) = (hbar v / 2) C(r, r)`` [eV]."""
        return 0.5 * self.hbar_v_eV_ang * self.total_rate

    # -- jumps ---------------------------------------------------------------

    def tds_phase(self, source: int, offset_ang) -> np.ndarray:
        """``phi_u(r) - <phi>(r)`` of one source displaced by ``offset_ang`` (real).

        ``offset_ang`` may be a stack of shape ``(n, 2)``; the result is then
        ``(n, nx, ny)``.
        """
        src = self.tds_sources[source]
        kx = self.grid.kx[:, None]
        ky = self.grid.ky[None, :]
        spectrum = self.phase_spectra[src.element]
        x0, y0 = src.position_ang
        base = spectrum * np.exp(-1j * (kx * x0 + ky * y0))
        mean = np.exp(-0.5 * src.displacement_ang2 * (kx**2 + ky**2))
        offsets = np.asarray(offset_ang, dtype=float)
        single = offsets.ndim == 1
        offsets = np.atleast_2d(offsets)
        moved = np.exp(
            -1j * (kx[None] * offsets[:, 0, None, None] + ky[None] * offsets[:, 1, None, None])
        )
        out = np.fft.ifft2(base[None] * (moved - mean[None]), axes=(1, 2)).real * spectrum.size
        return out[0] if single else out

    def tds_weight_spectrum(self, source: int, density: np.ndarray) -> np.ndarray:
        """Coefficients ``w_K`` with ``int |psi|^2 L_u^2 = Re sum_K w_K e^{-iK.u}``.

        ``density`` is ``|psi|^2`` on the grid. ``L_u(r_j)^2`` is a trigonometric
        polynomial in ``u`` of twice the grid band, so the coefficients live on
        a 2x-oversampled momentum grid (FFT layout, ``2n_x x 2n_y``) and the
        expansion is exact for every displacement.
        """
        src = self.tds_sources[source]
        nx, ny = self.grid.shape
        area = self.grid.cell_area_ang2
        spectrum = self.phase_spectra[src.element]
        fine = RealSpaceGrid(self.grid.extent_ang, (2 * nx, 2 * ny))
        kx, ky = self.grid.kx[:, None], self.grid.ky[None, :]
        fkx, fky = fine.kx[:, None], fine.ky[None, :]
        rows, cols = _pad_index(nx), _pad_index(ny)
        padded = np.zeros(fine.shape, dtype=complex)
        padded[np.ix_(rows, cols)] = spectrum
        # Self-convolution sum_{k + k' = K} S_k S_k' (no wrap: the band is half the fine one).
        convolution = np.fft.fft2(np.fft.ifft2(padded) ** 2) * padded.size
        mean_field = (
            np.fft.ifft2(
                spectrum
                * np.exp(-0.5 * src.displacement_ang2 * (kx**2 + ky**2))
                * np.exp(-1j * (kx * src.position_ang[0] + ky * src.position_ang[1]))
            ).real
            * spectrum.size
        )
        # sum_j D_j e^{iK.r_j} on the coarse grid, periodically extended in K.
        overlap = np.conj(np.fft.fft2(density))
        # Periodic extension onto the fine FFT layout (same K values).
        fx = np.rint(fine.kx / (2.0 * np.pi / self.grid.extent_ang[0])).astype(int) % nx
        fy = np.rint(fine.ky / (2.0 * np.pi / self.grid.extent_ang[1])).astype(int) % ny
        extended = overlap[np.ix_(fx, fy)]
        cross = np.zeros(fine.shape, dtype=complex)
        cross[np.ix_(rows, cols)] = spectrum * np.conj(np.fft.fft2(density * mean_field))
        shift = np.exp(-1j * (fkx * src.position_ang[0] + fky * src.position_ang[1]))
        out = shift * (convolution * extended - 2.0 * cross) * area
        out[0, 0] += np.sum(density * mean_field**2) * area
        return out

    def tds_weight(self, coefficients: np.ndarray, offsets) -> np.ndarray:
        """Evaluate :meth:`tds_weight_spectrum` coefficients at displacements ``(n, 2)``."""
        fine = RealSpaceGrid(self.grid.extent_ang, coefficients.shape)
        u = np.atleast_2d(np.asarray(offsets, dtype=float))
        ex = np.exp(-1j * np.multiply.outer(u[:, 0], fine.kx))
        ey = np.exp(-1j * np.multiply.outer(u[:, 1], fine.ky))
        return np.einsum("px,xy,py->p", ex, coefficients, ey).real

    def tds_weight_envelope(self, coefficients: np.ndarray, edges_ang) -> np.ndarray:
        """Rigorous maximum of the weight on each annulus ``edges[i] <= |u| <= edges[i+1]``.

        The weight is sampled on a square lattice of spacing ``delta =
        edges[-1] / 64`` and each annulus maximum is raised by the Lipschitz
        bound ``sum_K |w_K| |K| delta / sqrt 2`` (every point lies within
        ``delta / sqrt 2`` of a lattice node).
        """
        edges = np.asarray(edges_ang, dtype=float)
        fine = RealSpaceGrid(self.grid.extent_ang, coefficients.shape)
        delta = edges[-1] / 64.0
        axis = np.arange(-65, 66) * delta
        ex = np.exp(-1j * np.multiply.outer(axis, fine.kx))
        ey = np.exp(-1j * np.multiply.outer(axis, fine.ky))
        values = (ex @ coefficients @ ey.T).real
        radius = np.hypot(axis[:, None], axis[None, :])
        k = np.hypot(fine.kx[:, None], fine.ky[None, :])
        slack = float(np.sum(np.abs(coefficients) * k)) * delta / np.sqrt(2.0)
        out = np.empty(edges.size - 1)
        reach = delta / np.sqrt(2.0)
        for i in range(out.size):
            near = (radius >= edges[i] - reach) & (radius <= edges[i + 1] + reach)
            out[i] = values[near].max() + slack
        return out

    def smoothed_density(self, q_inv_ang: float) -> np.ndarray:
        """``rho_q(r)`` on the grid [1/angstrom^3], clipped at zero."""
        k2 = self.grid.kx[:, None] ** 2 + self.grid.ky[None, :] ** 2
        damp = np.exp(-0.5 * k2 / q_inv_ang**2)
        # ifft2 divides by n_x n_y once more than electron_density already does.
        values = np.fft.ifft2(self.electron_density * damp).real * self.electron_density.size**2
        return np.maximum(values, 0.0)

    def electronic_expectations(self, density: np.ndarray) -> np.ndarray:
        """``<rho_q>`` of a normalized ``|psi|^2`` sample at every transfer node."""
        spectrum = np.fft.fft2(density) * self.grid.cell_area_ang2  # int |psi|^2 e^{-ik.r}
        live = self.electron_density != 0.0
        overlap = (self.electron_density[live] * np.conj(spectrum[live])).real * spectrum.size
        return self.electronic_damping @ overlap


def _in_plane_isotropic(profile: AxialProfile, site: int) -> float:
    plane = profile.axes[:2]
    block = plane @ profile.displacements_ang2[site] @ plane.T
    a2 = 0.5 * np.trace(block)
    if np.abs(block - a2 * np.eye(2)).max() > _ISOTROPY_RTOL * a2:
        raise NotImplementedError(
            "the axial TDS diagonal needs an in-plane isotropic displacement tensor"
        )
    return float(a2)


def _site_displacements(profile: AxialProfile) -> dict[str, float]:
    from ...materials.crystal import CRYSTALS

    out: dict[str, float] = {}
    for site, (element, _) in enumerate(CRYSTALS[profile.crystal]["basis"]):
        a2 = _in_plane_isotropic(profile, site)
        if element in out and not np.isclose(out[element], a2, rtol=1e-9):
            raise NotImplementedError("per-site displacements must agree within an element")
        out[element] = a2
    return out


def _fold(grid: RealSpaceGrid, cells, coefficients: dict[tuple[int, int], complex]) -> np.ndarray:
    """Point samples of ``sum_G c_G e^{iG.r}`` (cell harmonics folded onto the grid)."""
    nx, ny = grid.shape
    spectrum = np.zeros(grid.shape, dtype=complex)
    for (m, n), value in coefficients.items():
        spectrum[(m * cells[0]) % nx, (n * cells[1]) % ny] += value
    return np.fft.ifft2(spectrum).real * (nx * ny)


def axial_dissipator(
    profile: AxialProfile,
    kinetic_energy_eV: float,
    cells: tuple[int, int],
    shape: tuple[int, int],
    *,
    mechanisms: tuple[str, ...] = ("tds", "electronic"),
    oscillators: MaterialShellOscillators | None = None,
) -> AxialDissipator:
    """Build the axial dissipator on a ``cells`` supercell sampled at ``shape``.

    Validation: cr-axial-realspace-dissipator
    """
    unknown = set(mechanisms) - {"tds", "electronic"}
    if unknown:
        raise ValueError(f"unknown mechanisms {sorted(unknown)}")
    cx, cy = (int(c) for c in cells)
    grid = RealSpaceGrid((profile.period_ang[0] * cx, profile.period_ang[1] * cy), tuple(shape))
    nx, ny = grid.shape
    hbar_v = _kinematics(kinetic_energy_eV)
    zero = np.zeros(grid.shape)
    kx = grid.kx[:, None]
    ky = grid.ky[None, :]
    k2 = kx**2 + ky**2
    gx, gy = profile.g_inv_ang

    sources: list[TDSSource] = []
    source_rates: list[np.ndarray] = []
    spectra: dict[str, np.ndarray] = {}
    grid_rate = zero.copy()
    sink_rate = zero.copy()
    if "tds" in mechanisms:
        a2 = _site_displacements(profile)
        band = (np.abs(np.fft.fftfreq(nx) * nx) < nx / 2)[:, None] & (
            np.abs(np.fft.fftfreq(ny) * ny) < ny / 2
        )[None, :]
        # Oversampled (2x) layout for E_u[phi_u^2].
        fine = RealSpaceGrid(grid.extent_ang, (2 * nx, 2 * ny))
        fkx, fky = fine.kx[:, None], fine.ky[None, :]
        for element in sorted(a2):
            v = _potential(element, np.sqrt(k2))
            spectra[element] = np.where(band, -v / hbar_v / grid.area_ang2, 0.0)
        full: dict[tuple[int, int], complex] = {}
        a2_min = min(a2.values())
        g_cut = np.sqrt(4.0 * -np.log(_TDS_TAIL) / a2_min)
        m_max, n_max = int(np.ceil(g_cut / gx)), int(np.ceil(g_cut / gy))
        mm, nn = np.meshgrid(
            np.arange(-m_max, m_max + 1), np.arange(-n_max, n_max + 1), indexing="ij"
        )
        g_mag = np.hypot(mm * gx, nn * gy)
        inside = g_mag <= g_cut
        unique, inverse = np.unique(np.round(g_mag[inside], 9), return_inverse=True)
        for element in sorted(a2):
            strings = [s for s in profile.strings if element in s.linear_density]
            transform = _tds_transform(element, a2[element], unique)[inverse]
            for s in strings:
                lam = s.linear_density[element]
                phase = np.exp(
                    -1j
                    * (mm[inside] * gx * s.position_ang[0] + nn[inside] * gy * s.position_ang[1])
                )
                values = lam / profile.area_ang2 * phase * transform / (2.0 * np.pi * hbar_v) ** 2
                for key, val in zip(zip(mm[inside], nn[inside], strict=True), values, strict=True):
                    key = (int(key[0]), int(key[1]))
                    full[key] = full.get(key, 0.0) + val
                # Grid sources: every image of this string in the supercell.
                for i in range(cx):
                    for j in range(cy):
                        pos = (
                            s.position_ang[0] + i * profile.period_ang[0],
                            s.position_ang[1] + j * profile.period_ang[1],
                        )
                        sources.append(TDSSource(element, pos, lam, a2[element]))
        # In-grid variance per source.
        for src in sources:
            spec = spectra[src.element]
            padded = np.zeros(fine.shape, dtype=complex)
            padded[np.ix_(_pad_index(nx), _pad_index(ny))] = spec
            phi0 = np.fft.ifft2(padded).real * padded.size
            square = np.fft.fft2(phi0**2) / padded.size
            shift = np.exp(-1j * (fkx * src.position_ang[0] + fky * src.position_ang[1]))
            damp = np.exp(-0.5 * src.displacement_ang2 * (fkx**2 + fky**2))
            second = (np.fft.ifft2(square * damp * shift).real * padded.size)[::2, ::2]
            mean = (
                np.fft.ifft2(
                    spec
                    * np.exp(-0.5 * src.displacement_ang2 * k2)
                    * np.exp(-1j * (kx * src.position_ang[0] + ky * src.position_ang[1]))
                ).real
                * spec.size
            )
            rate = src.linear_density * (second - mean**2)
            source_rates.append(rate)
            grid_rate += rate
        # Band-limited pointwise variance can exceed the full one on coarse grids;
        # the sink is clamped at zero (no gain).
        sink_rate = np.maximum(_fold(grid, (cx, cy), full) - grid_rate, 0.0)

    el_rate = zero.copy()
    q_nodes = np.zeros(0)
    q_weights = np.zeros(0)
    density = np.zeros(grid.shape, dtype=complex)
    m_max, n_max = profile.max_order
    for m in range(-m_max, m_max + 1):
        if 2 * abs(m * cx) >= nx:
            continue
        for n in range(-n_max, n_max + 1):
            if 2 * abs(n * cy) >= ny:
                continue
            density[(m * cx) % nx, (n * cy) % ny] = profile.electron_density[m + m_max, n + n_max]
    if "electronic" in mechanisms:
        if oscillators is None:
            oscillators = catalog_shell_oscillators(profile.crystal)
        channels = electronic_channels(oscillators, kinetic_energy_eV)
        q_nodes, q_weights = _electronic_nodes(channels)
        keep = q_weights > 0.0
        q_nodes, q_weights = q_nodes[keep], q_weights[keep]
        coefficients: dict[tuple[int, int], complex] = {}
        mm, nn = np.meshgrid(
            np.arange(-m_max, m_max + 1), np.arange(-n_max, n_max + 1), indexing="ij"
        )
        g_mag = np.hypot(mm * gx, nn * gy)
        transform = q_weights @ np.exp(-0.5 * (g_mag.ravel()[None, :] / q_nodes[:, None]) ** 2)
        for (m, n), value in zip(zip(mm.ravel(), nn.ravel(), strict=True), transform, strict=True):
            coefficients[(int(m), int(n))] = profile.electron_density[m + m_max, n + n_max] * value
        el_rate = _fold(grid, (cx, cy), coefficients)

    return AxialDissipator(
        grid=grid,
        hbar_v_eV_ang=hbar_v,
        tds_grid_rate=grid_rate,
        tds_sink_rate=sink_rate,
        electronic_rate=el_rate,
        tds_sources=tuple(sources),
        tds_source_rates=np.array(source_rates) if source_rates else np.zeros((0, *grid.shape)),
        phase_spectra=spectra,
        electronic_q_inv_ang=q_nodes,
        electronic_weights=q_weights,
        electron_density=density / (nx * ny),
        electronic_damping=np.exp(-0.5 * k2[density != 0.0][None, :] / q_nodes[:, None] ** 2)
        if q_nodes.size
        else np.zeros((0, int(np.count_nonzero(density)))),
    )


def _pad_index_n(n: int, size: int) -> np.ndarray:
    """Positions of an ``n``-point FFT spectrum inside a ``size``-point one."""
    freq = np.rint(np.fft.fftfreq(n) * n).astype(int)
    return np.where(freq >= 0, freq, freq + size)


def _pad_index(n: int) -> np.ndarray:
    """Positions of an ``n``-point FFT spectrum inside a ``2n``-point one."""
    return _pad_index_n(n, 2 * n)


__all__ = ["AxialDissipator", "TDSSource", "axial_dissipator"]
