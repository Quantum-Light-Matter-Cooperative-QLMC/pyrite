"""Carrier-inclusive probability flow in the periodic axial FFT supercell.

The paraxial generator is ``H = a (-i grad + q)^2 + U`` with
``a = HBAR2_2M_EV_ANG2 / gamma`` and ``q`` the physical carrier momentum.
For ``i hbar_v d_z phi = H phi`` the transverse probability current is

    j = 2 a / hbar_v * Im(phi.conj() * grad(phi) + i*q*abs(phi)**2).

The gradient uses *exactly* the FFT momenta of the kinetic propagator.
The current is evaluated at sample centres; face values are averages of
neighbouring samples. Divergence is the finite-volume difference of those
same faces and a contour encloses periodic pixel centres (not an absorbing
boundary). These finite-volume reductions satisfy the discrete divergence
THEOREM exactly, but the sampled continuum current need not satisfy an
exact pointwise continuity equation for under-resolved FFT states (spectral
aliasing). Convergence tests must check that residual separately.

Currents have units Å^-2 (transverse probability crossing a unit-length
segment per Å of propagation); divergences have units Å^-3. Contour side
fluxes have units Å^-1. Removed MCWF trajectories contribute zero to
all ensemble-averaged observables.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics


@dataclass(frozen=True)
class FluxContour:
    """Rectangle of grid samples; ``start`` may wrap on either periodic axis.

    ``size`` is the number of *pixels*, not a physical length. Each side
    lies halfway between included and excluded sample centres. The full
    supercell is admissible and has zero net contour flux by construction.
    """

    start: tuple[int, int]
    size: tuple[int, int]

    def validate(self, shape: tuple[int, int]) -> None:
        if (
            len(self.start) != 2
            or len(self.size) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, Integral)
                for v in (*self.start, *self.size)
            )
            or any(n < 1 or n > max_n for n, max_n in zip(self.size, shape, strict=True))
        ):
            raise ValueError("flux contour needs two integer starts and widths within the grid")


def samples_per_cell(
    cells: tuple[int, int], shape: tuple[int, int], *, warn: bool = True
) -> tuple[int, int]:
    """Require integer, useful resolution in each periodic projected cell.

    This is a conservative admission rule for the *diagnostic* FFT grid,
    not a validated physical convergence threshold. Compare 16, 24, 32,
    ... points per cell as needed for the specific potential and probe.
    """
    if (
        len(cells) != 2
        or len(shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in cells)
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 2 for v in shape)
    ):
        raise ValueError("cells and shape must be positive integer pairs")
    if any(n % c for n, c in zip(shape, cells, strict=True)):
        raise ValueError(
            "shape must be divisible by cells on both axes; hold samples/cell fixed when changing extent"
        )
    result = (shape[0] // cells[0], shape[1] // cells[1])
    if min(result) < 8:
        raise ValueError(
            "at least 8 samples per projected cell/axis are required; >=16 recommended"
        )
    if warn and min(result) < 16:
        warnings.warn(
            "fewer than 16 samples per cell/axis: check fixed-resolution convergence ",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def fixed_resolution_shape(cells: tuple[int, int], per_cell: tuple[int, int]) -> tuple[int, int]:
    """Total FFT shape for a supercell study at *fixed* per-cell resolution."""
    shape = (cells[0] * per_cell[0], cells[1] * per_cell[1])
    samples_per_cell(cells, shape, warn=False)
    return shape


class AxialFluxObservable:
    """Bounded CPU/CuPy callback returning current, optional divergence/flux.

    ``recording_width`` is expressed in *complex128* slots for the composite
    axial recording budget; each returned value is in fact real float64.
    The result is a flattened vector suitable for MCWF online averaging.
    """

    cuda_compatible = True

    def __init__(
        self,
        grid,
        energy_eV: float,
        *,
        divergence: bool = False,
        contour: FluxContour | None = None,
    ) -> None:
        if not np.isfinite(energy_eV) or energy_eV <= 0:
            raise ValueError("energy_eV must be positive and finite")
        if contour is not None:
            contour.validate(grid.shape)
        self.grid = grid
        self.divergence = bool(divergence)
        self.contour = contour
        self.coefficient = (
            2.0
            * HBAR2_2M_EV_ANG2
            / longitudinal_kinematics(energy_eV)[0]
            / (self._hbar_v(energy_eV))
        )
        pixels = int(np.prod(grid.shape))
        self.recording_width = (2 + int(self.divergence)) * pixels + (
            4 if contour is not None else 0
        )
        # FFT, two derivatives, currents, faces and optional divergence. CUDA
        # reserving this separately avoids silent oversubscription by callbacks.
        self.scratch_bytes = 12 * pixels * np.dtype(complex).itemsize
        self._device = None

    @staticmethod
    def _hbar_v(energy_eV):
        from pyrite.materials.crystal import HBARC_EV_ANG

        return HBARC_EV_ANG * longitudinal_kinematics(energy_eV)[1]

    def __call__(self, phi, tilt):
        xp = np
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as xp
        if self._device is None or self._device[0] is not xp:
            self._device = (
                xp,
                xp.asarray(self.grid.kx[:, None]),
                xp.asarray(self.grid.ky[None, :]),
            )
        _, kx, ky = self._device
        spectrum = xp.fft.fft2(phi)
        density = xp.abs(phi) ** 2
        jx = self.coefficient * (
            xp.imag(xp.conj(phi) * xp.fft.ifft2(1j * kx * spectrum)) + tilt[0] * density
        )
        jy = self.coefficient * (
            xp.imag(xp.conj(phi) * xp.fft.ifft2(1j * ky * spectrum)) + tilt[1] * density
        )
        parts = [jx.ravel(), jy.ravel()]
        if self.divergence or self.contour is not None:
            # Positive faces: between pixel i and i+1. Periodicity at N-1/0.
            face_x = 0.5 * (jx + xp.roll(jx, -1, axis=0))
            face_y = 0.5 * (jy + xp.roll(jy, -1, axis=1))
            dx = self.grid.extent_ang[0] / self.grid.shape[0]
            dy = self.grid.extent_ang[1] / self.grid.shape[1]
            if self.divergence:
                div = (face_x - xp.roll(face_x, 1, axis=0)) / dx
                div += (face_y - xp.roll(face_y, 1, axis=1)) / dy
                parts.append(div.ravel())
            if self.contour is not None:
                x0, y0 = self.contour.start
                nx, ny = self.contour.size
                xx = (x0 + xp.arange(nx)) % self.grid.shape[0]
                yy = (y0 + xp.arange(ny)) % self.grid.shape[1]
                # Four outward signed sides: -x, +x, -y, +y.
                sides = xp.stack(
                    (
                        -xp.sum(face_x[(x0 - 1) % self.grid.shape[0], yy]) * dy,
                        xp.sum(face_x[(x0 + nx - 1) % self.grid.shape[0], yy]) * dy,
                        -xp.sum(face_y[xx, (y0 - 1) % self.grid.shape[1]]) * dx,
                        xp.sum(face_y[xx, (y0 + ny - 1) % self.grid.shape[1]]) * dx,
                    )
                )
                parts.append(sides)
        return xp.concatenate(parts)

    def unpack(self, recorded):
        """Split a depth-indexed accumulated observable into physical arrays."""
        count = int(np.prod(self.grid.shape))
        jx = recorded[:, :count].real.reshape(-1, *self.grid.shape)
        jy = recorded[:, count : 2 * count].real.reshape(-1, *self.grid.shape)
        offset = 2 * count
        div = None
        if self.divergence:
            div = recorded[:, offset : offset + count].real.reshape(-1, *self.grid.shape)
            offset += count
        sides = None if self.contour is None else recorded[:, offset : offset + 4].real
        return np.stack((jx, jy), axis=1), div, sides


__all__ = ["AxialFluxObservable", "FluxContour", "fixed_resolution_shape", "samples_per_cell"]
