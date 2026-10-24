"""Independent mesh and FFT-window convergence of finite-PEC collected flux (#616).

The finite-screen source is sampled on a periodic FFT grid. Two numerical
errors are separable and must be converged independently before a collected
spectrum is quoted:

* **pitch** (source/near-field resolution): halve ``dx, dy`` at a fixed
  physical window ``nx dx``;
* **window** (periodic-image wraparound and truncated evanescent tails): double
  ``nx, ny`` at a fixed pitch, i.e. zero-pad the computational window.

The k-space flux density has an integrable 1/kz singularity on the light
circle, so FFT-grid sums converge slowly and *non-monotonically* in the window
for any source with near-grazing emission. ``hemisphere_fft_bias`` compares the
base-grid FFT hemisphere energy with the window-independent angular quadrature
of the same sampled source, and is an indicator of that sampling error. With
``green_pitch_m`` the base grid is also compared with the non-periodic direct
Green collection (:func:`collect_screen_green`), which has no window error.

Each step reports the relative change in collected photons/eV at the stated
detector plane. A step below ``rtol`` is a numerical-convergence diagnostic for
that geometry and energy only, not a physics validation or a statement about
other energies or distances. Each refinement quadruples memory; large meshes
belong on ``pyrite remote``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optical_finite_collection import PlaneAcceptance, collect_screen_field, collect_screen_green
from .optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    hemisphere_energy_per_eV,
    spectral_energy_per_eV,
)

__all__ = ["CollectionConvergence", "finite_screen_collection_convergence"]

_MAX_GRID = 4096


@dataclass(frozen=True)
class CollectionConvergence:
    """Collected photons/eV for the base grid and each independent refinement."""

    energy_eV: float
    plane_z_m: float
    base: tuple[int, int, float, float, float]
    pitch_refined: tuple[tuple[int, int, float, float, float], ...]
    window_refined: tuple[tuple[int, int, float, float, float], ...]
    pitch_relative_change: float
    window_relative_change: float
    hemisphere_fft_bias: float
    rtol: float
    green_photons_per_eV: float | None = None

    @property
    def green_relative_difference(self) -> float | None:
        """Base FFT collection relative to the direct Green reference."""
        if self.green_photons_per_eV is None:
            return None
        return self.base[-1] / self.green_photons_per_eV - 1.0

    @property
    def converged(self) -> bool:
        return (
            max(
                self.pitch_relative_change,
                self.window_relative_change,
                abs(self.hemisphere_fft_bias),
                abs(self.green_relative_difference or 0.0),
            )
            <= self.rtol
        )


def _collect(energy_eV, screen, nx, ny, dx_m, dy_m, plane_z_m, aperture):
    field = circular_pec_transition_field(energy_eV, screen, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m)
    photons = collect_screen_field(field, plane_z_m=plane_z_m, aperture=aperture).photons_per_eV
    return (nx, ny, float(dx_m), float(dy_m), float(photons))


def _relative_change(rows):
    a, b = rows[-2][-1], rows[-1][-1]
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(b - a) / scale


def finite_screen_collection_convergence(
    energy_eV: float,
    screen: PECCircularScreen,
    *,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    plane_z_m: float,
    aperture: PlaneAcceptance | None = None,
    refinements: int = 1,
    rtol: float = 0.01,
    green_pitch_m: float | None = None,
) -> CollectionConvergence:
    """Refine pitch and window separately ``refinements`` times each.

    The reported change is between the last two grids of each sequence.
    ``aperture`` must fit the base window (it then fits every refinement).

    Validation: optical-wavefront-propagation
    """
    if not isinstance(refinements, int) or isinstance(refinements, bool) or refinements < 1:
        raise ValueError("refinements must be a positive integer")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie in (0, 1)")
    if max(nx, ny) * 2**refinements > _MAX_GRID:
        raise ValueError("refined grid exceeds the local limit; use pyrite remote")
    args = (plane_z_m, aperture)
    field = circular_pec_transition_field(energy_eV, screen, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m)
    bias = spectral_energy_per_eV(field) / hemisphere_energy_per_eV(field) - 1.0
    green = None
    if green_pitch_m is not None:
        if aperture is None:
            raise ValueError("the direct Green reference needs a finite aperture")
        green = collect_screen_green(
            field, plane_z_m=plane_z_m, aperture=aperture, pitch_m=green_pitch_m
        ).photons_per_eV
    base = _collect(energy_eV, screen, nx, ny, dx_m, dy_m, *args)
    pitch, window = [base], [base]
    for k in range(1, refinements + 1):
        f = 2**k
        pitch.append(_collect(energy_eV, screen, nx * f, ny * f, dx_m / f, dy_m / f, *args))
        window.append(_collect(energy_eV, screen, nx * f, ny * f, dx_m, dy_m, *args))
    return CollectionConvergence(
        energy_eV=float(energy_eV),
        plane_z_m=float(plane_z_m),
        base=base,
        pitch_refined=tuple(pitch[1:]),
        window_refined=tuple(window[1:]),
        pitch_relative_change=_relative_change(pitch),
        window_relative_change=_relative_change(window),
        hemisphere_fft_bias=float(bias),
        rtol=float(rtol),
        green_photons_per_eV=green,
    )
