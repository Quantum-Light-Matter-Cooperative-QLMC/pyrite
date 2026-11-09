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

:func:`finite_screen_spectral_convergence` closes the remaining two axes for a
band: **energy** (nested composite-Simpson node doubling of the band photon
count) and **angle** (doubling both Gauss–Legendre ``cos theta`` and uniform
``phi`` orders of :func:`hemisphere_energy_per_eV` at every base node). The
window-independent quadrature removes the window axis; the fixed source pitch
must resolve the shortest wavelength and is refined by the pitch diagnostic.

:func:`finite_aperture_spectral_convergence` does the same for a band
collected through a finite detector-plane aperture with the window-free direct
Green sum (:func:`collect_screen_green`): energy nodes, detector pitch and
source pitch (at a fixed physical window) are refined independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optical_finite_collection import PlaneAcceptance, collect_screen_field, collect_screen_green
from .optical_finite_screen import (
    _E_CHARGE_C,
    PECCircularScreen,
    ScreenSpectralField,
    circular_pec_transition_field,
    hemisphere_energy_per_eV,
    spectral_energy_per_eV,
)

__all__ = [
    "ApertureSpectralConvergence",
    "CollectionConvergence",
    "SpectralConvergence",
    "finite_aperture_spectral_convergence",
    "finite_screen_collection_convergence",
    "finite_screen_spectral_convergence",
]

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


@dataclass(frozen=True)
class SpectralConvergence:
    """Band photon count per electron under energy-node and angular refinement.

    ``energy_rows`` are ``(nodes, photons)`` at the base angular order;
    ``angular_rows`` are ``((n_u, n_phi), photons)`` on the base energy nodes.
    ``angular_pointwise_change`` is the largest relative change of
    ``dN/dE`` over the base nodes. ``energy_eV``/``photons_per_eV`` hold the
    finest energy grid.
    """

    band_eV: tuple[float, float]
    energy_eV: np.ndarray
    photons_per_eV: np.ndarray
    energy_rows: tuple[tuple[int, float], ...]
    angular_rows: tuple[tuple[tuple[int, int], float], ...]
    energy_relative_change: float
    angular_relative_change: float
    angular_pointwise_change: float
    rtol: float

    @property
    def band_photons(self) -> float:
        """Best (finest energy grid) band photon count per electron."""
        return self.energy_rows[-1][1]

    @property
    def converged(self) -> bool:
        return (
            max(
                self.energy_relative_change,
                self.angular_relative_change,
                self.angular_pointwise_change,
            )
            <= self.rtol
        )


def _simpson(energy, values):
    h = (energy[-1] - energy[0]) / (energy.size - 1)
    return float(
        h / 3 * (values[0] + values[-1] + 4 * values[1:-1:2].sum() + 2 * values[2:-1:2].sum())
    )


def finite_screen_spectral_convergence(
    band_eV: tuple[float, float],
    screen: PECCircularScreen,
    *,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    n_energy: int = 5,
    n_u: int = 32,
    n_phi: int = 64,
    refinements: int = 1,
    rtol: float = 0.01,
) -> SpectralConvergence:
    """Converge the hemisphere band photon count in energy nodes and angle.

    ``n_energy`` (odd, >= 3) uniform Simpson nodes span ``band_eV``; each
    energy refinement halves the spacing (``2 (n - 1) + 1`` nodes, nested, so
    earlier nodes are reused); each angular refinement doubles ``n_u`` and
    ``n_phi``. The reported changes are between the last two levels. The same
    sampled source grid is used at every energy (module docstring). Photons
    per eV are ``(dU/dE)/(E e)`` per physical electron.

    Validation: optical-finite-boundary-field
    """
    lo, hi = (float(v) for v in band_eV)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("band_eV must be finite with 0 < low < high")
    if not isinstance(n_energy, int) or n_energy < 3 or n_energy % 2 == 0:
        raise ValueError("n_energy must be an odd integer >= 3")
    if not isinstance(refinements, int) or isinstance(refinements, bool) or refinements < 1:
        raise ValueError("refinements must be a positive integer")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie in (0, 1)")
    fields: dict[float, ScreenSpectralField] = {}
    cache: dict[tuple[float, int, int], float] = {}

    def density(energy: float, order_u: int, order_phi: int) -> float:
        key = (energy, order_u, order_phi)
        if key not in cache:
            if energy not in fields:
                fields[energy] = circular_pec_transition_field(
                    energy, screen, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m
                )
            joules = hemisphere_energy_per_eV(fields[energy], n_u=order_u, n_phi=order_phi)
            cache[key] = joules / (energy * _E_CHARGE_C)
        return cache[key]

    energy_rows = []
    for level in range(refinements + 1):
        nodes = (n_energy - 1) * 2**level + 1
        energy = np.linspace(lo, hi, nodes)
        values = np.array([density(float(e), n_u, n_phi) for e in energy])
        energy_rows.append((nodes, _simpson(energy, values)))
    finest_energy, finest = energy, values
    base = np.linspace(lo, hi, n_energy)
    angular_rows, spectra = [], []
    for level in range(refinements + 1):
        order = (n_u * 2**level, n_phi * 2**level)
        values = np.array([density(float(e), *order) for e in base])
        spectra.append(values)
        angular_rows.append((order, _simpson(base, values)))
    pointwise = float(np.max(np.abs(spectra[-1] - spectra[-2]) / np.abs(spectra[-1])))
    return SpectralConvergence(
        band_eV=(lo, hi),
        energy_eV=finest_energy,
        photons_per_eV=finest,
        energy_rows=tuple(energy_rows),
        angular_rows=tuple(angular_rows),
        energy_relative_change=_relative_change(energy_rows),
        angular_relative_change=_relative_change(angular_rows),
        angular_pointwise_change=pointwise,
        rtol=float(rtol),
    )


@dataclass(frozen=True)
class ApertureSpectralConvergence:
    """Band photon count per electron through a finite detector-plane aperture.

    ``energy_rows`` are ``(nodes, photons)`` at the base pitches;
    ``detector_rows`` are ``(detector_pitch_m, photons)`` and ``source_rows``
    ``(source_pitch_m, photons)`` on the base energy nodes. The ``*_pointwise``
    changes are the largest relative ``dN/dE`` change over those nodes.
    ``energy_eV``/``photons_per_eV`` hold the finest energy grid.
    """

    band_eV: tuple[float, float]
    plane_z_m: float
    energy_eV: np.ndarray
    photons_per_eV: np.ndarray
    energy_rows: tuple[tuple[int, float], ...]
    detector_rows: tuple[tuple[float, float], ...]
    source_rows: tuple[tuple[float, float], ...]
    energy_relative_change: float
    detector_pointwise_change: float
    source_pointwise_change: float
    rtol: float

    @property
    def band_photons(self) -> float:
        """Best (finest energy grid) band photon count per electron."""
        return self.energy_rows[-1][1]

    @property
    def converged(self) -> bool:
        return (
            max(
                self.energy_relative_change,
                self.detector_pointwise_change,
                self.source_pointwise_change,
            )
            <= self.rtol
        )


def _pointwise(spectra):
    return float(np.max(np.abs(spectra[-1] - spectra[-2]) / np.abs(spectra[-1])))


def finite_aperture_spectral_convergence(
    band_eV: tuple[float, float],
    screen: PECCircularScreen,
    *,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    plane_z_m: float,
    aperture: PlaneAcceptance,
    detector_pitch_m: float,
    n_energy: int = 5,
    refinements: int = 1,
    rtol: float = 0.01,
) -> ApertureSpectralConvergence:
    """Converge a finite-aperture band photon count in energy and both pitches.

    Energy refinement halves the Simpson node spacing (nested, ``n_energy``
    odd >= 3). Detector refinement halves ``detector_pitch_m``; source
    refinement halves ``dx_m, dy_m`` and doubles ``nx, ny`` (fixed physical
    window, which must keep the screen compact). Both pitch axes are evaluated
    on the base energy nodes. The direct Green sum has no window axis. Cost is
    O(source pixels x aperture points) per node; refinements multiply it by 4.

    Validation: optical-wavefront-propagation
    """
    lo, hi = (float(v) for v in band_eV)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("band_eV must be finite with 0 < low < high")
    if not isinstance(n_energy, int) or n_energy < 3 or n_energy % 2 == 0:
        raise ValueError("n_energy must be an odd integer >= 3")
    if not isinstance(refinements, int) or isinstance(refinements, bool) or refinements < 1:
        raise ValueError("refinements must be a positive integer")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie in (0, 1)")
    if not isinstance(aperture, PlaneAcceptance):
        raise TypeError("aperture must be a PlaneAcceptance")
    if max(nx, ny) * 2**refinements > _MAX_GRID:
        raise ValueError("refined grid exceeds the local limit; use pyrite remote")
    cache: dict[tuple[float, int, float], float] = {}

    def density(energy: float, source_level: int, pitch: float) -> float:
        key = (energy, source_level, pitch)
        if key not in cache:
            f = 2**source_level
            field = circular_pec_transition_field(
                energy, screen, nx=nx * f, ny=ny * f, dx_m=dx_m / f, dy_m=dy_m / f
            )
            cache[key] = collect_screen_green(
                field, plane_z_m=plane_z_m, aperture=aperture, pitch_m=pitch
            ).photons_per_eV
        return cache[key]

    energy_rows = []
    for level in range(refinements + 1):
        nodes = (n_energy - 1) * 2**level + 1
        energy = np.linspace(lo, hi, nodes)
        values = np.array([density(float(e), 0, detector_pitch_m) for e in energy])
        energy_rows.append((nodes, _simpson(energy, values)))
    finest_energy, finest = energy, values
    base = np.linspace(lo, hi, n_energy)
    detector_rows, detector_spectra, source_rows, source_spectra = [], [], [], []
    for level in range(refinements + 1):
        pitch = detector_pitch_m / 2**level
        values = np.array([density(float(e), 0, pitch) for e in base])
        detector_spectra.append(values)
        detector_rows.append((float(pitch), _simpson(base, values)))
        values = np.array([density(float(e), level, detector_pitch_m) for e in base])
        source_spectra.append(values)
        source_rows.append((float(dx_m / 2**level), _simpson(base, values)))
    return ApertureSpectralConvergence(
        band_eV=(lo, hi),
        plane_z_m=float(plane_z_m),
        energy_eV=finest_energy,
        photons_per_eV=finest,
        energy_rows=tuple(energy_rows),
        detector_rows=tuple(detector_rows),
        source_rows=tuple(source_rows),
        energy_relative_change=_relative_change(energy_rows),
        detector_pointwise_change=_pointwise(detector_spectra),
        source_pointwise_change=_pointwise(source_spectra),
        rtol=float(rtol),
    )
