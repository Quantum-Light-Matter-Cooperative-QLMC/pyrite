"""Returned value for the public single-shot simulation API."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from .._spectral_components import line_spectrum
from ..instrument import PlanarDetector
from ..montecarlo import Case

#: Spatial result components. ``*_total`` adds characteristic radiation.
COMPONENTS = ("line", "background", "coherent", "characteristic", "line_total", "coherent_total")


def _readonly_array(value: object, *, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class PixelRayMap:
    """Store shared pixel geometry for factorized photon scoring.

    Parameters
    ----------
    tile_index
        Integer array ``(ny, nx)`` mapping every pixel to an angular tile.
    solid_angle_sr
        Positive per-pixel solid angle in sr with shape ``(ny, nx)``.
    path_length_mm
        Non-negative filter path lengths with shape ``(ny, nx, n_filter)``.

    Notes
    -----
    Inputs are converted to read-only NumPy arrays.
    """

    tile_index: np.ndarray
    solid_angle_sr: np.ndarray
    path_length_mm: np.ndarray

    def __post_init__(self) -> None:
        tile_index = _readonly_array(self.tile_index, dtype=np.int64)
        solid_angle = _readonly_array(self.solid_angle_sr, dtype=float)
        paths = _readonly_array(self.path_length_mm, dtype=float)
        if tile_index.ndim != 2:
            raise ValueError("tile_index must be two-dimensional")
        if solid_angle.shape != tile_index.shape:
            raise ValueError("solid_angle_sr must match tile_index")
        if paths.shape[:2] != tile_index.shape or paths.ndim != 3:
            raise ValueError("path_length_mm must have shape (ny, nx, n_filter)")
        if np.any(tile_index < 0):
            raise ValueError("tile_index must be non-negative")
        if not np.all(np.isfinite(solid_angle)) or np.any(solid_angle <= 0.0):
            raise ValueError("solid_angle_sr must contain finite positive values")
        if not np.all(np.isfinite(paths)) or np.any(paths < 0.0):
            raise ValueError("path_length_mm must contain finite non-negative values")
        object.__setattr__(self, "tile_index", tile_index)
        object.__setattr__(self, "solid_angle_sr", solid_angle)
        object.__setattr__(self, "path_length_mm", paths)


@dataclass(frozen=True)
class SpectralFactors:
    """Store tile spectra and filters without an eager pixel-energy cube.

    Parameters
    ----------
    energy_eV
        One-dimensional photon-energy coordinate in eV.
    intrinsic_by_tile
        Pre-filter source density with shape ``(n_tile, n_energy)`` in photons
        per incident electron per eV per sr. Pixel solid angle, filter
        transmission, and detector response have not been applied.
    mu_by_filter_inv_mm
        Linear attenuation coefficients with shape ``(n_filter, n_energy)`` in
        inverse mm.
    """

    energy_eV: np.ndarray
    intrinsic_by_tile: np.ndarray
    mu_by_filter_inv_mm: np.ndarray

    def __post_init__(self) -> None:
        energy = _readonly_array(self.energy_eV, dtype=float)
        intrinsic = _readonly_array(self.intrinsic_by_tile, dtype=float)
        coefficient = _readonly_array(self.mu_by_filter_inv_mm, dtype=float)
        if energy.ndim != 1 or energy.size == 0:
            raise ValueError("energy_eV must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
            raise ValueError("energy_eV must contain finite non-negative values")
        if intrinsic.ndim != 2 or intrinsic.shape[1] != energy.size:
            raise ValueError("intrinsic_by_tile must have shape (n_tile, n_energy)")
        if coefficient.ndim != 2 or coefficient.shape[1] != energy.size:
            raise ValueError("mu_by_filter_inv_mm must have shape (n_filter, n_energy)")
        if not np.all(np.isfinite(intrinsic)):
            raise ValueError("intrinsic_by_tile must contain finite values")
        if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError("mu_by_filter_inv_mm must contain finite non-negative values")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "intrinsic_by_tile", intrinsic)
        object.__setattr__(self, "mu_by_filter_inv_mm", coefficient)


@dataclass(frozen=True)
class SpatialResult:
    """Provide lazy spatial scoring from factorized planar-detector data.

    Parameters
    ----------
    ray_map
        Shared pixel solid angles, angular-tile indices, and filter path lengths.
    line, background
        Pre-filter line and continuum source factors.
    detector
        Physical detector used for geometry and optional measured response.
    coherent_line
        Optional coherent line factors, available for coherent calculations.
    characteristic_line
        Optional characteristic-radiation factors. ``line`` and
        ``coherent_line`` exclude it; the ``"line_total"`` and
        ``"coherent_total"`` components add it.
    """

    ray_map: PixelRayMap
    line: SpectralFactors
    background: SpectralFactors
    detector: PlanarDetector
    coherent_line: SpectralFactors | None = None
    characteristic_line: SpectralFactors | None = None

    def __post_init__(self) -> None:
        for name in ("line", "background"):
            factor = getattr(self, name)
            if not isinstance(factor, SpectralFactors):
                raise TypeError(f"{name} must be SpectralFactors")
            self._validate_factor(factor)
        if self.coherent_line is not None:
            if not isinstance(self.coherent_line, SpectralFactors):
                raise TypeError("coherent_line must be SpectralFactors or None")
            self._validate_factor(self.coherent_line)
        if self.characteristic_line is not None:
            if not isinstance(self.characteristic_line, SpectralFactors):
                raise TypeError("characteristic_line must be SpectralFactors or None")
            self._validate_factor(self.characteristic_line)
        if not isinstance(self.detector, PlanarDetector):
            raise TypeError("detector must be a PlanarDetector")

    def _validate_factor(self, factor: SpectralFactors) -> None:
        n_tile = int(np.max(self.ray_map.tile_index)) + 1
        if factor.intrinsic_by_tile.shape[0] != n_tile:
            raise ValueError("spectral tile count must match ray_map.tile_index")
        if factor.mu_by_filter_inv_mm.shape[0] != self.ray_map.path_length_mm.shape[2]:
            raise ValueError("spectral filter count must match ray_map.path_length_mm")

    def _factor(self, component: str) -> SpectralFactors:
        if component == "line":
            return self.line
        if component == "background":
            return self.background
        if component == "coherent":
            if self.coherent_line is None:
                raise ValueError("coherent line spectrum is not available")
            return self.coherent_line
        if component == "characteristic":
            if self.characteristic_line is None:
                raise ValueError("characteristic line spectrum is not available")
            return self.characteristic_line
        if component in {"line_total", "coherent_total"}:
            base = self._factor(component.removesuffix("_total"))
            extra = self.characteristic_line
            if extra is None:
                return base
            # Both factors share the line grid and attenuation, so the
            # pre-filter densities add tile by tile.
            return SpectralFactors(
                base.energy_eV,
                base.intrinsic_by_tile + extra.intrinsic_by_tile,
                base.mu_by_filter_inv_mm,
            )
        raise ValueError(f"component must be one of {', '.join(map(repr, COMPONENTS))}")

    def _coordinates(self, *, pixels, region) -> np.ndarray:
        if (pixels is None) == (region is None):
            raise ValueError("provide exactly one of pixels or region")
        ny, nx = self.ray_map.tile_index.shape
        if region is not None:
            if (
                not isinstance(region, tuple)
                or len(region) != 2
                or not all(isinstance(part, slice) for part in region)
            ):
                raise TypeError("region must be a (row_slice, column_slice) tuple")
            rows = np.arange(ny)[region[0]]
            columns = np.arange(nx)[region[1]]
            yy, xx = np.meshgrid(rows, columns, indexing="ij")
            coordinates = np.column_stack((yy.ravel(), xx.ravel()))
        else:
            try:
                raw_coordinates = np.asarray(tuple(pixels), dtype=object)
            except TypeError as exc:
                raise TypeError("pixels must be an iterable of (row, column) pairs") from exc
            if raw_coordinates.ndim != 2 or raw_coordinates.shape[1] != 2:
                raise ValueError("pixels must have shape (n_pixel, 2)")
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_coordinates.ravel()
            ):
                raise TypeError("pixel coordinates must be integers")
            coordinates = raw_coordinates.astype(np.int64)
        if coordinates.size and (
            np.any(coordinates[:, 0] < 0)
            or np.any(coordinates[:, 0] >= ny)
            or np.any(coordinates[:, 1] < 0)
            or np.any(coordinates[:, 1] >= nx)
        ):
            raise IndexError("pixel coordinate is outside the detector grid")
        if not coordinates.size:
            raise ValueError("pixel selection must not be empty")
        return coordinates

    def _materialize(self, factor: SpectralFactors, coordinates: np.ndarray) -> np.ndarray:
        from ..instrument import primary_transmission

        rows, columns = coordinates.T
        tile = self.ray_map.tile_index[rows, columns]
        paths = self.ray_map.path_length_mm[rows, columns]
        transmission = primary_transmission(paths, factor.mu_by_filter_inv_mm)
        return (
            factor.intrinsic_by_tile[tile]
            * self.ray_map.solid_angle_sr[rows, columns, None]
            * transmission
        )

    def spectra(
        self,
        *,
        pixels=None,
        region=None,
        component: str = "line",
        measured: bool = False,
        fwhm_eV: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize selected accepted-flux spectra.

        Parameters
        ----------
        pixels
            Iterable of ``(row, column)`` pixel coordinates. Mutually exclusive
            with ``region``.
        region
            ``(row_slice, column_slice)`` selection. Mutually exclusive with
            ``pixels``.
        component
            One of :data:`COMPONENTS`. ``"line"`` and ``"coherent"`` exclude
            characteristic radiation; ``"line_total"`` and ``"coherent_total"``
            include it.
        measured
            Apply the detector response after geometric scoring.
        fwhm_eV
            Optional response-resolution override in eV.

        Returns
        -------
        energy_eV, spectra
            Photon-energy coordinate and an ``(n_pixel, n_energy)`` array in
            photons per incident electron per eV.
        """
        factor = self._factor(component)
        coordinates = self._coordinates(pixels=pixels, region=region)
        spectra = self._materialize(factor, coordinates)
        if measured:
            spectra = np.stack(
                [
                    np.asarray(
                        self.detector.score(factor.energy_eV, spectrum, fwhm_eV=fwhm_eV, scale=1.0)
                    )
                    for spectrum in spectra
                ]
            )
            if spectra.shape[1:] != (factor.energy_eV.size,):
                raise ValueError("detector response must preserve the requested energy grid")
        return factor.energy_eV, spectra

    def image(
        self,
        energy_range_eV: tuple[float, float],
        *,
        component: str = "line",
        measured: bool = False,
        fwhm_eV: float | None = None,
        pixel_chunk: int = 1024,
    ) -> np.ndarray:
        """Integrate an energy window into a detector image.

        Parameters
        ----------
        energy_range_eV
            Finite increasing ``(low, high)`` bounds in eV, inclusive.
        component
            One of :data:`COMPONENTS`. ``"line"`` and ``"coherent"`` exclude
            characteristic radiation; ``"line_total"`` and ``"coherent_total"``
            include it.
        measured
            Apply detector response before integration.
        fwhm_eV
            Optional response-resolution override in eV.
        pixel_chunk
            Positive maximum pixels materialized per working chunk.

        Returns
        -------
        numpy.ndarray
            ``(ny, nx)`` accepted photons per incident electron.
        """
        chunk = _positive_integer("pixel_chunk", pixel_chunk)
        factor = self._factor(component)
        low, high = map(float, energy_range_eV)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("energy_range_eV must be a finite increasing pair")
        selected = (factor.energy_eV >= low) & (factor.energy_eV <= high)
        if np.count_nonzero(selected) < 2:
            raise ValueError("energy window must contain at least two energy samples")
        ny, nx = self.ray_map.tile_index.shape
        image = np.empty(ny * nx, dtype=float)
        flat = np.arange(ny * nx)
        for start in range(0, flat.size, chunk):
            chosen = flat[start : start + chunk]
            coordinates = np.column_stack(np.unravel_index(chosen, (ny, nx)))
            spectra = self._materialize(factor, coordinates)
            if measured:
                spectra = np.stack(
                    [
                        np.asarray(
                            self.detector.score(
                                factor.energy_eV,
                                spectrum,
                                fwhm_eV=fwhm_eV,
                                scale=1.0,
                            )
                        )
                        for spectrum in spectra
                    ]
                )
                if spectra.shape[1:] != (factor.energy_eV.size,):
                    raise ValueError("detector response must preserve the requested energy grid")
            image[chosen] = np.trapezoid(spectra[:, selected], factor.energy_eV[selected], axis=1)
        return image.reshape(ny, nx)

    def average_density(self, component: str, *, pixel_chunk: int = 1024) -> np.ndarray:
        """Return the solid-angle-weighted detector-average density.

        Parameters
        ----------
        component
            One of :data:`COMPONENTS`. ``"line"`` and ``"coherent"`` exclude
            characteristic radiation; ``"line_total"`` and ``"coherent_total"``
            include it.
        pixel_chunk
            Positive maximum pixels materialized per working chunk.

        Returns
        -------
        numpy.ndarray
            Spectrum in photons per incident electron per eV per sr.
        """
        chunk = _positive_integer("pixel_chunk", pixel_chunk)
        factor = self._factor(component)
        ny, nx = self.ray_map.tile_index.shape
        total = np.zeros(factor.energy_eV.shape, dtype=float)
        flat = np.arange(ny * nx)
        for start in range(0, flat.size, chunk):
            chosen = flat[start : start + chunk]
            coordinates = np.column_stack(np.unravel_index(chosen, (ny, nx)))
            total += np.sum(self._materialize(factor, coordinates), axis=0)
        return total / np.sum(self.ray_map.solid_angle_sr)


@dataclass(frozen=True)
class Result:
    """Scalar spectral arrays, optional spatial factors, and provenance.

    ``spectrum`` and ``background`` are photon densities per incident electron
    per eV per sr. For scalar-detector simulations they are response-free source
    densities without acceptance or current scaling. For a
    physical planar detector they are filter-attenuated, solid-angle-weighted
    observation averages. Selected ``spatial`` spectra are accepted per-pixel
    flux, with pixel solid angle included. ``energy_eV`` and
    ``background_energy_eV`` are the respective photon-energy coordinates.

    Parameters
    ----------
    energy_eV, background_energy_eV
        One-dimensional line and background photon-energy coordinates in eV.
    spectrum, background
        Line and continuum densities in photons per incident electron per eV
        per sr.
    case
        Fully resolved transport input that produced the arrays.
    provenance
        Immutable-view metadata describing scene, numerics, identity, backend,
        and dependency versions.
    coherent_spectrum
        Optional coherent line density on ``energy_eV``.
    characteristic_spectrum
        Optional characteristic-radiation density on ``energy_eV``.
        ``spectrum`` and ``coherent_spectrum`` exclude it; use
        :meth:`line_total` for the sum.
    spatial
        Optional factorized planar-detector result.
    """

    energy_eV: np.ndarray
    spectrum: np.ndarray
    background_energy_eV: np.ndarray
    background: np.ndarray
    case: Case
    provenance: Mapping[str, Any]
    coherent_spectrum: np.ndarray | None = None
    characteristic_spectrum: np.ndarray | None = None
    spatial: SpatialResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_eV", np.asarray(self.energy_eV))
        object.__setattr__(self, "spectrum", np.asarray(self.spectrum))
        object.__setattr__(self, "background_energy_eV", np.asarray(self.background_energy_eV))
        object.__setattr__(self, "background", np.asarray(self.background))
        if self.coherent_spectrum is not None:
            object.__setattr__(self, "coherent_spectrum", np.asarray(self.coherent_spectrum))
        if self.characteristic_spectrum is not None:
            object.__setattr__(
                self,
                "characteristic_spectrum",
                np.asarray(self.characteristic_spectrum),
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

        if self.spatial is not None and not isinstance(self.spatial, SpatialResult):
            raise TypeError("spatial must be a SpatialResult or None")

    def line_total(self, *, coherent: bool = False, characteristic: bool = True) -> np.ndarray:
        """Return the line-grid emission density on ``energy_eV``.

        Parameters
        ----------
        coherent
            Use ``coherent_spectrum`` instead of ``spectrum``.
        characteristic
            Add ``characteristic_spectrum`` when present.

        Returns
        -------
        numpy.ndarray
            Photons per incident electron per eV per sr.
        """
        if coherent and self.coherent_spectrum is None:
            raise ValueError("coherent spectrum is not available")
        record = {
            "spec": self.spectrum,
            "spec_coherent": self.coherent_spectrum,
            "spec_characteristic": self.characteristic_spectrum,
        }
        return line_spectrum(record, coherent=coherent, characteristic=characteristic)


__all__ = ["PixelRayMap", "Result", "SpatialResult", "SpectralFactors"]
