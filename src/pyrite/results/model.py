"""Returned value for the public single-shot simulation API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from ..instrument import PlanarDetector
from ..montecarlo import Case


def _readonly_array(value: object, *, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PixelRayMap:
    """Shared pixel geometry for factorized downstream photon scoring."""

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
    """Tile spectra and filter coefficients without an eager pixel-energy cube."""

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
    """Lazy, factorized pixel result for a physical planar detector."""

    ray_map: PixelRayMap
    line: SpectralFactors
    background: SpectralFactors
    detector: PlanarDetector
    coherent_line: SpectralFactors | None = None

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
        raise ValueError("component must be 'line', 'background', or 'coherent'")

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
        """Materialize selected accepted-flux spectra [photons/electron/eV]."""
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
        """Integrate a true or measured energy window in bounded pixel chunks."""
        if pixel_chunk <= 0:
            raise ValueError("pixel_chunk must be positive")
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
        for start in range(0, flat.size, pixel_chunk):
            chosen = flat[start : start + pixel_chunk]
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
        """Return the solid-angle-weighted detector-average density [per sr]."""
        factor = self._factor(component)
        ny, nx = self.ray_map.tile_index.shape
        total = np.zeros(factor.energy_eV.shape, dtype=float)
        flat = np.arange(ny * nx)
        for start in range(0, flat.size, pixel_chunk):
            chosen = flat[start : start + pixel_chunk]
            coordinates = np.column_stack(np.unravel_index(chosen, (ny, nx)))
            total += np.sum(self._materialize(factor, coordinates), axis=0)
        return total / np.sum(self.ray_map.solid_angle_sr)


@dataclass(frozen=True)
class Result:
    """Intrinsic spectral arrays and resolved simulation provenance.

    ``spectrum`` and ``background`` are photon-density arrays per incident
    electron per eV per sr. ``energy_eV`` and ``background_energy_eV`` are their
    respective photon-energy coordinates in eV.
    """

    energy_eV: np.ndarray
    spectrum: np.ndarray
    background_energy_eV: np.ndarray
    background: np.ndarray
    case: Case
    provenance: Mapping[str, Any]
    coherent_spectrum: np.ndarray | None = None
    spatial: SpatialResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_eV", np.asarray(self.energy_eV))
        object.__setattr__(self, "spectrum", np.asarray(self.spectrum))
        object.__setattr__(self, "background_energy_eV", np.asarray(self.background_energy_eV))
        object.__setattr__(self, "background", np.asarray(self.background))
        if self.coherent_spectrum is not None:
            object.__setattr__(self, "coherent_spectrum", np.asarray(self.coherent_spectrum))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

        if self.spatial is not None and not isinstance(self.spatial, SpatialResult):
            raise TypeError("spatial must be a SpatialResult or None")


__all__ = ["PixelRayMap", "Result", "SpatialResult", "SpectralFactors"]
