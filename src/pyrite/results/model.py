"""Returned value for the public single-shot simulation API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from .._api_deprecations import renamed_fields, warn_renamed
from ..instrument import PlanarDetector
from ..montecarlo import Case

#: Spatial result components. ``*_total`` adds characteristic radiation.
COMPONENTS = ("cxr", "brem", "coherent", "characteristic", "cxr_total", "coherent_total")

if TYPE_CHECKING:
    import pandas as pd

    from ..instrument import Acquisition, AcquisitionBatch
    from ..instrument.geometry import PixelRays


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
        inverse mm; ``+inf`` marks an energy at which the filter is opaque.
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
        if np.any(np.isnan(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError(
                "mu_by_filter_inv_mm must contain non-negative values (+inf marks an opaque energy)"
            )
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "intrinsic_by_tile", intrinsic)
        object.__setattr__(self, "mu_by_filter_inv_mm", coefficient)


@dataclass(frozen=True)
class PixelMetadata:
    """Per-pixel geometry and filter factors for a pixel selection.

    Every array's first axis follows ``coordinates``. Lab-frame vectors use
    the source at the origin with ``+z`` along the beam; polar angle is
    measured from ``+z`` and azimuth from ``+x`` toward ``+y``. Detector-local
    positions are the pixel-centre offsets along the pose ``x``/``y`` axes.

    Parameters
    ----------
    coordinates
        ``(n_pixel, 2)`` integer ``(row, column)`` pairs.
    local_position_mm
        ``(n_pixel, 2)`` detector-local ``(x, y)`` pixel-centre offsets in mm.
    center_mm
        ``(n_pixel, 3)`` lab-frame pixel centres in mm.
    direction_lab
        ``(n_pixel, 3)`` unit source-to-pixel-centre directions.
    polar_deg, azimuth_deg
        ``(n_pixel,)`` lab polar and azimuth angles of ``direction_lab``.
    distance_mm
        ``(n_pixel,)`` source-to-pixel-centre distance in mm.
    solid_angle_sr
        ``(n_pixel,)`` scored pixel solid angle in sr.
    tile_index
        ``(n_pixel,)`` angular tile containing the pixel: its sole intrinsic
        spectrum under ``"nearest_tile"``, one of up to four blended tiles
        under ``"bilinear_tile"``.
    tile_direction_lab
        ``(n_pixel, 3)`` representative tile direction, or ``None`` when the
        result does not retain tile directions.
    path_length_mm
        ``(n_pixel, n_filter)`` filter path lengths in mm, in filter order.
    """

    coordinates: np.ndarray
    local_position_mm: np.ndarray
    center_mm: np.ndarray
    direction_lab: np.ndarray
    polar_deg: np.ndarray
    azimuth_deg: np.ndarray
    distance_mm: np.ndarray
    solid_angle_sr: np.ndarray
    tile_index: np.ndarray
    tile_direction_lab: np.ndarray | None
    path_length_mm: np.ndarray

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None:
                object.__setattr__(self, name, _readonly_array(value))


@renamed_fields(
    line="cxr",
    background="brem",
    coherent_line="coherent_cxr",
    characteristic_line="characteristic",
)
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
    coherent_cxr
        Optional coherent line factors, available for coherent calculations.
    characteristic
        Optional characteristic-radiation factors. ``line`` and
        ``coherent_cxr`` exclude it; the ``"cxr_total"`` and
        ``"coherent_total"`` components add it.
    tile_directions_lab
        Optional representative unit direction per angular tile with shape
        ``(n_tile, 3)`` in the lab frame; required to persist the result.
    reconstruction
        :class:`~pyrite.instrument.PixelScorer` reconstruction mode.
        ``"nearest_tile"`` gives each pixel its tile's intrinsic spectrum;
        ``"bilinear_tile"`` blends neighbouring tiles and requires
        ``tile_directions_lab`` and a pixelated detector.
    """

    ray_map: PixelRayMap
    cxr: SpectralFactors
    brem: SpectralFactors
    detector: PlanarDetector
    coherent_cxr: SpectralFactors | None = None
    characteristic: SpectralFactors | None = None
    tile_directions_lab: np.ndarray | None = None
    reconstruction: str = "nearest_tile"

    def __post_init__(self) -> None:
        from ..instrument.model import RECONSTRUCTIONS

        if self.reconstruction not in RECONSTRUCTIONS:
            raise ValueError(
                "reconstruction must be one of " + ", ".join(map(repr, RECONSTRUCTIONS))
            )
        if self.reconstruction == "bilinear_tile" and self.tile_directions_lab is None:
            raise ValueError("bilinear_tile reconstruction requires tile_directions_lab")
        if self.tile_directions_lab is not None:
            directions = _readonly_array(self.tile_directions_lab, dtype=float)
            n_tile = int(np.max(self.ray_map.tile_index)) + 1
            if directions.shape != (n_tile, 3):
                raise ValueError("tile_directions_lab must have shape (n_tile, 3)")
            if not np.all(np.isfinite(directions)):
                raise ValueError("tile_directions_lab must contain finite values")
            object.__setattr__(self, "tile_directions_lab", directions)
        for name in ("cxr", "brem"):
            factor = getattr(self, name)
            if not isinstance(factor, SpectralFactors):
                raise TypeError(f"{name} must be SpectralFactors")
            self._validate_factor(factor)
        if self.coherent_cxr is not None:
            if not isinstance(self.coherent_cxr, SpectralFactors):
                raise TypeError("coherent_cxr must be SpectralFactors or None")
            self._validate_factor(self.coherent_cxr)
        if self.characteristic is not None:
            if not isinstance(self.characteristic, SpectralFactors):
                raise TypeError("characteristic must be SpectralFactors or None")
            self._validate_factor(self.characteristic)
        if not isinstance(self.detector, PlanarDetector):
            raise TypeError("detector must be a PlanarDetector")

    def _validate_factor(self, factor: SpectralFactors) -> None:
        n_tile = int(np.max(self.ray_map.tile_index)) + 1
        if factor.intrinsic_by_tile.shape[0] != n_tile:
            raise ValueError("spectral tile count must match ray_map.tile_index")
        if factor.mu_by_filter_inv_mm.shape[0] != self.ray_map.path_length_mm.shape[2]:
            raise ValueError("spectral filter count must match ray_map.path_length_mm")

    @cached_property
    def _rays(self) -> PixelRays:
        from ..instrument.geometry import planar_detector_rays

        rays = planar_detector_rays(self.detector)
        if rays.shape != self.ray_map.tile_index.shape:
            raise ValueError("detector pixel grid does not match ray_map")
        return rays

    @cached_property
    def _blend(self) -> tuple[np.ndarray, np.ndarray]:
        from ..instrument.geometry import angular_tile_weights

        tile_index = self.ray_map.tile_index
        n_tile = int(np.max(tile_index)) + 1
        n_column = int(tile_index[0, -1]) + 1
        angular_shape = (n_tile // n_column, n_column)
        assert self.tile_directions_lab is not None
        tile, weight = angular_tile_weights(self.detector, self.tile_directions_lab, angular_shape)
        if tile.shape[:2] != tile_index.shape:
            raise ValueError("detector pixel grid does not match ray_map")
        return tile, weight

    @property
    def shape(self) -> tuple[int, int]:
        """Detector pixel-grid shape ``(ny, nx)``."""
        return self.ray_map.tile_index.shape  # type: ignore[return-value]

    def pixel_metadata(self, *, pixels=None, region=None) -> PixelMetadata:
        """Return geometry and filter factors for selected pixels.

        Geometry is recomputed from ``detector``; it reproduces the stored
        ``ray_map`` solid angles, so a reopened observation reports the same
        values that scored it.

        Parameters
        ----------
        pixels
            Iterable of ``(row, column)`` pixel coordinates. Mutually exclusive
            with ``region``.
        region
            ``(row_slice, column_slice)`` selection. Mutually exclusive with
            ``pixels``.
        """
        coordinates = self._coordinates(pixels=pixels, region=region)
        rows, columns = coordinates.T
        rays = self._rays
        pixel_grid = self.detector.pixels
        ny, nx = self.shape
        if pixel_grid is None:
            local = np.zeros((rows.size, 2))
        else:
            pitch_y, pitch_x = pixel_grid.pitch_mm
            local = np.column_stack(
                ((columns - (nx - 1) / 2.0) * pitch_x, (rows - (ny - 1) / 2.0) * pitch_y)
            )
        direction = rays.directions_lab[rows, columns]
        tile = self.ray_map.tile_index[rows, columns]
        return PixelMetadata(
            coordinates=coordinates,
            local_position_mm=local,
            center_mm=rays.centers_mm[rows, columns],
            direction_lab=direction,
            polar_deg=np.degrees(np.arccos(np.clip(direction[:, 2], -1.0, 1.0))),
            azimuth_deg=np.degrees(np.arctan2(direction[:, 1], direction[:, 0])),
            distance_mm=rays.distance_mm[rows, columns],
            solid_angle_sr=self.ray_map.solid_angle_sr[rows, columns],
            tile_index=tile,
            tile_direction_lab=(
                None if self.tile_directions_lab is None else self.tile_directions_lab[tile]
            ),
            path_length_mm=self.ray_map.path_length_mm[rows, columns],
        )

    def filter_coverage(self) -> np.ndarray:
        """Return ``(ny, nx, n_filter)`` booleans: the centre ray crosses the filter."""
        return self.ray_map.path_length_mm > 0.0

    def _energy_node(self, energy_eV: float, component: str) -> int:
        grid = self._factor(component).energy_eV
        energy = float(energy_eV)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite")
        if not grid[0] <= energy <= grid[-1]:
            raise ValueError(
                f"energy_eV {energy:g} is outside the {component} grid "
                f"[{grid[0]:g}, {grid[-1]:g}] eV"
            )
        return int(np.argmin(np.abs(grid - energy)))

    def energy_node(self, energy_eV: float, *, component: str = "cxr") -> float:
        """Return the stored energy node in eV nearest ``energy_eV`` on a component grid.

        Raises
        ------
        ValueError
            If ``energy_eV`` is not finite or lies outside that grid.
        """
        return float(self._factor(component).energy_eV[self._energy_node(energy_eV, component)])

    def transmission_image(self, energy_eV: float, *, component: str = "cxr") -> np.ndarray:
        """Return ``(ny, nx)`` primary filter transmission at one energy node.

        ``energy_eV`` selects the nearest sample of the component's energy
        grid (:meth:`energy_node`); attenuation coefficients are stored only
        at those nodes, so no value is interpolated, and an energy outside the
        grid is rejected rather than snapped to its end. Uncovered pixels and
        the no-filter case are 1.
        """
        from ..instrument import primary_transmission

        factor = self._factor(component)
        node = self._energy_node(energy_eV, component)
        return primary_transmission(
            self.ray_map.path_length_mm, factor.mu_by_filter_inv_mm[:, node : node + 1]
        )[..., 0]

    def _factor(self, component: str) -> SpectralFactors:
        aliases = {"line": "cxr", "background": "brem", "line_total": "cxr_total"}
        if component in aliases:
            warn_renamed(f"SpatialResult component {component!r}", repr(aliases[component]))
            component = aliases[component]
        if component == "cxr":
            return self.cxr
        if component == "brem":
            return self.brem
        if component == "coherent":
            if self.coherent_cxr is None:
                raise ValueError("coherent CXR spectrum is not available")
            return self.coherent_cxr
        if component == "characteristic":
            if self.characteristic is None:
                raise ValueError("characteristic CXR spectrum is not available")
            return self.characteristic
        if component in {"cxr_total", "coherent_total"}:
            base = self._factor(component.removesuffix("_total"))
            extra = self.characteristic
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
        elif isinstance(pixels, np.ndarray) and pixels.dtype.kind in "iu":
            # Integer arrays (the chunked image paths) need no per-element check.
            if pixels.ndim != 2 or pixels.shape[1] != 2:
                raise ValueError("pixels must have shape (n_pixel, 2)")
            coordinates = pixels.astype(np.int64, copy=False)
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
        """Validation: pixel-angular-interpolation"""
        from ..instrument import primary_transmission

        rows, columns = coordinates.T
        paths = self.ray_map.path_length_mm[rows, columns]
        transmission = primary_transmission(paths, factor.mu_by_filter_inv_mm)
        if self.reconstruction == "bilinear_tile":
            # Validation: pixel-angular-interpolation
            tiles, weights = self._blend
            tile, weight = tiles[rows, columns], weights[rows, columns]
            intrinsic = np.einsum("pk,pke->pe", weight, factor.intrinsic_by_tile[tile])
        else:
            intrinsic = factor.intrinsic_by_tile[self.ray_map.tile_index[rows, columns]]
        return intrinsic * self.ray_map.solid_angle_sr[rows, columns, None] * transmission

    def spectra(
        self,
        *,
        pixels=None,
        region=None,
        component: str = "cxr",
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
            One of :data:`COMPONENTS`. ``"cxr"`` and ``"coherent"`` exclude
            characteristic radiation; ``"cxr_total"`` and ``"coherent_total"``
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
        component: str = "cxr",
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
            One of :data:`COMPONENTS`. ``"cxr"`` and ``"coherent"`` exclude
            characteristic radiation; ``"cxr_total"`` and ``"coherent_total"``
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
            One of :data:`COMPONENTS`. ``"cxr"`` and ``"coherent"`` exclude
            characteristic radiation; ``"cxr_total"`` and ``"coherent_total"``
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

    def acquire(
        self,
        acquisition: Acquisition,
        *,
        rep_rate_hz: float,
        bunch_charge_pc: float,
        observation_digest: str,
        pixels=None,
        region=None,
        components: tuple[str, ...] = ("cxr", "brem"),
    ) -> AcquisitionBatch:
        """Score selected pixels into common reporting bins and count channels."""
        from ..instrument import combine_acquisitions, score_acquisition

        aliases = {"line": "cxr", "background": "brem", "line_total": "cxr_total"}
        resolved_components = tuple(aliases.get(name, name) for name in components)
        if not resolved_components:
            raise ValueError("components must not be empty")
        if len(set(resolved_components)) != len(resolved_components):
            raise ValueError("components must be unique")
        coordinates = self._coordinates(pixels=pixels, region=region)
        detector = self.detector.scalar_detector()
        batches = []
        for component in resolved_components:
            factor = self._factor(component)
            batch = score_acquisition(
                detector,
                factor.energy_eV,
                self._materialize(factor, coordinates),
                acquisition,
                rep_rate_hz=rep_rate_hz,
                bunch_charge_pc=bunch_charge_pc,
                observation_digest=observation_digest,
                coordinates=coordinates,
                # Component names seed stored Poisson observations. Keep
                # historical stream keys while returning canonical labels.
                component={"cxr": "line", "brem": "background", "cxr_total": "line_total"}.get(
                    component, component
                ),
            )
            batches.append(replace(batch, components=(component,)))
        return combine_acquisitions(batches)

    def acquisition_image(
        self,
        *,
        acquisition: Acquisition,
        rep_rate_hz: float,
        bunch_charge_pc: float,
        observation_digest: str,
        energy_range_eV: tuple[float, float] | None = None,
        components: tuple[str, ...] = ("cxr", "brem"),
        pixel_chunk: int = 1024,
    ) -> np.ndarray:
        """Return a total or reporting-window count image in bounded pixel chunks."""
        chunk = _positive_integer("pixel_chunk", pixel_chunk)
        ny, nx = self.ray_map.tile_index.shape
        dtype = np.int64 if acquisition.mode == "poisson" else float
        image = np.empty(ny * nx, dtype=dtype)
        flat = np.arange(ny * nx)
        for start in range(0, flat.size, chunk):
            chosen = flat[start : start + chunk]
            coordinates = np.column_stack(np.unravel_index(chosen, (ny, nx)))
            batch = self.acquire(
                acquisition,
                rep_rate_hz=rep_rate_hz,
                bunch_charge_pc=bunch_charge_pc,
                observation_digest=observation_digest,
                pixels=coordinates,
                components=components,
            )
            image[chosen] = (
                batch.total_counts
                if energy_range_eV is None
                else batch.window_counts(energy_range_eV)
            )
        return image.reshape(ny, nx)


@dataclass(frozen=True)
class SpectralDensity:
    """One radiation component paired with its own photon-energy grid.

    photon_energy_eV
        One-dimensional photon-energy coordinates in eV.
    density
        Density on exactly that grid, in photons / incident electron / eV / sr.
    normalization
        ``source`` means response-free source density without acceptance or
        current scaling. ``observation-average`` means filter-attenuated,
        solid-angle-weighted physical-detector average, still per sr.
    """

    photon_energy_eV: np.ndarray
    density: np.ndarray
    normalization: str = "source"

    def __post_init__(self) -> None:
        energy = np.asarray(self.photon_energy_eV)
        density = np.asarray(self.density)
        if energy.ndim != 1 or density.shape != energy.shape:
            raise ValueError("density must match its one-dimensional photon_energy_eV grid")
        if self.normalization not in {"source", "observation-average"}:
            raise ValueError("normalization must be 'source' or 'observation-average'")
        object.__setattr__(self, "photon_energy_eV", energy)
        object.__setattr__(self, "density", density)


@dataclass(frozen=True)
class CxrSpectra:
    """PXR/CBS densities in photons / incident electron / eV / sr.

    incoherent
        Incoherent CXR density paired with its photon-energy grid in eV.
    coherent
        Optional coherent CXR density on the same photon-energy grid; ``None``
        when not calculated. Neither component includes atomic characteristic
        radiation. Both carry their source/observation normalization.
    """

    incoherent: SpectralDensity
    coherent: SpectralDensity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.incoherent, SpectralDensity):
            raise TypeError("incoherent must be a SpectralDensity")
        if self.coherent is not None:
            if not isinstance(self.coherent, SpectralDensity):
                raise TypeError("coherent must be a SpectralDensity or None")
            if (
                not np.array_equal(self.incoherent.photon_energy_eV, self.coherent.photon_energy_eV)
                or self.incoherent.normalization != self.coherent.normalization
            ):
                raise ValueError("CXR modes must share their photon-energy grid and normalization")


@dataclass(frozen=True, init=False)
class Result:
    """Radiation families with explicit grids, normalization, and provenance.

    cxr
        PXR/CBS modes, each pairing photon-energy coordinates in eV with density
        in photons / incident electron / eV / sr. Mode selection never changes
        the meaning of ``cxr.incoherent``.
    brem
        Bremsstrahlung density in the same units, on its own photon-energy grid.
    characteristic
        Optional atomic emission density in the same units on the CXR grid.
        It is separate from both CXR modes.
    case
        Fully resolved transport input that produced the arrays.
    provenance
        Immutable-view scene, numerics, identity, backend, and version metadata.
    spatial
        Optional factorized pixel data. Selected spectra include pixel solid
        angle and have units photons / incident electron / eV, not per sr.

    Old flat constructor arguments and attributes remain warning aliases through
    0.13.0. For old constructors, ``spectrum`` supplies the incoherent density;
    callers constructing coherent-only legacy results should migrate to groups.
    """

    cxr: CxrSpectra
    brem: SpectralDensity
    characteristic: SpectralDensity | None
    case: Case
    provenance: Mapping[str, Any]
    spatial: SpatialResult | None

    def __init__(
        self,
        *args,
        cxr: CxrSpectra | None = None,
        brem: SpectralDensity | None = None,
        characteristic: SpectralDensity | None = None,
        case: Case | None = None,
        provenance: Mapping[str, Any] | None = None,
        spatial: SpatialResult | None = None,
        **legacy,
    ) -> None:
        names = (
            "energy_eV",
            "spectrum",
            "background_energy_eV",
            "background",
            "case",
            "provenance",
            "coherent_spectrum",
            "characteristic_spectrum",
            "spatial",
        )
        if len(args) > len(names):
            raise TypeError("too many positional Result arguments")
        for name, value in zip(names, args, strict=False):
            if (
                name in legacy
                or (name == "case" and case is not None)
                or (name == "provenance" and provenance is not None)
                or (name == "spatial" and spatial is not None)
            ):
                raise TypeError(f"multiple values for {name}")
            legacy[name] = value
        case = legacy.pop("case", case)
        provenance = legacy.pop("provenance", provenance)
        spatial = legacy.pop("spatial", spatial)
        unknown = set(legacy) - set(names)
        if unknown:
            raise TypeError(f"unknown Result arguments: {sorted(unknown)}")
        if legacy:
            if cxr is not None or brem is not None or characteristic is not None:
                raise TypeError("cannot combine grouped and legacy spectral arguments")
            warn_renamed(
                "Result flat spectral arguments", "Result(cxr=..., brem=..., characteristic=...)"
            )
            missing = [name for name in names[:4] if name not in legacy]
            if missing:
                raise TypeError(f"missing Result arguments: {missing}")
            energy = legacy["energy_eV"]
            norm = (
                "observation-average"
                if isinstance(
                    getattr((provenance or {}).get("scene"), "detector", None), PlanarDetector
                )
                else "source"
            )
            coherent = legacy.get("coherent_spectrum")
            atomic = legacy.get("characteristic_spectrum")
            cxr = CxrSpectra(
                SpectralDensity(energy, legacy["spectrum"], norm),
                None if coherent is None else SpectralDensity(energy, coherent, norm),
            )
            brem = SpectralDensity(legacy["background_energy_eV"], legacy["background"], norm)
            characteristic = None if atomic is None else SpectralDensity(energy, atomic, norm)
        if not isinstance(cxr, CxrSpectra) or not isinstance(brem, SpectralDensity):
            raise TypeError("Result requires cxr=CxrSpectra and brem=SpectralDensity")
        if characteristic is not None:
            if not isinstance(characteristic, SpectralDensity):
                raise TypeError("characteristic must be a SpectralDensity or None")
            if not np.array_equal(characteristic.photon_energy_eV, cxr.incoherent.photon_energy_eV):
                raise ValueError("characteristic must use the CXR photon-energy grid")
            if characteristic.normalization != cxr.incoherent.normalization:
                raise ValueError("characteristic and CXR normalization must match")
        if spatial is not None and not isinstance(spatial, SpatialResult):
            raise TypeError("spatial must be a SpatialResult or None")
        if case is None or provenance is None:
            raise TypeError("Result requires case and provenance")
        for name, value in (
            ("cxr", cxr),
            ("brem", brem),
            ("characteristic", characteristic),
            ("case", case),
            ("provenance", MappingProxyType(dict(provenance))),
            ("spatial", spatial),
        ):
            object.__setattr__(self, name, value)

    def combined_cxr(
        self, *, mode: str = "incoherent", include_characteristic: bool = True
    ) -> SpectralDensity:
        """Return the selected CXR mode plus optional atomic emission on its grid.

        The returned density is photons / incident electron / eV / sr, with
        the selected mode's source/observation normalization. Brem is excluded
        because it may use another photon-energy grid.
        """
        if mode not in {"incoherent", "coherent"}:
            raise ValueError("mode must be 'incoherent' or 'coherent'")
        component = getattr(self.cxr, mode)
        if component is None:
            raise ValueError("coherent spectrum is not available")
        density = component.density
        if include_characteristic and self.characteristic is not None:
            density = density + self.characteristic.density
        return SpectralDensity(component.photon_energy_eV, density, component.normalization)

    @property
    def energy_eV(self) -> np.ndarray:
        """Deprecated CXR photon-energy coordinates in eV."""
        warn_renamed("Result.energy_eV", "Result.cxr.incoherent.photon_energy_eV")
        return self.cxr.incoherent.photon_energy_eV

    @property
    def spectrum(self) -> np.ndarray:
        """Deprecated selected-mode CXR density, photons / electron / eV / sr."""
        warn_renamed(
            "Result.spectrum", "Result.cxr.incoherent.density or Result.cxr.coherent.density"
        )
        scene = self.provenance.get("scene")
        component = (
            self.cxr.coherent
            if getattr(scene, "emission", None) == "coherent"
            else self.cxr.incoherent
        )
        if component is None:
            raise ValueError("coherent spectrum is not available")
        return component.density

    @property
    def background_energy_eV(self) -> np.ndarray:
        """Deprecated brem photon-energy coordinates in eV."""
        warn_renamed("Result.background_energy_eV", "Result.brem.photon_energy_eV")
        return self.brem.photon_energy_eV

    @property
    def background(self) -> np.ndarray:
        """Deprecated brem density, photons / electron / eV / sr, on its grid."""
        warn_renamed("Result.background", "Result.brem.density")
        return self.brem.density

    @property
    def coherent_spectrum(self) -> np.ndarray | None:
        """Deprecated coherent CXR density on the CXR grid, photons / electron / eV / sr."""
        warn_renamed("Result.coherent_spectrum", "Result.cxr.coherent.density")
        return None if self.cxr.coherent is None else self.cxr.coherent.density

    @property
    def characteristic_spectrum(self) -> np.ndarray | None:
        """Deprecated atomic density on the CXR grid, photons / electron / eV / sr."""
        warn_renamed("Result.characteristic_spectrum", "Result.characteristic.density")
        return None if self.characteristic is None else self.characteristic.density

    def line_total(self, *, coherent: bool = False, characteristic: bool = True) -> np.ndarray:
        """Deprecated CXR-plus-atomic density on the CXR grid, photons / electron / eV / sr."""
        warn_renamed("Result.line_total", "Result.combined_cxr")
        selected_coherent = (
            coherent or getattr(self.provenance.get("scene"), "emission", None) == "coherent"
        )
        return self.combined_cxr(
            mode="coherent" if selected_coherent else "incoherent",
            include_characteristic=characteristic,
        ).density

    def to_dataframe(self) -> pd.DataFrame:
        """Return the separate spectral arrays as a tidy long-form DataFrame.

        Columns are ``component`` (``cxr``, ``brem``, ``characteristic``),
        ``mode`` (CXR ``incoherent``/``coherent``, missing for other families),
        ``photon_energy_eV`` (eV), and ``density`` (photons / incident electron /
        eV / sr). Each array keeps its own grid; nothing is interpolated or
        padded, and derived CXR-plus-characteristic totals are omitted.

        ``attrs`` carries a plain provenance copy, the resolved case, units,
        and each family's source/observation normalization. Some pandas
        operations drop ``attrs``; read them before combining frames. Spatial
        pixel spectra/counts stay on ``spatial`` and the acquisition methods.
        """
        import pandas as pd

        parts: list[tuple[str, str | None, SpectralDensity]] = [
            ("cxr", "incoherent", self.cxr.incoherent)
        ]
        if self.cxr.coherent is not None:
            parts.append(("cxr", "coherent", self.cxr.coherent))
        if self.characteristic is not None:
            parts.append(("characteristic", None, self.characteristic))
        parts.append(("brem", None, self.brem))
        families = list(dict.fromkeys(name for name, _, _ in parts))
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "component": pd.Categorical(
                            [name] * len(component.density), categories=families
                        ),
                        "mode": pd.Categorical(
                            [mode] * len(component.density), categories=["incoherent", "coherent"]
                        ),
                        "photon_energy_eV": np.asarray(
                            component.photon_energy_eV, dtype=np.float64
                        ),
                        "density": np.asarray(component.density, dtype=np.float64),
                    }
                )
                for name, mode, component in parts
            ],
            ignore_index=True,
        )
        frame.attrs.update(
            provenance=dict(self.provenance),
            case=self.case,
            units={"photon_energy_eV": "eV", "density": "photons / (electron eV sr)"},
            normalization={name: component.normalization for name, _, component in parts},
        )
        return frame

    def _acquisition_context(self):
        scene = self.provenance.get("scene")
        acquisition = getattr(scene, "acquisition", None)
        beam = getattr(scene, "beam", None)
        if acquisition is None or beam is None:
            raise ValueError("result was not produced with an acquisition configuration")
        if self.spatial is None:
            raise ValueError("result does not contain spatial factors")
        digest = self.provenance.get("observation_identity_digest")
        if not isinstance(digest, str):
            raise ValueError("result does not contain an observation identity digest")
        return acquisition, beam, digest, scene

    def acquire(
        self,
        *,
        pixels=None,
        region=None,
        components: tuple[str, ...] | None = None,
    ) -> AcquisitionBatch:
        """Score selected pixels using this result's frozen acquisition identity."""
        acquisition, beam, digest, scene = self._acquisition_context()
        assert self.spatial is not None
        selected = (
            (("coherent", "brem") if scene.emission == "coherent" else ("cxr", "brem"))
            if components is None
            else components
        )
        return self.spatial.acquire(
            acquisition,
            rep_rate_hz=beam.rep_rate_hz,
            bunch_charge_pc=beam.bunch_charge_pc,
            observation_digest=digest,
            pixels=pixels,
            region=region,
            components=selected,
        )

    def acquisition_image(
        self,
        energy_range_eV: tuple[float, float] | None = None,
        *,
        components: tuple[str, ...] | None = None,
        pixel_chunk: int = 1024,
    ) -> np.ndarray:
        """Return the configured total or reporting-window detector count image."""
        acquisition, beam, digest, scene = self._acquisition_context()
        assert self.spatial is not None
        selected = (
            (("coherent", "brem") if scene.emission == "coherent" else ("cxr", "brem"))
            if components is None
            else components
        )
        return self.spatial.acquisition_image(
            acquisition=acquisition,
            rep_rate_hz=beam.rep_rate_hz,
            bunch_charge_pc=beam.bunch_charge_pc,
            observation_digest=digest,
            energy_range_eV=energy_range_eV,
            components=selected,
            pixel_chunk=pixel_chunk,
        )


__all__ = ["PixelMetadata", "PixelRayMap", "Result", "SpatialResult", "SpectralFactors"]
