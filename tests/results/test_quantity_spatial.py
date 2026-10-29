"""Quantity-aware pixel spectra retain solid-angle and detector-response semantics."""

import subprocess
import sys

import numpy as np
import pytest

from pyrite.instrument import PixelGrid, PlanarDetector, PlanarPose
from pyrite.results import QuantifiedSpatialImage, QuantifiedSpatialSpectrum
from pyrite.results.model import PixelRayMap, SpatialResult, SpectralFactors
from pyrite.units import ureg


class HalfResponse:
    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        return np.asarray(intrinsic_density) * (0.5 * scale)


def _spatial() -> SpatialResult:
    ray_map = PixelRayMap(
        tile_index=np.array([[0, 1]]),
        solid_angle_sr=np.array([[0.1, 0.2]]),
        path_length_mm=np.array([[[0.0], [1.0]]]),
    )
    cxr = SpectralFactors(
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.full((1, 3), np.log(2.0)),
    )
    brem = SpectralFactors(
        np.array([1000.0, 3000.0]),
        np.array([[3.0, 3.0], [4.0, 4.0]]),
        np.full((1, 2), np.log(2.0)),
    )
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)),
        pixels=PixelGrid((1, 2), (1.0, 1.0)),
        response=HalfResponse(),
    )
    return SpatialResult(ray_map, cxr, brem, detector)


def test_pixel_axis_measure_density_conversion_and_integral() -> None:
    spatial = _spatial()
    before = spatial.spectra(pixels=[(0, 1), (0, 0)])
    view = spatial.quantify_spectra(pixels=[(0, 1), (0, 0)])
    converted = view.to_energy("keV")

    assert isinstance(view, QuantifiedSpatialSpectrum)
    assert view.axis_names == ("pixel", "photon_energy")
    assert view.sampling == "nodes"
    assert view.measure == "per_energy"
    assert view.normalization == "pixel-accepted"
    assert view.count_basis == "photons_per_incident_electron"
    assert view.component == "cxr" and not view.measured
    assert view.density.units == ureg("1/eV").units
    assert converted.density.units == ureg("1/keV").units
    np.testing.assert_array_equal(view.pixel_coordinates, [[0, 1], [0, 0]])
    np.testing.assert_array_equal(view.photon_energy.magnitude, before[0])
    np.testing.assert_allclose(view.density.magnitude, before[1])
    np.testing.assert_allclose(converted.density.magnitude, before[1] * 1000)
    np.testing.assert_array_equal(converted.photon_energy.magnitude, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        np.trapezoid(converted.density.magnitude, converted.photon_energy.magnitude, axis=1),
        np.trapezoid(before[1], before[0], axis=1),
    )
    np.testing.assert_allclose(spatial.image((1000, 3000)), [[200, 400]])
    # Plain numeric behavior and grid remain unchanged after conversion.
    np.testing.assert_array_equal(spatial.spectra(pixels=[(0, 1), (0, 0)])[0], before[0])


def test_measured_and_component_specific_grids_and_region_selection() -> None:
    spatial = _spatial()
    region = (slice(None), slice(None))
    brem = spatial.quantify_spectra(region=region, component="brem")
    measured = spatial.quantify_spectra(
        pixels=[(0, 0), (0, 1)], component="cxr", measured=True, fwhm_eV=12.0
    )
    assert brem.component == "brem" and len(brem.photon_energy.magnitude) == 2
    assert brem.density.magnitude.shape == (2, 2)
    np.testing.assert_array_equal(brem.pixel_coordinates, [[0, 0], [0, 1]])
    assert measured.measured is True and measured.fwhm_eV == 12.0
    np.testing.assert_allclose(
        measured.density.magnitude,
        0.5 * spatial.spectra(pixels=[(0, 0), (0, 1)])[1],
    )
    np.testing.assert_allclose(
        measured.to_energy("keV").density.magnitude,
        1000 * measured.density.magnitude,
    )


def test_snapshot_readback_and_conversion_are_independent() -> None:
    spatial = _spatial()
    view = spatial.quantify_spectra(pixels=[(0, 1)])
    converted = view.to_energy("keV")
    view.pixel_coordinates[:] = 99
    view.photon_energy.magnitude[:] = 99
    view.density.magnitude[:] = 99
    np.testing.assert_array_equal(view.pixel_coordinates, [[0, 1]])
    np.testing.assert_array_equal(converted.pixel_coordinates, [[0, 1]])
    np.testing.assert_array_equal(converted.photon_energy.magnitude, [1, 2, 3])
    np.testing.assert_allclose(converted.density.magnitude, [[200, 200, 200]])


@pytest.mark.parametrize("unit", ["second", "radian", "steradian", "dimensionless"])
def test_bad_energy_units_rejected(unit: str) -> None:
    with pytest.raises(ValueError, match="energy_unit.*energy"):
        _spatial().quantify_spectra(pixels=[(0, 0)]).to_energy(unit)


def test_original_pixel_selection_and_missing_component_errors_preserved() -> None:
    spatial = _spatial()
    with pytest.raises(ValueError, match="exactly one"):
        spatial.quantify_spectra()
    with pytest.raises(IndexError, match="outside"):
        spatial.quantify_spectra(pixels=[(100, 0)])
    with pytest.raises(ValueError, match="coherent"):
        spatial.quantify_spectra(pixels=[(0, 0)], component="coherent")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"_density_per_eV": np.zeros((3, 3))}, "shape"),
        ({"_pixel_coordinates": np.array([[0.0, 0.0]])}, "integers"),
        ({"_energy_eV": np.array([1j, 2j, 3j])}, "floating-point"),
        ({"_energy_eV": [1.0, 2.0, 3.0]}, "CPU NumPy"),
    ],
)
def test_direct_view_rejects_invalid_arrays(changes: dict, match: str) -> None:
    defaults = dict(
        _energy_eV=np.array([1.0, 2.0, 3.0]),
        _density_per_eV=np.ones((1, 3)),
        _pixel_coordinates=np.array([[0, 0]]),
        component="cxr",
        measured=False,
    )
    with pytest.raises((TypeError, ValueError), match=match):
        QuantifiedSpatialSpectrum(**(defaults | changes))


def test_integrated_image_quantity_and_window_units() -> None:
    spatial = _spatial()
    expected = spatial.image((1000.0, 3000.0), component="cxr")
    view = spatial.quantify_image((1000.0, 3000.0), component="cxr", pixel_chunk=1)
    kev = view.to_energy("keV")

    assert isinstance(kev, QuantifiedSpatialImage)
    assert view.axis_names == ("pixel_row", "pixel_column")
    assert view.sampling == "integrated_window"
    assert view.measure == "per_pixel"
    assert view.normalization == "pixel-accepted"
    assert view.count_basis == "photons_per_incident_electron"
    assert view.component == "cxr" and not view.measured
    assert view.photon_yield.units == ureg.dimensionless
    assert kev.photon_energy_window.units == ureg.keV
    np.testing.assert_array_equal(kev.photon_energy_window.magnitude, [1.0, 3.0])
    np.testing.assert_allclose(kev.photon_yield.magnitude, expected)
    np.testing.assert_allclose(expected, [[200.0, 400.0]])
    np.testing.assert_allclose(kev.to_energy("eV").photon_yield.magnitude, expected)
    np.testing.assert_array_equal(spatial.image((1000.0, 3000.0)), expected)
    # The legacy image selects native nodes inside bounds, without interpolation.
    off_grid = spatial.quantify_image((900.0, 3100.0))
    np.testing.assert_allclose(off_grid.photon_yield.magnitude, expected)
    np.testing.assert_array_equal(off_grid.photon_energy_window.magnitude, [900.0, 3100.0])
    np.testing.assert_allclose(
        spatial.quantify_image((1000.0, 3000.0), component="brem").photon_yield.magnitude,
        [[600.0, 800.0]],
    )


def test_measured_image_reuses_response_without_rescoring_on_conversion() -> None:
    spatial = _spatial()
    view = spatial.quantify_image((1000.0, 3000.0), measured=True, fwhm_eV=12.0, component="cxr")
    assert view.measured and view.fwhm_eV == 12.0
    np.testing.assert_allclose(
        view.photon_yield.magnitude,
        spatial.image((1000.0, 3000.0), measured=True, fwhm_eV=12.0),
    )
    np.testing.assert_allclose(view.photon_yield.magnitude, [[100.0, 200.0]])
    np.testing.assert_array_equal(
        view.to_energy("keV").photon_yield.magnitude, view.photon_yield.magnitude
    )


def test_integrated_image_snapshot_isolation_and_validation() -> None:
    spatial = _spatial()
    view = spatial.quantify_image((1000.0, 3000.0))
    converted = view.to_energy("keV")
    view.photon_yield.magnitude[:] = -1
    view.photon_energy_window.magnitude[:] = -1
    np.testing.assert_allclose(view.photon_yield.magnitude, [[200.0, 400.0]])
    np.testing.assert_allclose(converted.photon_yield.magnitude, [[200.0, 400.0]])
    np.testing.assert_array_equal(view.photon_energy_window.magnitude, [1000.0, 3000.0])
    with pytest.raises(ValueError, match="at least two energy samples"):
        spatial.quantify_image((1000.0, 1100.0))
    for unit in ("second", "steradian", "dimensionless"):
        with pytest.raises(ValueError, match="energy_unit.*energy"):
            view.to_energy(unit)
    with pytest.raises((ValueError, TypeError), match="finite"):
        QuantifiedSpatialImage(np.array([[np.nan]]), np.array([1000.0, 3000.0]), "cxr", False)


def test_numeric_import_stays_pint_lazy() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] == 'pint':
        raise AssertionError('Pint imported on numeric path')
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import pyrite
from pyrite.results import QuantifiedSpatialSpectrum
from pyrite.results.model import SpatialResult
import numpy as np
from pyrite.results import PixelMetadata
metadata = PixelMetadata(
    np.array([[0, 0]]), np.zeros((1, 2)), np.array([[0., 0., 1.]]),
    np.array([[0., 0., 1.]]), np.zeros(1), np.zeros(1), np.ones(1),
    np.ones(1), np.zeros(1, dtype=int), None, np.empty((1, 0)),
)
assert metadata.center_mm[0, 2] == 1
assert 'pint' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_pixel_metadata_quantities_preserve_selected_geometry() -> None:
    metadata = _spatial().pixel_metadata(pixels=[(0, 1), (0, 0)])
    # Two pixel centres at +/-0.5 mm, 10 mm downstream of the source.
    np.testing.assert_allclose(metadata.local_position.to("um").magnitude, [[500, 0], [-500, 0]])
    np.testing.assert_allclose(metadata.center.to("cm").magnitude, [[0.05, 0, 1], [-0.05, 0, 1]])
    np.testing.assert_allclose(metadata.distance.to("cm").magnitude, np.hypot(0.05, 1))
    np.testing.assert_allclose(metadata.polar.to("radian").magnitude, np.arctan2(0.5, 10))
    np.testing.assert_allclose(metadata.azimuth.to("radian").magnitude, [0, np.pi])
    assert metadata.solid_angle.units == ureg.steradian
    np.testing.assert_array_equal(metadata.solid_angle.magnitude, [0.2, 0.1])
    np.testing.assert_allclose(metadata.path_length.to("um").magnitude, [[1000], [0]])
    np.testing.assert_array_equal(metadata.coordinates, [[0, 1], [0, 0]])
    np.testing.assert_array_equal(metadata.tile_index, [1, 0])
    assert metadata.tile_direction_lab is None
    # Conversion must refuse a physical equivalency, such as length to energy.
    from pint import DimensionalityError

    with pytest.raises(DimensionalityError):
        metadata.distance.to("eV")


@pytest.mark.parametrize(
    "semantic,numeric,unit",
    [
        ("local_position", "local_position_mm", "mm"),
        ("center", "center_mm", "mm"),
        ("polar", "polar_deg", "degree"),
        ("azimuth", "azimuth_deg", "degree"),
        ("distance", "distance_mm", "mm"),
        ("solid_angle", "solid_angle_sr", "sr"),
        ("path_length", "path_length_mm", "mm"),
    ],
)
def test_pixel_geometry_readback_isolated(semantic: str, numeric: str, unit: str) -> None:
    metadata = _spatial().pixel_metadata(region=(slice(None), slice(None)))
    expected = getattr(metadata, numeric).copy()
    value = getattr(metadata, semantic)
    assert value.units == ureg.Unit(unit)
    assert value.shape == expected.shape
    value.magnitude[:] = -999
    np.testing.assert_array_equal(getattr(metadata, semantic).magnitude, expected)
    np.testing.assert_array_equal(getattr(metadata, numeric), expected)


def test_pixel_geometry_empty_filters_keep_axis() -> None:
    from dataclasses import replace

    spatial = _spatial()
    metadata = spatial.pixel_metadata(pixels=[(0, 1)])
    metadata = replace(metadata, path_length_mm=np.empty((1, 0)))
    assert metadata.path_length.to("cm").shape == (1, 0)
