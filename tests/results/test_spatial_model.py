from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyrite.instrument import PixelGrid, PlanarDetector, PlanarPose
from pyrite.results.model import PixelRayMap, SpatialResult, SpectralFactors


class HalfResponse:
    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        return np.asarray(intrinsic_density) * (0.5 * scale)


def _spatial() -> SpatialResult:
    energy = np.array([1.0, 2.0, 3.0])
    ray_map = PixelRayMap(
        tile_index=np.array([[0, 1]]),
        solid_angle_sr=np.array([[0.1, 0.2]]),
        path_length_mm=np.array([[[0.0], [1.0]]]),
    )
    factors = SpectralFactors(
        energy_eV=energy,
        intrinsic_by_tile=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        mu_by_filter_inv_mm=np.array([[np.log(2.0)] * 3]),
    )
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)),
        pixels=PixelGrid((1, 2), (1.0, 1.0)),
        response=HalfResponse(),
    )
    return SpatialResult(ray_map, factors, factors, detector)


def _materialize_with_chunk(spatial: SpatialResult, method: str, pixel_chunk: Any) -> object:
    if method == "image":
        return spatial.image((1.0, 3.0), pixel_chunk=pixel_chunk)
    return spatial.average_density("line", pixel_chunk=pixel_chunk)


def test_selected_spectra_include_solid_angle_and_filter_transmission() -> None:
    spatial = _spatial()

    energy, spectra = spatial.spectra(pixels=[(0, 0), (0, 1)])

    np.testing.assert_array_equal(energy, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(spectra[0], [0.1, 0.1, 0.1])
    np.testing.assert_allclose(spectra[1], [0.2, 0.2, 0.2], rtol=1.0e-15)


def test_image_materializes_bounded_chunks_and_scores_response(monkeypatch) -> None:
    spatial = _spatial()
    largest = 0
    original = spatial._materialize

    def bounded(factor, coordinates):
        nonlocal largest
        largest = max(largest, len(coordinates))
        return original(factor, coordinates)

    monkeypatch.setattr(SpatialResult, "_materialize", lambda self, f, c: bounded(f, c))

    true_image = spatial.image((1.0, 3.0), pixel_chunk=1)
    measured_image = spatial.image((1.0, 3.0), measured=True, pixel_chunk=1)

    np.testing.assert_allclose(true_image, [[0.2, 0.4]], rtol=1.0e-15)
    np.testing.assert_allclose(measured_image, 0.5 * true_image, rtol=1.0e-15)
    assert largest == 1


def test_average_density_recovers_sum_flux_over_total_solid_angle() -> None:
    spatial = _spatial()
    _, spectra = spatial.spectra(region=(slice(None), slice(None)))

    expected = np.sum(spectra, axis=0) / np.sum(spatial.ray_map.solid_angle_sr)

    np.testing.assert_allclose(spatial.average_density("line"), expected, rtol=1.0e-15)


@pytest.mark.parametrize("pixel_chunk", [0, -1])
@pytest.mark.parametrize("method", ["image", "average_density"])
def test_spatial_materialization_rejects_nonpositive_pixel_chunks(method, pixel_chunk) -> None:
    spatial = _spatial()

    with pytest.raises(ValueError, match="pixel_chunk must be positive"):
        _materialize_with_chunk(spatial, method, pixel_chunk)


@pytest.mark.parametrize("pixel_chunk", [True, 1.5])
@pytest.mark.parametrize("method", ["image", "average_density"])
def test_spatial_materialization_rejects_noninteger_pixel_chunks(method, pixel_chunk) -> None:
    spatial = _spatial()

    with pytest.raises(TypeError, match="pixel_chunk must be an integer"):
        _materialize_with_chunk(spatial, method, pixel_chunk)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pixels": None, "region": None},
        {"pixels": [(0, 0)], "region": (slice(None), slice(None))},
        {"pixels": [(1, 0)], "region": None},
        {"pixels": [], "region": None},
    ],
)
def test_spectra_rejects_ambiguous_or_invalid_selection(kwargs) -> None:
    with pytest.raises((ValueError, IndexError)):
        _spatial().spectra(**kwargs)
