from typing import Any

import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite.detectors import IdealPhotonCounter
from pyrite.instrument import Acquisition, PixelGrid, PlanarDetector, PlanarPose
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


def _counting_spatial() -> SpatialResult:
    spatial = _spatial()
    detector = PlanarDetector(
        pose=spatial.detector.pose,
        pixels=spatial.detector.pixels,
        response=IdealPhotonCounter(),
    )
    return SpatialResult(
        spatial.ray_map,
        spatial.line,
        spatial.background,
        detector,
    )


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


def test_acquisition_image_is_chunk_invariant_and_sums_component_draws(monkeypatch) -> None:
    spatial = _counting_spatial()
    largest = 0
    original = SpatialResult._materialize

    def bounded(self, factor, coordinates):
        nonlocal largest
        largest = max(largest, len(coordinates))
        return original(self, factor, coordinates)

    monkeypatch.setattr(SpatialResult, "_materialize", bounded)
    acquisition = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(0.5, 1.5, 2.5, 3.5),
        mode="poisson",
        seed=23,
    )
    kwargs = dict(
        acquisition=acquisition,
        rep_rate_hz=1.0,
        bunch_charge_pc=elementary_charge * 1.0e12 * 100.0,
        observation_digest="b" * 64,
        components=("line", "background"),
    )

    one = spatial.acquisition_image(pixel_chunk=1, **kwargs)
    assert largest == 1
    two = spatial.acquisition_image(pixel_chunk=2, **kwargs)
    assert largest == 2
    window = spatial.acquisition_image(
        energy_range_eV=(0.5, 1.5),
        pixel_chunk=1,
        **kwargs,
    )
    selected = spatial.acquire(pixels=[(0, 0), (0, 1)], **kwargs)

    np.testing.assert_array_equal(one, two)
    np.testing.assert_array_equal(one.ravel(), selected.total_counts)
    np.testing.assert_array_equal(window.ravel(), selected.window_counts((0.5, 1.5)))


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


def test_total_components_add_characteristic_to_each_line() -> None:
    base = _spatial()
    characteristic = SpectralFactors(
        energy_eV=base.line.energy_eV,
        intrinsic_by_tile=np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]),
        mu_by_filter_inv_mm=base.line.mu_by_filter_inv_mm,
    )
    spatial = SpatialResult(
        base.ray_map,
        base.line,
        base.background,
        base.detector,
        coherent_line=base.line,
        characteristic_line=characteristic,
    )

    for total, line in (("line_total", "line"), ("coherent_total", "coherent")):
        np.testing.assert_allclose(
            spatial.average_density(total),
            spatial.average_density(line) + spatial.average_density("characteristic"),
            rtol=1.0e-15,
        )


def test_total_component_without_characteristic_is_the_line() -> None:
    spatial = _spatial()

    np.testing.assert_array_equal(
        spatial.average_density("line_total"), spatial.average_density("line")
    )
    with pytest.raises(ValueError, match="coherent line spectrum is not available"):
        spatial.average_density("coherent_total")


def test_pixel_metadata_reports_local_and_lab_geometry() -> None:
    spatial = _spatial()

    metadata = spatial.pixel_metadata(pixels=[(0, 1), (0, 0)])

    np.testing.assert_array_equal(metadata.coordinates, [[0, 1], [0, 0]])
    np.testing.assert_allclose(metadata.local_position_mm, [[0.5, 0.0], [-0.5, 0.0]])
    np.testing.assert_allclose(metadata.center_mm, [[0.5, 0.0, 10.0], [-0.5, 0.0, 10.0]])
    np.testing.assert_allclose(metadata.distance_mm, np.hypot(0.5, 10.0))
    np.testing.assert_allclose(metadata.polar_deg, np.degrees(np.arctan2(0.5, 10.0)))
    np.testing.assert_allclose(metadata.azimuth_deg, [0.0, 180.0])
    np.testing.assert_array_equal(metadata.solid_angle_sr, [0.2, 0.1])
    np.testing.assert_array_equal(metadata.tile_index, [1, 0])
    np.testing.assert_array_equal(metadata.path_length_mm, [[1.0], [0.0]])
    assert metadata.tile_direction_lab is None
    assert not metadata.solid_angle_sr.flags.writeable


def test_filter_coverage_and_transmission_image_use_stored_nodes() -> None:
    spatial = _spatial()

    np.testing.assert_array_equal(spatial.filter_coverage(), [[[False], [True]]])
    # mu = ln 2 / mm over a 1 mm path halves the covered pixel; 2.4 eV snaps to 2 eV.
    np.testing.assert_allclose(spatial.transmission_image(2.4), [[1.0, 0.5]], rtol=1.0e-15)
    with pytest.raises(ValueError, match="finite"):
        spatial.transmission_image(float("nan"))


def test_512_square_detector_images_and_selections_stay_bounded() -> None:
    """No request allocates the (512, 512, n_energy) pixel-energy cube."""
    import tracemalloc

    ny = nx = 512
    n_energy = 256
    n_tile = 25
    energy = np.linspace(1_000.0, 20_000.0, n_energy)
    rows = np.minimum(np.arange(ny) * 5 // ny, 4)
    tile_index = rows[:, None] * 5 + rows[None, :]
    paths = np.zeros((ny, nx, 1))
    paths[:, : nx // 2, 0] = 0.1
    ray_map = PixelRayMap(tile_index, np.full((ny, nx), 1.0e-6), paths)
    factors = SpectralFactors(
        energy,
        np.linspace(1.0, 2.0, n_tile)[:, None] * np.ones(n_energy),
        np.full((1, n_energy), 1.0),
    )
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 100.0)),
        pixels=PixelGrid((ny, nx), (0.055, 0.055)),
        response=IdealPhotonCounter(),
    )
    spatial = SpatialResult(ray_map, factors, factors, detector)
    acquisition = Acquisition(exposure_s=1.0, measured_edges_eV=(1_000.0, 5_000.0, 20_000.0))
    cube_bytes = ny * nx * n_energy * np.dtype(float).itemsize

    tracemalloc.start()
    try:
        total = spatial.acquisition_image(
            acquisition=acquisition,
            rep_rate_hz=1.0,
            bunch_charge_pc=1.0,
            observation_digest="0" * 64,
            pixel_chunk=4096,
        )
        window = spatial.acquisition_image(
            acquisition=acquisition,
            rep_rate_hz=1.0,
            bunch_charge_pc=1.0,
            observation_digest="0" * 64,
            energy_range_eV=(1_000.0, 5_000.0),
            pixel_chunk=4096,
        )
        true_window = spatial.image((1_000.0, 5_000.0), pixel_chunk=4096)
        _, selected = spatial.spectra(pixels=[(0, 0), (511, 511)])
        metadata = spatial.pixel_metadata(pixels=[(256, 256)])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert total.shape == window.shape == true_window.shape == (ny, nx)
    assert selected.shape == (2, n_energy)
    assert metadata.coordinates.shape == (1, 2)
    assert np.all(window <= total) and np.all(window > 0.0)
    assert peak < cube_bytes / 8
