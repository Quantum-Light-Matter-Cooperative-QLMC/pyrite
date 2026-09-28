import numpy as np
import pytest

from pyrite.observations import observation_from_result
from pyrite.plots._pixel_frames import (
    ObservationImage,
    detector_image_frame,
    histogram_frame,
    observation_image,
    pixel_metadata_rows,
    pixel_spectrum_frame,
)
from pyrite.plots.altair.pixels import (
    detector_image_chart,
    histogram_chart,
    pixel_spectrum_chart,
)
from tests.observations.test_store import _simulate, fake_directional  # noqa: F401


def _image(values, reduce="sum") -> ObservationImage:
    return ObservationImage(np.asarray(values, dtype=float), "total", "t", "counts", reduce)


def test_display_blocks_conserve_counts_and_report_pixel_ranges() -> None:
    values = np.arange(5 * 7, dtype=float).reshape(5, 7)

    frame = detector_image_frame(_image(values), max_cells=3)

    assert frame["block"].unique().tolist() == [3]
    assert frame["value"].sum() == values.sum()
    last = frame.iloc[-1]
    assert (last.row0, last.row1, last.column0, last.column1) == (3, 4, 6, 6)
    assert last.value == values[3:5, 6:7].sum()


def test_mean_blocks_average_ragged_edges_and_full_resolution_is_exact() -> None:
    values = np.ones((5, 5))
    values[4, 4] = 3.0

    mean = detector_image_frame(_image(values, "mean"), max_cells=2)
    exact = detector_image_frame(_image(values), max_cells=128)

    # The ragged 2 x 2 corner averages over its own four pixels, not a full block.
    assert mean.iloc[-1].value == pytest.approx(1.5)
    assert mean.iloc[0].value == pytest.approx(1.0)
    assert exact["block"].unique().tolist() == [1]
    np.testing.assert_array_equal(exact["value"].to_numpy().reshape(5, 5), values)


def test_512_display_grid_is_bounded() -> None:
    frame = detector_image_frame(_image(np.zeros((512, 512))))
    assert len(frame) == 128 * 128
    assert frame["block"].iloc[0] == 4


def test_observation_images_spectra_histogram_and_metadata() -> None:
    observation = observation_from_result(_simulate())
    edges = observation.acquisition.measured_edges_eV

    total = observation_image(observation, "total")
    window = observation_image(observation, "window", energy_range_eV=(edges[1], edges[2]))
    transmission = observation_image(observation, "transmission")
    coverage = observation_image(observation, "coverage")

    assert total.values.shape == (4, 6) and total.value_title == "expected counts"
    assert np.all(window.values <= total.values)
    assert transmission.reduce == "mean" and np.all(transmission.values <= 1.0)
    np.testing.assert_array_equal(
        coverage.values, observation.spatial.filter_coverage().sum(axis=-1)
    )
    with pytest.raises(ValueError, match="needs energy_range_eV"):
        observation_image(observation, "window")

    spectrum = pixel_spectrum_frame(observation, 1, 2, components=("line", "background"))
    assert set(spectrum["component"]) == {"line", "background"}
    histogram, accounting = histogram_frame(observation.acquire(pixels=[(1, 2)]))
    assert histogram["counts"].sum() == pytest.approx(total.values[1, 2])
    assert set(accounting) == {"underflow", "overflow", "below_cut"}
    rows = pixel_metadata_rows(observation.spatial.pixel_metadata(pixels=[(1, 2)]))
    assert rows[0] == {"quantity": "row, column", "value": "1, 2"}
    assert any(row["quantity"] == "filter 1 path [mm]" for row in rows)

    for chart in (
        detector_image_chart(
            detector_image_frame(total), title="t", value_title="c", selected=(1, 2)
        ),
        pixel_spectrum_chart(spectrum, title="s", y_type="log"),
        histogram_chart(histogram, title="h", count_title="c", y_type="log"),
    ):
        assert chart.to_dict()


def test_uniform_image_chart_keeps_a_valid_color_domain() -> None:
    chart = detector_image_chart(
        detector_image_frame(_image(np.zeros((3, 3)))), title="t", value_title="c", scale_type="log"
    )
    spec = chart.to_dict()
    assert spec["layer"][0]["encoding"]["color"]["scale"]["domain"] == [0.0, 1.0]
