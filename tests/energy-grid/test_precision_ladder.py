"""Float32-versus-FP64 lineshape instrument: grids, deviations, pipeline identity.

The float32 measurement itself needs a GPU backend and runs remotely; these
tests pin the instrument. The pipeline test runs both "precisions" on the CPU
backend (float64 twice), so any nonzero deviation would be an instrument bug,
not floating-point distortion.
"""

import numpy as np
import pytest

from pyrite.energy_grid import precision_ladder as pl
from pyrite.energy_grid.convergence import lineshape_deviation


@pytest.mark.parametrize(("top", "ulp"), [(10000.0, 2.0**-10), (16500.0, 2.0**-9)])
def test_float32_ulp_is_constant_on_each_binade(top, ulp):
    assert pl.float32_ulp(top) == ulp
    assert pl.float32_ulp(top * 1.2) == ulp


@pytest.mark.parametrize("ratio", pl.DEFAULT_SPACING_OVER_ULP)
def test_ulp_window_grid_has_the_requested_spacing_on_and_off_the_ulp_lattice(ratio):
    ulp = pl.float32_ulp(9726.0)
    on = pl.ulp_window_grid(9726.0, ratio, 200.0, off_lattice=False)
    off = pl.ulp_window_grid(9726.0, ratio, 200.0)

    assert on[-1] == 9726.0
    assert np.diff(on).max() / ulp == pytest.approx(ratio, rel=1e-6)
    assert off[-1] == pytest.approx(9726.0 + pl.LATTICE_TOP_OFFSET_ULPS * ulp, abs=1e-9)
    assert np.diff(off).max() / ulp == pytest.approx(ratio, rel=1e-3)
    for grid in (on, off):
        assert 8192.0 <= grid[0]
        assert abs((grid[-1] - grid[0]) - 200.0) <= 0.5 * np.diff(grid).max()


def test_cast_statistics_see_lattice_jitter_and_collapse_below_one_ulp():
    lattice = pl.cast_statistics(pl.ulp_window_grid(9726.0, 8.0, 5.0, off_lattice=False))
    at_floor = pl.cast_statistics(pl.ulp_window_grid(9726.0, 8.0, 5.0))
    below = pl.cast_statistics(pl.ulp_window_grid(9726.0, 0.5, 5.0))

    assert lattice["max_node_shift_over_spacing"] == 0.0
    assert at_floor["collapsed_intervals"] == 0
    assert at_floor["spacing_over_ulp"] == pytest.approx(8.0, rel=1e-3)
    # Off the lattice every node rounds by up to half an ulp: 1/16 of an 8-ulp step.
    assert 0.0 < at_floor["max_node_shift_over_spacing"] <= 0.5 / 8.0 + 1e-12
    assert at_floor["min_cast_step_over_spacing"] < 1.0 < at_floor["max_cast_step_over_spacing"]
    assert below["collapsed_intervals"] > 0


def test_lineshape_deviation_reads_scale_shift_and_pointwise_change():
    E = np.linspace(9000.0, 9200.0, 4001)
    reference = np.exp(-(((E - 9100.0) / 10.0) ** 2))

    assert lineshape_deviation(E, reference, reference) == {
        "yield_rel": 0.0,
        "centroid_shift_eV": 0.0,
        "fwhm_rel": 0.0,
        "max_pointwise_rel": 0.0,
    }
    scaled = lineshape_deviation(E, reference, 1.001 * reference)
    assert scaled["yield_rel"] == pytest.approx(1e-3, rel=1e-6)
    assert scaled["fwhm_rel"] == pytest.approx(0.0, abs=1e-9)
    assert scaled["max_pointwise_rel"] == pytest.approx(1e-3, rel=1e-6)
    shifted = lineshape_deviation(E, reference, np.exp(-(((E - 9100.5) / 10.0) ** 2)))
    assert shifted["centroid_shift_eV"] == pytest.approx(0.5, rel=1e-3)


def test_locate_windows_refuses_a_band_without_lines():
    with pytest.raises(SystemExit, match="no CXR line density"):
        pl.locate_windows(lambda E: np.zeros_like(E), [(8192.0, 16384.0)], 200.0)


def test_locate_windows_picks_the_window_holding_the_most_line_yield():
    def lines(E):
        return np.exp(-(((E - 12000.0) / 5.0) ** 2)) + 0.5 * np.exp(-(((E - 9000.0) / 5.0) ** 2))

    (window,) = pl.locate_windows(lines, [(8192.0, 16384.0)], 200.0)

    assert window["top_eV"] - 200.0 <= 12000.0 <= window["top_eV"]
    assert window["peak_eV"] == pytest.approx(12000.0, abs=pl.LOCATOR_STEP_EV)
    assert window["window_integral"] == pytest.approx(5.0 * np.sqrt(np.pi), rel=1e-3)


def test_same_precision_pipeline_reports_exactly_zero_deviation(tmp_path):
    payload = tmp_path / "segments.pkl"
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    report = tmp_path / "report.json"

    assert (
        pl.main(
            [
                "transport",
                "--material",
                "hopg",
                "--energy",
                "30",
                "--tilt",
                "5",
                "--azimuth",
                "95",
                "--thickness",
                "1e4",
                "--ne",
                "3",
                "--bands",
                "1024:2048",
                "--window",
                "40",
                "--out",
                str(payload),
            ]
        )
        == 0
    )
    for out in (first, second):
        pl.main(["evaluate", "--payload", str(payload), "--ratios", "1000,3000", "--out", str(out)])

    table = pl.compare(first, second, require_precision_pair=False)

    assert len(table["rows"]) == 2
    for row in table["rows"]:
        assert row["collapsed_intervals"] == 0
        assert row["yield_rel"] == 0.0 and row["max_pointwise_rel"] == 0.0
        assert row["centroid_shift_eV"] == 0.0
    with pytest.raises(SystemExit, match="expected float64 reference and float32 candidate"):
        pl.compare(first, second)
    assert not report.exists()
