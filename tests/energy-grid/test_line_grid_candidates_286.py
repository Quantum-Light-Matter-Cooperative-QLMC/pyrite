"""Scientific accounting in the #286 measurement harness, without transport."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks.line_grid_candidates_286 import case_band, observables, uniform_cells
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths


@pytest.mark.parametrize("count", [2, 7, 28, 112])
def test_cell_axes_preserve_the_same_band_at_every_refinement(count):
    grid = uniform_cells(100, 100_000, count)
    edges, widths = _energy_bin_edges_and_widths(grid)
    assert edges[[0, -1]] == pytest.approx([100, 100_000], abs=2e-10)
    assert np.sum(widths) == pytest.approx(99_900)
    assert np.all(np.diff(grid) > 0)


class HalfEfficiency:
    def score(self, energy, density, **kwargs):
        return 0.5 * density


class FullNativeEfficiency:
    def native_score(self, energy, density):
        widths = _energy_bin_edges_and_widths(energy)[1]
        return SimpleNamespace(events=density * widths)


def test_bin_masses_centroid_and_detector_cut_have_analytic_values():
    # Cells [100,200,...,500] with means [1,2,3,4]: total=1000, first moment=350000.
    grid = uniform_cells(100, 500, 4)
    # The cut at 250 eV splits a source cell: mass = 100*1 + 50*2 = 200.
    detector_grid = uniform_cells(100, 250, 2)
    result = observables(
        grid,
        np.array([1.0, 2.0, 3.0, 4.0]),
        detector_grid,
        {
            "eaglexo_counts": HalfEfficiency(),
            "timepix3_counts": FullNativeEfficiency(),
        },
    )
    assert result["yield"] == pytest.approx(1000)
    assert result["centroid_eV"] == pytest.approx(350)
    assert result["eaglexo_counts"] == pytest.approx(100)
    assert result["timepix3_counts"] == pytest.approx(200)


def test_constant_density_is_invariant_under_coarsening():
    detector_grid = uniform_cells(100, 250, 5)
    for count in [2, 16, 64]:
        result = observables(
            uniform_cells(100, 500, count),
            np.ones(count),
            detector_grid,
            {"eaglexo_counts": HalfEfficiency()},
        )
        assert result == pytest.approx({"yield": 400, "centroid_eV": 300, "eaglexo_counts": 75})


@pytest.mark.parametrize("args", [(0, 10, 2), (10, 5, 2), (1, 10, 1)])
def test_invalid_band_or_insufficient_cells_refused(args):
    with pytest.raises(ValueError):
        uniform_cells(*args)


def test_measurement_refuses_cpu_before_starting_transport(tmp_path, monkeypatch):
    import sys

    from checks.line_grid_candidates_286 import main

    monkeypatch.setattr("pyrite._backend.xp", np)
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["grid-study", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


def test_compact_grid_encoding_is_decoded_before_band_selection():
    # The last tuple item is spacing (3 eV), not the upper coordinate.
    compact = {"E_grid": (50.0, 25703.0, 3.0)}
    explicit = {"E_grid": np.arange(50.0, 25703.0, 3.0)}
    assert case_band(compact) == case_band(explicit) == (50.0, 25700.0)
    # An explicit three-node array must remain coordinates.
    assert case_band({"E_grid": np.array([50.0, 100.0, 150.0])}) == (50.0, 150.0)
