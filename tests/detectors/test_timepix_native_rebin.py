"""Native Timepix scoring integrates source cells across input channels (#219)."""

import numpy as np
import pytest
from scipy.special import erf

from pyrite.detectors import Detector, Timepix3
from pyrite.detectors.timepix_response import TimepixResponse
from pyrite.energy_grid.semantics import node_bin_edges_and_widths


def test_native_line_mass_is_independent_of_source_cell_partition():
    uniform = np.arange(1_000.0, 4_001.0, 20.0)
    nonuniform = np.sort(
        np.concatenate(
            (
                uniform[(uniform < 1_980.0) | (uniform > 2_040.0)],
                [1_985.0, 1_995.0, 2_015.0, 2_045.0],
            )
        )
    )
    outputs = []
    for energy in (uniform, nonuniform):
        response = TimepixResponse(energy, n_mc=256, seed=7)
        # The same top-hat line occupies [1990, 2030] eV on both axes.
        # Its exact channel masses are 10 below 2000 eV and 30 above it.
        density = ((energy > 1_990.0) & (energy < 2_030.0)).astype(float)
        expected = (
            10.0 * response.R[:, np.searchsorted(response.in_edges, 2_000.0) - 1]
            + 30.0 * response.R[:, np.searchsorted(response.in_edges, 2_000.0)]
        )
        native = response.apply_native(density)
        np.testing.assert_allclose(native, expected, rtol=1e-13, atol=1e-14)
        outputs.append(native)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-13, atol=1e-14)


def test_native_gaussian_counts_agree_on_uniform_and_resonance_local_axes():
    from pyrite._line_windows import build_window_plan
    from pyrite.montecarlo.spectrum.line_seeds import ResonancePopulation, local_spacing_seeds

    centre, sigma = 2_000.3, 3.0
    population = ResonancePopulation("test-line", np.array([centre]), np.ones(1), np.array([0.5]))
    seeds, _ = local_spacing_seeds(
        [population],
        start_eV=1_000.0,
        stop_eV=4_000.0,
        floor_spacing_eV=0.1,
        max_spacing_eV=20.0,
        halo_limit=1e-4,
    )
    local = build_window_plan(1_000.0, 4_000.0, 20.0, seeds).coordinates()
    assert np.ptp(np.diff(local)) > 1.0
    detector = Detector(response=Timepix3(n_mc=256, seed=7))
    totals = []
    for energy in (np.linspace(1_000.0, 4_000.0, 30_001), local):
        edges, widths = node_bin_edges_and_widths(energy)
        # Exact Gaussian cell means isolate detector rebinning from source quadrature.
        cdf = 0.5 * (1.0 + erf((edges - centre) / (np.sqrt(2.0) * sigma)))
        density = np.diff(cdf) / widths
        totals.append(detector.native_score(energy, density).events.sum())
    assert totals[0] > 0.0
    assert totals[1] == pytest.approx(totals[0], rel=1e-5)


@pytest.mark.parametrize(
    "energy",
    [
        np.array([0.0, 40.0, 100.0, 2_000.0, 2_030.0, 4_000.0]),
        np.array([100.0, 140.0, 200.0, 2_000.0, 2_030.0, 4_000.0]),
    ],
)
def test_native_batches_and_apply_use_the_same_input_masses(energy):
    response = TimepixResponse(energy, n_mc=256, seed=7)
    density = np.linspace(0.5, 1.5, energy.size)
    batch = density * np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    native = response.apply_native(batch)
    assert native.shape == (2, 2, response.E_out.size)
    for index in np.ndindex(batch.shape[:-1]):
        single = response.apply_native(batch[index])
        np.testing.assert_allclose(native[index], single, rtol=1e-13, atol=1e-12)
        expected_density = np.interp(
            energy, response.E_out, single / response.dE_out, left=0.0, right=0.0
        )
        np.testing.assert_allclose(
            response.apply(batch[index]), expected_density, rtol=1e-13, atol=1e-12
        )
    assert response.apply_native(np.empty((0, energy.size))).shape == (0, response.E_out.size)


def test_native_and_apply_conserve_the_same_detected_total_on_measured_centres():
    energy = np.arange(12.5, 6_000.0, 25.0)
    response = TimepixResponse(energy, n_mc=256, seed=7)
    density = np.exp(-0.5 * ((energy - 2_000.3) / 100.0) ** 2)
    native_total = response.apply_native(density).sum()
    assert native_total > 0.0
    assert np.sum(response.apply(density) * 25.0) == pytest.approx(native_total, rel=1e-13)
