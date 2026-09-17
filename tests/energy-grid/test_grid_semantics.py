"""Uniform-only energy-grid consumers must refuse a nonuniform grid loudly.

Issue #98 pinned the grid semantics (see
``docs/physics/radiation-physics/energy-grid-semantics.md``) and audited every
consumer that reads a single spacing and applies it across the whole grid. Those
consumers now raise :class:`NonuniformEnergyGridError` instead of returning a
silently wrong number.

Two invariants are pinned here:

1. a nonuniform (logarithmic) grid raises the explicit unsupported error, and
2. the uniform path is untouched -- the guard returns exactly the
   ``float(E[1] - E[0])`` its callers used before, so existing results stay
   bit-for-bit identical.

The detector cache-identity test covers a separate correctness bug: a key of
(size, first, last) is not a grid identity, so a linear and a logarithmic grid
over the same interval with the same node count collided in the response cache.
"""

import numpy as np
import pytest

from pyrite.detectors import _si_sensor, response
from pyrite.energy_grid.semantics import (
    NonuniformEnergyGridError,
    grid_identity,
    is_uniform_grid,
    node_bin_edges_and_widths,
    rebin_piecewise_constant_density,
    require_uniform_grid,
    spacing_spread,
)
from pyrite.results import line_metrics

LINEAR_GRID = np.linspace(100.0, 10_000.0, 512)
LOG_GRID = np.logspace(np.log10(100.0), np.log10(10_000.0), 512)


def test_linspace_and_arange_grids_read_as_uniform():
    """Float rounding jitter must not trip the guard on a real catalog grid."""
    assert is_uniform_grid(LINEAR_GRID)
    assert is_uniform_grid(np.arange(10.0, 20_000.0, 10.0))
    assert spacing_spread(LINEAR_GRID) < 1e-12


def test_fine_step_at_high_energy_stays_uniform():
    """A 2.5e-5 eV step at 1600 eV is only ~1e8 float64 ulp wide, so its
    unavoidable rounding jitter is ~1e-8 relative. The representation floor is
    what keeps that real line-window grid (tests/montecarlo/test_xray_dispersion)
    on the uniform side of the guard."""
    fine = np.linspace(1599.0, 1601.5, 100_001)
    assert is_uniform_grid(fine)
    assert require_uniform_grid(fine, consumer="probe") == float(fine[1] - fine[0])


def test_log_grid_is_rejected_and_reports_the_spacing_range():
    assert not is_uniform_grid(LOG_GRID)
    with pytest.raises(NonuniformEnergyGridError) as excinfo:
        require_uniform_grid(LOG_GRID, consumer="probe")
    message = str(excinfo.value)
    assert "probe requires a UNIFORM energy grid" in message
    assert "not yet supported" in message


def test_guard_returns_the_exact_first_spacing_for_a_uniform_grid():
    """Bit-for-bit: the guard hands back the value its callers already used."""
    assert require_uniform_grid(LINEAR_GRID, consumer="probe") == float(
        LINEAR_GRID[1] - LINEAR_GRID[0]
    )


def test_node_bin_widths_preserve_uniform_behavior_and_follow_local_spacing():
    _, linear_widths = node_bin_edges_and_widths(LINEAR_GRID)
    assert np.allclose(linear_widths, LINEAR_GRID[1] - LINEAR_GRID[0], rtol=1e-13)
    edges, log_widths = node_bin_edges_and_widths(LOG_GRID)
    assert np.array_equal(log_widths, np.diff(edges))
    assert log_widths[-1] > log_widths[0]


def test_piecewise_constant_rebin_conserves_mass_and_reports_outside_window():
    masses, outside = rebin_piecewise_constant_density([0.0, 1.0, 3.0], [2.0, 4.0], [0.5, 2.0, 4.0])
    assert np.array_equal(masses, [5.0, 4.0])
    assert outside == (1.0, 0.0)
    assert np.sum(masses) + sum(outside) == pytest.approx(10.0)


def test_poisson_core_uses_local_widths_on_a_log_grid():
    rng = np.random.default_rng(0)
    density = np.ones_like(LOG_GRID)
    _, expected = _si_sensor.poisson_core(LOG_GRID, density, 3.0, rng)
    _, widths = node_bin_edges_and_widths(LOG_GRID)
    assert np.array_equal(expected, density * widths * 3.0)


def test_poisson_core_uniform_expected_counts_unchanged():
    rng = np.random.default_rng(0)
    density = np.full(LINEAR_GRID.shape, 2.0)
    _, expected = _si_sensor.poisson_core(LINEAR_GRID, density, 3.0, rng)
    dE = float(LINEAR_GRID[1] - LINEAR_GRID[0])
    assert np.array_equal(expected, np.clip(density * dE * 3.0, 0.0, None))


def test_convolve_detector_still_runs_on_a_uniform_grid():
    out = response.convolve_detector(LINEAR_GRID, np.ones_like(LINEAR_GRID), 50.0)
    assert out.shape == LINEAR_GRID.shape


def test_convolve_detector_conserves_mass_on_a_log_grid_away_from_edges():
    """#100: convolve_detector no longer requires a uniform grid. Away from the
    zero-padded edges, the graded-grid quadrature path (node_bin_edges_and_widths
    local weights) should conserve mass like the uniform sample-space path does."""
    density = np.exp(-0.5 * ((LOG_GRID - 3_000.0) / 20.0) ** 2)
    out = response.convolve_detector(LOG_GRID, density, 50.0)

    assert out.shape == LOG_GRID.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    _, widths = node_bin_edges_and_widths(LOG_GRID)
    input_mass = np.sum(density * widths)
    output_mass = np.sum(out * widths)
    assert output_mass == pytest.approx(input_mass, rel=0.05)


def test_eagle_response_resolve_energy_accepts_a_log_grid():
    eaglexo_response = pytest.importorskip("pyrite.detectors.eaglexo_response")

    resp = eaglexo_response.EagleResponse(LOG_GRID, resolve_energy=True)
    detected = resp.apply(np.ones_like(LOG_GRID))

    assert detected.shape == LOG_GRID.shape
    assert np.all(np.isfinite(detected))
    assert np.all(detected >= 0.0)


def test_timepix_response_accepts_a_log_grid_and_uses_local_input_widths():
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    response = timepix_response.TimepixResponse(LOG_GRID, n_mc=8)
    _, widths = node_bin_edges_and_widths(LOG_GRID)
    assert np.array_equal(response.dE_fine, widths)
    rebinned = np.bincount(
        response.idx_in,
        weights=np.ones_like(LOG_GRID) * response.dE_fine,
        minlength=response.n_in,
    )
    conservative, outside = rebin_piecewise_constant_density(
        response.fine_edges, np.ones_like(LOG_GRID), response.in_edges
    )
    assert np.sum(conservative) == pytest.approx(np.sum(widths), rel=1e-14)
    assert outside == (0.0, 0.0)
    assert np.sum(rebinned) == pytest.approx(np.sum(conservative), rel=1e-14)
    detected = response.apply(np.ones_like(LOG_GRID))
    assert detected.shape == LOG_GRID.shape
    assert np.all(np.isfinite(detected))


def test_timepix_response_channels_do_not_shift_with_source_mesh_refinement():
    """#100: detector channels and their seeded MC must be source-mesh independent."""
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    coarse_grid = np.linspace(1_000.0, 3_000.0, 101)
    fine_grid = np.linspace(1_000.0, 3_000.0, 201)
    coarse = timepix_response.TimepixResponse(coarse_grid, n_mc=64, seed=7)
    fine = timepix_response.TimepixResponse(fine_grid, n_mc=64, seed=7)

    np.testing.assert_array_equal(coarse.in_edges, fine.in_edges)
    np.testing.assert_array_equal(coarse.E_in, fine.E_in)
    np.testing.assert_array_equal(coarse.E_out, fine.E_out)
    np.testing.assert_array_equal(coarse.R, fine.R)
    assert coarse.R is fine.R

    coarse_density = np.exp(-0.5 * ((coarse_grid - 2_000.0) / 100.0) ** 2)
    fine_density = np.exp(-0.5 * ((fine_grid - 2_000.0) / 100.0) ** 2)
    _, coarse_widths = node_bin_edges_and_widths(coarse_grid)
    _, fine_widths = node_bin_edges_and_widths(fine_grid)
    coarse_counts = np.sum(coarse.apply(coarse_density) * coarse_widths)
    fine_counts = np.sum(fine.apply(fine_density) * fine_widths)
    assert coarse_counts == pytest.approx(fine_counts, rel=1e-3)


def test_timepix_response_uses_an_explicit_zero_energy_detector_edge():
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    grid = np.array([1.0, 97.0, 193.0])

    detector = timepix_response.TimepixResponse(grid, n_mc=8)

    assert detector.fine_edges[0] == 0.0
    assert detector.in_edges[0] == 0.0
    assert np.all(np.isfinite(detector.apply(np.ones_like(grid))))


def test_line_metrics_is_correct_on_a_log_grid():
    """#110 replaced the sample-space ``w_samp * dE`` peak-width conversion with
    physical-energy interpolation (``_sample_energy``, ``_integrate_energy_window``)
    and grid-coordinate trapezoids, so line_metrics no longer needs the uniform
    guard -- unlike the sample-space consumers above, a log grid is exact here,
    not merely tolerated."""
    spec = np.exp(-(((LOG_GRID - 3000.0) / 200.0) ** 2))
    brem = np.full_like(LOG_GRID, 1e-3)
    record = {"E_grid": LOG_GRID, "spec": spec, "brem": brem, "scale": 1.0}
    metrics = line_metrics(record, None)
    assert metrics["coherent_flux_per_na"] == pytest.approx(float(np.trapezoid(spec, LOG_GRID)))
    assert metrics["total_flux_per_na"] == pytest.approx(
        float(np.trapezoid(spec, LOG_GRID)) + float(np.trapezoid(brem, LOG_GRID))
    )
    assert metrics["line_eV"] == pytest.approx(3000.0, abs=50.0)
    assert metrics["fwhm_eV"] > 0.0


def test_grid_identity_separates_grids_that_share_size_and_endpoints():
    """The cache-correctness bug: (size, first, last) is not an identity."""
    assert LINEAR_GRID.size == LOG_GRID.size
    assert LINEAR_GRID[0] == LOG_GRID[0]
    assert LINEAR_GRID[-1] == pytest.approx(LOG_GRID[-1])
    legacy = (LINEAR_GRID.size, round(float(LINEAR_GRID[0]), 6), round(float(LINEAR_GRID[-1]), 6))
    legacy_log = (LOG_GRID.size, round(float(LOG_GRID[0]), 6), round(float(LOG_GRID[-1]), 6))
    assert legacy == legacy_log  # the collision the old grid_key had
    assert grid_identity(LINEAR_GRID) != grid_identity(LOG_GRID)
    assert _si_sensor.grid_key(LINEAR_GRID) != _si_sensor.grid_key(LOG_GRID)


def test_grid_key_is_stable_for_the_same_coordinates():
    assert _si_sensor.grid_key(LINEAR_GRID) == _si_sensor.grid_key(LINEAR_GRID.copy())
