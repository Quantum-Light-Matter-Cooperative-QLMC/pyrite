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


def test_poisson_core_refuses_a_log_grid():
    rng = np.random.default_rng(0)
    density = np.ones_like(LOG_GRID)
    with pytest.raises(NonuniformEnergyGridError, match="poisson_core"):
        _si_sensor.poisson_core(LOG_GRID, density, 1.0, rng)


def test_poisson_core_uniform_expected_counts_unchanged():
    rng = np.random.default_rng(0)
    density = np.full(LINEAR_GRID.shape, 2.0)
    _, expected = _si_sensor.poisson_core(LINEAR_GRID, density, 3.0, rng)
    dE = float(LINEAR_GRID[1] - LINEAR_GRID[0])
    assert np.array_equal(expected, np.clip(density * dE * 3.0, 0.0, None))


def test_convolve_detector_refuses_a_log_grid():
    with pytest.raises(NonuniformEnergyGridError, match="convolve_detector"):
        response.convolve_detector(LOG_GRID, np.ones_like(LOG_GRID), 50.0)


def test_convolve_detector_still_runs_on_a_uniform_grid():
    out = response.convolve_detector(LINEAR_GRID, np.ones_like(LINEAR_GRID), 50.0)
    assert out.shape == LINEAR_GRID.shape


def test_timepix_response_refuses_a_log_grid():
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    with pytest.raises(NonuniformEnergyGridError, match="TimepixResponse"):
        timepix_response.TimepixResponse(LOG_GRID, n_mc=8)


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
