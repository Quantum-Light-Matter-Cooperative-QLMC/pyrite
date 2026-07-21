import numpy as np
import pytest

from cxr_mc.line_grid_bounds import (
    CoverageGridTooNarrow,
    coverage_energy,
    margined_stop,
    spacing_num,
)


def test_coverage_energy_flat_spectrum_half_coverage_is_midpoint():
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.ones_like(E_grid)
    assert coverage_energy(E_grid, spec, coverage=0.5) == pytest.approx(2.0)


def test_coverage_energy_reaching_coverage_only_at_ceiling_refuses_silently():
    """Coverage met only in the final bin means the emission tail runs past the
    grid ceiling, so a reported bound would be a silent floor pinned to it.
    Without an explicit override this must raise, not clamp (issue_notes.md #1)."""
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.ones_like(E_grid)
    with pytest.raises(CoverageGridTooNarrow):
        coverage_energy(E_grid, spec, coverage=0.99)
    # allow_shortfall is the explicit opt-in that returns the ceiling-pinned floor.
    assert coverage_energy(E_grid, spec, coverage=0.99, allow_shortfall=True) == pytest.approx(4.0)


def test_coverage_energy_zero_spectrum_returns_grid_start():
    """A geometry with no coherent-line intensity places no requirement on
    grid width, and must not be mistaken for the widest requirement when
    candidates are ranked by this value -- so it returns the grid's floor,
    not its ceiling."""
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.zeros_like(E_grid)
    assert coverage_energy(E_grid, spec, coverage=0.99) == pytest.approx(0.0)


def test_coverage_energy_early_spike_is_captured_quickly():
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    spec = np.array([0.0, 100.0, 0.0, 0.0, 0.0])
    assert coverage_energy(E_grid, spec, coverage=0.99) == pytest.approx(2.0)


def test_margined_stop_exact_multiple_of_round_to():
    assert margined_stop(2000.0, margin=0.15, round_to=100.0) == pytest.approx(2300.0)


def test_margined_stop_rounds_up_past_round_to():
    assert margined_stop(2001.0, margin=0.15, round_to=100.0) == pytest.approx(2400.0)


@pytest.mark.parametrize(
    ("start_eV", "stop_eV", "expected_num"),
    [
        (10.0, 2500.0, 831),
        (10.0, 3000.0, 998),
        (50.0, 3500.0, 1151),
        (50.0, 4000.0, 1318),
        (50.0, 4500.0, 1484),
        (50.0, 5000.0, 1651),
    ],
)
def test_spacing_num_matches_existing_catalog_grids(start_eV, stop_eV, expected_num):
    assert spacing_num(start_eV, stop_eV) == expected_num
