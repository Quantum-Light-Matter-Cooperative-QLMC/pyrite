import numpy as np
import pytest

from pyrite.energy_grid.bounds import (
    CoverageGridTooNarrow,
    coverage_energy,
    line_shift_fraction,
    margined_stop,
    spacing_num,
)
from pyrite.energy_grid.semantics import resolution_num, validate_backend_spacing


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


def test_resolution_num_never_exceeds_requested_spacing():
    num = resolution_num(10.0, 20.0, 3.0)
    assert num == 5
    assert (20.0 - 10.0) / (num - 1) <= 3.0


def test_backend_spacing_rejects_unrepresentable_float32_grid():
    with pytest.raises(ValueError, match="PYRITE_FP64=1"):
        validate_backend_spacing(19_999.0, 20_000.0, 10_001, dtype=np.float32, safety_ulps=8.0)


def test_backend_spacing_accepts_resolved_grid_and_returns_actual_step():
    assert validate_backend_spacing(10.0, 20_000.0, 20_000, dtype=np.float32) == pytest.approx(
        (20_000.0 - 10.0) / 19_999
    )


# ---- beam energy spread vs the derived line window ------------------------
# Validation: beam-energy-spread-injection


def test_line_shift_is_zero_without_spread():
    assert line_shift_fraction(30.0, 0.0) == 0.0


def test_line_shift_is_first_order_in_the_spread():
    """domega/omega = S*delta, so doubling the spread doubles the shift."""
    single = line_shift_fraction(30.0, 0.01)
    assert line_shift_fraction(30.0, 0.02) == pytest.approx(2.0 * single)
    # Sign-free: a beam is broadened in both directions by one sigma.
    assert line_shift_fraction(30.0, -0.01) == single


def test_line_shift_sensitivity_falls_toward_the_nonrelativistic_half():
    """The gamma -> 1 limit is omega ~ beta ~ sqrt(T), so S -> 1/2.

    Above it the sensitivity only decreases, which is why the 30 keV end of the
    sweep -- not the 300 keV ceiling -- sets the worst case for grid width.
    """
    assert line_shift_fraction(1.0e-3, 1.0) == pytest.approx(0.5, rel=1e-5)
    sensitivities = [line_shift_fraction(T, 1.0) for T in (30.0, 100.0, 300.0)]
    assert sensitivities == sorted(sensitivities, reverse=True)
    assert sensitivities[0] < 0.5


def test_forward_observation_raises_the_sensitivity():
    """The Doppler denominator 1 - beta*cos(theta) is what makes it geometric."""
    side = line_shift_fraction(300.0, 0.01, cos_theta_obs=0.0)
    forward = line_shift_fraction(300.0, 0.01, cos_theta_obs=1.0)
    assert forward > side


def test_catalog_margin_covers_any_plausible_beam_spread():
    """The check task step H asks for: is a broadened line clipped?

    `margined_stop` cuts the window 15% above the measured coverage energy, and
    the spread displaces the line by S*delta of its own energy. At the 30 keV
    worst case S = 0.46, so the margin is only consumed once the beam spread
    passes ~33% RMS -- an order of magnitude beyond any real photoinjector. No
    gate on `energy_spread_frac` is needed; a run that does set a spread that
    large is already outside the first-order model this bound is derived under.
    """
    margin = margined_stop(1000.0, round_to=1.0) / 1000.0 - 1.0
    worst = max(line_shift_fraction(T, 0.05, cos_theta_obs=0.0) for T in (30.0, 100.0, 300.0))
    assert worst < margin
    breakeven = margin / line_shift_fraction(30.0, 1.0)
    assert breakeven > 0.3
