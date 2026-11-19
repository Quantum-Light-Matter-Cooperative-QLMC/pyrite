"""Newton inversion of the hard loss CDF against the pre-#916 bisection oracle."""

import numpy as np
import pytest

from pyrite.montecarlo.transport.hard_inelastic import (
    _hard_cdf_density,
    _hard_loss_bounds,
    _sample_hard_transfer_eV,
    _sample_hard_transfer_eV_bisection,
)

# (U, W_resonance) pairs spanning conduction band, valence and K shells.
SHELLS = ((0.0, 16.7), (99.0, 120.0), (1839.0, 2100.0), (69525.0, 75000.0))
DRAWS = (0.0, 1e-15, 1e-9, 0.13, 0.5, 0.87, 1.0 - 1e-9)


def _cases():
    for energy_eV in np.geomspace(2.0e3, 1.0e7, 17):
        for ionization_eV, resonance_eV in SHELLS:
            for branch in range(6):
                for cutoff_eV in (50.0, 1000.0):
                    yield energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV


def test_newton_reaches_the_bisection_root():
    ulps = []
    for case in _cases():
        for u in DRAWS:
            newton = _sample_hard_transfer_eV(*case, u)
            bisection = _sample_hard_transfer_eV_bisection(*case, u)
            lower, upper, _ = _hard_loss_bounds(*case)
            assert lower <= newton <= max(lower, upper)
            assert abs(newton - bisection) <= 1e-8 * bisection
            ulps.append(abs(newton - bisection) / np.spacing(bisection))
    assert np.median(ulps) <= 4.0


def test_newton_keeps_the_interval_limits():
    energy_eV, ionization_eV, resonance_eV = 1.0e6, 149.0, 170.0
    for branch in (0, 2, 5):
        lower, upper, _ = _hard_loss_bounds(energy_eV, ionization_eV, resonance_eV, branch, 50.0)
        args = (energy_eV, ionization_eV, resonance_eV, branch, 50.0)
        assert _sample_hard_transfer_eV(*args, 0.0) == lower
        top = _sample_hard_transfer_eV(*args, 1.0)
        # The distant density vanishes at its top, so the root there is set
        # to about sqrt(eps) by FP64 round-off of J (the bisection's too).
        assert top == pytest.approx(upper, rel=1e-7)


def test_density_is_the_cdf_derivative():
    for energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV in _cases():
        lower, upper, peak_end = _hard_loss_bounds(
            energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV
        )
        if not upper > lower * (1.0 + 1e-6) or (branch % 3 != 2 and ionization_eV == 0.0):
            continue
        kind = 0 if branch % 3 != 2 else (2 if branch >= 3 else 1)
        prime = energy_eV + ionization_eV
        w = np.sqrt(lower * upper) if kind else 0.5 * (lower + upper)
        h = 1e-6 * w
        plus, _ = _hard_cdf_density(kind, energy_eV, prime, peak_end, lower, w + h)
        minus, _ = _hard_cdf_density(kind, energy_eV, prime, peak_end, lower, w - h)
        _, density = _hard_cdf_density(kind, energy_eV, prime, peak_end, lower, w)
        assert (plus - minus) / (2.0 * h) == pytest.approx(density, rel=1e-5)
