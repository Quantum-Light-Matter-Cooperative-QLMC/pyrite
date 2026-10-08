"""Outward derivative and power arithmetic for captured coherent fields."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_normalization import _sampled_power_bounds
from pyrite.montecarlo.spectrum.coherent_sample_certificates import (
    dispersion_derivative_bound,
    row_derivative_bounds,
)
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


def _row():
    return CoherentRowField(
        label="directed derivative",
        energy_eV=np.array([1000.0]),
        amplitude=np.array([[1.0]]),
        duration_ang=np.array([100.0]),
        centre_ang=np.array([1e300]),
        electron=np.array([0]),
        start_transmission=np.ones(1),
        end_transmission=np.ones(1),
        mean_transmission=np.ones(1),
        attenuation_slope_ang=np.zeros(1),
        escape_mid_ang=np.zeros(1),
        escape_change_ang=np.zeros(1),
        phase_rad=np.zeros(1),
    )


def test_directed_derivative_preserves_a_small_flight_at_a_large_clock_origin():
    from mpmath import mp

    row = _row()
    bounds = row_derivative_bounds(row, 0.0)
    with mp.workdps(90):
        # Derivative of dd sinc(v), after removing the common clock phase,
        # at v=1. This is nonzero even when float64 centre +/-dd/2 collapses.
        actual = abs(mp.cos(1) - mp.sin(1)) * 100**2 / (2 * mp.mpf(HBARC_EV_ANG))
        assert all(mp.mpf(float(bound)) >= actual for bound in bounds)
    translated = row_derivative_bounds(replace(row, centre_ang=np.zeros(1)), 0.0)
    assert bounds == translated


def test_directed_power_enclosure_covers_power_below_float64_underflow():
    from mpmath import mp

    lower, upper = _sampled_power_bounds(
        np.array([0.5]),
        np.full((1, 2), 1e-320),
        sample_norm_errors=np.zeros((1, 2)),
        derivative_norm_upper=(0.0, 0.0),
        start_eV=0.0,
        stop_eV=1.0,
        form_factor_bounds=(0.4, 0.4),
        directed=True,
    )
    with mp.workdps(90):
        actual = mp.mpf(1e-320) ** 2
        assert mp.mpf(lower) <= actual <= mp.mpf(upper)
    assert lower == 0 and upper > 0


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_directed_constant_power_encloses_exact_norm_and_band_arithmetic(scale):
    from mpmath import mp

    norms = np.full((2, 2), scale)
    lower, upper = _sampled_power_bounds(
        np.array([0.2, 0.8]),
        norms,
        sample_norm_errors=np.zeros((2, 2)),
        derivative_norm_upper=(0.0, 0.0),
        start_eV=0.1,
        stop_eV=0.9,
        form_factor_bounds=(0.37, 0.37),
        directed=True,
    )
    with mp.workdps(90):
        actual = mp.mpf(float(scale)) ** 2 * (mp.mpf(0.9) - mp.mpf(0.1))
        assert mp.mpf(lower) <= actual <= mp.mpf(upper)
    assert upper - lower <= 1e-13 * upper


def test_directed_material_derivative_covers_a_separate_constant_forward_formula():
    from mpmath import mp

    law = CoherentDispersionLaw("hopg", 900.0, 1100.0, use_henke=False)
    bound = dispersion_derivative_bound(law, 950.0, 1050.0)
    with mp.workdps(90):
        A = mp.mpf(law.prefactor) * mp.mpf(law.forward_constant)
        for energy in np.linspace(950, 1050, 13):
            E = mp.mpf(float(energy))
            # w=(E-sqrt(E²-A))/H, so w'=(1-E/sqrt(E²-A))/H.
            actual = abs(1 - E / mp.sqrt(E**2 - A)) / mp.mpf(HBARC_EV_ANG)
            assert mp.mpf(bound) >= actual


def test_adjacent_float_band_below_a_knot_uses_the_left_derivative():
    from types import SimpleNamespace

    from mpmath import mp
    from scipy.interpolate import PPoly

    law = CoherentDispersionLaw("hopg", 999.0, 1001.0, use_henke=False)
    # A continuous synthetic linear forward factor has a derivative jump
    # at1000. A rounded band midpoint lands on the right, zero-slope piece.
    poly = PPoly(np.array([[1000.0, 0.0], [-1000.0, 0.0]]), np.array([999.0, 1000.0, 1001.0]))
    law.prefactor = 1.0
    law.forward_constant = 0.0
    law.breaks = np.array([999.0, 1000.0, 1001.0])
    law.atoms = [SimpleNamespace(count=1, f1=(poly,), energies=law.breaks, f2=np.ones(3))]
    lo, hi = np.nextafter(1000.0, -np.inf), 1000.0
    assert lo + 0.5 * (hi - lo) == hi
    bound = dispersion_derivative_bound(law, lo, hi)
    with mp.workdps(90):
        point = (mp.mpf(float(lo)) + mp.mpf(hi)) / 2

        def phase(E):
            chi = -(1000 * (E - 1000) + 1j) / E**2
            return (1 - mp.sqrt(1 + chi).real) * E / mp.mpf(HBARC_EV_ANG)

        assert mp.mpf(bound) >= abs(mp.diff(phase, point))


def test_directed_cone_cutoff_and_varying_blend_cover_crossing_norms():
    from scipy.integrate import quad

    energies = np.array([-0.8, 0.0, 0.8])
    norms = np.column_stack((np.sqrt(1 + energies**2), np.abs(1 + energies)))
    lower, upper = _sampled_power_bounds(
        energies,
        norms,
        sample_norm_errors=np.full((3, 2), 1e-12),
        derivative_norm_upper=(1.0, 1.0),
        start_eV=-1.0,
        stop_eV=1.0,
        form_factor_bounds=(0.0, 1.0),
        directed=True,
    )

    def power(E):
        F = 0.5 + 0.4 * np.sin(9 * E)
        return (1 - F) * (1 + E**2) + F * (1 + E) ** 2

    actual = quad(power, -1, 1, epsabs=1e-12)[0]
    assert 0 < lower <= actual <= upper


@pytest.mark.parametrize("invalid", ["duration", "transmission", "amplitude"])
def test_directed_derivative_refuses_invalid_captured_fields(invalid):
    row = _row()
    if invalid == "duration":
        row = replace(row, duration_ang=np.array([-1.0]))
    elif invalid == "transmission":
        row = replace(row, start_transmission=np.array([-1.0]))
    else:
        row = replace(row, amplitude=np.array([[np.nan]]))
    with pytest.raises(ValueError, match="finite aligned"):
        row_derivative_bounds(row, 0.0)
