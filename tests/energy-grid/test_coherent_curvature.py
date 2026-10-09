"""Analytic curvature and second-order power enclosures (#350)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_normalization import (
    _curvature_power_bounds,
    _sampled_power_bounds,
)
from pyrite.montecarlo.spectrum.coherent_sample_certificates import (
    dispersion_derivative_bound,
    row_derivative_bounds,
)
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


@pytest.mark.parametrize("count", [9, 33])
@pytest.mark.parametrize("factor", [(0.0, 0.0), (0.4, 0.4), (0.0, 1.0)])
def test_curvature_encloses_analytic_affine_vector_power(count, factor):
    # Both sectors A(E)=1+E: K1=1, K2=0 and integral is 7/3.
    nodes = np.linspace(0, 1, count)
    norms = np.repeat((1 + nodes)[:, None], 2, axis=1)
    args = dict(
        sample_norm_errors=np.zeros_like(norms),
        derivative_norm_upper=(1.0, 1.0),
        start_eV=0.0,
        stop_eV=1.0,
        form_factor_bounds=factor,
    )
    cone = _sampled_power_bounds(nodes, norms, directed=True, **args)
    bound = _curvature_power_bounds(nodes, norms, second_derivative_norm_upper=(0.0, 0.0), **args)
    assert cone[0] <= bound[0] <= 7 / 3 <= bound[1] <= cone[1]
    if factor[0] == factor[1]:
        # A classical second-order remainder, rather than first-order norm drift.
        assert bound[1] - bound[0] <= 1.01 / (3 * (count - 1) ** 2)


@pytest.mark.parametrize("factor", [0.0, 0.4, 1.0])
def test_curvature_covers_nonlinear_phase_interference_and_endpoint_strips(factor):
    from mpmath import mp

    nodes = np.linspace(0.01, 0.99, 33)
    # G=sqrt(2), C=1+exp(i E²). |C'|<=2; |C''|<=6.
    norms = np.column_stack((np.full(nodes.size, np.sqrt(2)), np.abs(1 + np.exp(1j * nodes**2))))
    args = dict(
        sample_norm_errors=np.full_like(norms, 1e-14),
        derivative_norm_upper=(0.0, 2.0),
        second_derivative_norm_upper=(0.0, 6.0),
        start_eV=0.0,
        stop_eV=1.0,
        form_factor_bounds=(factor, factor),
    )
    lo, hi = _curvature_power_bounds(nodes, norms, **args)
    with mp.workdps(70):
        F = mp.mpf(factor)
        actual = 2 + 2 * F * mp.quad(lambda E: mp.cos(E**2), [0, 1])
        assert mp.mpf(lo) <= actual <= mp.mpf(hi)


def test_material_curvature_encloses_independent_constant_forward_formula():
    from mpmath import mp

    law = CoherentDispersionLaw("hopg", 900, 1100, use_henke=False)
    bound = dispersion_derivative_bound(law, 950, 1050, order=2)
    with mp.workdps(80):
        A = mp.mpf(law.prefactor) * mp.mpf(law.forward_constant)
        # w=(E-sqrt(E²-A))/H => w''=A/[H(E²-A)^(3/2)].
        for energy in np.linspace(950, 1050, 13):
            actual = A / (mp.mpf(HBARC_EV_ANG) * (mp.mpf(float(energy)) ** 2 - A) ** mp.mpf("1.5"))
            assert mp.mpf(bound) >= abs(actual)


def test_row_curvature_preserves_gauge_and_covers_a_physical_sinc():
    from mpmath import mp

    row = CoherentRowField(
        label="sinc curvature",
        energy_eV=np.array([1000.0]),
        amplitude=np.ones((1, 1)),
        duration_ang=np.array([100.0]),
        centre_ang=np.array([1e300]),
        electron=np.array([0]),
        start_transmission=np.ones(1),
        end_transmission=np.ones(1),
        mean_transmission=np.ones(1),
        escape_mid_ang=np.zeros(1),
        escape_change_ang=np.zeros(1),
        phase_rad=np.zeros(1),
    )
    with pytest.raises(ValueError, match="explicit certified"):
        row_derivative_bounds(row, 0.0, order=2)
    bounds = row_derivative_bounds(row, 0.0, order=2, phase_second_derivative_upper=0.0)
    assert bounds == row_derivative_bounds(
        replace(row, centre_ang=np.zeros(1)), 0.0, order=2, phase_second_derivative_upper=0.0
    )
    with mp.workdps(80):
        actual = mp.mpf(100) ** 3 / (12 * mp.mpf(HBARC_EV_ANG) ** 2)
        assert all(mp.mpf(b) >= actual for b in bounds)


@pytest.mark.parametrize("material", ["hopg", "wse2"])
def test_material_curvature_covers_high_precision_stored_law(material):
    from mpmath import mp

    law = CoherentDispersionLaw(material, 900, 4000)
    for carrier in (1100.0, 2100.0, 3000.0):
        index = np.searchsorted(law.breaks, carrier, side="right") - 1
        lo, hi = max(carrier - 0.1, law.breaks[index]), min(carrier + 0.1, law.breaks[index + 1])
        bound = dispersion_derivative_bound(law, lo, hi, order=2)
        with mp.workdps(80):

            def phase(E, carrier=carrier):
                f = mp.mpf(law.forward_constant)
                for atom in law.atoms:
                    poly = atom.f1[0]
                    k = np.searchsorted(poly.x, carrier, side="right") - 1
                    k = int(np.clip(k, 0, poly.c.shape[1] - 1))
                    x = E - mp.mpf(float(poly.x[k]))
                    # Power sums rather than the owner's interval Horner expression.
                    real = sum(
                        mp.mpf(float(c)) * x ** (poly.c.shape[0] - j - 1)
                        for j, c in enumerate(poly.c[:, k])
                    )
                    k = int(
                        np.clip(
                            np.searchsorted(atom.energies, carrier, side="right") - 1,
                            0,
                            atom.energies.size - 2,
                        )
                    )
                    a, b = map(lambda x: mp.mpf(float(x)), atom.energies[k : k + 2])
                    ya, yb = map(lambda x: mp.mpf(float(x)), atom.f2[k : k + 2])
                    imaginary = ya * (E / a) ** (mp.log(yb / ya) / mp.log(b / a))
                    f += atom.count * mp.mpc(real, imaginary)
                return (
                    E
                    * (1 - mp.sqrt(1 - mp.mpf(law.prefactor) * f / E**2).real)
                    / mp.mpf(HBARC_EV_ANG)
                )

            for energy in np.linspace(lo, hi, 5)[1:-1]:
                actual = abs(mp.diff(phase, mp.mpf(float(energy)), 2))
                assert mp.mpf(bound) >= actual


def test_curvature_retains_subnormal_positive_power_and_exact_zero():
    from mpmath import mp

    for scale in (0.0, 1e-320):
        lo, hi = _curvature_power_bounds(
            np.array([0.0, 1.0]),
            np.full((2, 2), scale),
            sample_norm_errors=np.zeros((2, 2)),
            derivative_norm_upper=(0.0, 0.0),
            second_derivative_norm_upper=(0.0, 0.0),
            start_eV=0.0,
            stop_eV=1.0,
            form_factor_bounds=(0.4, 0.4),
        )
        with mp.workdps(80):
            assert mp.mpf(lo) <= mp.mpf(scale) ** 2 <= mp.mpf(hi)
        assert (hi == 0) == (scale == 0)


@pytest.mark.parametrize("second", [(np.nan, 1), (-1, 1), (np.inf, 1)])
def test_curvature_refuses_invalid_second_derivative(second):
    with pytest.raises(ValueError, match="curvature"):
        _curvature_power_bounds(
            np.array([0.0, 1.0]),
            np.ones((2, 2)),
            sample_norm_errors=np.zeros((2, 2)),
            derivative_norm_upper=(1, 1),
            second_derivative_norm_upper=second,
            start_eV=0,
            stop_eV=1,
        )
