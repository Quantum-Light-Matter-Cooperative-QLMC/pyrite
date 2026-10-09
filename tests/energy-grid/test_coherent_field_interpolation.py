"""Complex-field interpolation error against independent analytic fields."""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_normalization import (
    _certified_interpolated_row_power_bounds,
    _interpolated_power_bounds,
    _sample_row_fields,
)
from pyrite.montecarlo.spectrum.coherent_sample_certificates import (
    row_derivative_bounds,
    row_phase_gauges,
    sample_dispersion_bounds,
    sample_row_norm_certificate,
)
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


def test_an_affine_complex_field_has_exact_interpolated_power():
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    fields = [[ctx.mpc(1, 0)], [ctx.mpc(1, 1)]]
    bounds = _interpolated_power_bounds(
        ctx, np.array([0.0, 1.0]), (fields, fields), (0.0, 0.0), (0.4, 0.4)
    )
    with mp.workdps(70):
        actual = mp.mpf(4) / 3
        assert mp.mpf(bounds[0]) <= actual <= mp.mpf(bounds[1])
    assert bounds[1] - bounds[0] < 1e-13


@pytest.mark.parametrize("count", [9, 33])
def test_quadratic_field_encloses_analytic_power_with_second_order_charge(count):
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    nodes = np.linspace(0, 1, count)
    fields = [[1 + ctx.mpf(float(E)) ** 2] for E in nodes]
    lo, hi = _interpolated_power_bounds(ctx, nodes, (fields, fields), (2.0, 2.0), (0.0, 0.0))
    with mp.workdps(70):
        actual = mp.mpf(28) / 15
        assert mp.mpf(lo) <= actual <= mp.mpf(hi)
    assert hi - lo < 2.0 / (count - 1) ** 2


def _row(origin=0.0, signed_absorption=False):
    slopes = np.array([0.004, -0.003]) if signed_absorption else np.zeros(2)
    duration = np.array([100.0, 120.0])
    return CoherentRowField(
        label="two attenuated dispersive fields",
        energy_eV=np.array([1000.0, 1000.1]),
        amplitude=np.array([[1 + 0.3j, 0.4j], [0.1, 0.2]]),
        duration_ang=duration,
        centre_ang=np.array([50.0, 300.0]) + origin,
        electron=np.array([0, 1]),
        start_transmission=np.exp(np.minimum(slopes * duration, 0)),
        end_transmission=np.exp(np.minimum(-slopes * duration, 0)),
        mean_transmission=np.ones(2),
        attenuation_slope_ang=slopes,
        escape_mid_ang=np.array([100.0, 300.0]),
        escape_change_ang=np.array([20.0, -30.0]),
        phase_rad=np.array([0.3, 1.1]),
    )


@pytest.mark.parametrize("factor", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("signed_absorption", [False, True])
def test_composed_field_enclosure_contains_a_separate_dispersive_piece_integral(
    factor, signed_absorption
):
    from mpmath import mp

    row = _row(origin=1e7, signed_absorption=signed_absorption)
    law = CoherentDispersionLaw("hopg", 900, 1100, use_henke=False)
    lo, hi = _certified_interpolated_row_power_bounds(
        row,
        law,
        999.75,
        1000.25,
        np.linspace(999.751, 1000.249, 33),
        form_factor_bounds=(factor, factor),
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        material = mp.mpf(law.prefactor) * mp.mpf(law.forward_constant)

        def power(E):
            w = (E - mp.sqrt(E**2 - material)) / H
            polarizations = []
            for coefficients in row.amplitude:
                fields = []
                for j, c in enumerate(coefficients):
                    dd = mp.mpf(float(row.duration_ang[j]))
                    q = dd * mp.mpf(float(row.slope_ang[j])) / 2
                    v = dd * (E - mp.mpf(float(row.energy_eV[j]))) / (2 * H)
                    v -= mp.mpf(float(row.escape_change_ang[j])) * w / 2
                    z = -q + mp.j * v
                    formation = mp.exp(-abs(q)) * (mp.sinh(z) / z if z else 1)
                    phase = (
                        E
                        * (mp.mpf(float(row.centre_ang[j])) - mp.mpf(float(row.centre_ang[0])))
                        / H
                    )
                    phase -= mp.mpf(float(row.phase_rad[j])) - mp.mpf(float(row.phase_rad[0]))
                    phase -= (
                        mp.mpf(float(row.escape_mid_ang[j])) - mp.mpf(float(row.escape_mid_ang[0]))
                    ) * w
                    fields.append(mp.mpc(complex(c)) * dd * formation * mp.exp(mp.j * phase))
                polarizations.append(fields)
            grouped = sum(abs(z) ** 2 for fields in polarizations for z in fields)
            coherent = sum(abs(sum(fields)) ** 2 for fields in polarizations)
            F = mp.mpf(factor)
            return (1 - F) * grouped + F * coherent

        actual = mp.quad(power, [mp.mpf(999.75), mp.mpf(1000.25)])
        assert mp.mpf(lo) <= actual <= mp.mpf(hi)
        assert (mp.mpf(hi) - mp.mpf(lo)) / actual < mp.mpf("1e-3")


def test_field_capture_preserves_the_existing_norm_certificate():
    row = _row(signed_absorption=True)
    law = CoherentDispersionLaw("hopg", 900, 1100, use_henke=False)
    nodes = np.linspace(999.9, 1000.1, 5)
    samples = _sample_row_fields(row, law, nodes, remove_global_phase=True)
    phases = sample_dispersion_bounds(law, nodes)
    original = sample_row_norm_certificate(row, samples, nodes, phases)
    captured = []

    def capture(k, ctx, values, omega):
        captured.append((k, values))

    observed = sample_row_norm_certificate(row, samples, nodes, phases, field_capture=capture)
    assert len(captured) == nodes.size
    assert isinstance(captured[0][1], tuple)
    for a, b in zip(original, observed, strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_electron_dependent_coherent_gauge_is_refused():
    row = _row()
    gauges = row_phase_gauges(row)
    gauges[1, 1, 0] += 1.0
    with pytest.raises(ValueError, match="common coherent"):
        row_derivative_bounds(row, 0.0, phase_gauges=gauges)


def test_composed_field_enclosure_is_invariant_under_a_common_clock_translation():
    law = CoherentDispersionLaw("hopg", 900, 1100, use_henke=False)
    nodes = np.linspace(999.751, 1000.249, 9)
    bounds = [
        _certified_interpolated_row_power_bounds(
            _row(origin=origin), law, 999.75, 1000.25, nodes, form_factor_bounds=(0.4, 0.4)
        )
        for origin in (0.0, 1e7)
    ]
    np.testing.assert_array_equal(bounds[0], bounds[1])


def test_varying_form_factor_is_enclosed_without_approximating_it_as_constant():
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    nodes = np.array([0.0, 0.5, 1.0])
    grouped = [[ctx.mpc(1)] for E in nodes]
    coherent = [[ctx.mpc(2 * float(E))] for E in nodes]
    lo, hi = _interpolated_power_bounds(ctx, nodes, (grouped, coherent), (0.0, 0.0), (0.0, 1.0))
    with mp.workdps(70):
        # F=E², (1-F)|G|²+F|C|² = 1-E²+4E⁴.
        actual = mp.mpf(22) / 15
        assert mp.mpf(lo) <= actual <= mp.mpf(hi)


def test_directed_field_rectangles_include_endpoint_uncertainty():
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    fields = [[ctx.mpc([0.9, 1.1], [-0.1, 0.1])], [ctx.mpc([1.9, 2.1], [-0.1, 0.1])]]
    lo, hi = _interpolated_power_bounds(
        ctx, np.array([0.0, 1.0]), (fields, fields), (0.0, 0.0), (0.4, 0.4)
    )
    with mp.workdps(70):
        for real_a in (0.9, 1.0, 1.1):
            for real_b in (1.9, 2.0, 2.1):
                for imag_a, imag_b in ((-0.1, 0.1), (0.1, 0.1), (0.0, 0.0)):
                    a, b = mp.mpc(real_a, imag_a), mp.mpc(real_b, imag_b)
                    actual = (abs(a) ** 2 + (a.conjugate() * b).real + abs(b) ** 2) / 3
                    assert mp.mpf(lo) <= actual <= mp.mpf(hi)


@pytest.mark.parametrize("factors", [(0.0, np.inf), (-1.0, 1.0), (1.0, 0.0), [[0.0, 1.0]]])
def test_composed_interpolation_requires_valid_global_form_factor_bounds(factors):
    law = CoherentDispersionLaw("hopg", 900, 1100, use_henke=False)
    with pytest.raises(ValueError, match="global F"):
        _certified_interpolated_row_power_bounds(
            _row(), law, 999.75, 1000.25, np.array([999.8, 1000.2]), form_factor_bounds=factors
        )


@pytest.mark.parametrize("scale", [0.0, 1e-320])
def test_interpolated_field_retains_zero_and_subnormal_power(scale):
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    fields = [[ctx.mpc(scale)], [ctx.mpc(-scale)]]
    lo, hi = _interpolated_power_bounds(
        ctx, np.array([0.0, 1.0]), (fields, fields), (0.0, 0.0), (1.0, 1.0)
    )
    with mp.workdps(70):
        actual = mp.mpf(scale) ** 2 / 3
        assert mp.mpf(lo) <= actual <= mp.mpf(hi)
    assert (hi == 0) == (scale == 0)
