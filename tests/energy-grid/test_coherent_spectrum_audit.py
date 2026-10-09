"""Weighted captured-spectrum yields against independent analytic integrals."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


def _row(amplitude=1.0, weight=1.0):
    return CoherentRowField(
        label="one electron sinc",
        energy_eV=np.array([1000.0]),
        amplitude=np.array([[amplitude]]),
        duration_ang=np.ones(1),
        centre_ang=np.zeros(1),
        electron=np.array([0]),
        start_transmission=np.ones(1),
        end_transmission=np.ones(1),
        mean_transmission=np.ones(1),
        attenuation_slope_ang=np.zeros(1),
        escape_mid_ang=np.zeros(1),
        escape_change_ang=np.zeros(1),
        phase_rad=np.zeros(1),
        mosaic_weight=weight,
    )


def _law():
    return CoherentDispersionLaw("hopg", 990.0, 1010.0, use_henke=False)


@pytest.mark.parametrize("bands", [[(990.0, 1010.0)], [(990.0, 995.0), (1002.0, 1010.0)]])
def test_weighted_rows_use_incident_population_and_check_the_numerical_yield(bands):
    from mpmath import mp

    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    rows = [_row(1.0, 0.25), _row(2.0, 0.75)]
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        actual = (
            mp.mpf(13)
            / 12
            * sum(
                mp.quad(lambda E: mp.sinc((E - 1000) / (2 * H)) ** 2, [lo, hi]) for lo, hi in bands
            )
        )
        result = audit_captured_spectrum_yield(
            rows,
            _law(),
            electron_count=3,
            numerical_yield=float(actual),
            bands=iter(bands),
            max_evaluations=100,
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)
    assert result.within_tolerance
    assert result.relative_error_upper <= 1e-3
    assert result.evaluations == sum(cert.evaluations for cert in result.row_certificates)
    # The same valid integral enclosure must reject a poor supplied grid yield.
    bad = audit_captured_spectrum_yield(
        rows,
        _law(),
        electron_count=3,
        numerical_yield=float(actual) * 1.1,
        bands=iter(bands),
        max_evaluations=100,
    )
    assert not bad.within_tolerance
    assert bad.relative_error_upper > 0.09


def test_spectrum_budget_covers_every_contributing_row_before_refinement():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_captured_spectrum_yield,
    )

    with pytest.raises(SpectrumPowerBudgetError, match="initialization needs 6") as error:
        audit_captured_spectrum_yield(
            [_row(), _row()], _law(), electron_count=1, numerical_yield=40, max_evaluations=5
        )
    assert error.value.row_index is None
    assert error.value.evaluations == 0
    assert error.value.row_certificate is None


def test_exhausted_row_budget_is_not_presented_as_a_complete_spectrum_certificate():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_captured_spectrum_yield,
    )

    row = replace(_row(), duration_ang=np.array([100.0]))
    with pytest.raises(SpectrumPowerBudgetError, match="row 0") as error:
        audit_captured_spectrum_yield(
            [row, row],
            _law(),
            electron_count=1,
            numerical_yield=1,
            relative_tolerance=1e-10,
            max_evaluations=6,
        )
    assert error.value.row_index == 0
    assert error.value.evaluations == 3
    assert error.value.row_certificate.evaluations == 3


def test_later_row_exhaustion_counts_work_from_completed_rows():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_captured_spectrum_yield,
    )

    with pytest.raises(SpectrumPowerBudgetError, match="row 1") as error:
        audit_captured_spectrum_yield(
            [
                replace(_row(), duration_ang=np.array([0.1])),
                replace(_row(), duration_ang=np.array([100.0])),
            ],
            _law(),
            electron_count=1,
            numerical_yield=1,
            max_evaluations=6,
        )
    assert error.value.row_index == 1
    assert error.value.evaluations == 6
    assert error.value.row_certificate.evaluations == 3


@pytest.mark.parametrize("estimate,accepted", [(0.0, True), (1.0, False)])
def test_no_contributing_rows_have_exact_zero_power(estimate, accepted):
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    result = audit_captured_spectrum_yield(
        [_row(weight=0.0)], _law(), electron_count=2, numerical_yield=estimate
    )
    assert (result.lower, result.upper, result.evaluations) == (0.0, 0.0, 0)
    assert result.within_tolerance == accepted


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan])
def test_invalid_mosaic_weights_are_refused(weight):
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    with pytest.raises(ValueError, match="mosaic weights"):
        audit_captured_spectrum_yield(
            [_row(weight=weight)], _law(), electron_count=1, numerical_yield=20
        )


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_population_normalization_is_refused(count):
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    with pytest.raises(ValueError, match="electron count"):
        audit_captured_spectrum_yield([_row()], _law(), electron_count=count, numerical_yield=20)


def test_positive_weighted_underflow_is_never_certified_zero():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    result = audit_captured_spectrum_yield(
        [_row(weight=np.nextafter(0.0, np.inf))],
        _law(),
        electron_count=100,
        numerical_yield=0.0,
        max_evaluations=100,
    )
    assert result.lower == 0.0 and result.upper == np.nextafter(0.0, np.inf)
    assert result.relative_error_upper == 1.0
    assert not result.within_tolerance


def test_weighted_subnormal_bounds_enclose_the_independent_integral():
    from mpmath import mp

    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield

    weight = float(np.nextafter(0.0, np.inf))
    result = audit_captured_spectrum_yield(
        [_row(weight=weight)], _law(), electron_count=3, numerical_yield=0.0, max_evaluations=100
    )
    with mp.workdps(90):
        H = mp.mpf(HBARC_EV_ANG)
        actual = (
            mp.mpf(weight) / 3 * mp.quad(lambda E: mp.sinc((E - 1000) / (2 * H)) ** 2, [990, 1010])
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)


@pytest.mark.parametrize("scale", [1.0, 1e-300, np.nextafter(0.0, np.inf)])
def test_full_axis_quadrature_encloses_exact_nonuniform_stored_sample_integral(scale):
    from fractions import Fraction

    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import _trapezoid_bounds

    grid = np.array([990.0, 990.125, 993.0, 1001.0, 1010.0])
    source = np.array([1.0, 3.0, 0.0, 2.0, 1.0]) * scale
    exact = sum(
        (Fraction(float(b)) - Fraction(float(a))) * (Fraction(float(u)) + Fraction(float(v))) / 2
        for a, b, u, v in zip(grid[:-1], grid[1:], source[:-1], source[1:], strict=True)
    )
    lower, upper = _trapezoid_bounds(grid, source)
    assert Fraction(lower) <= exact <= Fraction(upper)


def test_full_axis_yield_checks_backbone_intervals_and_rejects_a_bad_grid():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_yield

    grid = np.array([990.0, 990.1, 999.9, 1000.0, 1000.1, 1009.9, 1010.0])
    source = np.sinc((grid - 1000) / (2 * np.pi * HBARC_EV_ANG)) ** 2
    result = audit_full_axis_spectrum_yield(
        [_row()], _law(), grid, source, electron_count=1, max_evaluations=100
    )
    assert result.within_tolerance
    assert result.points == grid.size
    assert result.quadrature_lower <= np.trapezoid(source, grid) <= result.quadrature_upper
    assert result.spectrum.lower <= 20 <= result.spectrum.upper * 1.00001
    assert (
        sum(
            interval.stop_eV - interval.start_eV
            for interval in result.spectrum.row_certificates[0].intervals
        )
        == 20
    )
    bad = audit_full_axis_spectrum_yield(
        [_row()], _law(), grid, source * 1.1, electron_count=1, max_evaluations=100
    )
    assert not bad.within_tolerance
    assert bad.relative_error_upper > 0.09


@pytest.mark.parametrize(
    "grid,source",
    [
        ([991, 1010], [1, 1]),
        ([990, 1009], [1, 1]),
        ([990, 1000, 1000, 1010], [1, 1, 1, 1]),
        ([990, 1010], [1, -1]),
        ([990, 1010], [1, np.nan]),
        ([990, 1010], [1]),
        ([990, np.inf], [1, 1]),
        ([[990, 1010]], [[1, 1]]),
    ],
)
def test_full_axis_audit_refuses_partial_or_invalid_axes(grid, source):
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_yield

    with pytest.raises(ValueError, match="full-axis audit needs"):
        audit_full_axis_spectrum_yield([_row()], _law(), grid, source, electron_count=1)


def test_full_axis_budget_exhaustion_never_returns_a_partial_acceptance():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_full_axis_spectrum_yield,
    )

    with pytest.raises(SpectrumPowerBudgetError):
        audit_full_axis_spectrum_yield(
            [_row(), _row()], _law(), [990, 1010], [1, 1], electron_count=1, max_evaluations=5
        )


def test_empty_full_axis_source_has_exact_zero_quadrature():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_yield

    result = audit_full_axis_spectrum_yield([], _law(), [990, 1010], [0, 0], electron_count=1)
    assert (result.quadrature_lower, result.quadrature_upper) == (0, 0)
    assert result.within_tolerance


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"_coherent_yield_audit": True}, "mapping"),
        ({"_coherent_yield_audit": {"bands": [(990, 1010)]}}, "mapping"),
        ({"coherent_emission": False}, "requires coherent_emission"),
        ({"layer_radiators": []}, "one radiator"),
        ({"sinc_cutoff": 10}, "without sinc_cutoff"),
    ],
)
def test_runner_audit_refuses_unsupported_configuration_before_spectrum_work(changes, match):
    from pyrite.montecarlo.runner.coherent_audit import CoherentGridAudit

    case = {"_coherent_yield_audit": {}, "coherent_emission": True, **changes}
    with pytest.raises(ValueError, match=match):
        CoherentGridAudit(case)


def test_full_axis_centroid_encloses_the_symmetric_analytic_source():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
    )

    grid = np.array([990.0, 993.0, 1000.0, 1007.0, 1010.0])
    source = np.sinc((grid - 1000) / (2 * np.pi * HBARC_EV_ANG)) ** 2
    result = audit_full_axis_spectrum_centroid(
        [_row()],
        _law(),
        grid,
        source,
        electron_count=1,
        integration_bins=32,
        max_evaluations=500,
    )
    assert result.centroid_lower_eV <= 1000 <= result.centroid_upper_eV
    assert result.quadrature_centroid_lower_eV <= 1000 <= result.quadrature_centroid_upper_eV
    assert result.within_tolerance
    assert result.yield_audit.within_tolerance


def test_yield_acceptance_does_not_accept_a_shifted_centroid():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
    )

    result = audit_full_axis_spectrum_centroid(
        [_row()],
        _law(),
        [990, 1000, 1010],
        [0, 1, 2],
        electron_count=1,
        integration_bins=32,
        max_evaluations=500,
    )
    assert result.yield_audit.within_tolerance
    assert not result.within_tolerance
    assert result.relative_error_upper > 0.004


def test_centroid_rejects_an_unresolved_moment_even_when_yield_converged():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
    )

    result = audit_full_axis_spectrum_centroid(
        [_row()],
        _law(),
        [990, 1010],
        [1, 1],
        electron_count=1,
        integration_bins=1,
        max_evaluations=100,
    )
    assert result.yield_audit.within_tolerance
    assert not result.within_tolerance


def test_centroid_has_no_zero_power_acceptance():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
    )

    with pytest.raises(ValueError, match="positive yield"):
        audit_full_axis_spectrum_centroid(
            [],
            _law(),
            [990, 1010],
            [0, 0],
            electron_count=1,
        )


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, 10**9])
def test_centroid_partition_must_fit_the_complete_evaluation_budget(bins):
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
    )

    with pytest.raises(ValueError, match="integration bins|initialization needs"):
        audit_full_axis_spectrum_centroid(
            [_row()],
            _law(),
            [990, 1010],
            [1, 1],
            electron_count=1,
            integration_bins=bins,
            max_evaluations=100,
        )


def test_weighted_centroid_encloses_an_independent_first_moment():
    from mpmath import mp

    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_centroid

    rows = [
        replace(_row(1, 0.25), energy_eV=np.array([995.0]), duration_ang=np.array([10.0])),
        replace(_row(2, 0.75), energy_eV=np.array([1005.0]), duration_ang=np.array([10.0])),
    ]
    grid = np.linspace(990, 1010, 65)
    source = sum(
        r.mosaic_weight
        * abs(r.amplitude[0, 0]) ** 2
        * 10**2
        * np.sinc((grid - r.energy_eV[0]) * 10 / (2 * np.pi * HBARC_EV_ANG)) ** 2
        / 3
        for r in rows
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)

        def power(E):
            return sum(
                mp.mpf(r.mosaic_weight)
                * abs(mp.mpc(complex(r.amplitude[0, 0]))) ** 2
                * 10**2
                * mp.sinc((E - mp.mpf(float(r.energy_eV[0]))) * 10 / (2 * H)) ** 2
                / 3
                for r in rows
            )

        centroid = mp.quad(lambda E: E * power(E), [990, 1010]) / mp.quad(power, [990, 1010])
        result = audit_full_axis_spectrum_centroid(
            iter(rows),
            _law(),
            grid,
            source,
            electron_count=3,
            integration_bins=64,
            max_evaluations=2000,
        )
        assert mp.mpf(result.centroid_lower_eV) <= centroid <= mp.mpf(result.centroid_upper_eV)
    assert result.within_tolerance


def test_centroid_preserves_positive_subnormal_source():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_centroid

    row = _row(1e-158)
    grid = np.linspace(990, 1010, 33)
    source = (
        abs(row.amplitude[0, 0]) ** 2 * np.sinc((grid - 1000) / (2 * np.pi * HBARC_EV_ANG)) ** 2
    )
    result = audit_full_axis_spectrum_centroid(
        [row],
        _law(),
        grid,
        source,
        electron_count=1,
        integration_bins=32,
        max_evaluations=500,
    )
    assert result.yield_audit.spectrum.lower > 0
    assert result.centroid_lower_eV <= 1000 <= result.centroid_upper_eV
    assert result.quadrature_centroid_lower_eV <= 1000 <= result.quadrature_centroid_upper_eV


def test_runner_requires_centroid_acceptance_before_returning_provenance(monkeypatch):
    from scipy.constants import elementary_charge

    from pyrite._line_grid_policy import LineGridToleranceError
    from pyrite.montecarlo.runner.coherent_audit import CoherentGridAudit

    monkeypatch.setattr(
        "pyrite.montecarlo.runner.coherent_audit.CoherentDispersionLaw", lambda *args: _law()
    )
    audit = CoherentGridAudit(
        {
            "_coherent_yield_audit": {
                "centroid_relative_tolerance": 1e-3,
                "integration_bins": 32,
                "max_evaluations": 500,
            },
            "coherent_emission": True,
            "crystal": "hopg",
            "bunch_charge_pc": elementary_charge * 1e12,
        }
    )
    audit.collector.rows.append(_row())
    with pytest.raises(LineGridToleranceError, match="centroid audit failed"):
        audit.check(np.array([990.0, 1000.0, 1010.0]), np.array([0.0, 1.0, 2.0]), 1)
    record = audit.check(np.array([990.0, 1000.0, 1010.0]), np.ones(3), 1)
    assert record["scope"] == "stored-input full finite-axis yield and centroid"
    assert record["centroid_relative_error_upper"] <= 1e-3
    yield_only = CoherentGridAudit(
        {
            "_coherent_yield_audit": {},
            "coherent_emission": True,
            "crystal": "hopg",
            "bunch_charge_pc": elementary_charge * 1e12,
        }
    )
    yield_only.collector.rows.append(_row())
    record = yield_only.check(np.array([990.0, 1000.0, 1010.0]), np.ones(3), 1)
    assert record["scope"] == "stored-input full finite-axis yield"
    assert "centroid_relative_error_upper" not in record


def test_centroid_refuses_unrepresentable_partition_boundaries():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_full_axis_spectrum_centroid

    stop = np.nextafter(990.0, np.inf)
    law = CoherentDispersionLaw("hopg", 990.0, stop, use_henke=False)
    with pytest.raises(ValueError, match="representable increasing"):
        audit_full_axis_spectrum_centroid(
            [_row()],
            law,
            [990, stop],
            [1, 1],
            electron_count=1,
            integration_bins=2,
        )


def test_subnormal_moment_budget_does_not_return_zero_power_acceptance():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_full_axis_spectrum_centroid,
    )

    with pytest.raises(SpectrumPowerBudgetError):
        audit_full_axis_spectrum_centroid(
            [_row(1e-160)],
            _law(),
            [990, 1010],
            [1e-320, 1e-320],
            electron_count=1,
            integration_bins=32,
            max_evaluations=96,
        )
