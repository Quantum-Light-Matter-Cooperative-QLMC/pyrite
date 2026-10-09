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
        actual = mp.mpf(weight) / 3 * mp.quad(
            lambda E: mp.sinc((E - 1000) / (2 * H)) ** 2, [990, 1010]
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)
