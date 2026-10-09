"""Complete-band sums and bounded refinement of physical field certificates."""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_band_audit import (
    BandPowerBudgetError,
    BandPowerCertificate,
    certify_row_band_power,
)
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


def _row():
    return CoherentRowField(
        label="physical sinc",
        energy_eV=np.array([1000.0]),
        amplitude=np.ones((1, 1)),
        duration_ang=np.array([100.0]),
        centre_ang=np.array([50.0]),
        electron=np.array([0]),
        start_transmission=np.ones(1),
        end_transmission=np.ones(1),
        mean_transmission=np.ones(1),
        attenuation_slope_ang=np.zeros(1),
        escape_mid_ang=np.zeros(1),
        escape_change_ang=np.zeros(1),
        phase_rad=np.zeros(1),
    )


def _law():
    return CoherentDispersionLaw("hopg", 990.0, 1010.0, use_henke=False)


def test_full_band_refines_to_the_intrinsic_integral_share():
    from mpmath import mp

    result = certify_row_band_power(
        _row(),
        _law(),
        relative_tolerance=1e-3,
        max_evaluations=500,
        form_factor_bounds=(0.0, 0.0),
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        actual = mp.quad(
            lambda E: (100 * mp.sinc(100 * (E - 1000) / (2 * H))) ** 2, [990, 1000, 1010]
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)
        assert result.relative_width_upper <= 1e-3
        assert result.relative_error_upper(float(actual)) <= 1e-3
        assert result.relative_error_upper(float(actual) * 1.1) > 0.09
    assert result.evaluations <= 500
    assert result.intervals[0].samples > 3


def test_budget_refuses_instead_of_returning_an_unqualified_integral():
    with pytest.raises(BandPowerBudgetError, match="sample evaluations") as error:
        certify_row_band_power(_row(), _law(), relative_tolerance=1e-8, max_evaluations=3)
    assert error.value.certificate.evaluations == 3
    assert error.value.certificate.relative_width_upper > 1e-8


def test_disjoint_bands_split_at_material_knots_and_never_double_count():
    from mpmath import mp

    law = _law()
    law.breaks = np.array([990.0, 998.0, 1000.0, 1005.0, 1010.0])
    bands = [(990.0, 999.0), (1001.0, 1010.0)]
    result = certify_row_band_power(
        _row(), law, bands=bands, max_evaluations=1000, form_factor_bounds=(0.0, 0.0)
    )
    assert [(b.start_eV, b.stop_eV) for b in result.intervals] == [
        (990.0, 998.0),
        (998.0, 999.0),
        (1001.0, 1005.0),
        (1005.0, 1010.0),
    ]
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        actual = sum(
            mp.quad(lambda E: (100 * mp.sinc(100 * (E - 1000) / (2 * H))) ** 2, [a, b])
            for a, b in bands
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)


@pytest.mark.parametrize(
    "bands", [[(989.0, 1000.0)], [(999.0, 1001.0), (1000.0, 1002.0)], [(1001.0, 999.0)]]
)
def test_invalid_or_overlapping_bands_are_refused(bands):
    with pytest.raises(ValueError, match="bands"):
        certify_row_band_power(_row(), _law(), bands=bands)


def test_insufficient_initial_budget_reports_no_complete_certificate():
    law = _law()
    law.breaks = np.array([990.0, 1000.0, 1010.0])
    with pytest.raises(BandPowerBudgetError, match="initialization needs 6") as error:
        certify_row_band_power(_row(), law, max_evaluations=5)
    assert error.value.certificate is None


def test_empty_requested_bands_have_exact_zero_integral():
    result = certify_row_band_power(_row(), _law(), bands=[])
    assert (result.lower, result.upper, result.evaluations, result.intervals) == (0, 0, 0, ())
    assert result.relative_width_upper == 0
    assert result.relative_error_upper(0) == 0
    assert np.isinf(result.relative_error_upper(1))


def test_adjacent_float_band_cannot_claim_an_unsampled_certificate():
    with pytest.raises(ValueError, match="representable"):
        certify_row_band_power(_row(), _law(), bands=[(1000.0, np.nextafter(1000.0, np.inf))])


def test_relative_error_covers_every_yield_inside_a_subnormal_interval():
    from mpmath import mp

    lower, upper = 1e-320, 2e-320
    result = BandPowerCertificate(lower, upper, 0, ())
    with mp.workdps(80):
        assert mp.mpf(result.relative_width_upper) >= (mp.mpf(upper) - mp.mpf(lower)) / mp.mpf(
            lower
        )
        for estimate in (0.0, lower, 1.5e-320, upper, 4e-320):
            bound = result.relative_error_upper(estimate)
            for Y in (mp.mpf(lower), (mp.mpf(lower) + mp.mpf(upper)) / 2, mp.mpf(upper)):
                assert mp.mpf(bound) >= abs(mp.mpf(estimate) - Y) / Y
    no_floor = BandPowerCertificate(0.0, np.nextafter(0.0, np.inf), 0, ())
    assert no_floor.relative_error_upper(0.0) == 1.0
    assert np.isinf(no_floor.relative_error_upper(upper))


@pytest.mark.parametrize("estimate", [-1.0, np.inf, np.nan])
def test_invalid_numerical_yields_are_refused(estimate):
    with pytest.raises(ValueError, match="nonnegative"):
        BandPowerCertificate(1.0, 2.0, 0, ()).relative_error_upper(estimate)


@pytest.mark.parametrize("bands", [[(990.0, 1010.0)], [(990.0, 996.0), (1001.0, 1010.0)]])
def test_varying_factor_refines_bands_instead_of_resampling_a_fixed_uncertainty(monkeypatch, bands):
    from pyrite.montecarlo.spectrum import coherent_band_audit as audit

    # Exact constant sector powers G=1, C=2 give P(E)=1+3F(E).
    # More field samples cannot shrink a whole-interval F enclosure.
    calls = []

    def constant_sectors(field, law, lo, hi, nodes, *, form_factor_bounds):
        calls.append(nodes.size)
        f_lo, f_hi = form_factor_bounds
        return (hi - lo) * (1 + 3 * f_lo), (hi - lo) * (1 + 3 * f_hi)

    monkeypatch.setattr(audit, "_certified_interpolated_row_power_bounds", constant_sectors)
    law = _law()
    law.breaks = np.array([990.0, 993.0, 1010.0])
    result = certify_row_band_power(
        _row(),
        law,
        bands=bands,
        form_factor_bounds=lambda lo, hi: ((lo / 2000) ** 2, (hi / 2000) ** 2),
        relative_tolerance=0.002,
        max_evaluations=500,
    )
    actual = sum(hi - lo + (hi**3 - lo**3) / 2000**2 for lo, hi in bands)
    assert result.lower <= actual <= result.upper
    assert result.relative_width_upper <= 0.002
    assert result.evaluations == sum(calls)
    assert len(result.intervals) > len(bands) + 1
    assert sum(b.stop_eV - b.start_eV for b in result.intervals) == sum(hi - lo for lo, hi in bands)
    for interval in result.intervals:
        assert any(lo <= interval.start_eV < interval.stop_eV <= hi for lo, hi in bands)
        assert not interval.start_eV < 993.0 < interval.stop_eV


def test_band_split_budget_preserves_a_complete_parent_certificate(monkeypatch):
    from pyrite.montecarlo.spectrum import coherent_band_audit as audit

    monkeypatch.setattr(
        audit, "_certified_interpolated_row_power_bounds", lambda *args, **kwargs: (1.0, 2.0)
    )
    with pytest.raises(BandPowerBudgetError) as error:
        certify_row_band_power(
            _row(), _law(), max_evaluations=8, form_factor_bounds=lambda lo, hi: (0.0, 1.0)
        )
    certificate = error.value.certificate
    assert certificate is not None
    assert certificate.lower <= 1.0 < 2.0 <= certificate.upper
    assert certificate.evaluations == 3
    assert len(certificate.intervals) == 1


def test_looser_child_enclosures_retain_the_valid_whole_band_intersection(monkeypatch):
    from pyrite.montecarlo.spectrum import coherent_band_audit as audit

    def constant_power(field, law, lo, hi, nodes, **kwargs):
        # Actual P=1: a narrower domain can still have a looser certificate.
        return (19.0, 21.0) if hi - lo == 20.0 else (0.0, 20.0)

    monkeypatch.setattr(audit, "_certified_interpolated_row_power_bounds", constant_power)
    with pytest.raises(BandPowerBudgetError) as error:
        certify_row_band_power(
            _row(), _law(), max_evaluations=9, form_factor_bounds=lambda lo, hi: (0.0, 1.0)
        )
    certificate = error.value.certificate
    assert certificate.lower <= 20.0 <= certificate.upper
    assert certificate.lower >= np.nextafter(19.0, -np.inf)
    assert certificate.upper <= np.nextafter(21.0, np.inf)
    assert certificate.evaluations == 9
    assert [(b.start_eV, b.stop_eV) for b in certificate.intervals] == [
        (990.0, 1000.0),
        (1000.0, 1010.0),
    ]


def _pair_row(opposed=False):
    from dataclasses import replace

    one = _row()
    return replace(
        one,
        amplitude=np.array([[1.0, -1.0 if opposed else 1.0]]),
        energy_eV=np.repeat(one.energy_eV, 2),
        duration_ang=np.repeat(one.duration_ang, 2),
        centre_ang=np.repeat(one.centre_ang, 2),
        electron=np.array([0, 1]),
        start_transmission=np.ones(2),
        end_transmission=np.ones(2),
        mean_transmission=np.ones(2),
        attenuation_slope_ang=np.zeros(2),
        escape_mid_ang=np.zeros(2),
        escape_change_ang=np.zeros(2),
        phase_rad=np.zeros(2),
    )


def test_physical_pair_scale_above_one_encloses_the_analytic_aligned_integral():
    from mpmath import mp

    certificate = certify_row_band_power(
        _pair_row(),
        _law(),
        form_factor_bounds=(11.0, 11.0),
        max_evaluations=2000,
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        expected = 24 * mp.quad(
            lambda E: (100 * mp.sinc(100 * (E - 1000) / (2 * H))) ** 2, [990, 1000, 1010]
        )
        assert mp.mpf(certificate.lower) <= expected <= mp.mpf(certificate.upper)
    assert certificate.relative_width_upper <= 1e-3
    assert certificate.evaluations <= 2000
    assert certificate.evaluations % 2 == 0


def test_negative_pair_integral_is_preserved_and_never_certified_as_zero():
    with pytest.raises(BandPowerBudgetError) as caught:
        certify_row_band_power(
            _pair_row(opposed=True),
            _law(),
            form_factor_bounds=(11.0, 11.0),
            max_evaluations=100,
        )
    certificate = caught.value.certificate
    assert certificate.lower < certificate.upper < 0
    assert certificate.relative_width_upper == float("inf")
    assert certificate.evaluations <= 100


def test_signed_pair_initialization_budgets_both_sector_evaluations():
    with pytest.raises(BandPowerBudgetError, match="needs 6 sample evaluations") as caught:
        certify_row_band_power(
            _pair_row(),
            _law(),
            form_factor_bounds=(11.0, 11.0),
            max_evaluations=5,
        )
    assert caught.value.certificate is None


@pytest.mark.slow
def test_varying_pair_weight_across_one_keeps_signed_contributions():
    from mpmath import mp

    def bounds(a, b):
        return 1 + 1e-4 * (a - 1000), 1 + 1e-4 * (b - 1000)

    result = certify_row_band_power(
        _pair_row(),
        _law(),
        form_factor_bounds=bounds,
        max_evaluations=3000,
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        # Odd weight variation integrates to zero on the symmetric sinc field.
        expected = 4 * mp.quad(
            lambda E: (100 * mp.sinc(100 * (E - 1000) / (2 * H))) ** 2, [990, 1000, 1010]
        )
        assert mp.mpf(result.lower) <= expected <= mp.mpf(result.upper)
    assert result.relative_width_upper <= 1e-3
    assert result.evaluations <= 3000


def test_full_axis_signed_yield_passes_but_centroid_requires_positive_weights():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        audit_full_axis_spectrum_centroid,
        audit_full_axis_spectrum_yield,
    )

    law = CoherentDispersionLaw("hopg", 999.0, 1001.0, use_henke=False)
    energy = np.linspace(law.start, law.stop, 201)
    source = 12 * (100 * np.sinc(100 * (energy - 1000) / (2 * np.pi * HBARC_EV_ANG))) ** 2
    options = dict(electron_count=2, form_factor_bounds=(11.0, 11.0), max_evaluations=3000)
    result = audit_full_axis_spectrum_yield([_pair_row()], law, energy, source, **options)
    assert result.within_tolerance
    assert result.spectrum.evaluations <= 3000
    with pytest.raises(ValueError, match="centroid audit requires convex pair weights"):
        audit_full_axis_spectrum_centroid([_pair_row()], law, energy, source, **options)


def test_signed_zero_field_stays_exact_zero():
    from dataclasses import replace

    row = replace(_pair_row(), amplitude=np.zeros((1, 2)))
    result = certify_row_band_power(row, _law(), form_factor_bounds=(11.0, 11.0), max_evaluations=6)
    assert result.lower == result.upper == 0
    assert result.relative_width_upper == result.relative_error_upper(0) == 0
    assert result.evaluations == 6


def test_weighted_spectrum_reserves_both_sectors_for_every_row():
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import (
        SpectrumPowerBudgetError,
        audit_captured_spectrum_yield,
    )

    with pytest.raises(SpectrumPowerBudgetError, match="needs 12 sample evaluations"):
        audit_captured_spectrum_yield(
            [_pair_row(), _pair_row()],
            _law(),
            electron_count=2,
            numerical_yield=1.0,
            form_factor_bounds=(11.0, 11.0),
            max_evaluations=11,
        )
