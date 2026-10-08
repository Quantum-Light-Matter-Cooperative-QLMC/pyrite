"""Physical finite-band normalization lower bounds (#350)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum import coherent_windows as cw
from pyrite.montecarlo.spectrum.coherent_dispersion import DispersionCertificate
from pyrite.montecarlo.spectrum.coherent_normalization import (
    _physical_row_power_bounds,
    _physical_row_power_lower,
    _sample_row_fields,
    _sampled_power_bounds,
    _triangle_power_integral,
)


def _row():
    duration = np.array([100.0, 120.0])
    centre = np.array([50.0, 160.0])
    return cw.CoherentRowField(
        label="two physical pieces",
        energy_eV=np.full(2, 1000.0),
        amplitude=np.array([[1.0, 0.4j]]),
        duration_ang=duration,
        centre_ang=centre,
        electron=np.array([0, 1]),
        start_transmission=np.ones(2),
        end_transmission=np.ones(2),
        mean_transmission=np.ones(2),
        escape_mid_ang=np.array([100.0, 300.0]),
        escape_change_ang=np.array([20.0, -30.0]),
        phase_rad=1000.0 * centre / HBARC_EV_ANG,
    )


def test_reverse_triangle_integral_has_the_exact_compact_triangle_limit():
    # A=2, B=1: integral of (2-|E|)^2 over its support is 16/3.
    assert _triangle_power_integral(2.0, 1.0, 0.0, -10.0, 10.0) == pytest.approx(16.0 / 3.0)
    assert _triangle_power_integral(2.0, 0.0, 0.0, -10.0, 10.0) == 80.0
    assert _triangle_power_integral(0.0, 1.0, 0.0, -10.0, 10.0) == 0.0
    assert _triangle_power_integral(2.0, 1.0, 0.0, -1.0, 0.0) == pytest.approx(7.0 / 3.0)
    with pytest.raises(ValueError, match="in-band sample"):
        _triangle_power_integral(2.0, 1.0, 11.0, -10.0, 10.0)


@pytest.mark.parametrize("F", [0.0, 0.4, 1.0, None])
def test_physical_lower_bound_stays_below_an_independent_mixed_sinc_integral(F):
    row = _row()
    slope, intercept = 1e-7, 2e-4

    def law(E):
        return slope * np.asarray(E) + intercept

    certificate = DispersionCertificate(900.0, 1100.0, slope, intercept, 0.0, abs(slope))
    bounds = (0.0, 1.0) if F is None else (F, F)
    lower = _physical_row_power_lower(
        row, law, certificate, 1000.0, sample_norm_errors=(1e-9, 1e-9), form_factor_bounds=bounds
    )
    energies = np.linspace(900.0, 1100.0, 20001)
    dw = law(energies)
    v = (
        row.duration_ang[:, None] * (energies - row.energy_eV[:, None]) / (2 * HBARC_EV_ANG)
        - row.escape_change_ang[:, None] * dw / 2
    )
    phase = (
        row.centre_ang[:, None] * energies / HBARC_EV_ANG
        - row.phase_rad[:, None]
        - row.escape_mid_ang[:, None] * dw
    )
    pieces = row.amplitude[0, :, None] * row.duration_ang[:, None] * np.sinc(v / np.pi)
    pieces *= np.exp(1j * phase)
    factor = (energies - 900.0) / 200.0 if F is None else F
    power = (1 - factor) * np.sum(np.abs(pieces) ** 2, axis=0) + factor * np.abs(
        pieces.sum(axis=0)
    ) ** 2
    actual = np.trapezoid(power, energies)
    assert 0.0 < lower < actual
    nodes = np.linspace(900.0, 1100.0, 33)
    band_lower, band_upper = _physical_row_power_bounds(
        row,
        law,
        certificate,
        nodes,
        sample_norm_errors=np.full((nodes.size, 2), 1e-9),
        form_factor_bounds=bounds,
    )
    assert 0.0 < band_lower <= actual <= band_upper


def test_nominal_row_sample_reproduces_the_complete_affine_midpoint_phase():
    row = _row()
    slope, intercept = 1e-7, 0.03

    def law(E):
        return slope * np.asarray(E) + intercept

    mapped = cw._affine_dispersion_row(row, slope, intercept)
    energies = np.linspace(950.0, 1050.0, 101)
    actual = _sample_row_fields(row, law, energies)
    affine = _sample_row_fields(mapped, lambda E: np.zeros_like(E), energies)
    np.testing.assert_allclose(affine, actual, rtol=2e-12, atol=2e-11)
    # Omitting b Lmid changes the inter-electron interference, even though
    # every piece's self-term is unchanged.
    incomplete = replace(mapped, phase_rad=row.phase_rad)
    wrong = _sample_row_fields(incomplete, lambda E: np.zeros_like(E), energies)
    assert np.max(np.abs(np.abs(wrong.sum(axis=1)) ** 2 - np.abs(actual.sum(axis=1)) ** 2)) > 1.0


def test_lower_bound_is_invariant_to_a_common_retardation_translation():
    row = _row()
    certificate = DispersionCertificate(900.0, 1100.0, 0.0, 0.0, 0.0, 0.0)

    def law(E):
        return np.zeros_like(E)

    kwargs = dict(sample_norm_errors=(1e-9, 1e-9), form_factor_bounds=(1.0, 1.0))
    original = _physical_row_power_lower(row, law, certificate, 1000.0, **kwargs)
    translated = _physical_row_power_lower(
        replace(row, centre_ang=row.centre_ang + 1e7), law, certificate, 1000.0, **kwargs
    )
    assert translated == pytest.approx(original, rel=1e-13)


def test_uncertified_samples_cannot_supply_a_positive_normalization():
    row = _row()

    def law(E):
        return np.zeros_like(E)

    certificate = DispersionCertificate(900.0, 1100.0, 0.0, 0.0, 0.0, 0.0)
    # An uncertainty covering the whole sample gives no positive claim.
    assert (
        _physical_row_power_lower(row, law, certificate, 1000.0, sample_norm_errors=(1e9, 1e9))
        == 0.0
    )
    with pytest.raises(ValueError, match="captured midpoint"):
        _sample_row_fields(replace(row, phase_rad=None), law, np.array([1000.0]))
    with pytest.raises(ValueError, match="certified nonnegative sample errors"):
        _physical_row_power_lower(row, law, certificate, 1000.0, sample_norm_errors=(-1.0, 0.0))


@pytest.mark.parametrize("blend", [0.0, 0.4, 1.0, None])
def test_sampled_power_bounds_enclose_a_nonlinear_interference_integral(blend):
    from scipy.integrate import quad

    # Two independent electron fields: 1 and 0.6 exp(i pi E**2).
    # Per-electron gauges make the grouped norm constant. The row-wide
    # coherent derivative is bounded by 1.2 pi on [0, 1].
    def factor(E):
        return 0.5 + 0.4 * np.sin(7 * E) if blend is None else blend

    def intensity(E):
        return 1.36 + factor(E) * 1.2 * np.cos(np.pi * E**2)

    actual = quad(intensity, 0.0, 1.0, epsabs=1e-12)[0]
    widths = []
    for count in (9, 33, 129):
        energies = np.linspace(0.0, 1.0, count)
        norms = np.column_stack(
            (np.full(count, np.sqrt(1.36)), np.abs(1 + 0.6 * np.exp(1j * np.pi * energies**2)))
        )
        edges = np.r_[0.0, 0.5 * (energies[:-1] + energies[1:]), 1.0]
        # Certified cell-specific F enclosures from |F'| <= 2.8, including
        # cells crossing extrema; no endpoint-monotonicity assumption.
        radii = np.maximum(energies - edges[:-1], edges[1:] - energies)
        factors = np.asarray([factor(E) for E in energies])
        drift = 2.8 * radii if blend is None else np.zeros(count)
        bounds = np.column_stack((np.maximum(0, factors - drift), np.minimum(1, factors + drift)))
        lower, upper = _sampled_power_bounds(
            energies,
            norms,
            sample_norm_errors=np.full((count, 2), 1e-10),
            derivative_norm_upper=(0.0, 1.2 * np.pi),
            start_eV=0.0,
            stop_eV=1.0,
            form_factor_bounds=bounds,
        )
        assert lower <= actual <= upper
        # Any quadrature estimate Q has |Q-Y| <= max(|Q-L|, |Q-U|).
        estimate = np.trapezoid([intensity(E) for E in energies], energies)
        assert abs(estimate - actual) <= max(abs(estimate - lower), abs(estimate - upper))
        widths.append(upper - lower)
    if blend == 0.0:
        assert widths[-1] < 1e-8
    else:
        assert widths[-1] < widths[0] / 8


def test_sampled_power_bounds_retain_uncertainty_and_cover_the_whole_band():
    # A constant single field, sampled away from both band endpoints.
    lower, upper = _sampled_power_bounds(
        np.array([0.25, 0.75]),
        np.full((2, 2), 2.0),
        sample_norm_errors=np.full((2, 2), 0.1),
        derivative_norm_upper=(0.0, 0.0),
        start_eV=0.0,
        stop_eV=1.0,
    )
    assert lower == pytest.approx(1.9**2)
    assert upper == pytest.approx(2.1**2)


@pytest.mark.parametrize("energies", [[0.7, 0.3], [0.3, 0.3], [-0.1, 0.5], []])
def test_sampled_power_bounds_refuse_invalid_sample_partitions(energies):
    count = len(energies)
    with pytest.raises(ValueError, match="increasing in-band"):
        _sampled_power_bounds(
            np.array(energies),
            np.ones((count, 2)),
            sample_norm_errors=np.zeros((count, 2)),
            derivative_norm_upper=(1.0, 1.0),
            start_eV=0.0,
            stop_eV=1.0,
        )


@pytest.mark.parametrize("invalid", ["norms", "errors", "derivative", "factors"])
def test_sampled_power_bounds_refuse_nonfinite_certificate_inputs(invalid):
    norms = np.ones((1, 2))
    errors = np.zeros((1, 2))
    derivative = np.ones(2)
    factors = np.array([0.0, 1.0])
    {"norms": norms, "errors": errors, "derivative": derivative, "factors": factors}[invalid].flat[
        0
    ] = np.nan
    with pytest.raises(ValueError):
        _sampled_power_bounds(
            np.array([0.5]),
            norms,
            sample_norm_errors=errors,
            derivative_norm_upper=derivative,
            start_eV=0.0,
            stop_eV=1.0,
            form_factor_bounds=factors,
        )
