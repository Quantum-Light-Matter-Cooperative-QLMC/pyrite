"""Charge-jitter moments and likelihood of averaged Gaussian CTR scans (#617)."""

import numpy as np
import pytest

from pyrite.detectors.optical_ctr_inference import (
    ChargeJitterModel,
    GaussianCTRTemplates,
    OpticalVoltageScan,
    _response_kernel,
    fit_gaussian_ctr,
)
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import IdealMartinPuplett

_HBAR_EV_FS = 0.6582119569509067


def _setup(noise=1.5e-5):
    e = np.linspace(0.0005, 0.025, 180)
    templates = GaussianCTRTemplates(
        e,
        4.0e9 * np.ones_like(e),
        2.0e10 * (0.7 + 0.3 * e / e[-1]),
        np.pi / 4,
        "synthetic mean bunch",
    )
    detector = CalibratedPulseEnergyDetector(
        [0.0001, 0.03], [0.72, 0.72], 1e12, "synthetic absorption", noise_rms_v=noise
    )
    instrument = IdealMartinPuplett("ideal regression")
    shifts = np.linspace(-0.0012, 0.0012, 81)
    return templates, instrument, (detector, detector), shifts


def _mean_bunch_voltages(templates, instrument, detectors, shifts, sigma):
    kernel = _response_kernel(templates, shifts, instrument, detectors, 4_000_000)
    temporal = np.exp(-np.square(templates.energy_eV * sigma / _HBAR_EV_FS))
    return (
        kernel @ templates.incoherent_photons_per_eV_per_pulse,
        kernel @ (templates.coherent_envelope_photons_per_eV_per_pulse * temporal),
    )


def _shots(rng, a, c, n_mean, r, shots, trials):
    """Per-trial shot means of raw and charge-normalized port voltages."""
    n = np.maximum(np.rint(rng.normal(n_mean, r * n_mean, (trials, shots))), 2.0)
    q = n / n_mean
    q2 = n * (n - 1) / (n_mean * (n_mean - 1))
    raw = a[None, None, :] * q[..., None] + c[None, None, :] * q2[..., None]
    return raw.mean(axis=1), (raw / q[..., None]).mean(axis=1)


@pytest.mark.parametrize("n_mean", [40.0, 6.0e6])
def test_port_moments_match_direct_shot_simulation(n_mean):
    rng = np.random.default_rng(617)
    a, c = np.array([0.3, 0.5]), np.array([1.1, 0.2])
    r, shots, trials = 0.08, 16, 40_000
    raw, normalized = _shots(rng, a, c, n_mean, r, shots, trials)
    for normalization, sample in (("none", raw), ("per-shot-charge", normalized)):
        model = ChargeJitterModel(n_mean, r, shots, normalization)
        mean = a + model.pair_mean_factor * c
        cov = model.port_covariance(a[None], c[None])[0]
        # Monte Carlo error: mean ~ sqrt(cov / trials); covariance ~ sqrt(2 / trials).
        np.testing.assert_allclose(sample.mean(axis=0), mean, atol=5 * np.sqrt(cov.max() / trials))
        np.testing.assert_allclose(np.cov(sample.T), cov, rtol=0.04, atol=0.04 * cov.max())
    assert ChargeJitterModel(n_mean, r, shots, "per-shot-charge").pair_mean_factor == 1.0
    assert ChargeJitterModel(n_mean, r, shots).pair_mean_factor == pytest.approx(
        1 + n_mean * r**2 / (n_mean - 1)
    )


def test_jitter_covariance_limits_and_rank():
    a, c = np.array([[0.3, 0.5]]), np.array([[1.1, 0.2]])
    assert np.all(ChargeJitterModel(1e6, 0.0, 4).port_covariance(a, c) == 0)
    one = ChargeJitterModel(1e6, 0.05, 1, "per-shot-charge").port_covariance(a, c)[0]
    many = ChargeJitterModel(1e6, 0.05, 25, "per-shot-charge").port_covariance(a, c)[0]
    np.testing.assert_allclose(many, one / 25)
    # One common random variable drives both ports: rank one, full correlation.
    assert abs(np.linalg.det(one)) < 1e-12 * one[0, 0] * one[1, 1]
    # No coherent part: raw jitter is the linear charge term only.
    raw = ChargeJitterModel(1e6, 0.05, 1).port_covariance(a, np.zeros_like(a))[0]
    np.testing.assert_allclose(raw, np.outer(a[0], a[0]) * 0.05**2)
    for bad in (dict(charge_rms_fraction=1.0), dict(shots_per_point=0), dict(mean_electrons=1.0)):
        with pytest.raises(ValueError):
            ChargeJitterModel(
                **{**dict(mean_electrons=1e6, charge_rms_fraction=0.1, shots_per_point=4), **bad}
            )
    with pytest.raises(ValueError, match="normalization"):
        ChargeJitterModel(1e6, 0.1, 4, "per-shot-q2")


def _scan(rng, sigma, n_mean, r, shots, normalization, noise=1.5e-5):
    templates, instrument, detectors, shifts = _setup(noise)
    a, c = _mean_bunch_voltages(templates, instrument, detectors, shifts, sigma)
    values = np.empty_like(a)
    for i in range(shifts.size):  # independent shots at every mirror position
        raw, normalized = _shots(rng, a[i], c[i], n_mean, r, shots, 1)
        values[i] = (raw if normalization == "none" else normalized)[0]
    values += rng.normal(0.0, noise, values.shape)
    scan = OpticalVoltageScan(shifts, values, np.full(values.shape, noise), "synthetic shots")
    return scan, templates, instrument, detectors


def test_zero_jitter_reproduces_the_chi_square_fit():
    rng = np.random.default_rng(1)
    scan, templates, instrument, detectors = _scan(rng, 200.0, 6e6, 0.0, 1, "none")
    common = dict(
        instrument=instrument, detectors=detectors, sigma_bounds_fs=(20, 3000), profile_points=41
    )
    plain = fit_gaussian_ctr(scan, templates, **common)
    jitter = fit_gaussian_ctr(
        scan, templates, charge_jitter=ChargeJitterModel(6e6, 0.0, 1), **common
    )
    assert jitter.rms_duration_fs == pytest.approx(plain.rms_duration_fs, rel=1e-6)
    assert jitter.chi_square == pytest.approx(plain.chi_square, rel=1e-6)
    assert jitter.relative_source_scale == pytest.approx(plain.relative_source_scale, rel=1e-6)
    np.testing.assert_allclose(
        jitter.confidence_interval_95_fs, plain.confidence_interval_95_fs, rtol=1e-5
    )
    assert plain.metadata["charge_jitter"] is None


@pytest.mark.parametrize("normalization", ["none", "per-shot-charge"])
def test_jitter_likelihood_recovers_duration_with_honest_interval(normalization):
    rng = np.random.default_rng(20261109)
    n_mean, r, shots = 6.0e6, 0.10, 8
    scan, templates, instrument, detectors = _scan(rng, 200.0, n_mean, r, shots, normalization)
    common = dict(
        instrument=instrument, detectors=detectors, sigma_bounds_fs=(20, 3000),
        profile_points=41, fit_scale=False,
    )  # fmt: skip
    model = ChargeJitterModel(n_mean, r, shots, normalization)
    fit = fit_gaussian_ctr(scan, templates, charge_jitter=model, **common)
    lo, hi = fit.confidence_interval_95_fs
    assert fit.is_identified_95 and lo < 200.0 < hi
    # The quadratic form is a chi-square with ~ data-count degrees of freedom.
    dof = fit.degrees_of_freedom
    assert abs(fit.chi_square - dof) < 5 * np.sqrt(2 * dof)
    assert fit.metadata["charge_jitter"]["normalization"] == normalization
    # Ignoring the jitter covariance claims a far tighter interval than the data support.
    naive = fit_gaussian_ctr(scan, templates, **common)
    assert naive.chi_square > 20 * naive.degrees_of_freedom
    n_lo, n_hi = naive.confidence_interval_95_fs
    assert (n_hi - n_lo) < 0.3 * (hi - lo)


def test_refuses_templates_with_an_existing_pair_factor():
    rng = np.random.default_rng(3)
    scan, templates, instrument, detectors = _scan(rng, 200.0, 6e6, 0.05, 4, "none")
    doubled = templates.with_charge_jitter(mean_electrons=6e6, charge_rms_fraction=0.05)
    with pytest.raises(ValueError, match="already carry"):
        fit_gaussian_ctr(
            scan, doubled, instrument=instrument, detectors=detectors,
            charge_jitter=ChargeJitterModel(6e6, 0.05, 4),
        )  # fmt: skip
    with pytest.raises(TypeError, match="ChargeJitterModel"):
        fit_gaussian_ctr(
            scan, templates, instrument=instrument, detectors=detectors, charge_jitter=0.05
        )
