"""Minimal-phase (Kramers--Kronig) CTR profile reconstruction (#617).

Validation: optical-phase-retrieval
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.detectors.optical_phase_retrieval import (
    FormFactorUncertainty,
    MissingBandModel,
    minimal_phase,
    propagate_reconstruction_uncertainty,
    reconstruct_minimal_phase_profile,
)

HBAR = 0.6582119569509067  # eV fs


def _gamma2(t, tau):
    """t exp(-t/tau)/tau^2 for t > 0; f = 1/(1 - i omega tau)^2 (minimal phase)."""
    tp = np.clip(t, 0.0, None)
    return np.where(t > 0, tp * np.exp(-tp / tau) / tau**2, 0.0)


def test_exact_minimal_phase_profile_round_trip():
    tau = 100.0
    w0 = HBAR / tau
    e = np.geomspace(0.01 * w0, 100 * w0, 800)
    t = np.linspace(-200.0, 1500.0, 400)
    r = reconstruct_minimal_phase_profile(
        e,
        1 / (1 + (e / w0) ** 2) ** 2,
        t,
        missing_band=MissingBandModel(high_power_exponent=2.0),
        provenance="analytic gamma-2",
    )
    # Eq. (6) for |f| = 1/(1+x^2): psi = 2 arctan(x) exactly.
    assert np.max(np.abs(r.phase_rad[r.measured] - 2 * np.arctan(e / w0))) < 2e-4
    exact = _gamma2(t, tau)
    assert np.max(np.abs(r.profile_per_fs - exact)) < 2e-3 * exact.max()
    assert r.window_area == pytest.approx(1.0, abs=2e-3)
    assert r.rms_duration_fs == pytest.approx(np.sqrt(2) * tau, rel=2e-3)
    assert not r.is_unique and r.spectral_minima_eV == ()


def _piecewise_log_f(lo, hi, p, inner):
    def lnf(x):
        if x < lo:
            return inner(lo) * (x / lo) ** 2
        if x <= hi:
            return inner(x)
        return inner(hi) - p * np.log(x / hi)

    return lnf


def test_gaussian_phase_matches_independent_cauchy_quadrature():
    """Gaussian band: Eq. (6) against scipy's Cauchy-weight PV quadrature."""
    sigma, p = 200.0, 3.0
    e = np.linspace(0.2 * HBAR / sigma, 3 * HBAR / sigma, 300)

    def inner(x):
        return -0.5 * (x * sigma / HBAR) ** 2

    model = MissingBandModel(high_power_exponent=p)
    r = reconstruct_minimal_phase_profile(
        e, np.exp(2 * inner(e)), np.linspace(-600, 1800, 300), missing_band=model, provenance="g"
    )
    lnf = _piecewise_log_f(e[0], e[-1], p, inner)
    cut = 50 * r.energy_eV[-1]
    for w in e[::50]:
        level = lnf(w)
        pv = quad(
            lambda x, w=w, c=level: (lnf(x) - c) / (x + w),
            0,
            cut,
            weight="cauchy",
            wvar=w,
            limit=400,
        )[0]
        tail = quad(lambda x, w=w, c=level: (lnf(x) - c) / (x * x - w * w), cut, np.inf, limit=200)[
            0
        ]
        psi = -(2 * w / np.pi) * (pv + tail)
        assert r.phase_rad[np.flatnonzero(r.energy_eV == w)[0]] == pytest.approx(psi, abs=2e-4)
    # The minimal-phase partner of a symmetric Gaussian is a nonnegative,
    # skewed profile of similar width -- not the Gaussian itself.
    assert r.negative_lobe_fraction < 1e-3
    assert r.rms_duration_fs == pytest.approx(sigma, rel=0.02)
    t, rho = r.time_fs, r.profile_per_fs
    mean = np.sum(t * rho) / np.sum(rho)
    skew = np.sum((t - mean) ** 3 * rho) / np.sum(rho) / r.rms_duration_fs**3
    assert abs(skew) > 0.1


def test_reconstructed_profile_reproduces_measured_magnitude():
    sigma = 200.0
    e = np.linspace(0.2 * HBAR / sigma, 3 * HBAR / sigma, 200)
    f = np.exp(-0.5 * (e * sigma / HBAR) ** 2)
    t = np.linspace(-1500, 3000, 2400)
    r = reconstruct_minimal_phase_profile(
        e, f**2, t, missing_band=MissingBandModel(high_power_exponent=3.0), provenance="g"
    )
    ft = np.trapezoid(r.profile_per_fs[None, :] * np.exp(1j * np.outer(e / HBAR, t)), t, axis=1)
    assert np.max(np.abs(np.abs(ft) - f)) < 5e-3


def _two_bunch(e, t, a, tau, sep):
    w = e / HBAR
    h = 1 / (1 - 1j * w * tau) ** 2
    f = (1 + a * np.exp(1j * w * sep)) * h / (1 + a)
    rho = (_gamma2(t, tau) + a * _gamma2(t - sep, tau)) / (1 + a)
    return np.abs(f) ** 2, rho


def test_non_minimal_phase_profile_is_returned_as_its_minimal_partner():
    """Trailing bunch twice the leading: zeros in the upper half plane."""
    tau, sep, a = 50.0, 300.0, 2.0
    e = np.linspace(0.01 * HBAR / tau, 30 * HBAR / tau, 3000)
    t = np.linspace(-200.0, 900.0, 500)
    f2, rho_true = _two_bunch(e, t, a, tau, sep)
    model = MissingBandModel(high_power_exponent=2.0)
    r = reconstruct_minimal_phase_profile(e, f2, t, missing_band=model, provenance="2 bunch")
    _, rho_min = _two_bunch(e, t, 1 / a, tau, sep)  # same |f|, leading bunch larger
    peak = rho_true.max()
    assert np.max(np.abs(r.profile_per_fs - rho_min)) < 0.02 * peak
    assert np.max(np.abs(r.profile_per_fs - rho_true)) > 0.3 * peak
    # The ambiguity is reported, with the modulation minima at (2k+1) pi hbar / T.
    assert not r.is_unique
    assert any("zero flipping" in text for text in r.ambiguities)
    minima = np.array(r.spectral_minima_eV)
    assert minima.size >= 3
    assert minima[0] == pytest.approx(np.pi * HBAR / sep, abs=2 * (e[1] - e[0]))
    order = np.rint((minima * sep / (np.pi * HBAR) - 1) / 2)
    expected = (2 * order + 1) * np.pi * HBAR / sep
    assert np.array_equal(order, np.arange(minima.size))  # none skipped
    assert np.allclose(minima, expected, atol=2 * (e[1] - e[0]))
    t_rev, rho_rev = r.time_reversed()
    assert np.allclose(t_rev, -t[::-1]) and np.allclose(rho_rev, r.profile_per_fs[::-1])


def test_minimal_phase_two_bunch_is_recovered_itself():
    tau, sep = 50.0, 300.0
    e = np.linspace(0.01 * HBAR / tau, 30 * HBAR / tau, 3000)
    t = np.linspace(-200.0, 900.0, 500)
    f2, rho_true = _two_bunch(e, t, 0.5, tau, sep)
    r = reconstruct_minimal_phase_profile(
        e, f2, t, missing_band=MissingBandModel(high_power_exponent=2.0), provenance="2 bunch"
    )
    assert np.max(np.abs(r.profile_per_fs - rho_true)) < 0.02 * rho_true.max()


def test_missing_band_continuation_changes_and_is_recorded():
    sigma = 200.0
    e = np.linspace(0.6 * HBAR / sigma, 3 * HBAR / sigma, 200)  # |f(E_lo)|^2 = 0.70
    f2 = np.exp(-((e * sigma / HBAR) ** 2))
    t = np.linspace(-1500, 3000, 600)
    results = {
        low: reconstruct_minimal_phase_profile(
            e,
            f2,
            t,
            missing_band=MissingBandModel(low=low, high_power_exponent=3.0),
            provenance="g",
        )
        for low in ("gaussian-to-unity", "constant")
    }
    unity, constant = results["gaussian-to-unity"], results["constant"]
    assert unity.missing_band["low"] == "gaussian-to-unity"
    assert constant.missing_band["low"] == "constant"
    assert unity.window_area == pytest.approx(1.0, abs=5e-3)
    assert constant.window_area == pytest.approx(np.sqrt(f2[0]), abs=5e-3)
    assert unity.rms_duration_fs == pytest.approx(sigma, rel=0.03)
    assert abs(constant.rms_duration_fs - unity.rms_duration_fs) > 0.1 * sigma
    assert unity.extrapolated_power_fraction > 0.1
    steep = reconstruct_minimal_phase_profile(
        e, f2, t, missing_band=MissingBandModel(high_power_exponent=6.0), provenance="g"
    )
    assert np.max(np.abs(steep.phase_rad[steep.measured] - unity.phase_rad[unity.measured])) > 0.05


def test_power_law_tail_integral_is_exact_for_flat_band():
    """|f| = 1 to X then X^p/E^p: Eq. (6) tail in closed form, w = X/2."""
    p = 2.0
    e = np.linspace(0.0, 1.0, 2001)
    phase = minimal_phase(e, np.zeros_like(e), high_power_exponent=p)
    w = 0.5
    expected = -(2 * w / np.pi) * quad(lambda x: -p * np.log(x) / (x * x - w * w), 1.0, np.inf)[0]
    assert phase[1000] == pytest.approx(expected, rel=1e-9)
    assert phase[0] == 0.0


def _uncertainty_case():
    sigma = 200.0  # |f(E_lo)|^2 = 0.53: no 1/q^2 draw reaches the |f| <= 1 cap
    e = np.linspace(0.8 * HBAR / sigma, 3 * HBAR / sigma, 120)
    return e, np.exp(-((e * sigma / HBAR) ** 2)), np.linspace(-1500, 3000, 300)


def test_charge_uncertainty_only_enters_through_unity_normalization():
    e, f2, t = _uncertainty_case()
    charge = FormFactorUncertainty(charge_rms_fraction=0.05)
    kwargs = dict(uncertainty=charge, provenance="g", draws=40, seed=3)
    constant = propagate_reconstruction_uncertainty(
        e, f2, t, missing_band=MissingBandModel(low="constant", high_power_exponent=3.0), **kwargs
    )
    unity = propagate_reconstruction_uncertainty(
        e, f2, t, missing_band=MissingBandModel(high_power_exponent=3.0), **kwargs
    )
    # The minimal phase is invariant to a common scale of |f|: every draw is
    # the nominal profile times 1/q^2, so std/mean is the same at every t.
    rho = constant.nominal.profile_per_fs
    big = np.abs(rho) > 0.05 * rho.max()
    ratio = constant.profile_std_per_fs[big] / np.abs(constant.profile_mean_per_fs[big])
    assert constant.clipped_draws == 0
    assert np.ptp(ratio) < 1e-9 and ratio[0] > 0.05
    assert unity.undefined_rms_draws == 0
    assert np.std(unity.rms_duration_draws_fs) > 1e-3 * unity.nominal.rms_duration_fs
    assert unity.uncertainty["charge_rms_fraction"] == 0.05


def test_response_and_noise_uncertainty_spread_and_reproducibility():
    e, f2, t = _uncertainty_case()
    tilt = 0.1 * (e - e.mean()) / np.ptp(e)
    unc = FormFactorUncertainty(
        form_factor_squared_rms=0.01 * f2,
        transverse_rms_fraction=0.02,
        response_rms_fraction=tilt,
    )
    model = MissingBandModel(high_power_exponent=3.0)
    run = propagate_reconstruction_uncertainty(
        e, f2, t, missing_band=model, uncertainty=unc, provenance="g", draws=40, seed=11
    )
    again = propagate_reconstruction_uncertainty(
        e, f2, t, missing_band=model, uncertainty=unc, provenance="g", draws=40, seed=11
    )
    assert np.array_equal(run.rms_duration_draws_fs, again.rms_duration_draws_fs)
    lo, hi = run.rms_duration_interval_95_fs
    assert lo < run.nominal.rms_duration_fs < hi and hi - lo > 1.0
    assert np.max(run.profile_std_per_fs) > 0
    peak = run.nominal.profile_per_fs.max()
    assert np.max(np.abs(run.profile_mean_per_fs - run.nominal.profile_per_fs)) < 0.1 * peak


@pytest.mark.parametrize(
    "energy, f2, model, message",
    [
        ([1, 2, 3, 2.5], [0.5] * 4, MissingBandModel(), "strictly increasing"),
        ([1, 2, 3, 4], [0.5, 0, 0.5, 0.5], MissingBandModel(), "positive"),
        ([1, 2, 3, 4], [1.2, 0.5, 0.4, 0.3], MissingBandModel(), r"\|f\(E_lo\)\| <= 1"),
    ],
)
def test_invalid_inputs_are_refused(energy, f2, model, message):
    with pytest.raises(ValueError, match=message):
        reconstruct_minimal_phase_profile(
            energy, f2, np.linspace(-1, 1, 16), missing_band=model, provenance="x"
        )


def test_invalid_missing_band_models_are_refused():
    with pytest.raises(ValueError, match="low continuation"):
        MissingBandModel(low="polynomial")
    with pytest.raises(ValueError, match="positive"):
        MissingBandModel(high_power_exponent=0.0)
