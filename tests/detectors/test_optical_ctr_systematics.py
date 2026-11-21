"""Transverse/response systematic modes in the Gaussian CTR fit (#617)."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pyrite.detectors.optical_ctr_inference import (
    ChargeJitterModel,
    CTRSystematicMode,
    GaussianCTRTemplates,
    OpticalVoltageScan,
    _profile_at_sigma,
    _response_kernel,
    fit_gaussian_ctr,
    gaussian_ctr_forward_voltage,
    load_gaussian_ctr_fit,
)
from pyrite.detectors.optical_ctr_systematics import systematic_voltage_modes
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import IdealMartinPuplett

NOISE = 1.5e-5


def _setup():
    e = np.linspace(0.0005, 0.025, 180)
    pair = 2.0e10 * (0.7 + 0.3 * e / e[-1])
    templates = GaussianCTRTemplates(
        e, np.full(e.size, 1e8), pair, np.pi / 4, "synthetic collected templates"
    )
    detector = CalibratedPulseEnergyDetector(
        [0.0001, 0.03], [0.72, 0.72], 1e12, "synthetic calibration placeholder"
    )
    instrument = IdealMartinPuplett("ideal regression")
    shifts = np.linspace(-0.0012, 0.0012, 81)
    return templates, instrument, (detector, detector), shifts


def _transverse(e, amplitude=0.25):
    # Shape of d ln|F_T|^2 for a Gaussian transverse size: -2 (k sigma_r theta)^2.
    return CTRSystematicMode(
        "transverse-size", "transverse", -amplitude * (e / e[-1]) ** 2, "synthetic 1-sigma"
    )


def _scan(templates, instrument, detectors, shifts, sigma, density=None):
    if density is not None:
        templates = GaussianCTRTemplates(
            templates.energy_eV,
            templates.incoherent_photons_per_eV_per_pulse,
            density,
            templates.polarization_angle_rad,
            templates.provenance,
        )
    v = gaussian_ctr_forward_voltage(
        templates, shifts, rms_duration_fs=sigma, instrument=instrument, detectors=detectors
    )
    return OpticalVoltageScan(shifts, v, np.full(v.shape, NOISE), "synthetic expectation")


def test_profiled_pulls_equal_marginal_covariance_chi_square():
    templates, instrument, detectors, shifts = _setup()
    e = templates.energy_eV
    modes = (
        _transverse(e),
        CTRSystematicMode("eta-port1", "response", 0.05 * np.sin(e / e[-1] * 3), "syn", port=1),
        CTRSystematicMode("eta-common", "response", np.full(e.size, 0.03), "syn"),
    )
    rng = np.random.default_rng(7)
    scan = _scan(templates, instrument, detectors, shifts, 230.0)
    noisy = OpticalVoltageScan(
        shifts,
        scan.voltage_v_ports + rng.normal(0, 4 * NOISE, scan.voltage_v_ports.shape),
        scan.uncertainty_v_ports,
        "synthetic noisy",
    )
    kernel = _response_kernel(templates, shifts, instrument, detectors, 4_000_000)
    sigma = 230.0  # pulls stay inside their physical box: the identity holds
    chi2, scale, _, predicted, pulls = _profile_at_sigma(
        sigma,
        templates,
        kernel,
        noisy,
        fit_scale=False,
        fit_offsets=False,
        scale_prior_rms_fraction=None,
        modes=modes,
    )
    assert scale == 1.0 and pulls.shape == (3,)
    assert np.all(1 + pulls[:, None] * np.array([m.relative_rms for m in modes]) > 0)
    d = systematic_voltage_modes(modes, templates, kernel, sigma).reshape(3, -1)
    r = (noisy.voltage_v_ports - kernel @ templates.photon_density(sigma)).flatten()
    cov = np.diag(noisy.uncertainty_v_ports.flatten() ** 2) + d.T @ d
    np.testing.assert_allclose(chi2, r @ np.linalg.solve(cov, r), rtol=1e-8)
    # Port-1-only response mode leaves port 0 untouched.
    single = systematic_voltage_modes(modes[1:2], templates, kernel, sigma)
    assert np.all(single[0, :, 0] == 0) and np.any(single[0, :, 1] != 0)


def test_pulls_are_confined_to_physical_templates():
    # At 200 fs the 230 fs data want a transverse pull > 4, which would make the
    # pair envelope 1 + a u (u = -0.25 at the band top) negative: the box binds.
    templates, instrument, detectors, shifts = _setup()
    mode = _transverse(templates.energy_eV)
    scan = _scan(templates, instrument, detectors, shifts, 230.0)
    kernel = _response_kernel(templates, shifts, instrument, detectors, 4_000_000)
    for fit_scale in (False, True):
        chi2, _, _, _, pulls = _profile_at_sigma(
            200.0,
            templates,
            kernel,
            scan,
            fit_scale=fit_scale,
            fit_offsets=False,
            scale_prior_rms_fraction=None,
            modes=(mode,),
        )
        assert pulls[0] <= 4.0 + 1e-9
        assert np.min(1 + pulls[0] * mode.relative_rms) >= -1e-9
        if not fit_scale:
            assert pulls[0] == pytest.approx(4.0, rel=1e-6)


def test_nondegenerate_mode_removes_bias_and_recovers_pull():
    templates, instrument, detectors, shifts = _setup()
    e = templates.energy_eV
    mode = CTRSystematicMode(
        "acceptance-ripple", "transverse", 0.2 * np.cos(2 * np.pi * e / 0.006), "synthetic"
    )
    true_pull = 1.5
    shifted = templates.coherent_envelope_photons_per_eV_per_pulse * (
        1 + true_pull * mode.relative_rms
    )
    scan = _scan(templates, instrument, detectors, shifts, 200.0, shifted)
    kw = dict(
        instrument=instrument,
        detectors=detectors,
        sigma_bounds_fs=(20, 3000),
        profile_points=75,
        fit_scale=False,
    )
    naive = fit_gaussian_ctr(scan, templates, **kw)
    lo, hi = naive.confidence_interval_95_fs
    assert not lo <= 200.0 <= hi
    fit = fit_gaussian_ctr(scan, templates, systematics=(mode,), **kw)
    lo2, hi2 = fit.confidence_interval_95_fs
    assert lo2 < 200.0 < hi2
    assert fit.rms_duration_fs == pytest.approx(200.0, rel=1e-4)
    assert fit.metadata["systematic_pulls"]["acceptance-ripple"] == pytest.approx(
        true_pull, rel=1e-3
    )
    assert "transverse/response" in fit.model_assumptions[3]
    assert fit.degrees_of_freedom == naive.degrees_of_freedom


def test_gaussian_transverse_mode_is_degenerate_with_duration():
    # u = -A (E/E0)^2 matches d ln F_t/d sigma * delta: delta = A hbar^2 / (2 sigma E0^2).
    # E0 = E_max keeps |u| <= A = 0.3 (pulls physical to a = 3.3); delta ~ 0.5 fs
    # stays inside the linear regime of this precise scan.
    templates, instrument, detectors, shifts = _setup()
    e, sigma, amplitude = templates.energy_eV, 200.0, 0.3
    e0 = e[-1]
    mode = CTRSystematicMode("transverse-size", "transverse", -amplitude * (e / e0) ** 2, "syn")
    scan = _scan(templates, instrument, detectors, shifts, sigma)
    kw = dict(instrument=instrument, detectors=detectors, sigma_bounds_fs=(20, 3000))
    fit = fit_gaussian_ctr(scan, templates, systematics=(mode,), fit_scale=False, **kw)
    plain = fit_gaussian_ctr(scan, templates, fit_scale=False, **kw)
    delta = amplitude * 0.6582119569509067**2 / (2 * sigma * e0**2)
    lo, hi = fit.confidence_interval_95_fs
    lo0, hi0 = plain.confidence_interval_95_fs
    width = np.sqrt(((hi - lo) / 2) ** 2 - ((hi0 - lo0) / 2) ** 2)
    assert width == pytest.approx(1.959964 * delta, rel=0.1)


def test_unperturbed_data_reproduce_plain_fit_and_persist(tmp_path):
    templates, instrument, detectors, shifts = _setup()
    scan = _scan(templates, instrument, detectors, shifts, 200.0)
    kw = dict(instrument=instrument, detectors=detectors, sigma_bounds_fs=(20, 3000))
    plain = fit_gaussian_ctr(scan, templates, **kw)
    mode = _transverse(templates.energy_eV)
    fit = fit_gaussian_ctr(scan, templates, systematics=(mode,), **kw)
    assert fit.rms_duration_fs == pytest.approx(plain.rms_duration_fs, rel=1e-6)
    assert abs(fit.metadata["systematic_pulls"]["transverse-size"]) < 1e-4
    loaded = load_gaussian_ctr_fit(fit.save(tmp_path / "fit"))
    assert loaded.metadata["systematic_modes"][0]["kind"] == "transverse"


def test_from_shifted_density_and_refusals():
    templates, instrument, detectors, shifts = _setup()
    e = templates.energy_eV
    nominal = templates.coherent_envelope_photons_per_eV_per_pulse
    mode = CTRSystematicMode.from_shifted_density(
        "t", "transverse", nominal, nominal * 0.9, provenance="regenerated at +1 sigma radius"
    )
    np.testing.assert_allclose(mode.relative_rms, -0.1)
    with pytest.raises(ValueError, match="transverse modes"):
        CTRSystematicMode("t", "transverse", np.ones(3), "p", port=0)
    with pytest.raises(ValueError, match="identically zero"):
        CTRSystematicMode("t", "response", np.zeros(3), "p")
    with pytest.raises(ValueError, match="kind"):
        CTRSystematicMode("t", "charge", np.ones(3), "p")
    with pytest.raises(ValueError, match="port"):
        CTRSystematicMode("t", "response", np.ones(3), "p", port=2)
    scan = _scan(templates, instrument, detectors, shifts, 200.0)
    kw = dict(instrument=instrument, detectors=detectors)
    with pytest.raises(ValueError, match="energy grid"):
        fit_gaussian_ctr(scan, templates, systematics=(_transverse(e[:-1]),), **kw)
    with pytest.raises(ValueError, match="unique"):
        fit_gaussian_ctr(scan, templates, systematics=(mode, mode), **kw)
    with pytest.raises(ValueError, match="charge-jitter"):
        fit_gaussian_ctr(
            scan,
            templates,
            systematics=(mode,),
            charge_jitter=ChargeJitterModel(1e6, 0.05, 10),
            **kw,
        )


def _exact_bilinear_chi2(scan, model, drift, scales, fit_offsets):
    """Brute force: at each s, GLS over offsets with covariance C + s^2 sum D D^T.

    The inverse is applied by Woodbury (diagonal C, low-rank D), which is exact.
    """
    y = scan.voltage_v_ports.flatten()
    c_inv = 1 / scan.uncertainty_v_ports.flatten() ** 2
    d = drift.reshape(drift.shape[0], -1).T
    port = np.indices(model.shape)[1].flatten()
    x = np.column_stack([(port == p).astype(float) for p in (0, 1)])

    def chi2(s):
        cd = c_inv[:, None] * d
        core = np.linalg.inv(np.eye(d.shape[1]) / (s * s) + d.T @ cd) if s > 0 else None

        def w(v):
            out = c_inv[:, None] * v if v.ndim == 2 else c_inv * v
            return out if core is None else out - cd @ (core @ (cd.T @ v))

        r = y - s * model.flatten()
        if fit_offsets:
            r = r - x @ np.linalg.solve(x.T @ w(x), x.T @ w(r))
        return float(r @ w(r))

    values = [chi2(s) for s in scales]
    i = int(np.argmin(values))
    refined = minimize_scalar(
        chi2, bounds=(scales[max(i - 1, 0)], scales[min(i + 1, len(scales) - 1)]), method="bounded"
    )
    return min(values[i], float(refined.fun))


@pytest.mark.parametrize("fit_offsets", [False, True])
def test_fitted_scale_profiles_the_exact_bilinear_model(fit_offsets):
    templates, instrument, detectors, shifts = _setup()
    e = templates.energy_eV
    mode = CTRSystematicMode("ripple", "transverse", 0.2 * np.cos(2 * np.pi * e / 0.006), "s")
    rng = np.random.default_rng(3)
    scan = _scan(templates, instrument, detectors, shifts, 200.0)
    noisy = OpticalVoltageScan(
        shifts,
        1.4 * scan.voltage_v_ports + rng.normal(0, 3e-3, scan.voltage_v_ports.shape),
        np.full(scan.voltage_v_ports.shape, 3e-3),
        "synthetic noisy, scale 1.4",
    )
    kernel = _response_kernel(templates, shifts, instrument, detectors, 4_000_000)
    for sigma in (60.0, 200.0):
        chi2, scale, _, _, pulls = _profile_at_sigma(
            sigma,
            templates,
            kernel,
            noisy,
            fit_scale=True,
            fit_offsets=fit_offsets,
            scale_prior_rms_fraction=None,
            modes=(mode,),
        )
        model = kernel @ templates.photon_density(sigma)
        drift = systematic_voltage_modes((mode,), templates, kernel, sigma)
        grid = np.linspace(0.5 * scale, 1.5 * scale, 41)
        exact = _exact_bilinear_chi2(noisy, model, drift, grid, fit_offsets)
        assert chi2 == pytest.approx(exact, rel=1e-6, abs=1e-6)
    assert pulls.shape == (1,)


def test_negligible_mode_is_well_conditioned():
    templates, instrument, detectors, shifts = _setup()
    e = templates.energy_eV
    # Nonzero only at the top of the band, where exp[-(E sigma/hbar)^2] underflows.
    u = np.where(e > 0.9 * e[-1], 0.2, 0.0)
    mode = CTRSystematicMode("high-band", "transverse", u, "synthetic")
    scan = _scan(templates, instrument, detectors, shifts, 1300.0)
    kernel = _response_kernel(templates, shifts, instrument, detectors, 4_000_000)
    for fit_scale in (False, True):
        kw = dict(fit_scale=fit_scale, fit_offsets=True, scale_prior_rms_fraction=None)
        plain = _profile_at_sigma(1300.0, templates, kernel, scan, **kw)
        with_mode = _profile_at_sigma(1300.0, templates, kernel, scan, modes=(mode,), **kw)
        assert with_mode[0] == pytest.approx(plain[0], rel=1e-6, abs=1e-9)
        assert abs(with_mode[4][0]) < 1e-6


def test_new_support_is_refused():
    with pytest.raises(ValueError, match="support"):
        CTRSystematicMode.from_shifted_density(
            "t", "transverse", np.array([0.0, 1.0, 1.0]), np.array([0.1, 1.0, 1.1]), provenance="p"
        )


@pytest.mark.parametrize("ripple", [0.0, 1e-3, 0.5])
def test_two_port_response_with_free_scale_is_refused(ripple):
    # Uniform or nearly uniform two-port modes trade against a free scale
    # (deep chi2(s) branch far out in s); any port=None response is refused.
    templates, instrument, detectors, shifts = _setup()
    scan = _scan(templates, instrument, detectors, shifts, 200.0)
    e = templates.energy_eV
    shape = 0.3 * (1 + ripple * np.cos(7 * e / e[-1]))
    flat = CTRSystematicMode("gain", "response", shape, "s")
    kw = dict(instrument=instrument, detectors=detectors, systematics=(flat,))
    with pytest.raises(ValueError, match="partly degenerate with a free source scale"):
        fit_gaussian_ctr(scan, templates, **kw)
    single = CTRSystematicMode("gain1", "response", shape, "s", port=1)
    fit_gaussian_ctr(
        scan, templates, instrument=instrument, detectors=detectors, systematics=(single,)
    )
    # With a scale prior or a fixed scale the mode is identified.
    fit_gaussian_ctr(scan, templates, scale_prior_rms_fraction=0.1, **kw)
    fit_gaussian_ctr(scan, templates, fit_scale=False, **kw)
