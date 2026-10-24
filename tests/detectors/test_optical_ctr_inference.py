"""Gaussian CTR forward and conditional inverse regressions (#617)."""

import numpy as np
import pytest

from pyrite.detectors.optical_ctr_inference import (
    GaussianCTRTemplates,
    OpticalVoltageScan,
    fit_gaussian_ctr,
    gaussian_ctr_forward_voltage,
    load_gaussian_ctr_fit,
)
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import (
    IdealMartinPuplett,
    OpticalSpectralCoherency,
    polarized_interferogram,
)


def _setup(emin=0.0005, emax=0.025, n=180):
    e = np.linspace(emin, emax, n)
    bg = 1.0e8 * np.ones_like(e)
    pair = 2.0e10 * (0.7 + 0.3 * e / emax)
    source = GaussianCTRTemplates(
        e, bg, pair, np.pi / 4, "synthetic longitudinal gaussian; collected"
    )
    detector = CalibratedPulseEnergyDetector(
        [0.0001, max(0.03, 1.01 * emax)],
        [0.72, 0.72],
        1e12,
        "synthetic measured absorption placeholder",
        noise_rms_v=1.5e-5,
    )
    instrument = IdealMartinPuplett("ideal regression")
    shifts = np.linspace(-0.0012, 0.0012, 81)
    return source, instrument, (detector, detector), shifts


def _synth(sigma=200, **kw):
    source, instr, detectors, shifts = _setup(**kw)
    predicted = gaussian_ctr_forward_voltage(
        source,
        shifts,
        rms_duration_fs=sigma,
        instrument=instr,
        detectors=detectors,
    )
    uncertainty = np.full(predicted.shape, 1.5e-5)
    scan = OpticalVoltageScan(shifts, predicted, uncertainty, "synthetic exact forward model")
    return scan, source, instr, detectors


def test_forward_matches_exact_existing_two_port_calibration():
    scan, template, instrument, detectors = _synth()
    density = template.photon_density(200)
    coherency = OpticalSpectralCoherency.linearly_polarized(
        template.energy_eV,
        density,
        angle_rad=template.polarization_angle_rad,
        provenance="synthetic explicit polarization",
    )
    existing = polarized_interferogram(
        coherency, scan.mirror_displacement_m, instrument=instrument, detectors=detectors
    )
    np.testing.assert_allclose(
        scan.voltage_v_ports, existing.expected_voltage_v_ports, rtol=3e-15, atol=1e-14
    )


def test_gaussian_200fs_recovered_and_conditional_interval():
    scan, template, instrument, detectors = _synth()
    result = fit_gaussian_ctr(
        scan,
        template,
        instrument=instrument,
        detectors=detectors,
        sigma_bounds_fs=(20, 3000),
        profile_points=75,
    )
    assert result.rms_duration_fs == pytest.approx(200, rel=0.003)
    assert result.is_identified_95
    lo, hi = result.confidence_interval_95_fs
    assert lo < 200 < hi
    assert result.chi_square < 1e-7
    assert result.degrees_of_freedom == scan.voltage_v_ports.size - 2
    assert result.turnover_energy_eV == pytest.approx(0.65821195695 / 200)
    assert 0 < result.temporal_factor_at_band_edges[0] < 1


def test_fit_scale_and_offsets_and_prior():
    scan, template, instrument, detectors = _synth(190)
    actual = gaussian_ctr_forward_voltage(
        template,
        scan.mirror_displacement_m,
        rms_duration_fs=190,
        instrument=instrument,
        detectors=detectors,
        relative_source_scale=1.15,
        voltage_offsets_v=(0.00017, -0.0003),
    )
    measured = OpticalVoltageScan(
        scan.mirror_displacement_m, actual, scan.uncertainty_v_ports, "synthetic scale and offsets"
    )
    fit = fit_gaussian_ctr(
        measured,
        template,
        instrument=instrument,
        detectors=detectors,
        fit_offsets=True,
        scale_prior_rms_fraction=0.5,
        sigma_bounds_fs=(30, 2000),
        profile_points=61,
    )
    assert fit.rms_duration_fs == pytest.approx(190, rel=0.04)
    assert fit.relative_source_scale == pytest.approx(1.15, rel=0.03)
    np.testing.assert_allclose(fit.voltage_offsets_v, [0.00017, -0.0003], atol=3e-5)


def test_high_frequency_band_cannot_fake_identified_duration():
    scan, template, instr, detectors = _synth(200, emin=0.08, emax=0.2)
    fit = fit_gaussian_ctr(
        scan,
        template,
        instrument=instr,
        detectors=detectors,
        sigma_bounds_fs=(20, 2000),
        profile_points=65,
    )
    assert not fit.is_identified_95
    assert fit.confidence_interval_95_fs[1] is None
    assert fit.temporal_factor_at_band_edges[0] < 1e-3 or fit.rms_duration_fs < 200


def test_zero_coherent_template_refused():
    source, _, _, _ = _setup()
    with pytest.raises(ValueError, match="identically zero"):
        GaussianCTRTemplates(
            source.energy_eV,
            source.incoherent_photons_per_eV_per_pulse,
            np.zeros_like(source.energy_eV),
            np.pi / 4,
            "zero pair",
        )


def test_zero_uncertainty_saturation_and_missing_band_refused():
    scan, template, instr, detectors = _synth()
    with pytest.raises(ValueError, match="strictly positive"):
        OpticalVoltageScan(
            scan.mirror_displacement_m,
            scan.voltage_v_ports,
            np.zeros_like(scan.uncertainty_v_ports),
            "bad",
        )
    with pytest.raises(ValueError, match="saturated"):
        OpticalVoltageScan(
            scan.mirror_displacement_m,
            scan.voltage_v_ports,
            scan.uncertainty_v_ports,
            "bad",
            np.ones_like(scan.voltage_v_ports, bool),
        )
    narrow = CalibratedPulseEnergyDetector([0.008, 0.01], [1, 1], 1e12, "narrow")
    with pytest.raises(ValueError, match="calibrated detector band"):
        fit_gaussian_ctr(scan, template, instrument=instr, detectors=(narrow, detectors[1]))


def test_nonphysical_templates_and_bounds_refused():
    s, i, dets, shifts = _setup()
    with pytest.raises(ValueError, match="nonnegative"):
        GaussianCTRTemplates(
            s.energy_eV,
            -s.incoherent_photons_per_eV_per_pulse,
            s.coherent_envelope_photons_per_eV_per_pulse,
            0,
            "bad",
        )
    with pytest.raises(ValueError, match="bounds"):
        fit_gaussian_ctr(_synth()[0], s, instrument=i, detectors=dets, sigma_bounds_fs=(200, 20))
    with pytest.raises(ValueError, match="scale prior"):
        fit_gaussian_ctr(
            _synth()[0],
            s,
            instrument=i,
            detectors=dets,
            scale_prior_rms_fraction=0.1,
            fit_scale=False,
        )


def test_no_random_noise_or_charge_factor_applied_twice():
    scan, template, i, dets = _synth()
    voltage = gaussian_ctr_forward_voltage(
        template, scan.mirror_displacement_m, rms_duration_fs=200, instrument=i, detectors=dets
    )
    np.testing.assert_array_equal(scan.voltage_v_ports, voltage)
    np.testing.assert_allclose(
        template.photon_density(0),
        template.incoherent_photons_per_eV_per_pulse
        + template.coherent_envelope_photons_per_eV_per_pulse,
    )


def test_persistence_provenance_and_open_interval(tmp_path):
    scan, template, instrument, detectors = _synth()
    fit = fit_gaussian_ctr(
        scan,
        template,
        instrument=instrument,
        detectors=detectors,
        profile_points=43,
        sigma_bounds_fs=(20, 3000),
    )
    path = fit.save(tmp_path / "inferred")
    loaded = load_gaussian_ctr_fit(path)
    assert loaded.metadata == fit.metadata
    assert loaded.metadata["detector_response_sha256"] == [d.response_sha256 for d in detectors]
    assert loaded.metadata["template_sha256"] and loaded.metadata["scan_sha256"]
    assert loaded.rms_duration_fs == fit.rms_duration_fs
    np.testing.assert_array_equal(loaded.predicted_voltage_v_ports, fit.predicted_voltage_v_ports)
    assert loaded.confidence_interval_95_fs == fit.confidence_interval_95_fs


def test_energy_dependent_detector_and_instrument_jitter_reused():
    scan, template, instrument, detectors = _synth()
    e = template.energy_eV
    measured_response = CalibratedPulseEnergyDetector(
        np.array([0.0001, 0.03]),
        np.array([0.1, 0.9]),
        3e12,
        "synthetic varying-absorption regression",
    )
    jittered = IdealMartinPuplett(
        "jitter", opd_jitter_rms_m=15e-6, arm_intensity_transmissions=(0.9, 0.7)
    )
    calculated = gaussian_ctr_forward_voltage(
        template,
        scan.mirror_displacement_m,
        rms_duration_fs=200,
        instrument=jittered,
        detectors=(measured_response, detectors[1]),
    )
    src = OpticalSpectralCoherency.linearly_polarized(
        e,
        template.photon_density(200),
        angle_rad=template.polarization_angle_rad,
        provenance=template.provenance,
    )
    exact = polarized_interferogram(
        src,
        scan.mirror_displacement_m,
        instrument=jittered,
        detectors=(measured_response, detectors[1]),
    )
    np.testing.assert_allclose(calculated, exact.expected_voltage_v_ports, rtol=2e-15)


def test_charge_jitter_scales_only_pair_term_by_exact_second_moment():
    source, *_ = _setup()
    assert source.with_charge_jitter(mean_electrons=1e9, charge_rms_fraction=0.0).photon_density(
        150
    ) == pytest.approx(source.photon_density(150), rel=1e-15)
    n_mean, r = 5.0, 0.4
    jittered = source.with_charge_jitter(mean_electrons=n_mean, charge_rms_fraction=r)
    np.testing.assert_array_equal(
        jittered.incoherent_photons_per_eV_per_pulse, source.incoherent_photons_per_eV_per_pulse
    )
    # Direct moment of a two-point population N in {N - d, N + d}, d = r N.
    d = r * n_mean
    pair_mean = np.mean([(n_mean + s * d) * (n_mean + s * d - 1) for s in (-1, 1)])
    np.testing.assert_allclose(
        jittered.coherent_envelope_photons_per_eV_per_pulse,
        source.coherent_envelope_photons_per_eV_per_pulse * pair_mean / (n_mean * (n_mean - 1)),
        rtol=1e-14,
    )
    big = source.with_charge_jitter(mean_electrons=1e12, charge_rms_fraction=0.1)
    np.testing.assert_allclose(
        big.coherent_envelope_photons_per_eV_per_pulse,
        1.01 * source.coherent_envelope_photons_per_eV_per_pulse,
        rtol=1e-11,
    )
    assert "charge jitter" in jittered.provenance
    with pytest.raises(ValueError):
        source.with_charge_jitter(mean_electrons=1.0, charge_rms_fraction=0.1)
    with pytest.raises(ValueError):
        source.with_charge_jitter(mean_electrons=10.0, charge_rms_fraction=-0.1)


def test_ignoring_charge_jitter_biases_fitted_duration():
    scan_source, instrument, detectors, shifts = _setup()
    jittered = scan_source.with_charge_jitter(mean_electrons=6.25e8, charge_rms_fraction=0.3)
    truth = gaussian_ctr_forward_voltage(
        jittered, shifts, rms_duration_fs=200, instrument=instrument, detectors=detectors
    )
    scan = OpticalVoltageScan(
        shifts, truth, np.full(truth.shape, 1.5e-5), "synthetic jittered shot average"
    )
    kw = dict(instrument=instrument, detectors=detectors, fit_scale=False, profile_points=61)
    matched = fit_gaussian_ctr(scan, jittered, **kw)
    naive = fit_gaussian_ctr(scan, scan_source, **kw)
    assert matched.rms_duration_fs == pytest.approx(200, rel=2e-3)
    assert naive.chi_square > matched.chi_square + 100
