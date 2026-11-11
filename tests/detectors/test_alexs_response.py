"""ALEX-s 2k512 BI UV1 charge-integrating response (#1081).

Datasheet constants are the public greateyes DS-ALEXs Rev05 values; QE curves
here are synthetic.
"""

import numpy as np
import pytest

from pyrite.detectors import AlexS, PixelGrid, PlanarDetector, PlanarPose, TabulatedEfficiency
from pyrite.detectors import alexs_response as ax
from pyrite.detectors.grating import qe_absorption

FLAT_QE = TabulatedEfficiency((100.0, 2000.0), (0.6, 0.6), name="synthetic-flat")


def test_datasheet_constants():
    assert ax.EV_PER_ELECTRON == 3.66
    assert ax.FULL_WELL_E == 100_000.0
    assert ax.read_noise_e(50e3) == 3.5
    assert ax.read_noise_e(1e6) == 6.8
    assert ax.read_noise_e(3e6) == 10.7
    assert ax.gain_adu_per_e("standard") == 1.0
    assert ax.gain_adu_per_e("high-capacity") == 0.34
    assert ax.DARK_CURRENT_E_PER_PX_S == 0.00025
    assert ax.EPI_THICKNESS_UM == 15.0
    fmt = ax.AlexsFormat.named("2k512")
    assert (fmt.columns, fmt.rows, fmt.pitch_mm) == (2048, 515, 0.0135)
    assert fmt.width_mm == pytest.approx(27.6, abs=0.05)
    assert fmt.height_mm == pytest.approx(6.9, abs=0.06)


def test_unknown_rate_and_gain_mode_are_rejected():
    with pytest.raises(ValueError, match="read-noise specification"):
        AlexS(readout_rate_hz=2e6)
    with pytest.raises(ValueError, match="gain mode"):
        AlexS(gain_mode="high")


@pytest.mark.parametrize("gain_mode, gain", [("standard", 1.0), ("high-capacity", 0.34)])
def test_monochromatic_adu_is_n_qe_e_over_w_times_gain(gain_mode, gain):
    # A narrow line of N photons at E: integrated ADU = N * QE * E / 3.66 * gain.
    E0, N = 700.0, 200.0  # below full well
    E = np.linspace(600.0, 800.0, 2001)
    sigma = 0.5
    density = N * np.exp(-0.5 * ((E - E0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    response = AlexS(gain_mode=gain_mode, qe=FLAT_QE)
    adu = np.trapezoid(response.score(E, density, fwhm_eV=None, scale=1.0), E)
    assert adu == pytest.approx(N * 0.6 * E0 / 3.66 * gain, rel=1e-6)
    frame = ax.alexs_frame(
        np.array([N * 0.6 * E0 / 3.66]), exposure_s=0.0, gain_mode=gain_mode, dark=False
    )
    assert frame[0] == pytest.approx(adu, rel=1e-6)


def test_signal_electrons_unit_photon_and_zero_qe_limits():
    assert ax.signal_electrons(3.66, 1.0, 1.0) == pytest.approx(1.0)
    assert ax.signal_electrons(1000.0, 5.0, 0.0) == 0.0


def test_full_well_clipping():
    signal = np.array([5.0e4, 1.0e5, 3.0e5])
    adu = ax.alexs_frame(signal, exposure_s=0.0, gain_mode="high-capacity", dark=False)
    assert adu == pytest.approx([0.34 * 5.0e4, 0.34 * 1.0e5, 0.34 * 1.0e5])


def test_dark_charge_is_rate_times_exposure():
    adu = ax.alexs_frame(np.zeros(3), exposure_s=400.0)
    assert adu == pytest.approx(np.full(3, 0.00025 * 400.0))
    rng = np.random.default_rng(7)
    zeros = np.zeros(200_000)
    drawn = ax.alexs_frame(zeros, exposure_s=2.0e4, photons=zeros, rng=rng, shot_noise=True)
    expected = 0.00025 * 2.0e4
    assert drawn.mean() == pytest.approx(expected, rel=0.01)
    assert drawn.var() == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("rate, gain_mode", [(50e3, "standard"), (3e6, "high-capacity")])
def test_read_noise_variance(rate, gain_mode):
    rng = np.random.default_rng(11)
    adu = ax.alexs_frame(
        np.full(400_000, 1.0e3),
        exposure_s=0.0,
        readout_rate_hz=rate,
        gain_mode=gain_mode,
        rng=rng,
        read_noise=True,
        dark=False,
    )
    gain = ax.gain_adu_per_e(gain_mode)
    assert adu.mean() == pytest.approx(gain * 1.0e3, rel=1e-3)
    assert adu.var() == pytest.approx((gain * ax.read_noise_e(rate)) ** 2, rel=0.01)


def test_photon_shot_noise_keeps_mean_charge_per_photon():
    rng = np.random.default_rng(3)
    photons = np.full(200_000, 50.0)
    signal = photons * 1000.0 / 3.66
    adu = ax.alexs_frame(
        signal, exposure_s=0.0, photons=photons, rng=rng, shot_noise=True, dark=False
    )
    per_photon = 1000.0 / 3.66
    assert adu.mean() == pytest.approx(signal[0], rel=2e-3)
    assert adu.var() == pytest.approx(50.0 * per_photon**2, rel=0.02)


def test_noise_requires_explicit_generator():
    with pytest.raises(ValueError, match="explicit numpy Generator"):
        ax.alexs_frame(np.zeros(2), exposure_s=1.0, read_noise=True)
    with pytest.raises(ValueError, match="photons > 0"):
        ax.alexs_frame(
            np.ones(2),
            exposure_s=0.0,
            photons=np.zeros(2),
            shot_noise=True,
            rng=np.random.default_rng(),
        )
    with pytest.raises(ValueError, match="photon image"):
        ax.alexs_frame(np.zeros(2), exposure_s=1.0, shot_noise=True, rng=np.random.default_rng())


def test_qe_fallback_is_beer_lambert_with_15_um_epi():
    E = np.array([300.0, 1500.0, 6000.0])
    assert ax.quantum_efficiency(E) == pytest.approx(qe_absorption(E, active_um=15.0))
    assert ax.quantum_efficiency(E, FLAT_QE) == pytest.approx([0.6, 0.6, 0.0])


def test_qe_curve_values_enter_response_identity():
    from pyrite.instrument.observation import response_payload

    a = response_payload(AlexS(qe=FLAT_QE))
    b = response_payload(AlexS(qe=TabulatedEfficiency((100.0, 2000.0), (0.6, 0.5), name="x")))
    assert a["config"]["qe"]["value"] == [0.6, 0.6]
    assert a != b


def test_2k512_pixel_grid_and_detector_factories():
    grid = PixelGrid.alexs_2k512()
    assert grid.shape == (515, 2048)
    assert grid.pitch_mm == (0.0135, 0.0135)
    pose = PlanarPose.from_observation(500.0, 90.0, 0.0, 0.0, (0.0, 0.0))
    detector = PlanarDetector.alexs_2k512(pose, response=AlexS(qe=FLAT_QE))
    assert detector.pixels == grid
    assert detector.size_mm == pytest.approx((2048 * 0.0135, 515 * 0.0135))
    E = np.array([500.0, 1000.0])
    scored = detector.score(E, np.ones(2))
    assert scored == pytest.approx(0.6 * E / 3.66)
