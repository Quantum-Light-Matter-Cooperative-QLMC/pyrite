"""Delay/spectral convergence diagnostic of the two-port scan (#617)."""

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferogram_convergence import interferogram_convergence
from pyrite.detectors.optical_interferometer import IdealMartinPuplett, OpticalSpectralCoherency

_HBAR_EV_PS = 0.6582119569509067e-3
_C_M_PS = 299792458.0e-12
_EV_J = 1.602176634e-19
_BAND = (2e-6, 2.4e-3)  # 0.5 GHz to 0.58 THz
_GAIN, _ETA = 3.0e11, 0.6


def _power(energy, sigma=2.0, xi=1.07):
    w = energy / _HBAR_EV_PS
    return 1e22 * np.exp(-((sigma * w) ** 2)) * (1 - np.exp(-((xi * w) ** 2))) ** 2  # eV/eV


def _spectrum(energy):
    return OpticalSpectralCoherency.linearly_polarized(
        energy, _power(energy) / energy, angle_rad=np.pi / 4, provenance="filtered Gaussian"
    )


def _run(step_ps, half_ps, **kw):
    detector = CalibratedPulseEnergyDetector([1e-6, 3e-3], [_ETA, _ETA], _GAIN, "flat synthetic")
    return interferogram_convergence(
        _spectrum,
        _BAND,
        mirror_step_m=0.5 * _C_M_PS * step_ps,
        mirror_half_range_m=0.5 * _C_M_PS * half_ps,
        instrument=kw.pop("instrument", IdealMartinPuplett("ideal")),
        detectors=(detector, detector),
        **kw,
    )


def test_resolved_scan_converges_and_recovers_the_fringe_spectrum():
    out = _run(0.25, 40.0)
    assert out.converged and out.nyquist_ratio < 1
    assert max(out.spectral_change, out.delay_step_change, out.delay_range_change) < 1e-3
    # Ideal 45-degree grids: W(E) = eta * gain * (energy spectrum in J/eV).
    expected = _ETA * _GAIN * _EV_J * _power(out.energy_eV)
    np.testing.assert_allclose(
        out.recovered_fringe_spectrum_v_per_eV, expected, atol=2e-3 * expected.max()
    )
    assert out.spectral_resolution_eV == pytest.approx(1.2398419843320026e-6 / (2 * _C_M_PS * 40.0))


def test_each_axis_is_flagged_independently():
    coarse_step = _run(2.0, 40.0)  # OPD step 0.6 mm > lambda_min / 2 = 0.26 mm
    assert coarse_step.nyquist_ratio > 1 and not coarse_step.converged
    assert coarse_step.delay_step_change > 0.05 > coarse_step.delay_range_change
    short = _run(0.25, 5.0)  # truncated inside the side lobes of Leissner Fig. 3
    assert short.delay_range_change > 0.05 > short.delay_step_change
    assert not short.converged and short.nyquist_ratio < 1
    sparse = _run(0.25, 40.0, n_energy=9)  # fringe unresolved in energy at 40 ps
    assert sparse.spectral_change > 0.05 and not sparse.converged


def test_refusals():
    with pytest.raises(ValueError, match="even scan"):
        _run(0.25, 40.0, instrument=IdealMartinPuplett("offset", zero_opd_m=1e-5))
    with pytest.raises(ValueError, match="mirror_step_m"):
        _run(30.0, 40.0)
    with pytest.raises(ValueError, match="n_energy"):
        _run(0.25, 40.0, n_energy=2)
    with pytest.raises(ValueError, match="band_eV"):
        interferogram_convergence(
            _spectrum, (0.0, 1e-3), mirror_step_m=1e-5, mirror_half_range_m=1e-3,
            instrument=IdealMartinPuplett("ideal"), detectors=(),
        )  # fmt: skip
