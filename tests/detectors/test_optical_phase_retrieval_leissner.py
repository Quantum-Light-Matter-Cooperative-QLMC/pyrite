"""Leissner thesis Sec. 7.1: minimal-phase reconstruction of high-pass-filtered Gaussians.

B. Leissner, Diplomarbeit RWTH Aachen, PITHA 99/4 (1998). A 3 ps RMS Gaussian
bunch spectrum ``exp(-omega^2 sigma^2)`` is multiplied by the high-pass filter
``1 - exp(-omega^2 b^2)`` (Eq. 7.1, cutoff Eq. 6.29), continued below the
falling flank by the intensity parabola of Eqs. (6.12)-(6.13) and
reconstructed by the minimal-phase Kramers-Kronig relation (Eqs. 6.20-6.22).
Fig. 7.5 curves are read from the PDF vector paths
(``tests/data/optical/leissner-1998-fig7-5.json``).

Not pinned: the high-frequency continuation (start point and exponent are not
stated for the simulation; it sets only the profile onset), the thesis's
discrete sum (6.21), and its Sec. 7.1 text RMS values (2.48 ps is the lobe RMS
of Fig. 7.3, a different, unstated 77 GHz reconstruction; none matches Fig. 7.5).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pyrite.detectors.optical_phase_retrieval import (
    MissingBandModel,
    reconstruct_minimal_phase_profile,
)

DATA = json.loads(
    (Path(__file__).parents[1] / "data" / "optical" / "leissner-1998-fig7-5.json").read_text()
)
_HBAR_EV_FS = 0.6582119569509067
_H_EV_PER_GHZ = 4.135667696e-6


def _filtered_band(cutoff_ghz, sigma_ps=3.0):
    """Falling-flank band from the steepest point to 1e-5 of the peak (arbitrary units)."""
    e = np.linspace(1e-7, 0.002, 20001)
    w = e / _HBAR_EV_FS
    b = np.sqrt(-np.log(1 - np.exp(-0.5))) / (2 * np.pi * cutoff_ghz * 1e9) * 1e15
    intensity = 7.0 * np.exp(-((w * sigma_ps * 1e3) ** 2)) * (1 - np.exp(-((b * w) ** 2)))
    slope = np.gradient(intensity, e)
    peak = int(np.argmax(intensity))
    start = peak + int(np.argmin(slope[peak:]))
    stop = int(np.flatnonzero(intensity > 1e-5 * intensity[peak])[-1])
    keep = np.unique(np.linspace(start, stop, 400).astype(int))
    return e[keep], intensity[keep]


def _reconstruct(cutoff_ghz):
    e, intensity = _filtered_band(cutoff_ghz)
    return reconstruct_minimal_phase_profile(
        e,
        intensity,
        np.linspace(-15000.0, 40000.0, 2201),
        missing_band=MissingBandModel(
            low="parabolic-intensity", high_power_exponent=3.0, low_nodes=200
        ),
        provenance="Leissner 1998 Sec. 7.1 synthetic filtered Gaussian",
    )


def _figure_lobe_rms(t, rho):
    k = int(np.argmax(rho))
    a = k
    while a > 0 and rho[a - 1] >= 0:
        a -= 1
    b = k
    while b < rho.size - 1 and rho[b + 1] >= 0:
        b += 1
    x, y = t[a : b + 1], rho[a : b + 1]
    area = np.trapezoid(y, x)
    mean = np.trapezoid(x * y, x) / area
    return float(np.sqrt(np.trapezoid((x - mean) ** 2 * y, x) / area))


# A^2 of the parabola (unit-prefactor spectrum), from the fresh-context
# independent reconstruction (docs/validation/detectors/optical-phase-retrieval.md).
A2_INDEPENDENT = {"154": 0.05577, "77": 0.19271, "51": 0.35638}


@pytest.mark.parametrize("cutoff", ["154", "77", "51"])
def test_fig7_5_reconstructed_profiles(cutoff):
    curve = DATA["fig7_5"][cutoff]
    t_fig = np.array(curve["t_ps"])
    rho_fig = np.array(curve["rho_normalized"])
    result = _reconstruct(float(cutoff))
    t = result.time_fs / 1e3
    rho = result.profile_per_fs / np.max(result.profile_per_fs)
    window = (t_fig > -4) & (t_fig < 24)

    def worst(shift):
        return np.max(np.abs(np.interp(t_fig[window], t + shift, rho) - rho_fig[window]))

    shift = minimize_scalar(worst, bounds=(-15, 15), method="bounded").x
    diff = np.interp(t_fig[window], t + shift, rho) - rho_fig[window]
    # Body of the profile to ~1-2 % of the peak; the onset carries the
    # unpinned high-frequency continuation (max residual there).
    assert np.max(np.abs(diff)) < 0.06
    assert np.sqrt(np.mean(diff**2)) < 0.025
    figure_rms = _figure_lobe_rms(t_fig, rho_fig)
    assert result.lobe_rms_duration_fs / 1e3 == pytest.approx(figure_rms, rel=0.04)
    assert result.metadata["input_intensity_divided_by"] / 7.0 == pytest.approx(
        A2_INDEPENDENT[cutoff], rel=1e-3
    )


def test_text_rms_values_are_not_the_figure_rms():
    # Documents the excluded source values (they belong to another reconstruction).
    text = {"154": 2.12, "77": 2.48, "51": 2.75}
    for key, quoted in text.items():
        curve = DATA["fig7_5"][key]
        rms = _figure_lobe_rms(np.array(curve["t_ps"]), np.array(curve["rho_normalized"]))
        assert rms < quoted - 0.05


def test_parabolic_continuation_matches_value_and_slope():
    e, intensity = _filtered_band(77.0)
    result = _reconstruct(77.0)
    a2 = result.metadata["input_intensity_divided_by"]
    low = ~result.measured & (result.energy_eV < e[0])
    e_low = result.energy_eV[low]
    f2_low = result.form_factor_magnitude[low] ** 2
    slope = np.gradient(intensity[:3], e[:3], edge_order=2)[0]
    curvature = -slope / (2 * e[0])
    np.testing.assert_allclose(f2_low, (a2 - curvature * e_low**2) / a2, rtol=1e-12)
    assert f2_low[0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.form_factor_magnitude[result.measured] ** 2, intensity / a2, rtol=1e-12
    )


def test_parabolic_continuation_refuses_a_rising_flank():
    e = np.linspace(0.0001, 0.0005, 8)
    with pytest.raises(ValueError, match="falling"):
        reconstruct_minimal_phase_profile(
            e,
            np.linspace(1.0, 2.0, 8),
            np.linspace(-1000.0, 1000.0, 64),
            missing_band=MissingBandModel(low="parabolic-intensity"),
            provenance="rising",
        )
