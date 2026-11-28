"""Deterministic checks of the zone-resolved W <001> line record (#585)."""

import numpy as np
import pytest

from checks import channeling_submev_w001_dispersion as disp
from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states
from pyrite.montecarlo.channeling.axial_emission import axial_line_catalog


def test_weighted_summary_of_two_lines():
    lines = np.array([[100.0, 102.0]])
    summary = disp.weighted_summary(lines, np.array([[3.0, 1.0]]))
    assert summary["mean_eV"] == pytest.approx(100.5)
    assert summary["rms_width_eV"] == pytest.approx(np.sqrt(0.75))
    assert summary["quantiles_eV"]["q50"] == 100.0
    assert summary["range_eV"] == [100.0, 102.0]
    with pytest.raises(ValueError):
        disp.weighted_summary(lines, np.zeros((1, 2)))


def test_medium_mean_matches_constant_index():
    beta, delta, vacuum = 0.6, 2e-3, 500.0
    out = disp.medium_mean(vacuum, beta, lambda energy: delta)
    expected = vacuum * (1.0 - beta) / (1.0 - beta * (1.0 - delta))
    assert out["line_eV"] == pytest.approx(expected, rel=1e-12)
    assert out["shift_eV"] < 0.0


@pytest.mark.slow  # ~40 s: three order-10 W <001> solves
def test_zone_centre_line_matches_flat_catalog():
    # At 300 keV the flat catalog admits 2p -> 1s; at k = 0 the vertical
    # record must reproduce its gap and doublet-summed dipole strength.
    profile = axial_profile("tungsten", (0, 0, 1), 293.15)
    gamma, _ = longitudinal_kinematics(300e3)
    order = 10  # the production fixture order; 6 is too coarse for a flat 1s band
    catalog = axial_line_catalog(profile, 300e3, max_order=order)
    line = next(ln for ln in catalog.lines if ln.final == 0 and ln.initial == 1)
    states = axial_states(profile, gamma, max_order=order)
    gaps, dipole = disp.vertical_lines(states, gamma, (0.0, 0.0), profile.g_inv_ang)
    assert gaps == pytest.approx([line.delta_eV] * 2, abs=1e-6)
    assert np.sum(dipole**2) == pytest.approx(np.sum(np.abs(line.dipole_ang) ** 2), rel=1e-8)


def test_zone_grid_must_avoid_the_centre():
    profile = axial_profile("tungsten", (0, 0, 1), 293.15)
    with pytest.raises(ValueError, match="even"):
        disp.zone_record(profile, 1.5, 0.7, (4, 4), np.array([0.1]), 3)


def test_coherence_vanishes_for_a_single_incident_band():
    # One incident order lying wholly in band 1 has no inter-band coherence.
    class States:
        orders = np.array([[0, 0], [1, 0], [0, 1]])
        energies_eV = np.array([-10.0, -5.0, -4.0])
        coefficients = np.eye(3, dtype=complex)

    coh, inc = disp.doublet_coherence(States(), 1.2, (0.01, 0.0), (1.0, 1.0), [0.5], 1.0)
    assert coh == pytest.approx(inc, rel=1e-12)
