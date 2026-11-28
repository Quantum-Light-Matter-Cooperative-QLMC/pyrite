"""Calibration of the on-string/inter-string pair classifier (#585)."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks import channeling_submev_multistring as ms
from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states


def test_nearest_string_distance_includes_images():
    strings = [SimpleNamespace(position_ang=(0.0, 0.0)), SimpleNamespace(position_ang=(1.0, 0.5))]
    profile = SimpleNamespace(period_ang=(4.0, 1.0), strings=strings)
    # The image at (0, 1) is 1.0 away; the second string is sqrt(1.25) away.
    assert ms.nearest_string_distance(profile) == pytest.approx(1.0)


def test_classify_thresholds():
    assert ms.classify(ms.ON_STRING_MIN) == "on_string"
    assert ms.classify(ms.INTER_STRING_MAX) == "inter_string"
    assert ms.classify(0.5 * (ms.ON_STRING_MIN + ms.INTER_STRING_MAX)) == "mixed"


def _pairs(crystal, uvw, kinetic_eV):
    profile = axial_profile(crystal, uvw, ms.TEMPERATURE_K)
    lo, barrier = profile.extrema()
    gamma, _ = longitudinal_kinematics(kinetic_eV)
    orders = ms._orders(profile, gamma, barrier - lo, ms.CUTOFF)
    shape = ms.grid_shape(orders)
    d_nn = ms.nearest_string_distance(profile)
    masks = {f: ms.disc_mask(profile, shape, f * d_nn / 2) for f in ms.DISC_FRACTIONS}
    states = axial_states(profile, gamma, (0.013, 0.021), max_order=orders)
    return ms.pair_records(states, barrier, np.inf, masks, shape)


@pytest.mark.slow
def test_single_string_2p_to_1s_is_on_string():
    rows = [r for r in _pairs("tungsten", (0, 0, 1), 300e3) if r["lower"] == 0]
    line = min(rows, key=lambda r: r["gap_eV"])
    assert line["gap_eV"] == pytest.approx(250.2, abs=1.0)
    assert line["class"] == "on_string"
    assert line["disc_fraction"][0.5] == pytest.approx(1.0, abs=0.05)
    assert line["grid_closure"] == pytest.approx(1.0, abs=1e-9)


def test_dumbbell_ground_splitting_is_inter_string():
    rows = _pairs("silicon", (1, 1, 0), 30e3)
    assert rows
    for r in rows:
        if r["gap_eV"] < 15.0:
            assert r["class"] == "inter_string"
            assert r["disc_fraction"][0.5] < ms.INTER_STRING_MAX
