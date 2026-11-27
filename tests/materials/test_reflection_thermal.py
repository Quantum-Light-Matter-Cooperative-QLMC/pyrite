"""Shared uniaxial ADP projection for reflection couplings (#1480)."""

import numpy as np
import pytest

from pyrite.materials._reflection_thermal import common_uniaxial_b
from pyrite.materials.channeling.thermal import ChannelingDataError, thermal_site
from pyrite.materials.crystal import CRYSTALS, U_g, debye_waller, reciprocal_g_vector
from pyrite.montecarlo.spectrum.brem_lattice_share import (
    _reflection_shell_weights,
    reflection_shell_weights,
)


@pytest.mark.parametrize("hkl", [(0, 0, 2), (1, 1, 0), (1, 0, 2), (-1, 0, -2)])
def test_graphite_projection_matches_cartesian_exponent(hkl):
    lattice = CRYSTALS["hopg"]["lattice"]
    g, norm = reciprocal_g_vector(hkl, lattice)
    axis, _ = reciprocal_g_vector((0, 0, 1), lattice)
    site = thermal_site("hopg", "C")
    covariance = site.covariance(axis, 300.0)
    b = common_uniaxial_b("hopg", hkl, 300.0)
    assert debye_waller(norm, b) == pytest.approx(np.exp(-0.5 * g @ covariance @ g))
    assert U_g("hopg", hkl, 5000.0, B_ang2=b) == pytest.approx(
        U_g("hopg", hkl, 5000.0) * np.exp(-0.5 * g @ covariance @ g)
    )


def test_projection_rejects_unavailable_or_invalid_data():
    with pytest.raises(ChannelingDataError):
        common_uniaxial_b("hbn", (1, 0, 2), 300.0)
    with pytest.raises(ChannelingDataError):
        common_uniaxial_b("hopg", (1, 0, 2), 400.0)
    with pytest.raises(ValueError, match="nonzero"):
        common_uniaxial_b("hopg", (0, 0, 0), 300.0)


def test_shell_weights_use_same_projected_coupling():
    hkls = [(0, 0, 2), (1, 1, 2), (-1, -1, -2), (1, 0, 2)]
    composition = [("C", 0.113)]
    g, weights = _reflection_shell_weights(
        "hopg", hkls, lambda hkl: common_uniaxial_b("hopg", hkl, 300.0), composition
    )
    expected = sum(
        abs(U_g("hopg", hkl, 5000.0, B_ang2=common_uniaxial_b("hopg", hkl, 300.0))) ** 2
        for hkl in hkls
    )
    assert np.sum(weights["C"]) == pytest.approx(expected)
    scalar_g, scalar_w = reflection_shell_weights("hopg", hkls, 1.30, composition)
    callback_g, callback_w = _reflection_shell_weights("hopg", hkls, lambda _: 1.30, composition)
    np.testing.assert_array_equal(scalar_g, callback_g)
    np.testing.assert_array_equal(scalar_w["C"], callback_w["C"])
    np.testing.assert_array_equal(g, scalar_g)
