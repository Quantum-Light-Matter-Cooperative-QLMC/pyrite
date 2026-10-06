"""Analytic photon Born screen anchors; no transport or random sampling."""

import numpy as np
import pytest

from pyrite.materials.crystal import (
    HBARC_EV_ANG,
    chi_g,
    crystal_absorption_length_ang,
    kinematic_validity,
)


def test_reference_parameters():
    # omega=2/A, |chi|=0.001, Delta=0.4/A^2: independent hand calculation.
    result = kinematic_validity(2 * HBARC_EV_ANG, 0.001, -0.4, 100, 50)
    assert result["dyn"] == pytest.approx(0.01)
    assert result["extinction_length_ang"] == pytest.approx(1000)
    assert result["extinction_ratio"] == pytest.approx(20)
    assert result["photon_mixing"]
    assert not result["extinction_reachable"]


def test_thick_silicon_and_thin_hopg():
    # Si (111), 8 keV, scalar Bragg-matched photon. |g|=2pi sqrt(3)/a;
    # choose n.g=-g^2/(2 omega) so vacuum Delta=0. 1 mm >> exchange length.
    energy = 8000.0
    si = kinematic_validity(
        energy,
        abs(chi_g("silicon", (1, 1, 1), energy, 0, True)),
        0.0,
        1e7,
        float(crystal_absorption_length_ang("silicon", energy)),
    )
    assert si["photon_mixing"] and si["extinction_reachable"]
    assert si["extinction_length_ang"] < 1e5
    # HOPG (002), 25 keV incident, 119 deg observation, t=100 nm.
    beta = np.sqrt(1 - (1 + 25_000 / 510_998.95) ** -2)
    g = 4 * np.pi / 6.708
    omega = beta * g / (1 - beta * np.cos(np.deg2rad(119)))
    energy = omega * HBARC_EV_ANG
    hopg = kinematic_validity(
        energy,
        abs(chi_g("hopg", (0, 0, 2), energy, 0, True)),
        g**2 + 2 * omega * g * np.cos(np.deg2rad(119)),
        1000,
        float(crystal_absorption_length_ang("hopg", energy)),
    )
    assert hopg["dyn"] < 0.01
    assert hopg["extinction_ratio"] > 1
    assert not hopg["photon_mixing"] and not hopg["extinction_reachable"]


def test_zero_coupling_thickness_and_absorption_limits():
    zero = kinematic_validity(1000, 0, 0, 100, np.inf)
    assert zero["dyn"] == 0
    assert zero["extinction_length_ang"] == np.inf
    thin = kinematic_validity(1000, 0.001, 1, 0, 100)
    assert thin["extinction_ratio"] == np.inf
    transparent = kinematic_validity(1000, 0.001, 1, 100, np.inf)
    assert transparent["extinction_ratio"] == pytest.approx(
        transparent["extinction_length_ang"] / 100
    )


@pytest.mark.parametrize(
    "args",
    [
        (0, 0.1, 1, 1, 1),
        (100, -1, 1, 1, 1),
        (100, 1, 1, -1, 1),
        (100, 1, np.nan, 1, 1),
        (100, 1, 1, 1, np.nan),
        (100, 1, 1, 1, 0),
    ],
)
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        kinematic_validity(*args)
