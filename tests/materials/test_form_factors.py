"""Atomic form factors, now sourced from xraydb (Waasmaier-Kirfel f0 + Chantler
f',f''). f0(g=0) == Z is the cheap regression that catches a backend/units slip;
load_henke must still return a finite, physical (E, f1, f2) table per element."""

import numpy as np
import pytest

import pyrite.materials.crystal as crystal_module
from pyrite.materials.atomic import (
    Z_TABLE,
    atomic_form_factor,
    cromer_mann_f0,
    henke_dispersion,
    load_henke,
)

# the structure-factor elements the project models (light + edge-prone + heavy)
ELEMENTS = [
    "C",
    "Li",
    "F",
    "Si",
    "Ge",
    "S",
    "Fe",
    "Mo",
    "Nb",
    "Pd",
    "Se",
    "Zr",
    "Te",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Pt",
    "Bi",
]


@pytest.mark.parametrize("element", ELEMENTS)
def test_f0_at_zero_equals_Z(element):
    # f0(g=0) must reproduce the atomic number (Waasmaier-Kirfel -> Z at s=0)
    assert float(cromer_mann_f0(element, 0.0)) == pytest.approx(Z_TABLE[element], abs=0.1)


def test_f0_decreases_with_g():
    f = cromer_mann_f0("Mo", np.array([0.0, 1.0, 3.0, 6.0]))
    assert np.all(np.diff(f) < 0)


def test_tellurium_registered():
    assert Z_TABLE["Te"] == 52
    assert "Te" in Z_TABLE


def test_niobium_registered_and_edge_prone():
    assert Z_TABLE["Nb"] == 41
    assert "Nb" in crystal_module._EDGE_PRONE

    E, f1, f2 = load_henke("Nb")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)


def test_phosphorus_registered_and_edge_prone():
    assert Z_TABLE["P"] == 15
    assert "P" in crystal_module._EDGE_PRONE

    E, f1, f2 = load_henke("P")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)


@pytest.mark.parametrize("element", ["Fe", "Bi", "Re", "Ta"])
def test_new_transport_elements_have_edge_aware_form_factors(element):
    # These elements have L (Fe) or M (Bi/Re/Ta) edges in the catalog's
    # 350--3500 eV line grid, so the default structure-factor path must retain
    # xraydb's anomalous terms rather than reducing them to f0.
    assert element in crystal_module._EDGE_PRONE

    F = atomic_form_factor(element, 1.0, np.array([1000.0, 3000.0]))
    assert F.shape == (2,)
    assert np.all(np.isfinite(F.real))
    assert np.all(np.isfinite(F.imag))


def test_unknown_element_raises():
    assert "Xx" not in Z_TABLE
    with pytest.raises(KeyError):
        _ = Z_TABLE["Xx"]


def test_henke_table_loads_and_is_complex_factor():
    E, f1, f2 = load_henke("Te")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)
    F = atomic_form_factor("Te", 1.0, 10000.0)  # in-range energy -> finite complex
    assert np.isfinite(F.real) and np.isfinite(F.imag)


def test_out_of_range_energy_is_nan():
    # E below the tabulated grid (and the brem-grid E=0 bin) must read as NaN,
    # not raise, so absorption_length_ang stays index-aligned.
    fp, fpp = henke_dispersion("Te", np.array([0.0, 5000.0]))
    assert np.isnan(fp[0]) and np.isnan(fpp[0])
    assert np.isfinite(fp[1]) and np.isfinite(fpp[1])
