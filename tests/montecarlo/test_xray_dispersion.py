"""``xray_dispersion`` switch: vacuum baseline vs in-medium line kinematics.

The refractive model replaces ``k = omega`` with the Maxwell dispersion relation
``k = n(omega) omega`` in the bulk crystal, which moves the CXR resonance from
``omega_res = v.g / (1 - v.n_hat)`` to ``omega_res = v.g / (1 - Re n (v.n_hat))``.
For ``delta = 1 - Re n << 1`` that is a fractional line shift of
``-delta (v.n_hat) / (1 - v.n_hat)``, which is what these tests measure.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, refractive_index
from pyrite.montecarlo.spectrum import mc_spectrum
from pyrite.montecarlo.spectrum.lines import _observation_direction
from pyrite.montecarlo.transport import beta_from_keV

CRYSTAL = "hopg"
HKL = (0, 0, 2)
B_ANG2 = 0.4
E_KEV = 100.0
THICKNESS_ANG = 1e5

# Long segment -> narrow sinc envelope, so the spectral peak resolves a shift of
# order delta * E_res (~6e-2 eV here) instead of drowning in a ~100 eV wide line.
SEG_LENGTH_ANG = 8000.0
E_GRID = np.linspace(1599.0, 1601.5, 100_001)


def _segments():
    return {
        "r_mid": np.array([[0.0, 0.0, 5000.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([SEG_LENGTH_ANG]),
        "E_keV": np.array([E_KEV]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": THICKNESS_ANG,
        "crystal_width_ang": 1e6,
        "crystal_height_ang": 1e6,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _single_layer():
    """One-layer absorber stack, which routes the run onto the per-hkl path."""
    info = CRYSTALS[CRYSTAL]
    n_atoms = len(info["basis"]) / info["V_cell"]
    return [(0.0, THICKNESS_ANG, [("C", n_atoms)])]


def _spectrum(**kwargs):
    return mc_spectrum(_segments(), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, **kwargs)


def _predicted_relative_shift(E_res_eV):
    v_dot_n = float(beta_from_keV(np.array([E_KEV]))[0]) * float(
        np.array([0.0, 0.0, 1.0]) @ _observation_direction(np.deg2rad(119.0), None)
    )
    delta = 1.0 - float(refractive_index(CRYSTAL, E_res_eV).real)
    return -delta * v_dot_n / (1.0 - v_dot_n)


def test_vacuum_is_the_default_and_leaves_the_spectrum_untouched():
    np.testing.assert_array_equal(_spectrum(), _spectrum(xray_dispersion="vacuum"))


@pytest.mark.parametrize("layers", [None, _single_layer()], ids=["batched", "per_hkl"])
def test_refractive_shifts_the_line_by_the_in_medium_denominator(layers):
    vac = _spectrum(layers=layers)
    ref = _spectrum(layers=layers, xray_dispersion="refractive")
    assert vac.max() > 0.0

    E_vac = E_GRID[vac.argmax()]
    E_ref = E_GRID[ref.argmax()]
    measured = (E_ref - E_vac) / E_vac
    # Positive: the detector looks upstream (v.n_hat < 0), so the in-medium
    # denominator is larger than the vacuum one and the line moves up in energy.
    assert measured > 0.0
    np.testing.assert_allclose(measured, _predicted_relative_shift(E_vac), rtol=2e-3)


def test_unknown_dispersion_model_is_rejected():
    with pytest.raises(ValueError, match="xray_dispersion must be one of"):
        _spectrum(xray_dispersion="in_medium")


def test_refractive_with_coherent_is_refused_until_the_phase_is_in_medium():
    with pytest.raises(NotImplementedError, match="propagation phase"):
        _spectrum(xray_dispersion="refractive", coherent=True)
