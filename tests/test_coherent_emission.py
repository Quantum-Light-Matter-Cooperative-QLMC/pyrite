"""Numerical limiting-case gates for opt-in coherent segment summation."""

import numpy as np
import pytest

from cxr_mc.montecarlo import mc_spectrum
from cxr_mc.montecarlo._backend import REAL

E_GRID = np.arange(700.0, 1500.0)
KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.01]),
}
RTOL = max(1e-12, 100.0 * float(np.finfo(REAL).eps))


def _segments(count=1):
    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
    }


def test_single_segment_coherent_equals_incoherent_self_term():
    segments = _segments()

    incoherent = mc_spectrum(segments, E_GRID, coherent=False, **KWARGS)
    coherent = mc_spectrum(segments, E_GRID, coherent=True, **KWARGS)

    assert np.max(incoherent) > 0.0
    np.testing.assert_allclose(coherent, incoherent, rtol=RTOL)


def test_identical_in_phase_electrons_reach_n_squared_limit():
    single = mc_spectrum(_segments(), E_GRID, coherent=True, **KWARGS)
    pair = mc_spectrum(_segments(2), E_GRID, coherent=True, **KWARGS)

    # Field doubles, intensity quadruples, then per-electron /Ne normalization
    # leaves twice the one-electron yield.
    np.testing.assert_allclose(pair, 2.0 * single, rtol=RTOL)


def test_coherent_components_are_rejected():
    with pytest.raises(ValueError, match="incompatible with components"):
        mc_spectrum(_segments(), E_GRID, coherent=True, components=True, **KWARGS)
