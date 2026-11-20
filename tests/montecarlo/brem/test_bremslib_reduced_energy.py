"""BremsLib ``k/T1`` sub-nodes between the library's 0 and 0.1 nodes.

The pinned values are BremsLib v2.0.8 rows for lithium and the output of the
library's own interpolation program (``Interpolate_DCS`` v1.0.7, Linux build,
default parameters, ``Z=3``) at the sub-nodes, so the scheme is anchored to
the program rather than re-derived from it.

Validation: bremslib-electron-electron
"""

import warnings

import numpy as np
import pytest

from pyrite._bremslib_reduced_energy import SMALL_REDUCED_ENERGY_NODES, refine_reduced_energy
from pyrite._bremslib_table import top_reduced_energy
from pyrite._physics_fallback import PhysicsFallbackWarning

NOMINAL = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0])

#: ``SDCS_3.txt`` scaled SDCS (mb) at ``T1`` = 0.5 and 1 MeV.
LI_SDCS = {
    0.5: [
        *(20.0460176585, 13.3901847611, 10.1536816257, 7.9998443808, 6.3954585831),
        *(5.1189283485, 4.056176832, 3.1374419696, 2.3091876511, 1.5048898021),
        *(1.0591781784, 0.7863080413, 0.3639802507),
    ],
    1.0: [
        *(17.541176247, 11.757324574, 8.917003765, 6.9944427832, 5.5568293776),
        *(4.4182102379, 3.47586383, 2.6651531228, 1.9357689904, 1.2290240927),
        *(0.8409780229, 0.6058857477, 0.2101696687),
    ],
}

#: ``SDCS_Born_3.txt`` at ``T1`` = 1 MeV, on its 20 ``k/T1`` values.
BORN_K = np.array(
    [0.0, 1e-4, 2.8e-4, 6.3e-4, 1.25e-3, 2.4e-3, 4.6e-3, 8.5e-3, 1.58e-2, 2.9e-2, 5.4e-2]
    + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
LI_BORN_1MEV = [
    *(17.4598816835, 17.4577890988, 17.4499833765, 17.4257724122, 17.3721263126),
    *(17.2618221907, 17.040987712, 16.6590366502, 16.0221319145, 15.0720633496),
    *(13.650680515, 11.6895516665, 8.8493278832, 6.9341250166, 5.5041350059),
    *(4.3724723001, 3.4370312833, 2.6338106468, 1.9130730265, 1.2163342761),
]

#: Interpolate_DCS scaled SDCS (mb) at the sub-nodes 0.004, 0.01, 0.02, 0.035, 0.06.
INTERPOLATE_DCS = {
    0.5: [19.610788, 18.994055, 18.056917, 16.842858, 15.239932],
    1.0: [17.180968, 16.596642, 15.770046, 14.765766, 13.423354],
}


def _refine(t1_MeV, sdcs, born=None):
    t1 = np.asarray(t1_MeV, dtype=float)
    sdcs = np.atleast_2d(np.asarray(sdcs, dtype=float))
    reduced = np.broadcast_to(NOMINAL, sdcs.shape).copy()
    reduced[:, -1] = top_reduced_energy(t1)
    shape = np.ones((*sdcs.shape, 3))
    return refine_reduced_energy(t1, reduced, sdcs, shape, born)


def _born(values=LI_BORN_1MEV, t1_MeV=1.0):
    return {
        "born_t1_MeV": np.array([t1_MeV]),
        "born_k_over_t1": BORN_K,
        "born_sdcs_mb": np.atleast_2d(values),
    }


def test_below_one_mev_sub_nodes_follow_the_library_sdcs_spline():
    sdcs, _ = _refine([0.5], LI_SDCS[0.5])
    np.testing.assert_allclose(sdcs[0, 1:6], INTERPOLATE_DCS[0.5], rtol=3e-4)
    # Linear interpolation, the previous scheme, sits 5.6 % high at 0.06.
    linear = np.interp(SMALL_REDUCED_ENERGY_NODES, NOMINAL[:2], LI_SDCS[0.5][:2])
    assert linear[-1] / INTERPOLATE_DCS[0.5][-1] - 1.0 > 0.05


def test_above_one_mev_sub_nodes_follow_the_born_ratio_spline():
    sdcs, _ = _refine([1.0], LI_SDCS[1.0], _born())
    np.testing.assert_allclose(sdcs[0, 1:6], INTERPOLATE_DCS[1.0], rtol=3e-4)


def test_library_nodes_are_kept_and_the_sub_nodes_inserted_after_the_first():
    sdcs, shape = _refine([1.0], LI_SDCS[1.0], _born())
    assert sdcs.shape == (1, 18) and shape.shape == (1, 18, 3)
    np.testing.assert_array_equal(sdcs[0, [0, *range(6, 18)]], LI_SDCS[1.0])


def test_below_thirty_kev_the_library_scheme_is_linear():
    curved = np.array(LI_SDCS[0.5])
    sdcs, _ = _refine([0.02], curved)
    w = np.asarray(SMALL_REDUCED_ENERGY_NODES) / 0.1
    np.testing.assert_allclose(sdcs[0, 1:6], (1 - w) * curved[0] + w * curved[1], rtol=1e-15)


def test_a_linear_cross_section_is_reproduced_exactly():
    """Both fits contain straight lines, and the node anchoring keeps them."""
    line = 10.0 * (1.0 - 0.5 * NOMINAL)
    plain, _ = _refine([0.5], line)
    born, _ = _refine([1.0], line, _born())
    expected = 10.0 * (1.0 - 0.5 * np.asarray(SMALL_REDUCED_ENERGY_NODES))
    np.testing.assert_allclose(plain[0, 1:6], expected, rtol=1e-12)
    # The Born ratio is not linear, so only the anchored nodes are exact here.
    np.testing.assert_allclose(born[0, [0, 6]], line[:2], rtol=1e-15)


def test_sub_node_shapes_mix_their_neighbours_linearly():
    t1 = np.array([0.5])
    sdcs = np.atleast_2d(LI_SDCS[0.5])
    reduced = np.broadcast_to(NOMINAL, sdcs.shape).copy()
    shape = np.zeros((1, 13, 2))
    shape[0, :, 0] = 1.0
    shape[0, 1:, 0] = 0.0
    shape[0, 1:, 1] = 1.0
    _, refined = refine_reduced_energy(t1, reduced, sdcs, shape)
    w = np.asarray(SMALL_REDUCED_ENERGY_NODES) / 0.1
    np.testing.assert_allclose(refined[0, 1:6], np.c_[1 - w, w], rtol=1e-15)


def test_a_non_positive_born_value_falls_back_to_the_sdcs_spline_with_a_warning():
    """BremsLib v2.0.8 has a zero Born SDCS at Z = 5, T1 = 3 MeV, k/T1 = 0.0046."""
    broken = np.array(LI_BORN_1MEV)
    broken[6] = 0.0
    with pytest.warns(PhysicsFallbackWarning, match="T1 = 1 MeV"):
        guarded, _ = _refine([1.0], LI_SDCS[1.0], _born(broken))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain, _ = _refine([1.0], LI_SDCS[1.0])
    np.testing.assert_array_equal(guarded, plain)
    # The plain spline stays within 1.5 % of the Born-ratio scheme here.
    np.testing.assert_allclose(plain[0, 1:6], INTERPOLATE_DCS[1.0], rtol=1.5e-2)
