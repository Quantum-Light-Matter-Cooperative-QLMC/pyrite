"""Focused structural regressions for the Nb/Ti/V telluride/selenide batch."""

from collections import Counter

import numpy as np
import pytest

from cxr_mc.materials import CATALOG
from cxr_mc.materials.crystal import CRYSTALS, U_g, chi_g, structure_factor


@pytest.mark.parametrize(
    ("key", "lattice", "volume", "counts", "label"),
    [
        (
            "nbte2",
            (19.390, 3.642, 9.375, 90.0, 134.58, 90.0),
            471.557,
            {"Nb": 6, "Te": 12},
            "NbTe2",
        ),
        (
            "tite2",
            (3.777, 3.777, 6.498, 90.0, 90.0, 120.0),
            80.279,
            {"Ti": 1, "Te": 2},
            "1T-TiTe2 (001)",
        ),
        (
            "vse2",
            (3.357, 3.357, 6.104, 90.0, 90.0, 120.0),
            59.573,
            {"V": 1, "Se": 2},
            "1T-VSe2 (001)",
        ),
        (
            "vte2",
            (18.984, 3.5947, 9.069, 90.0, 134.62, 90.0),
            440.510,
            {"V": 6, "Te": 12},
            "1T''-VTe2",
        ),
    ],
)
def test_nb_ti_v_structures_match_source_cells_and_stoichiometry(
    key, lattice, volume, counts, label
):
    """Pin the selected ambient bulk phases rather than formula-only aliases."""
    info = CRYSTALS[key]

    actual_lattice = tuple(
        info["lattice"][name] for name in ("a", "b", "c", "alpha", "beta", "gamma")
    )
    assert actual_lattice == pytest.approx(lattice, abs=1e-8)
    assert info["V_cell"] == pytest.approx(volume, abs=0.06)
    assert Counter(element for element, _ in info["basis"]) == counts
    assert len(info["basis"]) == sum(counts.values())
    assert CATALOG.material(key).label == label


@pytest.mark.parametrize("key", ["nbte2", "tite2", "vse2", "vte2"])
def test_nb_ti_v_basal_surface_couplings_are_finite_and_nonzero(key):
    """The source-setting basal (001) must be a usable symmetric-cut harmonic."""
    reflection = (0, 0, 1)
    spec = CATALOG.crystal(key)

    assert spec.surface_hkl == reflection
    assert spec.beam_uvw is None
    assert spec.hkl_families == (reflection,)
    assert spec.hkl_list == (reflection, (0, 0, -1))
    assert spec.B_ang2 == pytest.approx(0.6)
    assert spec.hkl_reason

    structure, g = structure_factor(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    susceptibility = chi_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    potential = U_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)

    assert np.isfinite(g) and g > 0.0
    for coupling in (structure, susceptibility, potential):
        assert np.isfinite(abs(coupling)) and abs(coupling) > 0.0
