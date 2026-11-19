"""Fiber texture restricts candidates before thermal factors and shell selection."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials.crystal import CRYSTALS, dominant_reflections, reciprocal_g_vector
from pyrite.montecarlo.runner.reflection_cutoff import iter_reflection_shells, reflection_shells


@pytest.mark.parametrize("crystal", ["hopg", "hbn"])
def test_fiber_shells_are_basal_and_begin_with_the_retired_pins(crystal):
    spec = CATALOG.crystal(crystal)
    assert spec.texture == "fiber"
    assert not spec.hkl_list
    shells = list(iter_reflection_shells(crystal, spec.B_ang2, 12.0))
    assert shells[:2] == [(2.0, [(0, 0, -2), (0, 0, 2)]), (4.0, [(0, 0, -4), (0, 0, 4)])]
    hkls = [h for _, members in shells for h in members]
    assert all(h == k == 0 and order % 2 == 0 for h, k, order in hkls)
    assert (0, 0, 6) in hkls and (0, 0, 8) in hkls
    assert shells == reflection_shells(crystal, spec.B_ang2, 12.0)
    assert list(iter_reflection_shells(crystal, spec.B_ang2, 0.1)) == []


def test_fiber_axis_is_a_reciprocal_direction_in_a_nonorthogonal_cell(monkeypatch):
    info = dict(CRYSTALS["hopg"], texture="fiber", texture_axis_hkl=(2, 2, 0))
    monkeypatch.setitem(CRYSTALS, "test-fiber", info)
    members = dominant_reflections("test-fiber", n_families=10**9, g_max_invang=12.0)
    assert members
    axis, _ = reciprocal_g_vector((1, 1, 0), info["lattice"])
    for hkl in members:
        g, _ = reciprocal_g_vector(hkl, info["lattice"])
        np.testing.assert_allclose(np.cross(g, axis), 0.0, atol=1e-12)


def test_single_crystal_enumeration_keeps_in_plane_families():
    spec = CATALOG.crystal("mos2")
    assert spec.texture == "single-crystal"
    members = dominant_reflections("mos2", n_families=10**9, B_ang2=spec.B_ang2)
    assert any(h or k for h, k, _ in members)


def test_crystal_texture_validates_direct_construction():
    spec = CATALOG.crystal("hopg")
    with pytest.raises(ValueError, match="texture"):
        replace(spec, texture="powder")
    with pytest.raises(ValueError, match="surface_hkl"):
        replace(spec, surface_hkl=None, beam_uvw=(0, 0, 1))
    with pytest.raises(ValueError, match="nonzero surface_hkl"):
        replace(spec, surface_hkl=(0, 0, 0))
    with pytest.raises(ValueError, match="parallel"):
        replace(spec, hkl_families=((1, 0, 0),))
