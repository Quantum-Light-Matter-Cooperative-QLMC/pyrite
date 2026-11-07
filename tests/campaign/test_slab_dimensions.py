"""Slab dimension compatibility and explicit infinite geometry."""

import pickle
import warnings
from dataclasses import asdict, replace

import pytest

from pyrite.geometry import Footprint, Slab
from pyrite.units import ureg


def test_slab_dimensions_match_footprint():
    slab = Slab("hopg", width=0.5 * ureg.cm, height=6 * ureg.mm)
    assert slab == Slab("hopg", footprint=Footprint(5, 6))
    assert slab.width.to("mm").magnitude == 5
    assert slab.height.to("mm").magnitude == 6
    assert slab.with_width(7 * ureg.mm).footprint == Footprint(7, 6)


def test_infinite_slab_and_removed_footprint_none():
    slab = Slab("hopg", width=None, height=None)
    assert slab.width is None and slab.height is None
    with pytest.raises(TypeError, match=r"removed in 0\.29\.0.*width=None, height=None"):
        Slab("hopg", footprint=None)


def test_slab_dimension_conflicts():
    with pytest.raises(TypeError, match="both width and height"):
        Slab("hopg", width=1 * ureg.mm)
    with pytest.raises(ValueError, match="both be None"):
        Slab("hopg", width=None, height=1 * ureg.mm)
    with pytest.raises(TypeError, match="mutually exclusive"):
        Slab("hopg", width=None, height=None, footprint=None)


def test_slab_quantity_sweep_matches_footprint_sweep():
    from pyrite.campaign.model import Scene, Sweep
    from pyrite.campaign.sweep import BeamSpec

    scene = Scene(
        beam=BeamSpec(energy_keV=30), target=Slab("hopg", width=5 * ureg.mm, height=6 * ureg.mm)
    )
    direct = Sweep(scene, axes={"target.width": [4 * ureg.mm, 5 * ureg.mm]})
    numeric = Sweep(scene, axes={"target.footprint.width_mm": [4, 5]})
    assert [item.target for _, item in direct.expand()] == [
        item.target for _, item in numeric.expand()
    ]


def test_infinite_slab_replacement_and_pickle_do_not_warn():
    slab = Slab("hopg", width=None, height=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert replace(slab, tilt_deg=40).width is None
        assert pickle.loads(pickle.dumps(slab)).with_tilt(40 * ureg.degree).width is None
    assert "_infinite_via_dimensions" not in asdict(slab)
