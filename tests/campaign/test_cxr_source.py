"""``Physics(cxr_source="none")`` opts out of line scoring (#1327)."""

import pickle

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign.profiles import case_content_key


def scene(**kwargs):
    return pr.Scene(
        pr.Beam(30.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("silicon", thickness_ang=1.0e5, tilt_deg=30.0, width=None, height=None),
        **kwargs,
    )


def digest(case):
    return case_content_key(case, xsgen_tables={})


def test_default_is_lines_on_and_leaves_identity_unchanged():
    case = api.build_case(scene())
    assert "cxr_source" not in case
    assert case["hkl_list"]
    assert api.build_case(scene(physics=pr.Physics(cxr_source="mc"))) == case


def test_none_normalizes_and_pickles():
    off = pr.Physics(cxr_source=None)
    assert off == pr.Physics(cxr_source="none")
    assert off.cxr_source == "none"
    assert pickle.loads(pickle.dumps(off)) == off
    with pytest.raises(ValueError, match="cxr_source"):
        pr.Physics(cxr_source="external")


def test_off_case_has_no_reflections_selection_or_line_policy_and_new_identity():
    on = api.build_case(scene())
    off = api.build_case(scene(physics=pr.Physics(cxr_source="none")))
    assert off["cxr_source"] == "none"
    assert off["hkl_list"] == []
    for key in ("reflection_selection", "line_grid_policy"):
        assert key not in off
    assert off["Ne"] == off["Ne_brem"]
    assert np.array_equal(off["E_grid_brem"], on["E_grid_brem"])
    assert digest(off) != digest(on)
    assert pickle.loads(pickle.dumps(off)) == off


@pytest.mark.parametrize(
    ("setting", "match"),
    [
        ({"emission": "coherent"}, "coherent"),
        ({"transition_radiation": "history"}, "transition_radiation"),
    ],
)
def test_physics_refuses_line_only_settings(setting, match):
    with pytest.raises(ValueError, match=match):
        pr.Physics(cxr_source="none", **setting)


def test_numerics_refuse_line_only_settings():
    off = pr.Physics(cxr_source="none")
    with pytest.raises(ValueError, match="temporal_profile"):
        api.build_case(scene(physics=off), pr.Numerics(temporal_profile=True))
    adaptive = pr.Numerics(sampling=pr.config.AdaptiveSampling(target_rse=0.05, max_histories=200))
    with pytest.raises(ValueError, match="brem adaptive-precision"):
        api.build_case(scene(physics=off), adaptive)
