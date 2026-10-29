"""Explicit target tilt quantities retain numeric orientation and records (#755)."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite import Beam, Scene, Sweep
from pyrite.beams import PointBunch, Spot
from pyrite.geometry import Footprint, Layer, Slab, Stack
from pyrite.optimization import Continuous, Objective, OptimizationStudy, ParameterSpace
from pyrite.units import Q_, ureg


def _slab(**kwargs):
    return Slab("hopg", 2000.0, footprint=None, **kwargs)


def _stack(**kwargs):
    layers = (Layer("hopg", 2000.0), Layer("silicon", 3000.0))
    return Stack(layers, footprint=None, **kwargs)


def _lowered(target):
    from pyrite.campaign.geometry import crystal_params

    cp = crystal_params("hopg")
    return [
        point.case_keys() for point in target.lower(cp, name_stem="hopg", beam_uvw=cp["beam_uvw"])
    ]


@pytest.mark.parametrize("build", [_slab, _stack])
def test_semantic_tilts_match_numeric_fields_records_and_cases(build):
    numeric = build(tilt_deg=45.0, tilt_azim_deg=180.0)
    semantic = build(tilt=0.125 * ureg.turn, tilt_azimuth=np.pi * ureg.radian)
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert _lowered(semantic) == _lowered(numeric)
    assert semantic.tilt == Q_(45.0, "degree")
    assert semantic.tilt_azimuth == Q_(180.0, "degree")
    assert build().tilt == Q_(30.0, "degree")
    assert build().tilt_azimuth == Q_(0.0, "degree")


@pytest.mark.parametrize("build", [_slab, _stack])
def test_replacement_preserves_other_settings_and_original(build):
    target = build(tilt_deg=45.0, tilt_azim_deg=180.0)
    tilted = target.with_tilt(-20 * ureg.degree)
    assert tilted == replace(target, tilt_deg=-20.0)
    rotated = target.with_tilt_azimuth(-0.5 * ureg.turn)
    assert rotated == replace(target, tilt_azim_deg=-180.0)
    assert (target.tilt_deg, target.tilt_azim_deg) == (45.0, 180.0)


def test_sweep_shape_and_order_are_preserved():
    slab = _slab(tilt=Q_([0.5, 0.1, 0.2], "radian"), tilt_azimuth=Q_([180, 0], "degree"))
    np.testing.assert_allclose(slab.tilt_deg, np.degrees([0.5, 0.1, 0.2]))
    np.testing.assert_array_equal(slab.tilt_azim_deg, [180.0, 0.0])
    readback = slab.tilt
    assert readback.shape == (3,)
    readback.magnitude[0] = 99.0
    assert slab.tilt_deg[0] == pytest.approx(np.degrees(0.5))
    numeric = _slab(tilt_deg=list(np.degrees([0.5, 0.1, 0.2])), tilt_azim_deg=[180.0, 0.0])
    assert _lowered(slab) == _lowered(numeric)


def test_external_input_and_readback_are_detached():
    external = pint.UnitRegistry()
    angle = external.Quantity(np.array([45.0, 60.0]), "degree")
    slab = _slab(tilt=angle)
    angle.magnitude[...] = 80.0
    returned = slab.tilt
    returned.ito("radian")
    np.testing.assert_array_equal(slab.tilt.magnitude, [45.0, 60.0])
    assert slab.tilt._REGISTRY is ureg


@pytest.mark.parametrize("keyword", ["tilt", "tilt_azimuth"])
@pytest.mark.parametrize(
    "angle, error",
    [
        (45.0, TypeError),
        ("45 degree", TypeError),
        (Q_([[45.0]], "degree"), TypeError),
        (Q_(np.array(True), "degree"), TypeError),
        (Q_(1 + 2j, "degree"), TypeError),
        (Q_(1, "second"), ValueError),
        (Q_(1, "dimensionless"), ValueError),
        (Q_(1, "steradian"), ValueError),
        (Q_(np.array([], dtype=float), "degree"), ValueError),
        (Q_(np.inf, "degree"), ValueError),
        (Q_(np.nan, "degree"), ValueError),
        (Q_(np.longdouble("1e-4000"), "degree"), ValueError),
        (Q_(np.longdouble("1e4000"), "degree"), ValueError),
    ],
)
def test_semantic_tilts_refuse_invalid_quantities(keyword, angle, error):
    with pytest.raises(error, match=f"Slab.{keyword}"):
        _slab(**{keyword: angle})
    with pytest.raises(error, match=f"Stack.{keyword}"):
        getattr(_stack(), f"with_{keyword}")(angle)


@pytest.mark.parametrize("build", [_slab, _stack])
def test_semantic_and_numeric_keywords_conflict_even_at_default(build):
    with pytest.raises(TypeError, match="tilt_deg and tilt"):
        build(tilt_deg=30.0, tilt=30 * ureg.degree)
    with pytest.raises(TypeError, match="tilt_azim_deg and tilt_azimuth"):
        build(tilt_azim_deg=0.0, tilt_azimuth=0 * ureg.degree)


def test_degenerate_geometry_guards_apply_to_quantities():
    with pytest.raises(ValueError, match="tilt_deg == 0"):
        _slab(tilt=0 * ureg.radian)
    with pytest.raises(ValueError, match="tilt_azim_deg == 90"):
        _stack().with_tilt_azimuth(0.25 * ureg.turn)
    assert _slab(tilt=0 * ureg.degree, allow_normal_incidence=True).tilt_deg == 0.0


def test_on_substrate_forwards_semantic_tilts():
    stack = Stack.on_substrate(
        "hopg",
        2000.0,
        "silicon",
        tilt=0.5 * ureg.radian,
        tilt_azimuth=180 * ureg.degree,
        footprint=None,
    )
    assert stack.tilt_deg == pytest.approx(np.degrees(0.5))
    assert stack.tilt_azim_deg == 180.0


def test_sweep_and_optimization_share_quantity_replacement():
    scene = Scene(
        Beam(40, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 2000.0, footprint=Footprint(5.0, 5.0)),
    )
    scenes = Sweep(scene, {"target.tilt": [0.5 * ureg.radian, 20 * ureg.degree]}).expand()
    assert [point.target.tilt_deg for _, point in scenes] == pytest.approx([np.degrees(0.5), 20])
    study = OptimizationStudy(
        scene,
        ParameterSpace({"target.tilt": Continuous(10 * ureg.degree, 1 * ureg.radian)}),
        (Objective("f", "maximize", ""),),
    )
    assert study.parameters.parameters["target.tilt"] == Continuous(10, np.degrees(1))
    candidate = study.scene({"target.tilt": 0.5 * ureg.radian})
    assert candidate.target.tilt_deg == pytest.approx(np.degrees(0.5))
    assert scene.target.tilt_deg == 30.0


def test_numeric_target_path_keeps_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.geometry import Slab, Stack, Layer; "
            "assert Slab('hopg', 2000, 45, 180, footprint=None).tilt_deg == 45; "
            "s = Stack((Layer('hopg', 2000), Layer('silicon', 3000)), 45, footprint=None); "
            "assert s.tilt_deg == 45; "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
