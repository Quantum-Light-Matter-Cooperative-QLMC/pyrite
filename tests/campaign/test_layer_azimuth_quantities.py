"""Explicit layer rotation quantities retain numeric orientation and records (#755)."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite import Beam, Scene, Sweep
from pyrite.beams import PointBunch, Spot
from pyrite.geometry import Footprint, Layer, Stack
from pyrite.optimization import Continuous, Objective, OptimizationStudy, ParameterSpace
from pyrite.units import Q_, ureg


def test_semantic_rotation_preserves_positional_fields_replacement_and_records():
    numeric = Layer("silicon", 2000.0, (1, 1, 1), 45.0)
    layer = Layer("silicon", 2000.0, (1, 1, 1), azimuth=0.125 * ureg.turn)
    assert layer == numeric
    assert asdict(layer) == asdict(numeric)
    assert layer.spec(2000.0) == numeric.spec(2000.0)
    assert pickle.loads(pickle.dumps(layer)) == numeric
    assert replace(layer) == numeric
    assert layer.azimuth == Q_(45.0, "degree")
    changed = layer.with_azimuth(-0.25 * ureg.turn)
    assert changed == replace(numeric, azimuth_deg=-90.0)
    assert layer.azimuth_deg == 45.0
    assert Layer("silicon", 2000.0).azimuth == Q_(0, "degree")
    assert layer.with_azimuth(2 * ureg.turn).azimuth_deg == 720.0


def test_external_input_and_readback_are_detached():
    external = pint.UnitRegistry()
    angle = external.Quantity(np.array(45.0), "degree")
    layer = Layer("silicon", 2000.0, azimuth=angle)
    angle.magnitude[...] = 80.0
    returned = layer.azimuth
    returned.ito("radian")
    assert layer.azimuth == Q_(45, "degree")
    assert layer.azimuth._REGISTRY is ureg


@pytest.mark.parametrize(
    "angle, error",
    [
        (45.0, TypeError),
        ("45 degree", TypeError),
        (Q_([45], "degree"), TypeError),
        (Q_(np.array(True), "degree"), TypeError),
        (Q_(1 + 2j, "degree"), TypeError),
        (Q_(1, "second"), ValueError),
        (Q_(1, "dimensionless"), ValueError),
        (Q_(1, "steradian"), ValueError),
        (Q_(np.inf, "degree"), ValueError),
        (Q_(np.nan, "degree"), ValueError),
        (Q_(np.longdouble("1e-4000"), "degree"), ValueError),
        (Q_(np.longdouble("1e4000"), "degree"), ValueError),
    ],
)
def test_semantic_rotation_refuses_invalid_quantities(angle, error):
    with pytest.raises(error, match="Layer.azimuth"):
        Layer("silicon", 2000.0, azimuth=angle)
    with pytest.raises(error, match="Layer.azimuth"):
        Layer("silicon", 2000.0).with_azimuth(angle)


def test_semantic_and_numeric_keywords_conflict_even_at_default():
    with pytest.raises(TypeError, match="azimuth_deg and azimuth"):
        Layer("silicon", 2000.0, azimuth_deg=0.0, azimuth=0 * ureg.degree)


def test_indexed_sweep_and_optimization_share_quantity_replacement():
    scene = Scene(
        Beam(40, transverse=Spot(), longitudinal=PointBunch()),
        Stack(
            layers=(Layer("hopg", 2000.0), Layer("silicon", 3000.0)),
            footprint=Footprint(5.0, 5.0),
        ),
    )
    path = "target.layers[1].azimuth"
    scenes = Sweep(scene, {path: [-0.125 * ureg.turn, 45 * ureg.degree]}).expand()
    assert [point.target.layers[1].azimuth_deg for _, point in scenes] == [-45, 45]
    study = OptimizationStudy(
        scene,
        ParameterSpace(
            {
                path: Continuous(-0.125 * ureg.turn, 45 * ureg.degree),
            }
        ),
        (Objective("f", "maximize", ""),),
    )
    assert study.parameters.parameters[path] == Continuous(-45, 45)
    assert study.scene({path: 0.1 * ureg.radian}).target.layers[1].azimuth_deg == pytest.approx(
        np.degrees(0.1)
    )
    assert scene.target.layers[1].azimuth_deg == 0.0


def test_numeric_layer_path_keeps_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.geometry import Layer; "
            "assert Layer('silicon', 2000, azimuth_deg=45).azimuth_deg == 45; "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
