"""Explicit crystal-cut and body-target angles retain numeric orientation (#755)."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite import Beam, Scene, Sweep
from pyrite.beams import PointBunch, Spot
from pyrite.geometry import CrystalCut, RectangularBody, Slab, SubstanceRef, Target
from pyrite.optimization import Continuous, Objective, OptimizationStudy, ParameterSpace
from pyrite.units import Q_, ureg

BODY = RectangularBody(SubstanceRef("crystal", "hopg"), (5, 5), 0.05)
NUMERIC_CUT = CrystalCut((1, 0, 1), None, 17.0, 5.5, 200.0)
CUT_FIELDS = {"roll": "roll_deg", "miscut": "miscut_deg", "miscut_azimuth": "miscut_azim_deg"}
TARGET_FIELDS = {"tilt": "tilt_deg", "tilt_azimuth": "tilt_azim_deg"}


def _lowered(target):
    from pyrite.campaign.geometry import crystal_params

    cp = crystal_params("hopg")
    lowered = target.lower(cp, name_stem="hopg", beam_uvw=cp["beam_uvw"])
    return [point.case_keys() for point in lowered]


def test_semantic_cut_matches_numeric_fields_records_and_lowering():
    cut = CrystalCut(
        surface_hkl=(1, 0, 1),
        roll=17 * ureg.degree,
        miscut=Q_(5.5, "degree").to("radian"),
        miscut_azimuth=Q_(200, "degree"),
    )
    assert cut.roll_deg == 17.0
    assert cut.miscut_deg == pytest.approx(5.5)
    cut = replace(cut, miscut_deg=5.5)
    assert cut == NUMERIC_CUT
    assert asdict(cut) == asdict(NUMERIC_CUT)
    assert pickle.loads(pickle.dumps(cut)) == NUMERIC_CUT
    assert (cut.roll, cut.miscut, cut.miscut_azimuth) == (
        Q_(17, "degree"),
        Q_(5.5, "degree"),
        Q_(200, "degree"),
    )
    assert CrystalCut().roll == Q_(0, "degree")
    slab = Slab("hopg", 500.0, tilt_deg=84.5, footprint=None, cut=cut)
    assert _lowered(slab) == _lowered(replace(slab, cut=NUMERIC_CUT))


def test_cut_replacement_preserves_axis_and_other_angles():
    rolled = NUMERIC_CUT.with_roll(-0.25 * ureg.turn)
    assert rolled == replace(NUMERIC_CUT, roll_deg=-90.0)
    assert NUMERIC_CUT.with_miscut(1 * ureg.degree) == replace(NUMERIC_CUT, miscut_deg=1.0)
    rotated = NUMERIC_CUT.with_miscut_azimuth(2 * ureg.turn)
    assert rotated == replace(NUMERIC_CUT, miscut_azim_deg=720.0)
    assert NUMERIC_CUT.roll_deg == 17.0
    with pytest.raises(ValueError, match="0 <= miscut_deg < 90"):
        NUMERIC_CUT.with_miscut(0.25 * ureg.turn)


def test_semantic_target_matches_numeric_fields_and_slab_lowering():
    numeric = Target(BODY, 45.0, 180.0)
    semantic = Target(BODY, tilt=0.125 * ureg.turn, tilt_azimuth=np.pi * ureg.radian)
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert semantic.to_slab() == numeric.to_slab()
    assert (semantic.tilt, semantic.tilt_azimuth) == (Q_(45, "degree"), Q_(180, "degree"))
    assert Target(BODY).tilt == Q_(30, "degree")
    assert numeric.with_tilt(-20 * ureg.degree) == replace(numeric, tilt_deg=-20.0)
    rotated = numeric.with_tilt_azimuth(-0.5 * ureg.turn)
    assert rotated == replace(numeric, tilt_azim_deg=-180.0)
    assert numeric.tilt_deg == 45.0


def test_external_input_and_readback_are_detached():
    external = pint.UnitRegistry()
    angle = external.Quantity(np.array(17.0), "degree")
    cut = CrystalCut(roll=angle)
    target = Target(BODY, tilt=angle)
    angle.magnitude[...] = 80.0
    returned = cut.roll
    returned.ito("radian")
    assert (cut.roll, target.tilt) == (Q_(17, "degree"), Q_(17, "degree"))
    assert cut.roll._REGISTRY is ureg
    assert target.tilt._REGISTRY is ureg


INVALID = [
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
]


@pytest.mark.parametrize("keyword", sorted(CUT_FIELDS))
@pytest.mark.parametrize("angle, error", INVALID)
def test_cut_angles_refuse_invalid_quantities(keyword, angle, error):
    with pytest.raises(error, match=f"CrystalCut.{keyword}"):
        CrystalCut(**{keyword: angle})
    with pytest.raises(error, match=f"CrystalCut.{keyword}"):
        getattr(NUMERIC_CUT, f"with_{keyword}")(angle)


@pytest.mark.parametrize("keyword", sorted(TARGET_FIELDS))
@pytest.mark.parametrize("angle, error", INVALID)
def test_target_angles_refuse_invalid_quantities(keyword, angle, error):
    with pytest.raises(error, match=f"Target.{keyword}"):
        Target(BODY, **{keyword: angle})
    with pytest.raises(error, match=f"Target.{keyword}"):
        getattr(Target(BODY), f"with_{keyword}")(angle)


@pytest.mark.parametrize(
    "build, semantic, numeric",
    [(CrystalCut, s, n) for s, n in CUT_FIELDS.items()]
    + [(lambda **kw: Target(BODY, **kw), s, n) for s, n in TARGET_FIELDS.items()],
)
def test_semantic_and_numeric_keywords_conflict_even_at_default(build, semantic, numeric):
    with pytest.raises(TypeError, match=f"{numeric} and {semantic} are mutually exclusive"):
        build(**{numeric: 0.0 if semantic != "tilt" else 30.0, semantic: 0 * ureg.degree})


def test_sweep_and_optimization_reach_nested_cut_and_target_angles():
    scene = Scene(
        Beam(40, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 500.0, tilt_deg=84.5, footprint=None, cut=NUMERIC_CUT),
    )
    path = "target.cut.miscut"
    scenes = Sweep(scene, {path: [1 * ureg.degree, 0.05 * ureg.radian]}).expand()
    assert [point.target.cut.miscut_deg for _, point in scenes] == pytest.approx(
        [1.0, np.degrees(0.05)]
    )
    study = OptimizationStudy(
        scene,
        ParameterSpace({path: Continuous(1 * ureg.degree, 0.1 * ureg.radian)}),
        (Objective("f", "maximize", ""),),
    )
    assert study.parameters.parameters[path] == Continuous(1, np.degrees(0.1))
    assert scene.target.cut == NUMERIC_CUT
    body_scene = Scene(Beam(40, transverse=Spot(), longitudinal=PointBunch()), Target(BODY))
    tilted = Sweep(body_scene, {"target.tilt": [0.5 * ureg.radian]}).expand()
    assert [point.target.tilt_deg for _, point in tilted] == pytest.approx([np.degrees(0.5)])


def test_numeric_orientation_paths_keep_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.geometry import CrystalCut, RectangularBody, SubstanceRef, "
            "Target; assert CrystalCut((1, 0, 1), None, 17, 5, 200).roll_deg == 17; "
            "b = RectangularBody(SubstanceRef('crystal', 'hopg'), (5, 5), 0.05); "
            "assert Target(b, 45, 180).tilt_deg == 45; "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
