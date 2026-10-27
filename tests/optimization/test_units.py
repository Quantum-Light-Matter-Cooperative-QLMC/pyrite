"""B3 (#755): Pint units on optimization objectives, constraints and spaces."""

from __future__ import annotations

import pytest

from pyrite.campaign.geometry import Footprint, Slab
from pyrite.campaign.model import Scene
from pyrite.campaign.sweep import BeamSpec
from pyrite.optimization import (
    Constraint,
    Continuous,
    FixedWindowYield,
    Integer,
    Objective,
    OptimizationStudy,
    ParameterSpace,
    PyriteEvaluator,
)
from pyrite.optimization.identity import study_definition_digest
from pyrite.units import ureg


def _base():
    return Scene(
        beam=BeamSpec(energy_keV=40),
        target=Slab("hopg", thickness_ang=2_000, footprint=Footprint(5.0, 5.0)),
    )


def test_units_parse_and_keep_caller_text():
    assert Objective("f", "maximize", "1/sr").pint_unit == ureg.Unit("1/sr")
    assert Objective("f", "maximize", "").pint_unit == ureg.dimensionless
    assert Objective("f", "maximize", ureg.Unit("eV")).unit == "electron_volt"
    assert Constraint("w", "<=", 1.0, "eV").unit == "eV"


@pytest.mark.parametrize("text", ["photons/(sr incident electron)", "photons/electron", "eV/"])
def test_unparseable_units_are_refused(text):
    with pytest.raises(ValueError, match=r"Objective\.unit could not parse"):
        Objective("f", "maximize", text)
    with pytest.raises(ValueError, match=r"Constraint\.unit could not parse"):
        Constraint("f", "<=", 1.0, text)
    with pytest.raises(TypeError, match="unit string"):
        Objective("f", "maximize", 3)


def test_constraint_bound_quantities_lower_to_declared_unit():
    numeric = Constraint("fwhm", "<=", 1.0, "eV")
    for bound in (1.0 * ureg.eV, 1e-3 * ureg.keV, 1000.0 * ureg.meV):
        assert Constraint("fwhm", "<=", bound, "eV") == numeric
    assert Constraint("t", "<=", 50 * ureg.nm, "angstrom").bound == pytest.approx(500.0)
    assert Constraint("t", "<=", 500 * ureg.angstrom, "angstrom").bound == 500
    with pytest.raises(ValueError, match=r"Constraint\.bound received unit 'nanometer'"):
        Constraint("fwhm", "<=", 1 * ureg.nm, "eV")
    # Pint treats sr as dimensionless; root units keep a ratio distinct from 1/sr.
    with pytest.raises(ValueError, match=r"Constraint\.bound received unit"):
        Constraint("y", ">=", 1 / ureg.sr, "")
    with pytest.raises(TypeError, match="scalar"):
        Constraint("fwhm", "<=", [1, 2] * ureg.eV, "eV")


def test_equivalent_quantity_definition_keeps_pinned_study_digest():
    # Pinned from the numeric-only implementation (before #755 B3): an existing
    # persisted study resumes with plain or equivalent quantity definitions.
    pinned = "850d4e64f4cb670a26294967cf91df41b7d5f100ededda79735039e0058c6f3f"
    space = ParameterSpace({"beam.energy_keV": Continuous(30.0, 100.0)})
    objectives = (Objective("flux", "maximize", "1/s"),)
    for fwhm, thickness in ((1.0, 500), (1000.0 * ureg.meV, 500 * ureg.angstrom)):
        constraints = (
            Constraint("fwhm", "<=", fwhm, "eV"),
            Constraint("t", "<=", thickness, "angstrom"),
        )
        digest = study_definition_digest(
            _base(), space, objectives, constraints, evaluation_id="flux-window-v1"
        )
        assert digest == pinned


def _study(*objectives, constraints=()):
    return OptimizationStudy(
        _base(),
        ParameterSpace({"beam.energy_keV": Continuous(30.0, 100.0)}),
        objectives,
        constraints,
    )


@pytest.mark.parametrize(
    ("unit", "ok"),
    [("1/sr", True), ("sr**-1", True), ("", False), ("1/mrad**2", False), ("eV", False)],
)
def test_objective_must_match_scorer_declared_unit(unit, ok):
    score = FixedWindowYield(500.0, 1500.0, name="flux")
    evaluator = PyriteEvaluator(score=score)
    assert evaluator.output_units == {"flux": "1/sr"}
    study = _study(Objective("flux", "maximize", unit))

    class NeverAsk:
        def ask(self):
            raise AssertionError("units mismatch must be refused before proposing work")

    if ok:
        with pytest.raises(AssertionError, match="before proposing"):
            study.run(evaluator, NeverAsk(), max_trials=1)
    else:
        with pytest.raises(ValueError, match="objective 'flux' declares unit"):
            study.run(evaluator, NeverAsk(), max_trials=1)


def test_constraint_on_scorer_output_is_unit_checked():
    evaluator = PyriteEvaluator(score=FixedWindowYield(500.0, 1500.0, name="flux"))
    study = _study(
        Objective("other", "maximize", ""), constraints=(Constraint("flux", ">=", 0.0, "eV"),)
    )
    with pytest.raises(ValueError, match="constraint 'flux' declares unit 'eV'"):
        study.run(evaluator, object(), max_trials=1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Continuous(30 * ureg.keV, 100.0),
        lambda: Continuous(30.0, 0.1 * ureg.MeV),
        lambda: Integer(1 * ureg.dimensionless, 4),
        lambda: Continuous(30.0, 100.0).validate(40 * ureg.keV),
        lambda: Integer(1, 4).validate(2 * ureg.dimensionless),
    ],
)
def test_parameter_quantities_refused_without_declared_path_unit(build):
    with pytest.raises(TypeError, match="paths declare no unit"):
        build()
