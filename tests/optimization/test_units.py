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
        lambda: Integer(1 * ureg.dimensionless, 4),
        lambda: Continuous(30.0, 100.0).validate(40 * ureg.keV),
        lambda: Integer(1, 4).validate(2 * ureg.dimensionless),
    ],
)
def test_parameter_quantities_refused_without_declared_path_unit(build):
    with pytest.raises(TypeError):
        build()


def test_quantity_parameter_domains_bind_to_owner_units():
    space = ParameterSpace(
        {
            "beam.kinetic_energy": Continuous(0.03 * ureg.MeV, 100 * ureg.keV),
            "target.thickness": Integer(
                100 * ureg.angstrom, 200 * ureg.angstrom, step=50 * ureg.angstrom
            ),
        }
    )
    study = OptimizationStudy(_base(), space, (Objective("f", "maximize", ""),))
    assert study.parameters.parameters["beam.kinetic_energy"] == Continuous(30, 100)
    assert study.parameters.parameters["target.thickness"] == Integer(100, 200, step=50)
    scene = study.scene({"beam.kinetic_energy": 40, "target.thickness": 150})
    assert scene.beam.energy_keV == 40
    assert scene.target.thickness_ang == 150
    assert _base().target.thickness_ang == 2000
    quantity_scene = study.scene(
        {
            "beam.kinetic_energy": 0.04 * ureg.MeV,
            "target.thickness": 150 * ureg.angstrom,
        }
    )
    assert quantity_scene == scene
    assert study.parameters.bind(_base()) == study.parameters


def test_quantity_parameter_digest_has_numeric_storage_and_canonical_owner_unit():
    objectives = (Objective("f", "maximize", ""),)
    digests = []
    for lo, hi in ((30 * ureg.keV, 100 * ureg.keV), (0.03 * ureg.MeV, 0.1 * ureg.MeV)):
        space = ParameterSpace({"beam.kinetic_energy": Continuous(lo, hi)})
        digests.append(study_definition_digest(_base(), space, objectives, (), evaluation_id="v1"))
    assert digests[0] == digests[1]


@pytest.mark.parametrize(
    "path, interval, error",
    [
        ("beam.energy_keV", Continuous(30 * ureg.keV, 100 * ureg.keV), TypeError),
        ("beam.kinetic_energy", Continuous(30, 100), TypeError),
        ("beam.kinetic_energy", Continuous(30 * ureg.nm, 100 * ureg.nm), ValueError),
        ("beam.kinetic_energy", Integer(30.5 * ureg.keV, 40 * ureg.keV), ValueError),
        (
            "beam.kinetic_energy",
            Integer(30 * ureg.keV, 40 * ureg.keV, step=0.5 * ureg.keV),
            ValueError,
        ),
    ],
)
def test_invalid_quantity_parameter_domains(path, interval, error):
    with pytest.raises(error):
        ParameterSpace({path: interval}).bind(_base())


def test_quantity_parameter_candidates_keep_bounds_lattice_and_dimensions():
    space = ParameterSpace(
        {"beam.kinetic_energy": Integer(30 * ureg.keV, 40 * ureg.keV, step=2 * ureg.keV)}
    ).bind(_base())
    assert space.validate({"beam.kinetic_energy": 32 * ureg.keV}) == {"beam.kinetic_energy": 32}
    for value in (31 * ureg.keV, 30.5 * ureg.keV, 50 * ureg.keV, 32 * ureg.nm):
        with pytest.raises(ValueError):
            space.validate({"beam.kinetic_energy": value})


@pytest.mark.parametrize(
    "bounds",
    [
        (30 * ureg.keV, 100),
        (30, 100 * ureg.keV),
        (30 * ureg.keV, 100 * ureg.nm),
    ],
)
def test_mixed_or_incompatible_quantity_bounds_are_refused(bounds):
    with pytest.raises((TypeError, ValueError)):
        Continuous(*bounds)


def test_external_quantity_bounds_are_detached_and_nested_paths_bind():
    import numpy as np
    import pint

    external = pint.UnitRegistry()
    low = external.Quantity(np.array(4.0), "mm")
    high = external.Quantity(0.6, "cm")
    space = ParameterSpace({"target.footprint.width": Continuous(low, high)})
    low.magnitude[...] = 20.0
    bound = space.bind(_base())
    assert bound.parameters["target.footprint.width"] == Continuous(4, 6)
    study = OptimizationStudy(_base(), bound, (Objective("f", "maximize", ""),))
    assert (
        study.scene(
            {"target.footprint.width": external.Quantity(0.5, "cm")}
        ).target.footprint.width_mm
        == 5.0
    )


@pytest.mark.parametrize(
    "value",
    [
        [1, 2] * ureg.keV,
        float("nan") * ureg.keV,
        float("inf") * ureg.keV,
        (1 + 2j) * ureg.keV,
    ],
)
def test_quantity_bounds_reject_non_scalar_non_real_or_unrepresentable_values(value):
    with pytest.raises((TypeError, ValueError)):
        Continuous(value, 100 * ureg.keV)


def test_quantity_integer_magnitudes_do_not_silently_round_to_float_storage():
    with pytest.raises(ValueError, match="exactly representable"):
        Integer((2**53 + 1) * ureg.keV, (2**54) * ureg.keV)
