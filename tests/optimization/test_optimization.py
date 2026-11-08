"""Fast CPU-only tests of the #682 study/Optuna boundary."""

from dataclasses import FrozenInstanceError

import pytest

from pyrite.campaign.geometry import Footprint, Slab
from pyrite.campaign.model import Scene
from pyrite.campaign.sweep import BeamSpec
from pyrite.optimization import (
    Categorical,
    Constraint,
    Continuous,
    Integer,
    Objective,
    OptimizationStudy,
    OptunaOptimizer,
    ParameterSpace,
    TrialState,
)


def scene():
    return Scene(
        beam=BeamSpec(energy_keV=40),
        target=Slab("hopg", thickness_ang=2_000, footprint=Footprint(5.0, 5.0)),
    )


def study():
    return OptimizationStudy(
        base=scene(),
        parameters=ParameterSpace(
            {
                "beam.energy_keV": Continuous(30, 100),
                "target.thickness_ang": Integer(1_000, 4_000, step=100),
            }
        ),
        objectives=(Objective("flux", "maximize", "1/sr"),),
        constraints=(Constraint("line_fwhm_eV", "<=", 1.0, "eV"),),
    )


def test_scene_replacement_without_mutating_base():
    s = study()
    new = s.scene({"beam.energy_keV": 65.0, "target.thickness_ang": 2500})
    assert new.beam.energy_keV == 65
    assert new.target.thickness_ang == 2500
    assert s.base.beam.energy_keV == 40
    assert s.base.target.thickness_ang == 2000
    with pytest.raises(FrozenInstanceError):
        s.base.beam = new.beam


def test_parameters_reject_bad_values_without_rounding():
    with pytest.raises(ValueError, match="positive"):
        Continuous(0, 1, log=True)
    with pytest.raises(ValueError, match="bounds"):
        Integer(3, 2)
    with pytest.raises(ValueError, match="refusing truncation"):
        Integer(1, 10, step=2)
    with pytest.raises(TypeError, match="implicit rounding"):
        Integer(0, 10).validate(3.5)
    with pytest.raises(ValueError, match="outside allowed lattice"):
        Integer(0, 10, step=2).validate(3)
    with pytest.raises(ValueError, match="duplicate"):
        Categorical(("a", "a"))
    with pytest.raises(ValueError, match="outside"):
        Continuous(0, 1).validate(1.01)
    assert Categorical(("hopg", "mos2")).validate("hopg") == "hopg"
    with pytest.raises(ValueError, match="not an allowed"):
        Categorical((False, 1)).validate(True)
    with pytest.raises(ValueError, match="missing"):
        study().parameters.validate({"beam.energy_keV": 50})


def test_rejects_invalid_path_or_accuracy_tuning_in_scientific_scene():
    with pytest.raises(ValueError, match="has no field"):
        OptimizationStudy(
            scene(),
            ParameterSpace({"target.typo": Continuous(0, 1)}),
            (Objective("f", "maximize", ""),),
        )
    with pytest.raises(ValueError, match="not a physical"):
        OptimizationStudy(
            scene(),
            ParameterSpace({"numerics.sampling": Integer(2, 100)}),
            (Objective("f", "maximize", ""),),
        )


def test_optuna_optimization_and_constraints():
    pytest.importorskip("optuna")
    s = study()
    sampler = OptunaOptimizer(s.parameters, s.objectives, constraints=s.constraints, seed=7)
    result = s.run(
        lambda candidate, number: {
            "flux": 1 - abs(candidate.beam.energy_keV - 60) / 40,
            "line_fwhm_eV": 0.5 if candidate.beam.energy_keV < 70 else 2,
        },
        sampler,
        max_trials=16,
    )
    assert len(result.trials) == 16
    assert len(sampler.study.trials) == 16
    assert all(t.state in (TrialState.COMPLETED, TrialState.INFEASIBLE) for t in result.trials)
    assert all(t.constraints for t in result.trials)
    assert result.best_trial is not None
    assert result.best_trial.constraints["line_fwhm_eV"] <= 1.0


@pytest.mark.parametrize(
    ("method", "expected"),
    [("tpe", "TPESampler"), ("multivariate-tpe", "TPESampler"), ("random", "RandomSampler")],
)
def test_optuna_sampler_selection(method, expected):
    pytest.importorskip("optuna")
    definition = study()
    optimizer = OptunaOptimizer(
        definition.parameters,
        definition.objectives,
        constraints=definition.constraints,
        seed=3,
        sampler=method,
    )
    assert type(optimizer.study.sampler).__name__ == expected
    if method in ("tpe", "multivariate-tpe"):
        assert bool(optimizer.study.sampler._multivariate) is (method == "multivariate-tpe")
    result = definition.run(
        lambda candidate, _: {
            "flux": candidate.beam.energy_keV,
            "line_fwhm_eV": 0.5,
        },
        optimizer,
        max_trials=3,
    )
    assert len(result.trials) == 3
    assert all(trial.state is TrialState.COMPLETED for trial in result.trials)


def test_constraint_storage_uses_optuna5_native_api_when_available():
    """The new API uses named violations, not deprecated sampler callbacks."""
    optuna = pytest.importorskip("optuna")
    if not callable(getattr(optuna.trial.Trial, "set_constraint", None)):
        pytest.skip("requires Optuna >=5")
    definition = study()
    optimizer = OptunaOptimizer(
        definition.parameters, definition.objectives, constraints=definition.constraints, seed=3
    )
    assert optimizer._native_constraints
    assert optimizer.study.sampler._constraints_func is None
    definition.run(
        lambda candidate, _: {
            "flux": candidate.beam.energy_keV,
            "line_fwhm_eV": 2.0 if candidate.beam.energy_keV > 60 else 0.5,
        },
        optimizer,
        max_trials=8,
    )
    for trial in optimizer.study.trials:
        expected = 1.0 if trial.params["beam.energy_keV"] > 60 else -0.5
        assert trial.constraints["line_fwhm_eV"] == expected
        assert tuple(trial.user_attrs["pyrite_constraint_violations"]) == (expected,)


@pytest.mark.parametrize("native", [False, True])
def test_constraint_storage_api_dispatch(native):
    """Smoke-test both the old and named Optuna constraint APIs."""
    pytest.importorskip("optuna")
    definition = study()
    optimizer = OptunaOptimizer(
        definition.parameters,
        definition.objectives,
        constraints=definition.constraints,
        seed=3,
        sampler="random",
    )

    class Token:
        def __init__(self):
            self.user_attrs = {}
            self.constraints = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

        def set_constraint(self, key, value):
            self.constraints[key] = value

    token = Token()
    optimizer._native_constraints = native
    optimizer._write_constraints(token, (0.5,))
    assert token.user_attrs["pyrite_constraint_violations"] == (0.5,)
    assert token.constraints == ({"line_fwhm_eV": 0.5} if native else {})


def test_optuna_sampler_identity_on_persistent_resume(tmp_path):
    pytest.importorskip("optuna")
    definition = study()
    storage = f"sqlite:///{tmp_path / 'sampling.db'}"
    kwargs = dict(
        parameters=definition.parameters,
        objectives=definition.objectives,
        constraints=definition.constraints,
        storage=storage,
        study_name="physical-study",
        seed=3,
    )
    first = OptunaOptimizer(**kwargs, sampler="random")
    definition.run(
        lambda _candidate, _: {"flux": 1.0, "line_fwhm_eV": 0.5},
        first,
        max_trials=2,
        evaluation_id="sampler-test-v1",
    )
    resumed = OptunaOptimizer(**kwargs, sampler="random")
    result = definition.run(
        lambda _candidate, _: {"flux": 1.0, "line_fwhm_eV": 0.5},
        resumed,
        max_trials=3,
        evaluation_id="sampler-test-v1",
    )
    assert len(result.trials) == 3
    with pytest.raises(ValueError, match="different sampler"):
        OptunaOptimizer(**kwargs, sampler="tpe")
    with pytest.raises(ValueError, match="sampler must be"):
        OptunaOptimizer(**kwargs, sampler="invalid")


def test_failure_never_gets_rewarded_and_missing_observable_fails():
    pytest.importorskip("optuna")
    s = study()
    sampler = OptunaOptimizer(s.parameters, s.objectives, constraints=s.constraints, seed=4)
    result = s.run(lambda *_: {"flux": float("nan"), "line_fwhm_eV": 0.0}, sampler, max_trials=3)
    assert all(t.state is TrialState.FAILED for t in result.trials)
    assert result.best_trial is None
    assert all(t.state.name == "FAIL" for t in sampler.study.trials)


def test_multiobjective_pareto_and_selection():
    pytest.importorskip("optuna")
    s = OptimizationStudy(
        scene(),
        ParameterSpace({"beam.energy_keV": Continuous(30, 100)}),
        (Objective("flux", "maximize", "1/s"), Objective("width", "minimize", "eV")),
    )
    sampler = OptunaOptimizer(s.parameters, s.objectives, seed=1)
    result = s.run(
        lambda candidate, _: {
            "flux": candidate.beam.energy_keV,
            "width": candidate.beam.energy_keV / 100,
        },
        sampler,
        max_trials=5,
    )
    assert len(result.pareto_front) == 5
    with pytest.raises(ValueError, match="multi-objective"):
        _ = result.best_trial


def test_pyrite_evaluator_routes_through_existing_run_sweep(monkeypatch):
    """No physics kernels needed: assert seeds, record names and executor handoff."""
    import sys
    from dataclasses import dataclass
    from types import ModuleType

    from pyrite.campaign.model import Sweep
    from pyrite.optimization import PyriteEvaluator

    @dataclass(frozen=True)
    class FakeCase:
        name: str = "case"
        seed: int = 1

    monkeypatch.setattr(Sweep, "cases", lambda self: [FakeCase()])
    calls = []
    fake_runs = ModuleType("pyrite.runs")
    fake_runs.__path__ = []
    fake_runner = ModuleType("pyrite.runs.run")

    def fake_run_sweep(cases, results, *, executor, **kwargs):
        calls.append((cases[0], executor, kwargs))
        results["flux"] = cases[0].seed
        return True

    fake_runner.run_sweep = fake_run_sweep
    monkeypatch.setitem(sys.modules, "pyrite.runs", fake_runs)
    monkeypatch.setitem(sys.modules, "pyrite.runs.run", fake_runner)
    executor = object()
    evaluator = PyriteEvaluator(
        lambda records: {"flux": records["flux"]},
        executor=executor,
        base_seed=10,
        run_options={"checkpoint_path": "dummy.h5"},
    )
    assert evaluator(scene(), 0) == {"flux": 10}
    assert evaluator(scene(), 1) == {"flux": 11}
    assert [item[0].seed for item in calls] == [10, 11]
    assert calls[0][0].name != calls[1][0].name
    assert all(item[1] is executor for item in calls)
    assert all(item[2]["resume"] is False for item in calls)
    assert all(item[2]["checkpoint_path"] == "dummy.h5" for item in calls)


def test_optimizer_definition_mismatch_cannot_silently_change_objectives():
    pytest.importorskip("optuna")
    s = study()
    bad = OptunaOptimizer(s.parameters, (Objective("flux", "minimize", "1/sr"),))
    with pytest.raises(ValueError, match="optimizer objectives"):
        s.run(lambda *_: {"flux": 1, "line_fwhm_eV": 0}, bad, max_trials=1)
    assert len(bad.study.trials) == 0
