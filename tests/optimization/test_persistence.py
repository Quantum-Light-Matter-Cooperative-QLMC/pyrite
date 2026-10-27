"""SQLite resume tests: inexpensive synthetic scoring, no MC or GPU."""

from dataclasses import replace

import pytest

from pyrite.campaign.geometry import Footprint, Slab
from pyrite.campaign.model import Scene
from pyrite.campaign.sweep import BeamSpec
from pyrite.optimization import (
    Constraint,
    Continuous,
    Objective,
    OptimizationStudy,
    OptimizationTrial,
    OptunaOptimizer,
    ParameterSpace,
    TrialState,
)

pytest.importorskip("optuna")


def _study():
    return OptimizationStudy(
        Scene(
            beam=BeamSpec(energy_keV=40),
            target=Slab("hopg", thickness_ang=2_000, footprint=Footprint(5.0, 5.0)),
        ),
        ParameterSpace({"beam.energy_keV": Continuous(30.0, 100.0)}),
        (Objective("flux", "maximize", "1/sr"),),
        (Constraint("fwhm", "<=", 1.0, "eV"),),
    )


def _opt(s, tmp_path):
    return OptunaOptimizer(
        s.parameters,
        s.objectives,
        constraints=s.constraints,
        storage=f"sqlite:///{tmp_path / 'study.sqlite3'}",
        study_name="pxr-test",
        seed=12,
    )


def _score(scene, n):
    return {"flux": scene.beam.energy_keV, "fwhm": 0.5 if n % 2 else 2.0}


def test_resumes_without_recomputing_completed_trials(tmp_path):
    s = _study()
    first = s.run(_score, _opt(s, tmp_path), max_trials=5, evaluation_id="flux-window-v1")
    assert len(first.trials) == 5
    seen = []
    new = s.run(
        lambda scene, n: (seen.append(n), _score(scene, n))[1],
        _opt(s, tmp_path),
        max_trials=8,
        evaluation_id="flux-window-v1",
    )
    assert seen == [5, 6, 7]
    assert new.trials[:5] == first.trials
    assert len(new.trials) == 8
    assert all(trial.state in (TrialState.COMPLETED, TrialState.INFEASIBLE) for trial in new.trials)
    assert new.best_trial is not None
    assert new.best_trial.constraints["fwhm"] <= 1
    assert (
        s.run(
            lambda *_: pytest.fail("unexpected evaluation"),
            _opt(s, tmp_path),
            max_trials=8,
            evaluation_id="flux-window-v1",
        )
        == new
    )


def test_rejects_unsafe_resume_before_scheduling_work(tmp_path):
    s = _study()
    s.run(_score, _opt(s, tmp_path), max_trials=2, evaluation_id="score-v1")
    cases = [
        (s, "score-v2"),
        (replace(s, base=replace(s.base, beam=replace(s.base.beam, energy_keV=41))), "score-v1"),
        (
            replace(s, parameters=ParameterSpace({"beam.energy_keV": Continuous(31, 100)})),
            "score-v1",
        ),
        (replace(s, objectives=(Objective("flux", "maximize", "1/s"),)), "score-v1"),
        (replace(s, constraints=(Constraint("fwhm", "<=", 0.5, "eV"),)), "score-v1"),
    ]
    for altered, evaluator_id in cases:
        with pytest.raises(ValueError, match="definition mismatch|optimizer"):
            altered.run(
                lambda *_: pytest.fail("should not execute"),
                _opt(altered, tmp_path),
                max_trials=3,
                evaluation_id=evaluator_id,
            )
    with pytest.raises(ValueError, match="nonempty evaluation_id"):
        s.run(_score, _opt(s, tmp_path), max_trials=3)
    with pytest.raises(ValueError, match="total study budget"):
        s.run(_score, _opt(s, tmp_path), max_trials=1, evaluation_id="score-v1")


def test_trial_with_recorded_result_finalized_after_interruption(tmp_path):
    s = _study()
    opt = _opt(s, tmp_path)
    # Bind definition while not yet proposing any work.
    s.run(_score, opt, max_trials=1, evaluation_id="value-v1")
    trial = opt.ask()
    record = OptimizationTrial(
        1,
        dict(trial.parameters),
        TrialState.COMPLETED,
        {"flux": 4.0},
        {"fwhm": 0.25},
    )
    opt.record_trial(trial, record)
    # Process dies here: metadata persisted, but tell() never executed.
    reopened = _opt(s, tmp_path)
    result = s.run(
        lambda *_: pytest.fail("must use persisted completed observation"),
        reopened,
        max_trials=2,
        evaluation_id="value-v1",
    )
    assert result.trials[1] == record
    assert reopened.study.trials[1].state.name == "COMPLETE"
    assert reopened.study.trials[1].user_attrs["pyrite_constraint_violations"] == [-0.75]


def test_interrupted_before_record_is_failed_not_reevaluated(tmp_path):
    s = _study()
    opt = _opt(s, tmp_path)
    s.run(_score, opt, max_trials=1, evaluation_id="value-v1")
    opt.ask()  # trial remains RUNNING without any result
    next_opt = _opt(s, tmp_path)
    history = s.run(
        lambda *_: pytest.fail("interrupted should not be recomputed"),
        next_opt,
        max_trials=2,
        evaluation_id="value-v1",
    )
    assert history.trials[1].state is TrialState.FAILED
    assert "Interrupted" in history.trials[1].error
    assert next_opt.study.trials[1].state.name == "FAIL"


def test_failed_evaluations_persist_with_errors(tmp_path):
    s = _study()
    first = s.run(
        lambda *_: {"flux": float("nan"), "fwhm": 0.5},
        _opt(s, tmp_path),
        max_trials=2,
        evaluation_id="fail-v1",
    )
    reopened = s.run(
        lambda *_: pytest.fail("do not rerun failed attempts"),
        _opt(s, tmp_path),
        max_trials=2,
        evaluation_id="fail-v1",
    )
    assert reopened == first
    assert all(t.state is TrialState.FAILED and "non-finite" in t.error for t in first.trials)
    assert reopened.best_trial is None


def test_on_trial_exception_after_commit_is_recoverable(tmp_path):
    s = _study()
    with pytest.raises(KeyboardInterrupt):
        s.run(
            _score,
            _opt(s, tmp_path),
            max_trials=3,
            evaluation_id="interrupt-v1",
            on_trial=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    result = s.run(
        lambda *_: pytest.fail("already committed"),
        _opt(s, tmp_path),
        max_trials=1,
        evaluation_id="interrupt-v1",
    )
    assert len(result.trials) == 1


def test_resume_fails_for_changed_evaluator_seed(tmp_path, monkeypatch):
    from pyrite.optimization import PyriteEvaluator

    s = _study()
    monkeypatch.setattr(
        PyriteEvaluator, "__call__", lambda self, scene, number: _score(scene, number)
    )
    evaluator = PyriteEvaluator(lambda _: {"flux": 10, "fwhm": 0.5}, base_seed=41)
    s.run(evaluator, _opt(s, tmp_path), max_trials=1, evaluation_id="flux-v1")
    different = PyriteEvaluator(lambda _: {"flux": 10, "fwhm": 0.5}, base_seed=42)
    with pytest.raises(ValueError, match="definition mismatch"):
        s.run(different, _opt(s, tmp_path), max_trials=2, evaluation_id="flux-v1")


def test_in_memory_optimizer_can_continue_with_total_budget():
    s = _study()
    opt = OptunaOptimizer(s.parameters, s.objectives, constraints=s.constraints, seed=10)
    assert len(s.run(_score, opt, max_trials=2).trials) == 2
    result = s.run(_score, opt, max_trials=4)
    assert [trial.number for trial in result.trials] == [0, 1, 2, 3]


def test_numeric_study_resumes_with_equivalent_quantity_definition(tmp_path):
    from pyrite.units import ureg

    s = _study()
    first = s.run(_score, _opt(s, tmp_path), max_trials=3, evaluation_id="flux-window-v1")
    quantified = replace(s, constraints=(Constraint("fwhm", "<=", 1000.0 * ureg.meV, "eV"),))
    resumed = quantified.run(
        lambda *_: pytest.fail("persisted trials must load, not re-run"),
        _opt(quantified, tmp_path),
        max_trials=3,
        evaluation_id="flux-window-v1",
    )
    assert resumed.trials == first.trials
    assert all(isinstance(v, float) for t in resumed.trials for v in t.parameters.values())
