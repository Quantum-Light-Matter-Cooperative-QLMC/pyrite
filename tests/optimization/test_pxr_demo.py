"""Comparison protocol checks; physical coverage is a separately selected slow node."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.optimization import FixedWindowYield, OptunaOptimizer, TrialState

pytest.importorskip("optuna")

from checks.pxr_optimization import (  # noqa: E402
    BUDGET,
    GRID,
    SCORER,
    ComparisonOptimizer,
    CountingExecutor,
    scene,
    study,
)


def test_baselines_have_equal_attempted_budgets_and_resume_reproducibly(tmp_path):
    definition = study(scene())
    for method in ("grid", "random", "tpe"):

        def optimizer(name, method=method):
            return ComparisonOptimizer(
                definition,
                method,
                21,
                f"sqlite:///{tmp_path / 'studies.sqlite3'}",
                name,
            )

        def score(candidate, _):
            return {
                SCORER.name: candidate.target.thickness_ang,
                "thickness_ang": candidate.target.thickness_ang,
            }

        first = definition.run(score, optimizer(method), max_trials=7, evaluation_id=method)
        continued = definition.run(
            score,
            optimizer(method),
            max_trials=BUDGET,
            evaluation_id=method,
        )
        whole = definition.run(
            score,
            optimizer(method + "-whole"),
            max_trials=BUDGET,
            evaluation_id=method,
        )
        assert len(continued.trials) == BUDGET
        assert continued.trials[:7] == first.trials
        assert continued.trials == whole.trials
        assert continued.best_trial.constraints["thickness_ang"] == 500
        if method == "grid":
            assert [t.parameters for t in continued.trials] == GRID
            assert sum(t.state is TrialState.INFEASIBLE for t in continued.trials) == 5


@pytest.mark.slow
def test_real_evaluator_cache_seed_executor_checkpoint_and_resume(tmp_path):
    """Two four-history physical runs; later duplicate requests must use CAS.

    Exact same-seed comparisons are bitwise: cache/executor routing cannot
    alter the realization. Independent seeds are asserted through identities,
    not by requiring unequal stochastic spectra (a short slab can have none).
    """
    from pyrite.campaign.profiles import case_content_key
    from pyrite.checkpoints import _checkpoint_store
    from pyrite.execution import LocalExecutor
    from pyrite.optimization import Integer, OptimizationStudy, ParameterSpace, PyriteEvaluator
    from pyrite.results import records

    received = []

    def score(result):
        (record,) = records(result)
        received.append(record)
        return SCORER(result)

    execution = CountingExecutor(LocalExecutor())
    options = {"checkpoint_path": str(tmp_path / "physical"), "max_workers": 0}
    evaluator = PyriteEvaluator(score, execution, options, base_seed=41)
    base = scene(4)
    a = evaluator(base, 0)
    b = evaluator(base, 1)
    assert execution.submitted == 2
    keys = [case_content_key(r["case"]) for r in received]
    assert keys[0] != keys[1]
    assert [r["case"]["seed"] for r in received] == [41, 42]
    assert all(_checkpoint_store.cas_contains("hopg", key, str(tmp_path)) for key in keys)
    assert (tmp_path / "physical" / "meta.json").is_file()

    # A second executor receives no work for an exact same-seed CAS request.
    def unexpected_execution(*args, **kwargs):
        pytest.fail("a cached same-seed case must not be dispatched")

    cached = replace(evaluator, executor=LocalExecutor(unexpected_execution))
    assert cached(base, 0) == a
    np.testing.assert_array_equal(received[0]["spec"], received[-1]["spec"])

    # Executor routing agrees with a fresh same-seed direct local run.
    direct = replace(evaluator, executor=None, run_options={**options, "cache_read": False})
    assert direct(base, 0) == a
    np.testing.assert_array_equal(received[0]["spec"], received[-1]["spec"])

    # Real study interruption and resume, reusing the two independent physical
    # realizations above. The on_trial exception occurs after durable tell().
    definition = OptimizationStudy(
        base,
        ParameterSpace({"target.tilt_deg": Integer(45, 45)}),
        study(base).objectives,
    )

    def optimizer():
        return OptunaOptimizer(
            definition.parameters,
            definition.objectives,
            seed=21,
            storage=f"sqlite:///{tmp_path / 'resume.sqlite3'}",
            study_name="physical",
        )

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        definition.run(
            cached,
            optimizer(),
            max_trials=2,
            on_trial=interrupt,
            evaluation_id=SCORER.evaluation_id,
        )
    before = len(received)
    result = definition.run(
        cached,
        optimizer(),
        max_trials=2,
        evaluation_id=SCORER.evaluation_id,
    )
    assert len(received) == before + 1  # committed trial zero was not evaluated
    assert [t.objectives for t in result.trials] == [a, b]
    assert all(t.state is TrialState.COMPLETED for t in result.trials)
    assert (
        definition.run(
            cached,
            optimizer(),
            max_trials=2,
            evaluation_id=SCORER.evaluation_id,
        )
        == result
    )
    with pytest.raises(ValueError, match="definition mismatch"):
        definition.run(
            cached,
            optimizer(),
            max_trials=3,
            evaluation_id=FixedWindowYield(821, 900).evaluation_id,
        )
