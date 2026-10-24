# Scientific optimization studies (experimental)

The `pyrite.optimization` module chooses **physical Scene parameters**. It does
not replace the existing Monte Carlo runner, executor, or case checkpoints.
Optuna is optional: install `pyrite-mc[optimization]` or use
`uv sync --extra optimization` in a checkout.

## Persisting and resuming a study

```python
from pyrite.optimization import (
    Constraint, Continuous, Objective, OptimizationStudy,
    OptunaOptimizer, ParameterSpace, PyriteEvaluator,
)

# `scene` is an existing, fully resolved pyrite.campaign.model.Scene.
study = OptimizationStudy(
    base=scene,
    parameters=ParameterSpace({
        "beam.energy_keV": Continuous(30, 100),
        "target.tilt_deg": Continuous(0, 60),
    }),
    objectives=(Objective("window_flux", "maximize", "photons/electron"),),
    constraints=(Constraint("line_fwhm_eV", "<=", 1.0, "eV"),),
)

# Compute these observables from normal run_sweep result records. Keep the
# energy integration window, normalizations and analysis settings fixed.
def score_results(records):
    ...  # return {"window_flux": ..., "line_fwhm_eV": ...}

evaluator = PyriteEvaluator(
    score_results,
    run_options={"checkpoint_dir": "pyrite-output/checkpoints"},
)
optimizer = OptunaOptimizer(
    study.parameters,
    study.objectives,
    constraints=study.constraints,
    seed=21,
    storage="sqlite:///pyrite-optimization.sqlite3",
    study_name="pxr-500ev-design",
)
result = study.run(
    evaluator, optimizer, max_trials=50,
    evaluation_id="window-490-510ev-per-electron-v1",
)
print(result.best_trial)
```

Run the same code with the **same study name, storage URL, scene, parameter
space, objectives, constraints, evaluator seed policy and `evaluation_id`** to
resume. `max_trials` is the **total study budget**, not an additional number of
trials: after 20 committed trials, changing 50 to 80 runs at most 60 more.
In-memory optimizers also support repeated `run` calls with a total budget.

On resume, the optimization definition is checked before any new candidate is
submitted. An incompatible definition fails rather than silently mixing
results from different physical setups. The definition includes the scene's
numerics, so changing statistical fidelity also forces a new study.

`evaluation_id` is an explicitly controlled version of the **scorer**: PyRITE
cannot infer arbitrary Python callback behavior or safely hash source code.
**Change the ID whenever the score implementation, integration energy window,
detector processing, source-current normalization, or scoring settings change.**
Keep evaluator `run_options` that affect physics/observables consistent too.
The evaluator's `base_seed` is automatically included in the compatibility ID.
Executor choice, SLURM/Dask scheduling, and checkpoint storage locations can be
changed independently so long as their results are physically equivalent.

## Crash recovery

Optuna's storage is the authoritative optimizer history. A trial's observables
and constraint violations are committed to its user attributes **before** the
Optuna `tell()` transition. If the coordinator is interrupted between these
steps, the next study invocation finalizes the recorded trial without another
simulation. A trial abandoned before observables are durably recorded is marked
**failed** on resume rather than implicitly re-run using a potentially changed
Monte Carlo seed. Existing physical case checkpoints remain owned by
`run_sweep`, and may contain recoverable physics data even for such failures.

Only **one coordinating process** should write to a given persistent study at
a time. The current implementation does not implement leases or multi-process
coordinator recovery. Optuna's storage is not itself an exclusive lock.

Errors in validation, physics, or scoring are recorded as failed trials rather
than assigned an artificial bad objective. Infeasible finite results are kept
as completed Optuna evaluations but excluded from PyRITE's feasible Pareto
front and `best_trial`.

## Limitations and next steps

This is a serial, synchronous ask/evaluate/tell implementation. It can delegate
individual simulations to an existing `Executor` through `PyriteEvaluator` but
does not yet submit multiple *optimizer trials* concurrently. Dask integration
under #681 should be implemented below the existing executor interface.
Independent high-precision validation of final configurations, noise-aware
multi-fidelity methods and the physics-validated fixed-window PXR scorer remain
separate follow-up slices.
