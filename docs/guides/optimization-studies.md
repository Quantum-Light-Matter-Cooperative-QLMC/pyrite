# Scientific optimization studies (experimental)

The `pyrite.optimization` module chooses **physical Scene parameters**. It does
not replace the existing Monte Carlo runner, executor, or case checkpoints.
Optuna is optional: install `pyrite-mc[optimization]` or use
`uv sync --extra optimization` in a checkout.

## Persisting and resuming a study

```python
from pyrite.optimization import (
    Constraint,
    Continuous,
    Objective,
    OptimizationStudy,
    OptunaOptimizer,
    ParameterSpace,
    PyriteEvaluator,
)

# `scene` is an existing, fully resolved pyrite.campaign.model.Scene.
study = OptimizationStudy(
    base=scene,
    parameters=ParameterSpace(
        {
            "beam.energy_keV": Continuous(30, 100),
            "target.tilt_deg": Continuous(0, 60),
        }
    ),
    objectives=(Objective("window_flux", "maximize", "photons/electron"),),
    constraints=(Constraint("line_fwhm_eV", "<=", 1.0, "eV"),),
)


# Compute these observables from normal run_sweep result records. Keep the
# energy integration window, normalizations and analysis settings fixed.
def score_results(records): ...  # return {"window_flux": ..., "line_fwhm_eV": ...}


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
    evaluator,
    optimizer,
    max_trials=50,
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
does not yet submit multiple *optimizer trials* concurrently. Individual trials
can use the existing #681 executor adapters. Noise-aware multi-fidelity methods
remain outside this implementation. The example below provides a fixed-window
scorer and an explicit independent-finalist validation protocol.

## Fixed-window PXR/CBS example

`FixedWindowYield(lower_eV, upper_eV)` is a maintained `PyriteEvaluator` scorer:

```python
from pyrite.optimization import FixedWindowYield, Objective, PyriteEvaluator

score = FixedWindowYield(820.0, 900.0)
objective = Objective(score.name, "maximize", score.units)
evaluator = PyriteEvaluator(score, base_seed=21)
# Pass evaluation_id=score.evaluation_id to study.run for durable resume.
```

It integrates the separate incoherent `spec` array in physical eV, with exact
window endpoints interpolated between samples. The result is
**photons/(sr incident electron)**. No current, sampled-history count,
solid angle or detector response is applied. Other emission components are
excluded. The window must be covered by strictly increasing finite nodes;
nonfinite densities, multiple records, bin-mean quadrature and two-beam
joint-field corrections are errors. It integrates the stored interpolant;
it cannot recover a narrow line missed by the nodes. See the
[independent integral verification](../validation/radiation-physics/optimization-fixed-window-yield.md).
The scorer's `evaluation_id` changes with its bounds and observable name;
version any additional scoring/constraint logic separately.

The checkout example `checks/pxr_optimization.py` supplies one complete bounded
comparison through the existing Python study API. It fixes:

- 30 keV point electrons; ideal graphite using the HOPG catalog's (001) cut
  and pinned basal (002)/(004) reflection families, with mosaic disabled.
- A 5 × 5 mm slab at azimuth 180°, a scalar observation direction at polar
  angle 90°, and an 820–900 eV incoherent PXR/CBS objective. The two mechanisms
  remain combined. This is a directional source design, without angular
  integration or detector response; scalar aperture metadata does not turn
  the objective into accepted photons.
- Tilt choices 41°, 43°, 45°, 47°, 49° and thickness choices 250, 500, 750 Å.
  A fabrication constraint accepts only thickness ≤500 Å. The wider search
  domain deliberately exercises finite, infeasible candidates.
- ELSEPA elastic scattering, continuous stopping, uncoupled radiative
  transport and no continuum scoring. These are explicit demonstration
  choices, held fixed for all methods, not new simulation defaults.
- 32 histories per search evaluation on 750–1000 eV nodes spaced by 2 eV.
  Search seeds 21, 22 and 23 each receive **15 attempted evaluations per
  method**, including failed and infeasible trials. Methods are constrained
  TPE (10 startup trials), uniform random sampling with replacement and the
  exhaustive 15-point grid. Small budgets need not favor TPE.

Select the CPU backend before importing PyRITE. The following commands are
for a checkout with the optional optimization extra and required ELSEPA data.
A bounded local preflight runs only one four-history case:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu uv run --extra optimization python checks/pxr_optimization.py \
  preflight --output /tmp/pxr-optimization
```

Run the comparison and finalist validation on the lab worker through the
[remote-job workflow](../computation/native-slurm-execution.md), from a clean,
committed and synced revision, with explicit catalog environment and memory
request:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 uv run --extra optimization \
  python checks/pxr_optimization.py search --output pyrite-output/optimization/pxr-demo
PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 uv run --extra optimization \
  python checks/pxr_optimization.py validate --output pyrite-output/optimization/pxr-demo
```

`search` uses one SQLite study per method/seed inside `studies.sqlite3`, with
shared physical CAS and separate checkpoint stems. Repeating it resumes each
study to its total budget. Proposals are reseeded by seed plus trial number;
TPE also consumes the completed history, so the example's suggestions match
an uninterrupted run with the same Optuna version. This explicit example
policy does not change the general adapter's sampler-state behavior. Search
Monte Carlo seeds are `10_000 * search_seed + trial_number`. Identical physical
cases with the same seed may share CAS data; independent seeds remain distinct.
`PyriteEvaluator` uses the standard case-content key by default. Custom content
keys must retain seeds, numerical fidelity, physics and table identity.

`search.json` contains all trial statuses, failures, feasible best candidates,
effective cases, content keys, checkpoint references, submitted-case counts and
wall times. A completed evaluation with zero submitted cases reused physical
results; failed attempts retain cost evidence too. JSON contains scalar metadata,
not duplicated spectra. The report records source revision, version, backend,
precision and platform. This is a scientific evaluation-budget comparison,
not a performance benchmark: cache hits, imports and JIT warm-up affect runtime.

`validate` independently re-evaluates each method/seed's feasible finalist with
four new 512-history replicates. It pairs 1 eV and 0.5 eV energy grids at the
same replicate seed, records each physical case, and reports the fine-grid mean,
sample standard deviation divided by √4 (SEM), and maximum absolute paired
grid difference. Validation seeds start at 1,000,000,000 and are disjoint from
search seeds. Rerunning validation can reuse completed same-seed CAS entries.
A central-direction observable has no angular quadrature to converge; no
finite detector-acceptance accuracy is claimed. Check grid differences against
the required scientific accuracy before treating the result as converged;
SEM does not include discretization error or model uncertainty. Search maxima
are selected noisy estimates, not evidence of a global optimum. Runtime and
independent validation results must be measured before claiming an optimizer
improvement.

The example declares a 2% maximum paired energy-grid change and 5% relative SEM
as demonstration precision targets before collecting results. `precision_pass`
records these checks; they are design tolerances, not claims of experimental
accuracy. A failed target requires more resolution or independent histories
before interpreting the finalist comparison.

### Measured bounded comparison

SLURM job 2511 completed successfully in 29 s at source revision `c941146c`
(CPU, two cores, 8 GiB allocation). The
[scalar evidence record](../validation/check-records/pxr-optimization.json)
pins the full source/table identities, study digests, original artifact hashes,
validation case keys and declared settings; operator-specific paths are omitted.
All methods attempted 15 evaluations per search seed. All 135 attempts completed
without failures, with 41 infeasible outcomes excluded from selection. Exact
same-seed cases reused CAS 33 times, leaving 102 actual search submissions.
The methods shared the same Monte Carlo seed schedule; the random baseline
reused many of TPE's startup cases. This saves work without creating independent
statistical replicates.

The table reports **independent 512-history finalist means ± SEM**, in units
of $10^{-8}$ photons/(sr incident electron), from four replicates on the 0.5 eV
grid. The underlying search used 32 histories; its noisy maxima are not the
values in this table.

| Method / search seed | Finalist tilt (°) | Thickness (Å) | Validated yield ×10⁸ | Infeasible attempts | CAS reuses |
|---|---:|---:|---:|---:|---:|
| TPE / 21 | 43 | 500 | 5.313 ± 0.013 | 3 | 0 |
| Random / 21 | 43 | 500 | 5.328 ± 0.050 | 5 | 11 |
| Grid / 21 | 43 | 500 | 5.337 ± 0.012 | 5 | 1 |
| TPE / 22 | 45 | 500 | 5.590 ± 0.037 | 4 | 0 |
| Random / 22 | 43 | 500 | 5.328 ± 0.060 | 5 | 10 |
| Grid / 22 | 45 | 500 | 5.529 ± 0.018 | 5 | 1 |
| TPE / 23 | 43 | 500 | 5.306 ± 0.028 | 4 | 0 |
| Random / 23 | 43 | 500 | 5.407 ± 0.020 | 5 | 10 |
| Grid / 23 | 43 | 500 | 5.290 ± 0.029 | 5 | 0 |

All nine finalists remained feasible and passed the declared precision targets.
The largest relative SEM was 1.126%; the largest paired 1 eV / 0.5 eV grid
change was 0.00484%. The 72 validation evaluations used independent seeds from
search, with paired seeds only across the two energy grids. Search-evaluator
wall times summed to 4.52 s and validation-evaluator times to 3.64 s; these
exclude optimizer/storage overhead and include cache/checkpoint work. They are
not hardware performance measurements.

This small comparison demonstrates the workflow and the effect of selection
noise: even exhaustive low-history grid search chose different finalists across
seeds. It does not establish a TPE advantage or a global optimum. The precision
checks cover this fixed energy integral and Monte Carlo variability only;
transport discretization, upstream-model accuracy, real mosaic material and
detector-accepted flux require their own validation.
