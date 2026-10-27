# Scientific optimization studies (experimental)

Choose crystal tilt and thickness to maximize the 820–900 eV directional
PXR/CBS yield, then re-evaluate the selected designs with independent seeds.
Compare constrained TPE, random sampling and a 15-point grid at equal budgets.

:::{warning}
The optimization API is **experimental**. This ideal-crystal simulation
demonstrates search and validation; it establishes neither a global optimum nor
experimental accuracy. Yield is **photons/(sr incident electron)** in one
direction, without detector response or angular integration.
:::

(fixed-window-pxr-cbs-example)=
## Run the study

Use a source checkout with the optimization extra and
[ELSEPA tables](getting-started.md). The maintained
{download}`workflow <../../checks/pxr_optimization.py>` supplies the complete
Scene, scorer, constraints, seeded optimizer and independent validation.
Start with one four-history CPU preflight:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
uv sync --extra optimization
PYRITE_MC_BACKEND=cpu uv run python checks/pxr_optimization.py \
  preflight --output /tmp/pxr-optimization
```

The search uses a 30 keV point beam, an ideal graphite slab (mosaic disabled),
a 90° observation direction and the catalog's basal (002)/(004) reflections.
Each method attempts 15 evaluations at 32 histories for each seed 21, 22 and 23.
Tilts are 41°–49° in 2° steps; thicknesses are 250, 500 and 750 Å, with a
fabrication constraint accepting only thickness ≤500 Å. Failed and infeasible
attempts count against the budget.

Run search and validation on the lab worker using the
[remote-job workflow](running-on-a-cluster.md), with a committed,
synced checkout, explicit catalog environment and memory allocation:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 uv run --extra optimization \
  python checks/pxr_optimization.py search --output pyrite-output/optimization/pxr-demo
PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 uv run --extra optimization \
  python checks/pxr_optimization.py validate --output pyrite-output/optimization/pxr-demo
```

(measured-bounded-comparison)=
## Compare finalists

The recorded CPU run (SLURM job 2511, revision `c941146c`) completed 135 search
attempts with no failures; 41 were infeasible. The plot contrasts selected search
values with independent four-replicate, 512-history means on a 0.5 eV grid.

```{eval-rst}
.. plot::
   :filename-prefix: optimization-finalists
   :alt: Nine selected search values compared with independently validated finalist means and SEM, grouped by search seed and method.
   :caption: Recorded simulated yields. Error bars show SEM of four independent finalist replicates.

   from _tutorials import optimization_studies
   optimization_studies(view="finalists")
```

Seven selections used 43° tilt and two used 45°; all selected 500 Å thickness.
Even exhaustive grid search chose different finalists across seeds. These small
budgets show selection noise, without establishing a TPE advantage. The record
contains selected values, not full trial histories.

## Check precision

Validation pairs 1 eV and 0.5 eV grids at the same replicate seed; all validation
seeds are disjoint from search. The declared targets are ≤5% relative SEM and
≤2% maximum paired grid change relative to the fine-grid mean.

```{eval-rst}
.. plot::
   :filename-prefix: optimization-precision
   :alt: Relative SEM and paired energy-grid changes for all nine finalists lie below their five-percent and two-percent targets.
   :caption: Monte Carlo SEM and paired energy-grid sensitivity test different error sources.

   from _tutorials import optimization_studies
   optimization_studies(view="precision")
```

All nine passed: maximum relative SEM **1.126%**, maximum paired grid change
**0.00484%**. SEM excludes discretization and model error. These tests do not
establish transport convergence, real mosaic-material accuracy or accepted flux.
The [evidence record](../validation/check-records/pxr-optimization.json) preserves
model/table identities, case keys and original artifact hashes.

(persisting-and-resuming-a-study)=
(crash-recovery)=
(limitations-and-next-steps)=
Repeating `search` resumes SQLite studies to the same **total** budget. Use one
coordinator per study. See the [study reference](../reference/optimization-studies.md)
for scorer normalization, durable identity, crash recovery and precision rules.
Download the {download}`plotting source <../examples/optimization_studies.py>`
and {download}`cached output <../examples/data/optimization_studies.json>`;
normal docs builds render checked evidence without running transport.
