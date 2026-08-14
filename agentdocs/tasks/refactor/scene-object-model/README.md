# Scene object model and public `pr.simulate`

Branch: `refactor/scene-object-model`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Changes 2 and 1 — sequencing steps 4 and 5.
Depends on: `refactor/target-geometry-surface`, `refactor/detector-scorer`, and
(for risk control) `refactor/typed-case-record`. Blocks:
`refactor/cli-noun-surface`.

The RFC states steps 4 and 5 "are the structural core and should be treated as
one project." They are one branch here for that reason. This is the largest and
highest-risk task in the set.

## Problem and scope

### Fields have no principled owner

`Sweep` (`src/pyrite/campaign/sweep.py:344`) carries, in one dataclass: scene
inputs (`material`, `thickness_ang`, `tilt_deg`, `tilt_azim_deg`, `beam`,
`detector`, `substrate`, `stack`); sweep axes (every field typed `ScalarOrSeq`);
numerics (`spec_chunk`, `brem_chunk`, `n_electrons`, `n_electrons_brem`);
physics-model switches (`mosaic`, `mosaic_route`, `mosaic_nodes`, `n_families`,
`max_reflections`); tally binning (`E_grid_line`, `E_grid_line_by_energy`,
`E_grid_brem`); and compatibility debris (`e_grid_eV`, plus the `theta_obs_deg` /
`dtheta_obs_deg` / `domega_sr` `InitVar` reconciliation in `__post_init__`).

`Settings` (`src/pyrite/results/store.py:71`) is documented as "analysis /
detector / unit knobs shared by post-processing and plots" but owns `emission`,
`xray_dispersion`, `n_electrons`, `n_electrons_brem`, and `brem_source` — all of
which change the result, not its presentation.

The diagnostic symptom is `build_cases` (`src/pyrite/campaign/sweep.py:625`):

```python
def build_cases(sweep, n_electrons=450, n_electrons_brem=100,
                coherent_emission=False, xray_dispersion="vacuum"):
```

Four run-affecting parameters re-threaded as loose keyword arguments because
they live on the wrong object. Every future run-affecting option inherits the
pattern.

### Sweepability is a property of the field type

A field is swept by being typed `ScalarOrSeq`, so only fields declared that way
can be swept. Today it is not possible to sweep `stack[0].thickness_ang`, a
per-layer orientation, `beam.transverse.eps_n`, or
`detector.observation_angle_deg` — each would need a new field, new `_seq()`
plumbing, and a new case-name suffix rule. Compare the `ne=` suffix introduced
for the electron-count grids.

### There is no first-class Python entry point

The documented entry points are the CLI and the marimo apps. `Sweep`,
`build_cases`, and `run_sweep` are importable, but the API is sweep-shaped: a
single simulation must be expressed as a one-element sweep, and its result is
retrieved through a checkpoint rather than returned.

`BeamSpec` (`src/pyrite/campaign/sweep.py:66`) is explicitly **not** a problem —
it is a well-formed frozen physical object and is the model the other scene
objects should follow.

## Target state

Four objects with distinct lifetimes:

| Object | Owns | Enters dataset identity |
| --- | --- | --- |
| `Scene` | `beam`, `target`, `detector` | Yes |
| `Sweep` | a base `Scene` plus named axes over paths into it | Yes, per expanded scene |
| `Numerics` | electron counts, chunk sizes, transport core, backend policy | No |
| `Numerics.convergence` | `n_families`, `max_reflections`, `mosaic_nodes`, `mosaic_route` | Yes |
| `Analysis` | unit scaling, presentation-time convolution, plotting knobs | No |

The `convergence` nesting is what makes the identity column enforceable:
everything directly under `Numerics` must be result-invariant and stay out of
`dataset_identity`; everything under `Numerics.convergence` enters it. A field
that cannot be placed by that test is misclassified.

Axes addressed by dotted path rather than by field type:

```python
sweep = pr.Sweep(
    base=scene,
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
        "target.layers[1].thickness_ang": [2.85e3, 5.0e3],
    },
)
```

This removes `ScalarOrSeq` from the object model and gives case naming one
mechanical rule — the axis path and its value — instead of hand-written label
concatenation.

`Settings` dissolves: run-affecting fields (`emission`, `xray_dispersion`,
`brem_source`) move to `Scene` or `Numerics` by whether they describe the
physical configuration or the sampling of it; presentation fields move to
`Analysis`.

Then the public entry point:

```python
import pyrite as pr

result = pr.simulate(beam, target, detector, numerics=pr.Numerics(n_electrons=450))
result.spectrum          # units documented on the object
result.provenance        # resolved scene, identity digest, backend, versions
```

`pr.simulate` is a **thin composition** of existing pieces: build one `Case`,
call the existing `run_case`, wrap the returned arrays. It must not duplicate
physics.

## Implementation path

Likely owners: `src/pyrite/campaign/sweep.py` (1 160 lines),
`src/pyrite/campaign/config.py`, `src/pyrite/campaign/profiles.py`,
`src/pyrite/results/store.py` (215 lines), `src/pyrite/runs/run.py`,
`src/pyrite/runs/scan.py`, `src/pyrite/runs/blaze.py`, the CLI command modules
that construct sweeps, and `src/pyrite/apps/` as the first consumer.

## Checklist

- [x] A — Field-by-field disposition table for `Sweep` and `Settings`: which of
      the four new objects each field lands on, and whether it enters dataset
      identity. This is the decision document; nothing else starts before it.
      See [slice-a-field-disposition.md](slice-a-field-disposition.md).
- [x] B — Land `Scene`, `Numerics`, `Analysis` alongside the existing classes.
      No removals yet.
- [x] C — Dotted-path axis resolution, including indexed segments
      (`target.layers[1].thickness_ang`), with the case-naming rule derived
      mechanically from path + value.
- [ ] D — `Sweep.from_legacy(old_sweep, settings)` plus deprecated shims for the
      old classes through one D7 support window.
- [x] E — Equivalence: every existing catalog profile round-trips to an
      identical expanded case list, and the resolution order (catalog profile →
      per-material override → explicit argument) is unchanged.
- [x] F — Demonstrate at least one previously unreachable axis. Per-layer
      thickness is the suggested target.
- [x] G — Public `pr.simulate` + `Result` with `spectrum` and `provenance`. No
      filesystem writes on the single-shot path.
- [ ] H — Reimplement `runs.scan` and `runs.blaze` over `pr.simulate`'s
      internals, so the sweep driver and the single-shot path share one code
      path rather than two.
- [ ] I — Port the marimo apps to the public API. The trace app already runs
      transport directly from catalog scan grids and is the natural first
      consumer. Per the RFC's GUI non-goal, apps **must not** construct case
      payloads directly.
- [ ] J — `docs/api.md` documents the public names; extend the export-freeze
      tests (`tests/montecarlo/test_exports.py`, `tests/results/test_exports.py`,
      `tests/plots/test_exports.py`) rather than relaxing them.

## Decisions and open questions

- **Decided:** no new `ScalarOrSeq` field anywhere.
- **Decided:** `pr.simulate` composes; it does not reimplement physics.
- **Decided:** land `refactor/typed-case-record` first. The RFC risk note is
  explicit — the split touches the object every physics module reads, so
  equivalence must be machine-checkable at the boundary before the objects
  above it move.
- **Decided (review):** the model switches are two categories, not one, and no
  fifth object is needed. `mosaic` — whether the crystal is modelled as mosaic —
  is a target property and moves to `Target` in
  `refactor/target-geometry-surface`. `n_families` (default 4),
  `max_reflections`, `mosaic_nodes` (Gauss–Hermite nodes), and `mosaic_route`
  (`analytic` vs `mc`) are convergence and truncation parameters: more is more
  correct, and a value is chosen for cost. They go to `Numerics.convergence`.
  `n_families` and `max_reflections` already feed `dataset_identity`
  (`campaign/profiles.py:236`), which confirms the classification.
- **Worth a test while in here:** `mosaic_route="analytic"` and `"mc"` evaluate
  the same mosaic integral, so they should agree within tolerance at high
  `mosaic_nodes`. If they do not converge, that is a physics bug this refactor
  would otherwise paper over. Route it to the physics ledger rather than
  absorbing it.
- **Open:** the `theta_obs_deg` / `dtheta_obs_deg` / `domega_sr` `InitVar`
  reconciliation. Once the detector owns acceptance, does this reconciliation
  survive at all, or is it deleted? Check against `refactor/detector-scorer`.
- **Open:** `e_grid_eV` — confirm it is genuinely dead before deleting it.
- **Open:** how `Analysis` reaches the plotting layer, which reads `Settings`
  today in `store.py` and across `src/pyrite/plots/`.
- **Open:** dotted-path axes must fail loudly on a typo. Decide whether paths
  are validated against the base `Scene` at `Sweep` construction (preferred) or
  at expansion.

## Delegation slices and required skills

- A → `lead-task`. The disposition table is the whole task's foundation and has
  at least three open questions in it. Never `one-shot`.
- B, C → `implement-task`; `scientific-library`. C is the novel machinery — give
  it its own review.
- D, E → `implement-task`; `catalog-golden` + `regression`.
- F → `implement-task-lite`; `catalog-golden`. `one-shot` once C and E land.
- G → `lead-task`; `scientific-library`. The public surface is a durable
  contract.
- H → `implement-task`; `monte-carlo`. Merging two drivers into one code path is
  where bit-for-bit reproduction is most likely to break.
- I → `implement-task`; `notebooks`. Run `uv run marimo check <app.py>` after
  every app edit.
- J → `implement-task-lite`; `documentation-maintenance` + `scientific-library`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
```

- Every existing catalog profile round-trips through the new objects to an
  identical expanded case list.
- At least one previously unreachable axis is demonstrated in a test.
- No new `ScalarOrSeq` field is introduced anywhere.
- A documented example runs end to end in fewer than twenty lines with no
  filesystem writes.
- `pr.simulate` on a scene equivalent to an existing catalog case reproduces
  that case's stored spectrum **bit for bit**.
- Export-freeze tests extended, not relaxed.

## Related

- `refactor/typed-case-record`, `refactor/target-geometry-surface`,
  `refactor/detector-scorer` — all three are prerequisites.
- `refactor/cli-noun-surface` — should follow this task so every removed command
  has a documented public-API replacement.
- TODO P3 "Parameter-space sampling review" becomes tractable once axes are
  paths rather than field types; it is not in scope here.
