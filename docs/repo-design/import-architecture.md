# Import architecture and dependency-cycle reduction

Issue [#708](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/708)
tracks incremental import-graph improvements. Namespace/API ownership alone is
not a dependency direction: source imports are scanned by `grimp`, and the
package DAG in [the repository map](../repo_map.md) condenses strongly connected
components (SCCs). Imports inside functions still count as static imports;
`TYPE_CHECKING` imports and package-facade re-exports are excluded. Dynamic
`import_module`/`__getattr__` dispatch and runtime side effects need separate
review.

## Reproducible baseline

From a checkout, before and after each slice:

```bash
uv run pyrite-dev repo-map --json > /tmp/pyrite-package-imports.json
uv run pyrite-dev repo-map --check
uv run lint-imports
```

The JSON contains sorted exact module-level import pairs, collapsed
package-level edges, the SCC membership, all within-cycle package edges, and
summary counts. Its stable schema is `pyrite.package-dependencies.v1`. Save a
copy of the baseline from the parent commit and compare to the new branch;
`module_edges` identifies the actual direct imports, rather than attributing
edges to every module in a package. `pyrite-dev verify` also runs the configured
import-linter checks.

The checked-in [machine-readable baseline](import-architecture-baseline.json)
records main at `a61e37b5`, including exact edges and all configured import-linter
contracts, ignored imports and indirect-import exceptions. It is reconstructed
from this slice's graph by restoring its sole removed cross-package edge;
the source diff identifies that edge explicitly. The largest static SCC contained
11 packages: `campaign`, `checkpoints`, `energy_grid`, `execution`,
`montecarlo`, `observations`, `perf`, `remote`, `results`, `runs`, `validation`.
The generated condensation graph had 18 component nodes. These are static
counts from the checked-in map, not an assertion about the runtime call graph.

| Metric | Baseline | Initial slice | SCC removal |
| --- | ---: | ---: | ---: |
| Cross-package module edges | 850 | 849 | 847 |
| Package edges | 124 | 123 | 121 |
| Components | 18 | 18 | 28 |
| Cyclic components | 1 | 1 | 0 |
| Largest component | 11 | 11 | 1 |

## Dependency direction

Design direction for *implementation imports*, from low to high:

1. Import-light scientific values, geometry, units, material data, configuration
   and identity contracts (root-level private leaves and `materials`).
2. Scientific computation kernels and transport/radiation engines. Engines
   should return portable value types, not import analysis drivers.
3. Observation/analysis value types and simulation case construction.
4. Case/sweep execution and persistence (`runs`, `execution`, `checkpoints`,
   `remote`), depending on the narrower value contracts below them.
5. User-facing API, CLI, apps, plots, interactive reporting and developer tools.

These are **intended** layers, not a claim the current source already conforms.
`results.model` still owns several value types imported by observations. Avoid
moving serialized dataclasses or changing `__module__`/pickle paths merely to
satisfy the diagram; a separate, reviewed type-ownership slice is needed.

## Initial cycle removal

A concrete direct-import cycle on main was:

```text
campaign.model -> observations.spec
observations.inventory -> checkpoints.persistence  # default_checkpoint_dir only
checkpoints.campaign_lock -> campaign.profiles
```

The second import is unnecessary: both checkpoint and observation locations are
resolved by `console.outputs.output_dir`. `observations.inventory` now resolves
`output_dir("checkpoints")` at call time, preserving workspace precedence,
legacy-migration opt-in, and the sibling observation-root policy. An
import-linter contract forbids any new *direct* `observations -> checkpoints`
edge. Indirect imports are temporarily allowed because other graph cycles
remain; otherwise this targeted contract would fail for unrelated existing
paths. The focused `repo_map` and inventory tests guard both source dependency
and path equivalence. No stored format, case identity, worker import, or
numerical operation changes in this slice.

A fresh spawned-process test imports the observation facade, resolves its lazy
inventory export and calls the default-path inventory without loading any
checkpoint module. This covers the deferred import as well as worker startup;
it does not claim that all other dynamic import paths are acyclic.

The initial *cycle edge* disappeared while the broad 11-package SCC remained:
other paths still connected these three packages. Compare `module_edges` and
`cycle_edges`, not only the largest-SCC size.

## Remaining SCC removal

Two upward dependencies kept the entire remaining component connected:

- `montecarlo.runner.transition_emission -> energy_grid.component_support`
  reached grid derivation, remote execution and validation through a package
  that also owned a small supported-band value contract.
- `campaign.resolution -> execution._simulation` reached execution artifacts
  and checkpoint drivers just to validate an executor and describe its policy.

The supported-band values and validation now live in `_component_support`,
beside the existing `_grid_semantics` leaf. Executor admission and provenance
live in `_executor_contract`; its type-only `Executor` reference does not load
execution drivers. The original modules re-export the same objects, with their
original `__module__` values retained for pickle/global lookup compatibility.
No algorithms, signatures, error messages or numerical defaults change.

These are real lower-level implementations, rather than deferred imports back
into the drivers. Fresh-process tests import and exercise both contracts without
loading any of the former SCC packages. Compatibility tests cover alias and
pickle identity; existing executor simulation/artifact and optical scorer tests
cover their consumers.

Three import-linter contracts prevent either upward edge from returning and
prevent the new leaves from importing any former SCC package, including through
indirect paths. A graph regression test requires every top-level package to
remain a singleton component. The generated graph has no static package cycles.
This remains a package-level result: the scanner excludes facade re-exports,
type-only imports and root-level leaves, and does not resolve dynamic imports.

## Follow-up priorities

- Split domain value contracts out of driver-heavy `results.store` and review
  `results.model`/`campaign.model` imports without changing public aliases or
  pickle identities.
- Separate engine-facing observation data from orchestration and persistence;
  examine `campaign -> observations -> results -> campaign` paths.
- Keep the now-directional `energy_grid`/`checkpoints` and `remote`/`runs`
  dependencies below their callers; the acyclic graph test guards regressions.
- Audit deferred/dynamic imports, subprocess worker startup, and package
  `__getattr__` facades separately from static SCC metrics.
