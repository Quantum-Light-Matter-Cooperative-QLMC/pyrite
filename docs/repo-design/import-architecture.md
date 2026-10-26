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

Baseline on main at `8816d484` (v0.21.0): the largest static SCC contained
11 packages: `campaign`, `checkpoints`, `energy_grid`, `execution`,
`montecarlo`, `observations`, `perf`, `remote`, `results`, `runs`, `validation`.
The generated condensation graph had 17 component nodes. These are static
counts from the checked-in map, not an assertion about the runtime call graph.

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

The *cycle edge* disappears even if the broad 11-package SCC remains: other
paths may still connect these three packages. Compare `module_edges` and
`cycle_edges`, not only the largest-SCC size.

## Follow-up priorities

- Split domain value contracts out of driver-heavy `results.store` and review
  `results.model`/`campaign.model` imports without changing public aliases or
  pickle identities.
- Separate engine-facing observation data from orchestration and persistence;
  examine `campaign -> observations -> results -> campaign` paths.
- Audit `energy_grid`/`checkpoints` and `remote`/`runs` host-side drivers, then
  add narrow *direct* import contracts after individual edges are removed.
- Audit deferred/dynamic imports, subprocess worker startup, and package
  `__getattr__` facades separately from static SCC metrics.
