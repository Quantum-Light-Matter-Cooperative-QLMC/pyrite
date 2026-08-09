# uv workspace and test-suite partitioning

## Problem and scope

`cxr-mc` is one distribution and one pytest collection root. Core numerical
code, the `cxr`/`cxr-dev` command surfaces, plotting, and marimo applications
therefore share one dependency set and one default repository-wide test path.
This makes ownership boundaries unclear and encourages broad verification for
small changes.

Evaluate and, where justified, introduce a `uv` workspace that separates core
computation from CLI orchestration and possibly analysis/application tooling.
Partition tests into stable domain suites so contributors can run the smallest
relevant suite without weakening the full verification gate.

Current `uv` behavior constrains the design: a workspace has one lockfile and
shared environment, `uv run`/`uv sync` default to the root member, and
`--package` selects a member. Workspaces do not enforce Python dependency
isolation. Treat package separation and faster tests as measured outcomes, not
assumptions.

In scope:

- package/dependency/import-boundary inventory and timing baseline;
- root/member `pyproject.toml` design and workspace commands;
- core versus CLI, remote, plotting, and app ownership;
- test-directory or marker partitioning, targeted `cxr-dev` commands, and CI
  mapping;
- compatibility for `import cxr_mc`, packaged data, `cxr`, `cxr-dev`, extras,
  editable installs, lockfile, and release artifacts;
- documentation and agent-command updates required by the chosen layout.

Out of scope:

- unrelated source refactors or CLI redesign;
- changing numerical or physics behavior;
- claiming dependency isolation or test-time improvement without measurement;
- splitting analysis/apps into a separate member when measured coupling or
  packaging cost does not justify it.

## Implementation path and likely owners

1. Packaging/dev-tool baseline: `pyproject.toml`, `uv.lock`,
   `src/cxr_mc/_dev.py`, build metadata, scripts, CI, and Docker/remote install
   paths.
2. Core boundary: `materials/`, `montecarlo/`, `detectors/`, `results/`,
   `sweep.py`, `run.py`, and their minimal dependencies/data.
3. CLI boundary: `cli/`, `_remote/`, command handlers, `cxr` entry point, and
   CLI contract/reference generators.
4. Analysis/app boundary candidate: `plots/`, `analyze.py`, `check.py`,
   exporters, and `notebooks/`; keep this optional until import and dependency
   measurements support a separate member.
5. Test/tooling boundary: `tests/`, `checks/`, `cxr-dev`, CI workflows, and
   contributor/agent documentation.

Preserve the `cxr_mc` public import surface. Before moving files, choose and
document how multiple distributions avoid package-path ownership collisions;
prefer unique distribution ownership with explicit dependencies over relying
on accidental shared-site-packages behavior.

## Checklist

- [x] Record baseline install artifacts, dependency groups, import graph,
      collection time, and representative core/CLI/app/full test runtimes.
- [x] Define explicit member boundaries and a one-way dependency DAG; list
      every cross-boundary import and packaged-data owner.
- [x] Decide whether goal needs multiple published distributions, internal
      workspace members, dependency groups, or only test partitioning.
- [x] Prototype root/member metadata and verify `uv lock`, root `uv sync`, and
      `uv run --package <member>` behavior without committing source moves.
- [x] Add packaging smoke tests for clean wheels/editable installs, public
      imports, data lookup, extras, and `cxr`/`cxr-dev` entry points.
- [x] Implement the smallest justified core/CLI split; keep analysis/apps
      together unless the decision gate supports a third member.
- [x] Reorganize tests into documented domain suites or equivalent stable
      selectors; retain one full-suite command as release gate.
- [x] Map changed paths/members to focused `cxr-dev` and CI commands; ensure
      cross-member integration tests still run when boundaries change.
- [x] Update README/contributor/agent commands, generated CLI reference if its
      generation path changes, repository map, Docker/remote setup, and release
      configuration.
- [x] Re-measure install, collection, and test runtimes; document gains,
      regressions, and remaining shared-environment limitations.

## Decisions and open questions

- Primary objective must be chosen from dependency/install separation,
  ownership clarity, release modularity, and faster local verification; these
  imply different layouts.
- Should workspace root remain the publishable `cxr-mc` umbrella, become a
  virtual project, or remain the core distribution?
- Must core be independently installable/publishable, or is an internal member
  sufficient?
- Should analysis/apps become a third member now, remain an optional extra, or
  wait for a later task? Default: defer unless baseline shows a clean boundary.
- Can separate distributions preserve the `cxr_mc` namespace safely across
  wheel installers, or should CLI/app implementation packages use distinct
  import namespaces behind compatibility entry points?
- Should focused test selection use directories, pytest markers, member-local
  suites, a dependency map, or a combination? Avoid duplicated tests and
  marker-only conventions that silently omit integration coverage.
- Full verification remains mandatory before merge/release even if routine
  checkpoint commits use focused suites.

## Delegation slices and required skills

1. Packaging/import investigation: `repo-orientation`, `scientific-library`;
   read-only inventory plus proposed dependency DAG.
2. CLI and entry-point migration: `cli-ui-ux`, `regression-testing`;
   preserve frozen CLI/help/output contracts.
3. App/plotting feasibility: `notebook-workflow`, `documentation-maintenance`;
   no split unless independently justified.
4. Test partition and timing: `regression-testing`, `performance`;
   baseline and compare focused/full commands.
5. Integration/verification owner: `lead-task`, `run-cxr-mc`,
   `verifying-changes`; coordinate packaging, remote/Docker, and release checks.

Slices 2-4 may investigate independently after the dependency DAG is fixed.
One integrating owner controls package moves and workspace metadata.

## Acceptance checks

- `uv lock --check`, root sync, and each supported member sync/run path succeed
  from a clean environment using the repository cache convention.
- Built wheels contain intended modules/data only and install together without
  overwritten files or undeclared dependency leakage.
- `import cxr_mc`, public export guards, `cxr --help`, `cxr-dev --help`, CLI
  contract/reference tests, and representative headless app smoke tests pass.
- Core-only installation can import and execute documented numerical APIs
  without CLI/app-only dependencies, if independent core installation is an
  accepted goal.
- Documented focused suites cover core, CLI/remote, plotting/apps, packaging,
  and cross-member integration; full `cxr-dev verify` remains available and
  passes.
- Runtime report compares baseline and final collection/focused/full times;
  no unmeasured speed claim.
- Repo map, setup/install, CI, Docker/remote, agent commands, and release docs
  match the final layout.
- No numerical, checkpoint-schema, CLI-contract, or public-import regression.

## Implemented decision and evidence (2026-08-01)

Retain one publishable root `cxr-mc` distribution as sole owner of the
`cxr_mc` namespace, bundled data, extras, and both console scripts. Add one
internal, non-package `cxr-mc-tests` member with the only cross-member edge
`cxr-mc-tests -> cxr-mc`. It owns pytest dependencies and explicit member test
paths. Do not split CLI or apps into another wheel: core feeds results and
drivers, CLI lazily spans drivers/remote/apps, and all data resolves through
`cxr_mc.DATA_DIR`; shared namespace wheels would overwrite files/data while
uv's shared environment would still provide no isolation. Durable rationale,
commands, data ownership, and timing report:
[`docs/development-workspace.md`](../../../../docs/development-workspace.md).

Implemented:

- Workspace/member metadata and lockfile; no source/package move.
- `cxr-dev test-suite {core,cli,apps,packaging}` partitions every test module
  exactly once; a regression test enforces exhaustive non-overlap.
  `integration` intentionally overlaps public boundaries.
- `cxr-dev package-smoke` builds the wheel, checks namespace/data/extras/scripts,
  and installs wheel and editable source into separate clean uv environments.
- CI runs four focused jobs, a package job, and the full verify gate. Docker
  selects the test member; remote sync selects runtime-only `cxr-mc`.
- README, agent commands, docs navigation, repository map, and workspace guide
  match the layout. `cxr` did not change; CLI reference/contract tests pass, so
  generated CLI artifacts required no rewrite.

Verification:

- `uv lock --check`; clean root-only and test-member sync/run paths: pass.
- Clean wheel/editable smoke: pass for `import cxr_mc`, packaged
  `materials.toml`, extras metadata, `cxr`, and `cxr-dev`.
- Domain suites: core 945 passed, 39 skipped, one exact sandbox-only
  deselection; CLI 839 passed; apps 279 passed; packaging 173 passed.
  Integration: 634 passed.
- Final `cxr-dev verify` with only the sandbox-incompatible forkserver test
  deselected: Ruff/ty pass; 2,237 passed, 39 skipped, 1 deselected in 85.10 s.
- Outside restricted sandbox: all four `marimo check` paths and real
  `cxr app analysis --smoke` pass.
- Strict Sphinx renders the new page but remains red on 11 pre-existing
  repository warnings; none points at this guide.
- Tokensave served `main` because the branch is unindexed; live inspection
  verified results. Context/affected queries saved about 15,792 tokens.

No numerical, physics, checkpoint-schema, public-import, or `cxr` CLI behavior
changed. No speed/isolation claim: warm-cache measurements are recorded but not
causally attributed to workspace metadata.
