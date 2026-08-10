# PyRITE project conversion

## Problem and scope

`cxr-mc` describes the current coherent-X-ray Monte Carlo implementation but is
generic, difficult to remember, and too narrow for the repository's radiation,
electron-transport, detector, materials, analysis, and validation surfaces.
Adopt the public brand **PyRITE**:

> **PyRITE: a Python toolkit for Radiation from Interactions and Transport of
> Electrons**

Use the short scientific tagline **Coherent X-ray radiation and electron
transport in crystals.** Use `pyrite` for lowercase repository infrastructure;
reserve `PyRITE` for prose and visible product identity.

This is a compatibility-sensitive conversion, not a blind text replacement.
The current identity spans the `cxr-mc` distribution, `cxr_mc` import package,
`cxr` / `cxr-dev` entry points, `CXR_*` environment variables, Click's app
directory, remote scripts and paths, checkpoint/schema identifiers, generated
CLI/API/repository references, installation examples, and GitHub URLs.

In scope:

- settle and record one identity/compatibility matrix for brand, repository,
  distribution, import namespace, commands, environment variables, config and
  cache paths, persisted schemas, and release artifacts;
- rebrand public prose, UI chrome, package metadata, generated references,
  contributor tooling, examples, and external repository metadata;
- relocate the implementation from `src/cxr_mc/` to `src/pyrite/`, only after
  revising the approved import/pickle compatibility policy and providing a
  tested `cxr_mc` compatibility surface for existing users and artifacts;
- introduce any approved new distribution/import/CLI/config identities with
  explicit compatibility aliases or migrations for existing users and data;
- update tests and runtime probes so both the new canonical surface and every
  intentionally retained compatibility path are enforced;
- coordinate the GitHub repository rename and local/remote URL updates without
  reusing the old repository name while redirects are needed.

Out of scope:

- physics, numerical, detector, or material-model changes;
- unrelated CLI restructuring or removal of compatibility paths already
  governed by the accepted CLI deprecation policy;
- renaming literature terminology (`PXR`, `CBS`, CXR as a scientific concept),
  validation IDs, or persisted schema keys merely because they contain `cxr`;
- publishing a release or claiming a package name before ownership and
  availability are verified.

## Implementation path and likely owners

The prerequisite `refactor/repo-structure-cleanup` landed before implementation
started. The conversion therefore builds on its package boundaries without
overlapping that completed refactor.

1. Inventory identity-bearing surfaces and classify each as public canonical,
   compatibility alias, internal-only, persisted/on-disk, generated, or
   historical. Start from `pyproject.toml`, `src/cxr_mc/cli/`,
   `src/cxr_mc/paths.py`, `src/cxr_mc/remote/`, `src/cxr_mc/devtools/`, tests,
   `README.md`, `docs/`, scripts, Docker/CI metadata, and repository URLs.
2. Write and approve the identity/compatibility matrix before moving packages
   or changing commands. Apply the CLI RFC's additive alias and deprecation
   policy to command/environment/config changes.
3. Revise the identity matrix for the `src/pyrite/` implementation move, then
   land the source/package transition in bounded checkpoints: packaging/import
   compatibility first; CLI/config/path compatibility second; public brand and
   generated documentation third. Keep persisted checkpoint and remote-job
   interoperability explicit throughout.
4. Perform the authenticated GitHub rename only after repository-contained URLs,
   CI/release settings, clone instructions, and rollback notes are ready. Update
   local remotes and verify old-location redirects.
5. Regenerate the CLI reference and repository map, build/install the wheel in a
   clean environment, and exercise real old/new command/import workflows.

Likely owners:

- packaging/imports: `pyproject.toml`, `src/cxr_mc/`, package smoke and export
  guards;
- CLI and compatibility: `src/cxr_mc/cli/`, `src/cxr_mc/_dev.py`, completion,
  deprecation registry, CLI contract snapshot, remote-generated commands;
- config and persisted paths: `src/cxr_mc/paths.py`, `src/cxr_mc/cli/_config.py`,
  `src/cxr_mc/remote/`, checkpoint and campaign-lock readers;
- branding/docs: `README.md`, `docs/`, app design chrome, scripts, Dockerfile,
  generated CLI/API/repository references;
- external cutover: GitHub organization settings and any eventual package-index
  release, requiring repository-owner credentials.

Related accepted design constraints:

- [`docs/cli-redesign-rfc.md`](../../../../docs/cli-redesign-rfc.md), especially
  additive aliases and the minimum deprecation window;
- [`docs/package-structure-rfc.md`](../../../../docs/package-structure-rfc.md) and
  [`docs/adr/0004-package-and-repository-structure.md`](../../../../docs/adr/0004-package-and-repository-structure.md),
  which establish the current single-distribution/package ownership and
  compatibility re-export policy;
- [`docs/development-workspace.md`](../../../../docs/development-workspace.md),
  which documents the root wheel, import namespace, entry points, and clean
  package-smoke contract.

## Checklist

- [x] A — Produce a complete identity inventory and proposed compatibility
      matrix. Include exact old/new spellings, canonical-vs-alias status,
      deprecation/removal policy, persisted-data handling, and external
      availability/ownership evidence.
- [x] B — Record the approved naming decision in the durable owning artifacts:
      display brand, long form, tagline, lowercase repository name,
      distribution, import namespace, commands, environment variables, and
      config/cache directory policy.
- [x] C — Implement the approved distribution/import transition. Preserve or
      intentionally migrate public imports, package data, version discovery,
      editable/wheel installs, pickled module paths, entry shims, and frozen
      exports.
- [x] D — Implement the approved CLI/config/environment transition additively.
      Cover root and developer commands, help, completion, JSON contracts,
      remote-generated invocations, environment precedence, Click app
      directories, and deprecation warnings.
- [x] E — Update persisted/on-disk compatibility: checkpoint readers, campaign
      locks, remote job metadata, config/cache/workspace paths, and any schema
      identifiers selected for migration. Old valid artifacts must remain
      readable unless the approved matrix explicitly documents otherwise.
- [x] F — Rebrand README/docs/apps/scripts/container and repository metadata.
      Regenerate `docs/cli-reference.md`, `docs/cli-deprecations.md`,
      `docs/api.md` where affected, and `docs/repo_map.md`; repair every
      maintained cross-link and clone/install example.
- [ ] G — Execute the authenticated GitHub repository rename, update remotes and
      configured integrations, and verify old web/Git operations redirect. Do
      not reuse `cxr-mc` while the redirect is part of the compatibility plan.
- [x] H — Run focused compatibility probes and the full release gate; review a
      scoped identity search so remaining `cxr-mc` / `cxr_mc` / `cxr` /
      `CXR_*` occurrences are intentional compatibility or historical records.
- [ ] I — Relocate the implementation package from `src/cxr_mc/` to
      `src/pyrite/`. First revise and approve the import/pickle portion of the
      identity matrix; then provide a tested `cxr_mc` compatibility surface
      for old imports, deep imports, package resources, and existing pickle
      module paths. Update packaging, tooling, generated references, tests,
      and the repository map. Do not claim rebrand completion until G and I
      both pass.

## Decisions

Decided from user review:

- Visible brand: **PyRITE**.
- Long form: **a Python toolkit for Radiation from Interactions and Transport
  of Electrons**.
- Short tagline: **Coherent X-ray radiation and electron transport in
  crystals.**
- Repository spelling: lowercase `pyrite`; do not use `Pyrite` or `PyRITE` as
  the operational repository name.
- Treat the conversion as a compatibility migration, not global replacement.
- Bare `pyrite` is already owned on PyPI; do not depend on acquiring it.

Accepted A/B matrix: [`project-identity-rfc.md`](project-identity-rfc.md),
recorded by [ADR-0007](../../../../docs/adr/0007-project-identity.md).

- Distribution: `pyrite-xray`; normalized wheel/dist-info stem `pyrite_xray`.
  Do not create a new `cxr-mc` compatibility distribution. Recheck and control
  the name immediately before publication.
- Import: keep `cxr_mc` canonical indefinitely. Do not add `pyrite` or
  `pyrite_xray` facades; this preserves the public API and pickle module paths.
  **Supersession required before I:** the requested move to `src/pyrite/`
  conflicts with this decision. Approve a replacement import and pickle
  compatibility matrix before implementation; a directory rename alone is not
  acceptable because it breaks deep imports, resource lookup, and unpickling.
- Commands: `pyrite` and `pyrite-dev` canonical from 0.2.0; `cxr` and
  `cxr-dev` remain installed compatibility executables, removable no earlier
  than 0.4.0 after the accepted two-minor window. Normal invocation warns;
  completion-mode invocation stays silent.
- Environment: add exact `PYRITE_*` counterparts for maintained `CXR_*`
  variables. Resolve per-call > `PYRITE_*` > `CXR_*` > canonical store >
  legacy-store fallback > built-in. Differing new/old values select `PYRITE_*`
  and warn once; legacy variables have no scheduled removal.
- User state: canonical Click application stem `pyrite`; read the canonical
  store first, fall back to `cxr-mc`, and atomically copy on first write without
  deleting legacy state. Cache/data resolution uses `platformdirs` with app
  name `pyrite`; caches may read through and regenerate. Never move a workspace
  or configured remote checkout automatically.
- Persisted protocols: keep `cxr_mc` pickle paths, all current `cxr.*` schema
  IDs, `cxr.lock.json`, `CXR_REMOTE_V1`, `CXR_REMOTE_FRAME_END`, checkpoint
  layouts, campaign locks, energy-grid artifacts, job metadata, and existing
  directories readable. New schema families use `pyrite.*`.
- Release identity: `PyRITE X.Y.Z`; tags remain `vX.Y.Z`; distribution files
  use `pyrite_xray-X.Y.Z`, containers use `pyrite:X.Y.Z`, and prior artifacts
  remain immutable.

Credential-only questions remain outside repository implementation:

- PyPI returned HTTP 404 for `pyrite-xray` on 2026-08-09, but an authenticated
  owner must still confirm/reserve it and recheck immediately before the first
  publication. No package was reserved or published.
- Anonymous GitHub API checks returned HTTP 404 for both the proposed `pyrite`
  path and the known-private current `cxr-mc` path. Only an authenticated
  organization owner can exclude a private collision and confirm rename
  authority. No repository, remote, integration, or redirect was changed.

## A/B evidence and inventory

Completed 2026-08-09 from the clean `chore/pyrite-rebrand` worktree. Exact-text
search covered `pyproject.toml`, `uv.lock`, README/docs, `src/cxr_mc`, tests,
scripts, Docker/CI metadata, agent tooling, GitHub URLs, every maintained
`CXR_*` spelling, `cxr.*` schemas, Click directories, and release/container
names. Serena ownership checks covered the CLI, developer runner, shared config
resolver, path resolver, remote script builders, package smoke, checkpoint
serialization, campaign locks, and energy-grid/performance/cache identifiers.

Classifications and exact compatibility/removal rules are durable in the RFC:
public canonical, packaging/import, CLI contract, config/mutable paths,
persisted/on-disk, remote/generated commands, generated docs/data,
repository/release, scientific/internal, and historical. Notable inventory
anchors:

- package/distribution/scripts: `pyproject.toml`, `src/cxr_mc/__init__.py`,
  `src/cxr_mc/cli/__init__.py`, `src/cxr_mc/_dev.py`, and
  `src/cxr_mc/devtools/package_smoke.py`;
- precedence/state: `src/cxr_mc/cli/_config.py`, `src/cxr_mc/paths.py`, app
  launcher state, and `src/cxr_mc/plots/plotly/render.py`;
- compatibility protocols: checkpoint/campaign/energy-grid/performance/Zhai
  owners plus `src/cxr_mc/remote/` framing, metadata, and generated commands;
- derived identities: CLI reference/deprecation generators, repository-map
  generator, API autosummary, CLI contract snapshot, and `uv.lock`;
- public/external identities: README/Sphinx/app chrome, clone URLs, Docker
  tags, current Git remote, and repository/release metadata.

Authoritative read-only availability evidence on 2026-08-09: PyPI `pyrite`
returned HTTP 200 and identifies an unrelated version 0.1 project attributed to
Mark Ramm; `pyrite-xray` and `cxr-mc` returned HTTP 404. Anonymous GitHub API
checks for the proposed and current repository paths both returned HTTP 404,
which is inconclusive for this private organization repository. Full URLs and
the credential boundary are recorded in the RFC.

C is checkpointed by `7cedf60`: distribution `pyrite-xray`, stable `cxr_mc`
imports/pickle paths, and clean-install package guards. D/E are checkpointed by
`268a9ad`: canonical and compatibility executables, deterministic environment
precedence, completion compatibility, canonical config/state/cache writes with
legacy read-through, and stable persisted/wire identifiers. F updates current
public and contributor surfaces, release metadata/version 0.2.0, generated CLI
and repository references, app chrome, scripts, Docker/CI, and canonical skill
mirrors. G remains an explicitly credentialed external step. H completed after
the F checkpoint: the full `pyrite-dev verify` gate and 232-test packaging suite
passed. `package-smoke` validated clean wheel and editable installs, package
metadata, `cxr_mc` imports and data, canonical `pyrite` / `pyrite-dev` entry
points, warned legacy executables, and packaged app launch probes. Focused CLI,
app, config, completion, remote-command, import, version, and transport-
environment regressions passed. A final exact-text audit found no maintained
old GitHub or GitHub Pages URLs; remaining old spellings are the RFC's retained
import, protocol, compatibility, scientific, internal, or historical classes.
The only full-suite warning was the pre-existing line-spectrum square-root
runtime warning; no physics or numerical implementation changed.

## Delegation slices and required skills

- A/B — `lead-task` + `repo-orientation` + `cli-ui-ux` +
  `documentation-maintenance`; one integrating owner, no parallel package or
  command edits before the matrix is approved.
- C — `implement-task` + `scientific-library`; package/API compatibility,
  serialization/import probes, and package smoke.
- D — `implement-task` + `cli-ui-ux`; command, completion, environment/config,
  remote invocation, and generated CLI contracts.
- E — `implement-task` + `regression-testing`; persisted artifacts and old/new
  compatibility fixtures.
- F — `implement-task` + `documentation-maintenance` + `repo-orientation`;
  maintained prose, generated references, links, and repository inventory.
- G — integrating owner plus repository owner; no delegated credentials, push,
  publication, or external rename authority unless explicitly granted.
- H — `run-cxr-mc` plus the owning skills above; focused probes first, then the
  full verification gate.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite packaging
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite apps
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev package-smoke
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev repo-map --check
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
```

- The approved identity matrix has exactly one canonical spelling per surface
  and names every retained compatibility spelling and removal condition.
- Clean editable and wheel installs expose the approved distribution, import,
  package-data, and entry-point surfaces.
- Approved new and legacy CLI invocations have tested help, completion,
  stream/exit, JSON, environment/config precedence, and remote command behavior.
- Existing valid config, checkpoints, campaign locks, pickles, and remote job
  records remain usable or have a tested, documented migration path.
- README, docs navigation, API/CLI references, app chrome, scripts, Docker/CI,
  repository map, and external repository metadata consistently use PyRITE.
- Every remaining old-name occurrence is reviewed and classified as an
  intentional compatibility surface, stable protocol identifier, historical
  record, or defect.
- GitHub old-location web and ordinary Git operations redirect after the rename;
  local remotes and integrations point to the new location.
- No physics/numerical result changes.
