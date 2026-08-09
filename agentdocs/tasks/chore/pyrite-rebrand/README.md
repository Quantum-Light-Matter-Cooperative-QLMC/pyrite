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

Sequence after `refactor/repo-structure-cleanup` lands. That branch is actively
splitting `montecarlo/runner.py` and `montecarlo/spectrum.py`; an import-package
rename in parallel would create broad, artificial conflicts.

1. Inventory identity-bearing surfaces and classify each as public canonical,
   compatibility alias, internal-only, persisted/on-disk, generated, or
   historical. Start from `pyproject.toml`, `src/cxr_mc/cli/`,
   `src/cxr_mc/paths.py`, `src/cxr_mc/remote/`, `src/cxr_mc/devtools/`, tests,
   `README.md`, `docs/`, scripts, Docker/CI metadata, and repository URLs.
2. Write and approve the identity/compatibility matrix before moving packages
   or changing commands. Apply the CLI RFC's additive alias and deprecation
   policy to command/environment/config changes.
3. Land the source/package transition in bounded checkpoints: packaging/import
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

- [ ] A — Produce a complete identity inventory and proposed compatibility
      matrix. Include exact old/new spellings, canonical-vs-alias status,
      deprecation/removal policy, persisted-data handling, and external
      availability/ownership evidence.
- [ ] B — Record the approved naming decision in the durable owning artifacts:
      display brand, long form, tagline, lowercase repository name,
      distribution, import namespace, commands, environment variables, and
      config/cache directory policy.
- [ ] C — Implement the approved distribution/import transition. Preserve or
      intentionally migrate public imports, package data, version discovery,
      editable/wheel installs, pickled module paths, entry shims, and frozen
      exports.
- [ ] D — Implement the approved CLI/config/environment transition additively.
      Cover root and developer commands, help, completion, JSON contracts,
      remote-generated invocations, environment precedence, Click app
      directories, and deprecation warnings.
- [ ] E — Update persisted/on-disk compatibility: checkpoint readers, campaign
      locks, remote job metadata, config/cache/workspace paths, and any schema
      identifiers selected for migration. Old valid artifacts must remain
      readable unless the approved matrix explicitly documents otherwise.
- [ ] F — Rebrand README/docs/apps/scripts/container and repository metadata.
      Regenerate `docs/cli-reference.md`, `docs/cli-deprecations.md`,
      `docs/api.md` where affected, and `docs/repo_map.md`; repair every
      maintained cross-link and clone/install example.
- [ ] G — Execute the authenticated GitHub repository rename, update remotes and
      configured integrations, and verify old web/Git operations redirect. Do
      not reuse `cxr-mc` while the redirect is part of the compatibility plan.
- [ ] H — Run focused compatibility probes and the full release gate; review a
      scoped identity search so remaining `cxr-mc` / `cxr_mc` / `cxr` /
      `CXR_*` occurrences are intentional compatibility or historical records.

## Decisions and open questions

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

Open and material before implementation:

- Distribution name: `pyrite-xray` is the current candidate, but ownership and
  availability must be rechecked immediately before reservation/publication.
- Import namespace: keep `cxr_mc`, add a `pyrite`/`pyrite_xray` facade, or move
  canonically with compatibility modules? Pickle compatibility and the value of
  an otherwise-costly namespace rename must drive this decision.
- CLI: make `pyrite` canonical with `cxr`/`cxr-dev` aliases, retain `cxr` as the
  stable scientific command under the new brand, and/or introduce a distinct
  developer command. Follow the accepted minimum deprecation window.
- Environment/config: whether to add `PYRITE_*` names and migrate the Click app
  directory, while defining deterministic precedence and conflict behavior
  with existing `CXR_*` variables and stores.
- Persisted identifiers: which `cxr.*` schema IDs, lockfiles, checkpoint
  metadata, remote job records, and on-disk directory names remain stable
  protocol identifiers rather than branding.
- External cutover timing and authenticated confirmation that the organization
  can use `pyrite`; anonymous API results cannot reveal private-name collisions.

No implementation slice is Serena `one-shot` eligible while A/B remain open.
After approval, C, E, and the repository-contained portion of F may become
self-contained; D remains cross-contract review work, and G remains a
credentialed coordination step.

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
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite packaging
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite cli
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite apps
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev package-smoke
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev repo-map --check
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
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
