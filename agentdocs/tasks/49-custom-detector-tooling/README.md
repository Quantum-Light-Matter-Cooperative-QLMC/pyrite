# Issue 49 slice: named detector CLI objects

Issue: #49

Work branch: `49-custom-detector-tooling`

## Problem

Detector geometry exists as a `Detector` value, but catalog profiles still own
three manually edited scalar fields and the CLI has no top-level detector noun.
Users cannot define one detector once, inspect it, reuse it across profiles, or
rename/delete it safely. This is the immediate named-object slice of #49.

## Decisions

- Add top-level `[detectors.NAME]` catalog objects. A profile may use
  `detector = "NAME"`; legacy inline `[profiles.NAME.detector]` remains readable.
- Named and inline forms use the same geometry decoder and resolve to the same
  `Detector` value. Object name and display-only label never affect simulation
  identity; resolved field values do. Both forms preserve the catalog's
  `Timepix3` default response introduced by issue #52 without serializing a
  response choice in detector TOML.
- Add visible `pyrite detector` with `list`, `show`, `create`, `set`, `rename`,
  and `delete`, matching established `pyrite beam` output, JSON, prompt,
  dry-run, completion, atomic-write, reference-update, and deletion-safety
  contracts.
- Add `pyrite profile create|set --detector NAME`. It conflicts with inline
  geometry flags. Retain those flags as compatibility paths with targeted
  deprecation warnings pointing to `pyrite detector create|set`.
- Replace the bundled standard inline detector block with a named default
  detector reference while preserving resolved 90-degree behavior and the
  standard dataset/checkpoint identity bit-for-bit.
- This slice covers the fields currently serializable in profile TOML:
  observation angle, full polar acceptance, and solid angle. Energy bins and
  response-specific schemas remain later #49 slices; do not invent a lossy
  serialization for runtime protocol objects.

## Plan

- [x] Add catalog parsing/storage for `[detectors.*]`, named-reference
  resolution, unknown-reference errors, and value-only identity behavior.
- [x] Add shared detector CLI options/writers so profile compatibility flags
  and named detector verbs validate identically.
- [x] Implement the top-level detector noun, shell completion, lazy command
  registration/help, text/JSON schemas, dry-run, rename propagation, and
  reference-safe deletion.
- [x] Add `profile create|set --detector NAME`; show named reference and
  resolved geometry. Deprecate but preserve inline detector flags.
- [x] Migrate bundled standard configuration to `[detectors.default]` plus
  `detector = "default"`; prove standard identity compatibility.
- [x] Add focused catalog, detector CLI, profile CLI, completion, identity, and
  generated-contract regression tests.
- [x] Regenerate CLI reference and catalog golden; run focused suites, lint,
  typecheck, and real CLI smoke checks.

## Implementation and acceptance evidence

Completed 2026-08-20 as the immediate named-geometry slice of #49.

- `MaterialCatalog` now parses and freezes named `[detectors.NAME]` geometry,
  labels, and ordered keys. `profiles.NAME.detector = "NAME"` resolves through
  the same decoder as an inline detector table. Unknown references report the
  profile path. Tests prove a named object and equivalent inline block resolve
  equal `Detector` values and equal `parameter_sha256`; labels and names never
  enter the resolved value.
- Visible lazy `pyrite detector list|show|create|set|rename|delete` commands
  implement text and versioned JSON results, validation, dry-run diffs,
  overwrite/destructive prompts, optimistic revalidation before deletion,
  suggestion-rich unknown-name errors, atomic writes, rename propagation, and
  reference-safe deletion. `tests/cli/test_detector.py` covers the complete
  verb lifecycle.
- `profile create|set --detector NAME` stores named references and conflicts
  with the three inline geometry flags. Profile show reports `detector_ref`
  plus resolved geometry in JSON and marks named references in text. Legacy
  geometry flags remain functional, warn through the deprecation registry, and
  detach a named reference into an equivalent inline table before applying the
  requested scalar so unspecified geometry is preserved.
- Offline completion reads only `[detectors.*]`; existing-name arguments and
  both profile attachment options use it, while create/new rename targets do
  not. Root help exposes the noun without importing scientific or hardware
  modules.
- Bundled `standard` now uses `detector = "default"` with
  `[detectors.default] observation_angle_deg = 90.0`. Named and inline geometry
  both retain issue #52's default `Timepix3` response. Existing resolved
  standard payload/digest/stem pins pass unchanged relative to that contract;
  catalog golden regeneration produced no diff because the named-object
  migration did not change resolved physical data.
- Maintained documentation covers the tenth root noun, named-detector profile
  workflow, identity boundary, CLI owner, and runtime-only response/bin
  boundary. CLI reference, deprecation reference, and frozen Click contract
  were regenerated.

Checks:

- Focused catalog/profile/detector/completion/contracts/docs-block suite:
  611 passed, 1 existing deprecation warning in 22.33 s.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli`:
  1227 passed in 36.75 s.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/energy-grid/`:
  235 passed in 10.59 s.
- `pyrite-dev lint`, `pyrite-dev typecheck`, and `pyrite-dev docs`: passed;
  strict offline Sphinx build succeeded.
- `pyrite-dev cli-reference --check`, `pyrite-dev cli-deprecations --check`,
  `pyrite-dev regen-golden --check`, `pyrite-dev repo-map --check`, frozen CLI
  contract check, and `git diff --check`: passed.
- Real read-only smokes passed: `pyrite detector list`,
  `pyrite detector show default -o json`, `pyrite profile show standard -o
  json`, and `pyrite --help`.

## Remaining issue #49 work

Named response objects, response-specific parameters, detector energy-bin
objects, and any portable resource/provenance schema remain later #49 slices.
This slice deliberately serializes only the three established geometry fields;
it adds no detector physics or configurable response behavior. Geometry
decoding preserves the existing issue #52 `Timepix3` default response.

## Acceptance

- `pyrite detector create standard-90 --observation-angle 90` creates a valid
  reusable object; list/show/set/rename/delete obey human and JSON contracts.
- `pyrite profile set NAME --detector standard-90` stores a reference and the
  loaded profile resolves the same `Detector` value as an equivalent legacy
  inline block.
- Rename updates every referencing profile atomically; delete refuses while
  referenced and names all referents.
- Unknown names suggest close matches and point to `pyrite detector create`.
- Named detector names/labels do not alter dataset identity; changed geometry
  does. Bundled standard payload/digest/stem remain compatible.
- Root/nested help remains lazy and documents the visible noun; shell
  completion is offline and side-effect-free.
- Focused tests, CLI reference check, catalog golden check, lint, and typecheck
  pass.

## Non-goals

- New detector response physics, response kernels, QE curves, or pixel models.
- Serializing arbitrary `DetectorResponse` implementations or ndarray energy
  bins before a format/provenance contract exists.
- Heavy Monte Carlo, remote jobs, issue edits, or pushes.

## Authority

One writer in this worktree. Checkpoint commits allowed. No push, issue edits,
delegation, remote execution, or unrelated cleanup.
