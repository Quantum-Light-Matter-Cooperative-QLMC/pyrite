# Documentation hierarchy cleanup

- **Branch:** `docs/create-hierarchy`
- **Authority:** direct user invocation of `lead-task`; task-local edits and
  checkpoint commits permitted; no push, TODO edit, delegation, or retirement
  authority.
- **Goal:** finish the Sphinx/MyST hierarchy migration, repair stale links,
  generators, and tests, retire the package/project RFCs in favor of concise
  ADR-0004/0007, and leave strict documentation checks green.
- **Non-goals:** physics/numerical behavior, CLI or public API redesign,
  weakening warning policy, exhaustive document inventory.

## Checklist

- [x] Establish intended filesystem and Sphinx navigation hierarchy.
- [x] Categorize strict docs and relevant test failures.
- [x] Repair canonical paths, links, configuration, and generators.
- [x] Update stale tests without weakening structural invariants.
- [x] Audit ADR-0004/0007 and retired RFC consumers.
- [x] Run strict docs, relevant suites, generator checks, and residual-link audit.

## Decisions

- `docs/validation/methodology.md` is the canonical validation-method document;
  `docs/validation/index.md` is its landing page.
- Published navigation follows document class: guides, physics, validation,
  repository design, ADRs, and API reference.
- Existing task records may retain historical path mentions where they are
  point-in-time evidence; maintained/public consumers must use current paths.
- Exploratory models and unimplemented proposals live under `docs/research/`;
  current physical-model descriptions remain under `docs/physics/`.

## Completion 2026-08-10

- Added site-native getting-started and result/checkpoint guides.
- Added the published research hierarchy; moved grazing-grating, future
  relativistic/channeling designs, and parameter-space proposal into it.
- Moved performance history under repository compute design, external
  bremsstrahlung comparison under validation, and grazing-beam projection under
  validation geometry.
- Repaired maintained links, source/check references, validation self-paths,
  TODO pointers, and the generated CLI-deprecation preamble.
- Strict offline Sphinx build, lint, repository-map check, CLI-deprecation
  regeneration/check, and focused CLI generator tests pass.

## Checkpoint 2026-08-09

Paused at user request before final completion.

- Strict offline Sphinx build passes with no warnings under
  `CXR_MC_BACKEND=cpu`.
- Skill sync/check and repository-map check pass.
- CLI reference and deprecation outputs were regenerated at their new canonical
  paths; focused CLI/dev tests reached 140 passing with one environment-caused
  CUDA selection failure before the CPU override was added.
- Full CPU test run was interrupted at 59% at user request; two failures had
  appeared but their reports were not reached. Lint, typecheck, generator
  rechecks, residual-path audit, and final diff review remain for the next slice.
