# Clearer user control of calculation numerics (issue #33)

Branch: `33-clearer-user-control-of-calculation-numerics`

## Problem

The issue asks for clearer user control of calculation numerics, but its migrated
backlog text does not define which controls or workflow are intended. The current
surface is capable but fragmented:

- the Python API has typed `Numerics` and `Convergence` objects;
- `pyrite profile create|set` stores line/bremsstrahlung electron-count grids and
  transport controls (`straggling`, `energy_model`, `max_dE_frac`);
- `pyrite run --n-families` is a per-invocation convergence override;
- fidelity presets also select electron counts, reflection limits, and grid
  reductions;
- worker/chunk/backend/core controls tune execution and should not be presented
  as scientific convergence controls.

The user therefore cannot inspect calculation numerics as one coherent effective
policy, and several result-affecting `Convergence` fields have no profile CLI.

## Proposed design for review

### Canonical workflow

Add a nested `pyrite profile numerics` workflow rather than another top-level
noun or a reusable named object:

```text
pyrite profile numerics show PROFILE [--fidelity full|survey] [-o table|json|wide]
pyrite profile numerics set PROFILE [OPTIONS] [--dry-run] [--yes]
pyrite profile numerics reset PROFILE [FIELD]... [--dry-run] [--yes]
```

`show` reports both explicit profile values and effective values after fidelity
resolution, with the source of each value (`profile`, `fidelity`, or built-in).
`set` and `reset` reuse the existing atomic catalog-edit path and validation.
Human output groups sampling, convergence, and transport controls; JSON uses an
explicit versioned payload.

The canonical user-facing terms should be descriptive (`line electrons`,
`bremsstrahlung electrons`, `reflection families`, `maximum reflections`,
`mosaic nodes/route`, `energy model`, `maximum fractional energy loss`, and
`straggling`). Existing `profile create|set --ne-line/--ne-brem` and transport
flags remain compatible during the normal deprecation window; this task must not
silently break scripts or existing catalog TOML.

### Scope boundary

This task owns result-affecting calculation numerics:

- sampling counts: `n_electrons`, `n_electrons_brem`;
- reflection/mosaic convergence: `n_families`, `max_reflections`,
  `mosaic_nodes`, `mosaic_route`;
- transport integration/model controls already exposed on profiles:
  `straggling`, `energy_model`, `max_dE_frac`.

It does not own result-invariant execution tuning: `workers`, `spec_chunk`,
`brem_chunk`, backend selection, or transport-core selection. Issue #55 owns
moving performance-only run flags to `pyrite-dev`.

### Precedence and identity

Use one documented resolution order:

```text
per-invocation override > explicit profile numerics > fidelity preset > built-in
```

Every result-affecting effective value must participate in dataset identity and
checkpoint collision protection. Result-invariant execution tuning must not.
Existing profiles that do not set new fields must retain their current resolved
behavior and identity. Do not change physics algorithms or default numerical
values in this task.

### Storage

Prefer extending the existing profile representation and compatibility decoder
over introducing `[numerics.NAME]` shared objects. Numerics are campaign policy,
already partly profile-owned, and their fidelity interaction makes indirection
less clear. Final key placement (compatible flat keys versus a nested profile
table with legacy decoding) is an implementation decision only if it preserves
old catalogs byte-semantically and keeps the CLI contract above.

## Checklist

- [ ] Confirm the canonical nested profile workflow and scope boundary with the
  user; this task is not one-shot while that decision is open.
- [ ] Write CLI contract tests first: help, validation, effective-value/source
  reporting, JSON envelope, dry-run, reset, standard-profile confirmation, and
  compatibility paths.
- [ ] Add one reusable resolver that returns explicit/effective numerics plus
  provenance for both CLI display and run lowering.
- [ ] Extend catalog/profile validation for the missing convergence fields by
  reusing `Numerics`/`Convergence` domain validation rather than duplicating it
  in Click callbacks.
- [ ] Implement `profile numerics show|set|reset`; preserve old flags and TOML.
- [ ] Make fidelity/profile/per-invocation precedence consistent across line and
  bremsstrahlung counts, reflection limits, and mosaic controls.
- [ ] Include every newly configurable result-affecting value in resolved
  provenance and dataset identity without perturbing implicit-default identity.
- [ ] Add focused catalog, profile-resolution, dataset-identity, and CLI
  regressions.
- [ ] Update the sweep-profile/configuration guide and regenerate the CLI
  reference and deprecation reference if compatibility aliases are added.
- [ ] Run focused tests, CLI suite, core suite, docs checks, and final verify.

## Likely owners

- `src/pyrite/cli/commands/profile.py`: nested command surface and output.
- `src/pyrite/campaign/profile_edit.py`: atomic profile mutations and payloads.
- `src/pyrite/materials/catalog.py`: profile schema decode/validation.
- `src/pyrite/campaign/profiles.py`, `src/pyrite/campaign/config.py`,
  `src/pyrite/runs/scan.py`: effective resolution, fidelity precedence, identity.
- `src/pyrite/campaign/model.py`, `src/pyrite/campaign/sweep.py`: domain
  validation/lowering only where existing seams cannot own it.
- `tests/cli/test_profile.py`, `tests/materials/test_material_catalog.py`,
  `tests/materials/test_profiles.py`, and focused scan/identity tests.
- `docs/guides/sweep-profiles.md` and generated CLI references.

## Required skills and dispatch

- Supervisor: `dispatch-task` after user approval.
- Worker: `lead-task` because precedence and checkpoint identity cross CLI,
  catalog, and campaign lowering.
- Domain skills: `cli-ui-ux`, `scientific-library`,
  `documentation-maintenance`, `regression-testing`.
- Serena execution: interactive until the design decision is approved; the
  reviewed implementation may use a one-shot contract.
- Default authority after direct user approval: checkpoint commits; no push,
  issue writing, or delegation unless explicitly granted by dispatch.

## Acceptance checks

- A CLI-only user can inspect and persist every in-scope result-affecting
  numerical control without editing TOML.
- `show --fidelity full|survey` identifies each effective value and its source.
- Invalid or coupled values fail at the boundary with the field, allowed domain,
  and correction; no traceback or partial write.
- Existing profile commands/catalogs continue to work, and implicit defaults
  retain current results and dataset identities.
- Explicit profile values override fidelity defaults; explicit run overrides
  override the profile where supported.
- Distinct effective result-affecting numerics cannot share a checkpoint
  identity; execution-only tuning remains identity-neutral.
- Generated CLI docs are current and focused CLI/core/docs/verify checks pass.

## Open review decision

Approve the proposed nested `pyrite profile numerics` workflow, or choose one
of these narrower alternatives before dispatch:

1. Keep `profile create|set|show` as the only commands and improve grouping,
   help, and missing convergence options there.
2. Introduce reusable named `[numerics.NAME]` objects attachable to profiles,
   analogous to named beams/detectors (larger schema and lifecycle surface).

