# CLI profile/material ownership split

Current CLI divides campaign configuration between `cxr profile` and
`cxr sweep`, while also exposing profile membership through both mixed
`profile add/remove --materials` options and dedicated
`profile add-material/remove-material` commands. Replace that structure with
one campaign-oriented `profile` surface and one per-material override surface.
Keep profile membership single-owned: profiles contain materials; materials do
not contain or mutate a reverse list of profiles.

This task builds on `feature/profile-rename`. It supersedes the membership
direction proposed in the originating worktree's
`docs/cli-material-command-split.md`: do **not** add
`material add-profile`, `material remove-profile`, or a second persisted or
user-facing membership model.

## Ownership decisions

- `cxr profile` owns campaign identity, default ranges, electron-count grids,
  and optional material membership.
- `[profiles.NAME].materials` remains the only persisted membership relation.
  An absent key retains current implicit-all-in-use semantics.
- `cxr profile show NAME` is the authoritative place to list that profile's
  explicit or implicit members.
- `cxr material` owns operations addressed to one material: effective range
  inspection and per-profile override editing.
- `cxr material show MATERIAL --profile NAME` shows effective ranges and their
  inherited/overridden sources. It does not list “profiles this material
  belongs to.”
- Any future reverse lookup is derived read-only data, labelled “referenced by”
  rather than membership, and is outside this task unless a demonstrated
  workflow needs it.
- `cxr catalog` remains owner of catalog-wide material discovery. Do not add a
  duplicate `cxr material list`.

## Target command model

### Profile-owned campaign operations

- Keep `profile list`, `show`, `create`, `set`, `add`, `remove`, `rename`, and
  `delete`.
- Add one canonical profile-owned membership mutation group:
  - `profile members set NAME MATERIAL...` replaces explicit membership;
  - `profile members add NAME MATERIAL...` extends explicit membership;
  - `profile members remove NAME MATERIAL...` shrinks explicit membership;
  - `profile members reset NAME` removes the `materials` key and restores
    implicit all-in-use membership.
- Keep `profile create NAME --materials ...` for atomic initial campaign
  creation. Remove visible `--materials` mixing from `profile set/add/remove`;
  retain hidden compatibility options for the migration window.
- `profile show NAME` remains the authoritative membership display and must
  distinguish explicit membership from implicit all-in-use membership. Do not
  add a second `profile members show` path.
- Hide and deprecate `profile add-material` and `profile remove-material`;
  route them to the canonical profile-owned path for compatibility.

### Material-owned override operations

- Add `cxr material show MATERIAL [--profile NAME]`; default profile is
  `standard`.
- Add `cxr material set MATERIAL [--profile NAME]` with current
  thickness/energy/polar/azimuth and `--reset` behavior.
- Thread `--profile` through override lookup and mutation. Current
  `sweep set` hardcodes `standard`; the replacement must support
  `[profiles.NAME.overrides.MATERIAL]` for any valid profile.
- Do not add membership mutation commands under `material`.
- Require explicit `show`; do not add a bare-name dynamic alias in the first
  version. This avoids command/profile/material-name ambiguity.

### Sweep migration

- Retire visible top-level `cxr sweep`.
- Preserve hidden, deprecated `sweep show` and `sweep set` compatibility paths
  for an announced migration window.
- Move shared TOML/catalog helpers from `cli/sweep.py` into a non-command owner
  such as `cli/_catalog_io.py`; neither `profile.py` nor new `material.py`
  should import a retired command module.

## Help and compatibility requirements

- Use “profile” only for named catalog campaigns. Describe `full|survey` as
  fidelity presets, never “sweep profiles.”
- Explain membership ownership once in `profile --help`; material help should
  discuss effective ranges and overrides only.
- Compatibility warnings go to stderr and name the exact replacement command.
- Existing command defaults, confirmation behavior, dry-run behavior, atomic
  TOML writes, completion, exit codes, and JSON envelopes remain stable unless
  explicitly changed and tested.
- Regenerate `docs/cli-reference.md` and the frozen CLI contract; do not edit
  generated help snapshots manually.

## Implementation checklist

- [ ] Inventory current `profile`, `sweep`, catalog-profile resolution, and
      completion contracts.
- [ ] Implement canonical `profile members set|add|remove|reset` mutations.
- [ ] Add `src/cxr_mc/cli/material.py` and register visible `material`.
- [ ] Implement profile-aware material `show` and override `set/reset`.
- [ ] Move shared catalog TOML helpers out of `cli/sweep.py`.
- [ ] Hide/deprecate `sweep` and redundant material-membership aliases.
- [ ] Update source help, completions, generated CLI reference, repository map,
      and CLI contract fixture.
- [ ] Migrate `tests/test_sweep_cli.py` coverage to material/profile ownership
      tests.
- [ ] Test implicit membership, explicit membership, membership restoration,
      non-standard profile overrides, compatibility warnings, dry-run,
      confirmation, JSON, and unknown-name suggestions.
- [ ] Run focused CLI tests, real subprocess help/error probes, lint,
      typecheck, and generated-reference checks.

## Acceptance criteria

1. User can answer “which materials will profile NAME run?” from one
   authoritative profile command.
2. User can inspect and edit MATERIAL's effective ranges under profile NAME
   without changing profile membership.
3. No command presents a material-owned reverse membership relation.
4. No persisted reverse mapping duplicates `[profiles.NAME].materials`.
5. Non-standard profile overrides work; `standard` is only the default.
6. `cxr --help` no longer shows `sweep`; `cxr profile --help` shows
   `members` but not `add-material` or `remove-material`. Old paths remain
   script-compatible through hidden deprecations.
7. Generated reference and frozen CLI contract match runtime help.

## TODO reconciliation

Do not edit `TODO.md` from this branch while the originating
`feature/profile-rename` worktree has user-owned uncommitted TODO/design
changes. Reconcile the main backlog separately with `/todo-sync` once that
worktree is clean, keeping `TODO.md` byte-identical on `main` and this branch.
