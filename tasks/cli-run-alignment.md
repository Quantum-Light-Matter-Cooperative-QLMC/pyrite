# CLI run alignment

Branch: `feature/cli-run-alignment`

TODO scope: P1 CLI Work sub-items 1–5.

## Goal

Expose one local and one remote sweep surface with the same selection model:

```text
cxr run [PROFILE] [-m MATERIAL]
cxr remote run [PROFILE] [-m MATERIAL]
```

`PROFILE` defaults to `standard`. Without `-m/--material`, run the profile's
resolved membership. Material-group composition belongs to `cxr profile
members`, not run/submit commands.

Also rename performance controls and add transfer compression:

```text
-r, --perf-reps N
-i, --perf-interval SECONDS
--level9
```

## Decisions

- Canonical verb: `run`. Same noun locally and remotely; locality is expressed
  only by the `remote` group.
- Fully remove `cxr remote scan`, including registration, help, completion,
  contract snapshots, tests, and docs.
- Replace `cxr scan` and `cxr remote submit` with canonical `run` paths. Do not
  retain multiple public names for the same operation.
- Remove run-time material selectors `-a/--all`, `-A/--actually-all`,
  `--include-unverified-dw`, `--include-high-energy` /
  `--including-high-energy`, and `--high-energy-min-kev`.
- Add group selectors to each applicable `profile members set|add|remove`
  operation:
  - `--unverified-dw`: operate on the catalog's unverified-Debye-Waller group.
  - `--high-energy-only`: operate on the catalog's high-energy group.
  - Existing explicit `MATERIAL...` arguments remain composable with group
    selectors; de-duplicate in catalog order.
- `--level9` controls the automatic post-run remote checkpoint preparation and
  pull path, matching `cxr remote pull --level9`.
- Rename public flags and every user-facing diagnostic/help example. Internal
  parameter names may remain stable if that reduces churn.
- Remove manual beam override options from the local run command. Profile and
  per-material configuration remain their owners.

## Owning paths

- Root/local CLI: `src/cxr_mc/cli/__init__.py`, `src/cxr_mc/scan.py`
- Remote CLI and lifecycle: `src/cxr_mc/_remote/cli.py`,
  `src/cxr_mc/_remote/lifecycle.py`, `src/cxr_mc/_remote/scripts.py`
- Profile membership: `src/cxr_mc/cli/profile.py`,
  `src/cxr_mc/cli/_catalog_io.py`
- Entrypoint text: `src/cxr_mc/_entry/scan.py`
- Contracts/tests: `tests/test_local_click_cli.py`, `tests/test_scan_beam_options.py`,
  `tests/test_remote.py`, `tests/test_remote_click.py`,
  `tests/test_cli_profile.py`, `tests/test_profiles.py`,
  `tests/test_cli_completion.py`, `tests/data/cli_contract.json`
- Generated docs: `docs/cli-reference.md`; examples in `README.md` and `docs/`

## Implementation path

1. Extract one profile/material selection resolver usable by local and remote
   boundary commands. Preserve lazy imports and catalog validation.
2. Extend profile membership mutation with catalog-group expansion and atomic
   TOML validation. Cover set/add/remove, mixed explicit/group input, empty
   results, duplicates, `standard` confirmation, and dry-run.
3. Register `cxr run`; make positional profile plus optional single material
   the only selection model. Delete obsolete local selection/beam options and
   diagnostics.
4. Register `cxr remote run` over the same selection contract. Delete
   `remote scan` and obsolete submit selectors.
5. Rename performance flags and diagnostics. Forward `--level9` through
   automatic pull without changing dry-run/headless/no-pull behavior.
6. Update shell completion, examples, generated CLI contract, and reference.
7. Search for stale command/flag spellings. Retain historical mentions only
   where explicitly documenting migrations.

## Verification

- Focused Click tests for local/remote help, dispatch, default `standard`
  profile, `-m`, unknown profile/material, incompatible options, and removed
  paths/flags.
- Profile tests for each group selector across set/add/remove and dry-run.
- Remote dry-run test proving `--level9`, `--perf-reps`, and `--perf-interval`
  reach generated workflow arguments without network access.
- Completion tests for `run`, profile, and material positions.
- Regenerate `tests/data/cli_contract.json` and `docs/cli-reference.md`.
- Run focused CLI/profile/remote suites, lint, then real `cxr --help`,
  `cxr run --help`, and `cxr remote run --help` subprocess probes.

## Non-goals

- Manual beam overrides on run commands.
- Changes to checkpoint identity, sweep physics, scheduler policy, or profile
  range semantics.
- Recompute command selection (`rebrem`/`reline --all`) unless shared removal
  is required by the exact TODO wording and confirmed during implementation.

## Integration

Land before `feature/cli-app-suite` and `feature/material-command-tree`; both
touch root command inventory and generated CLI artifacts. Rebase those branches
after this branch lands. `feature/checkpoint-prune` is otherwise independent.
