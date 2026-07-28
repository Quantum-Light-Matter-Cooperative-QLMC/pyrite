# TODO — feature/cli-profile-rework

CLI rework: `line-grid` → `energy-grid`, `cxr sweep`, `cxr profile`, remote
integration. Two colliding "profile" concepts (`SweepProfile` full/survey
fidelity policies vs catalog `[profiles.*]` scan defaults) get disentangled;
catalog keeps the word "profile", full/survey become `--fidelity`. Schema
inverts so profiles reference materials (campaign model) with per-material
overrides scoped inside each profile; derived line-grid bounds move to a
shared per-material store with provenance so profile edits can never delete
them.

Authoritative plan with decisions and full subitem→phase mapping:
[`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md).

## Phase 0 — independent bugfixes (no schema change)

1. Rework pending `cxr sweep set` diff before commit: drop line-grid pruning
   (`_reconcile_line_grids` deletes expensive derived rows) and
   copy-on-create grid duplication; drop `[profiles.test]` from
   `materials.toml`; keep `--yes`, overwrite confirmation, and test-helper
   `input=` support.
2. Fix `cxr remote pull` for survey checkpoints: pull misses
   `<material>--survey-<hash>/` identity dirs; resolve via existing identity
   regex (`profiles.py:283`) + checkpoint `meta.json`.
3. Interim fix for `sweep set --profile sub_200keV --energy 150` error
   ("no derived line grid for energy 150 keV" despite 150 in standard):
   consult standard/source profile grid rows before erroring. Root cause is
   per-profile grid copies; structural fix in Phase 1.

## Phase 1 — schema centerpiece

1. Rename `full`/`survey` reduction policies to `--fidelity full|survey`
   across `scan.py`, `_remote/cli.py`, `reline.py`, `rebrem.py`,
   `recompute_defaults.py`; legacy `--profile full|survey` aliases warn and
   forward; checkpoint dirname format unchanged.
2. Invert schema: profiles reference materials
   (`materials = [...]`, absent ⇒ all `mats_to_sim.toml` in-use materials);
   per-material overrides move to `[profiles.NAME.overrides.MAT]`;
   `[materials.*].profile` field removed. Catalog parse changes, migration,
   golden regen.
3. Shared per-material derived-grid store with
   `source = "derived" | "manual"` provenance; profiles reference beam
   energies only; deletion only via explicit `cxr energy-grid` command with
   confirmation. Derived upper bounds (long Monte Carlo runs) become
   undeletable via profile edits.
4. Docs: `docs/sweep-profiles.md` fidelity rename, CLI reference regen,
   repo map.

## Phase 2 — CLI surface (`cxr profile` group)

1. `cxr profile list|show|create|set|add|remove|delete` verb group,
   retiring `cxr sweep set --profile`; `cxr profile NAME` aliases `show`.
   - Explicit `create` required; `set` on unknown name errors with
     suggestions (kills silent-create footgun).
   - `add`/`remove` for incremental CSV edits (e.g. add 75 keV to
     sub_100keV) without overwrite prompt except on `standard`.
   - `delete NAME` requires `-y/--yes`; forbid `standard`; block while
     referenced, listing referents.
   - Material membership verbs; shell completion for profile names
     everywhere NAME appears.
2. Range syntax `start:stop:step`, stop-inclusive, mixable with CSV
   (`--energy 30,50:100:25`) on all range options.
3. `cxr energy-grid set MATERIAL --energy E --stop S` manual-bound escape
   hatch, stored `source = "manual"` (force-set per-material/per-profile
   bounds).
4. `-l/--ne-line` / `-b/--ne-brem` as profile settings: single-value grids,
   sweepable by default.
5. Every command through `cli-ui-ux` skill (design, implementation, tests).

## Phase 3 — remote integration [DONE]

1. `cxr remote submit --profile` accepts catalog profile names;
   `--fidelity full|survey` orthogonal. No auto-generated `survey` catalog
   profile — fidelity covers that role.
2. Pull selector `cxr remote pull MATERIAL@PROFILE` resolving via
   `meta.json` `dataset_identity`; on-disk names stay hash-based; multiple
   hashes for one profile → newest wins, `--hash` pins.
3. Run-time `cxr scan --all --profile NAME` flag; never writes catalog; no
   hidden session state (en-masse transient swap mostly dissolves under
   inverted schema).

Landed: `MaterialCatalog.profile_names`/`profile_memberships`/`profile_materials()`
wired from `[profiles.*]`; `scan.validate_catalog_profile()` (shared by
`scan._selected` and remote `_start_selected`) rejects an unknown `--profile`
and either silently intersects `--all` selections against a profile's
`materials` list or hard-errors an explicit non-member material.
`dataset_identity()`/`named_profile_stem()`/`high_energy_floor_*` gained a
`catalog_profile` kwarg (conditional hash key, `variant`-style — standard-profile
hashes stay bit-for-bit identical); non-standard profiles are never canonical
stems. `_remote/scripts.py` emits `--profile NAME` in queue scripts and
predicts the matching qualified stem; `_remote/lifecycle.pull()` splits
`MATERIAL@PROFILE` before the shell-token check and resolves it via
`resolve_profile_stem()` (reads each candidate directory's remote `meta.json`,
newest `mtime` wins, `--hash` pins one digest). Tests added across
`test_material_catalog.py`, `test_profiles.py`, `test_local_click_cli.py`,
`test_remote.py`, `test_remote_click.py`; full suite 1773 passed / 39 skipped.

Follow-up: `cxr profile add-material NAME --all` (seed/extend membership from
`mats_to_sim.toml`'s verified list, including implicit all-in-use profiles);
`cxr remote submit --profile NAME` now runs with no `MATERIAL`/`--all`/`-A`
when the profile has explicit membership. Also fixed a real bug found via
manual testing: `scan._resolved_run` never passed `catalog_profile` into
`material_sweep()`, so non-standard profiles ran standard's grid despite
correct identity/stem tagging. Full suite 1779 passed / 39 skipped.
