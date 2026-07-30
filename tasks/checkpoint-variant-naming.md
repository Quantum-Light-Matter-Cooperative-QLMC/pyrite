# Checkpoint variant naming + analyze-menu visibility

## Problem and scope

`docs/scan-config-cli-plan.md` (status: planned, decisions locked 2026-07-26,
owner: Alex) specifies checkpoint stems as `checkpoints/<material>@<set>.pkl` /
`<material>@<preset>.pkl`, with `analyze.material_menu` globbing
`<material>@*.pkl` (doc lines 7-9, 97-108, 127-129). What actually shipped
(`src/cxr_mc/profiles.py::variant_stem`, `profiles.py:304-317`) is a different
scheme: any non-canonical run gets `<material>--<fidelity>-<parameter_sha256[:12]>`
(a hash suffix, no profile name in the stem at all). Non-canonical means: any
non-`standard` `catalog_profile`, `--quick` combined with other overrides, a
high-energy-floored grid, `coherent_emission=True`, or custom beam params --
see the `canonical_full` condition in `scan._resolved_run` (`scan.py:512-607`).

`analyze.material_menu` (`analyze.py:65-98`) only ever recognizes an exact
`<material>.pkl` / `<material>/` stem match against `CATALOG.material_keys` --
its docstring says this "deliberately excludes ... unknown pickle stems." Net
effect: **every checkpoint from a named catalog_profile run is permanently
invisible to `cxr app analysis`**, with no UI path to select it, regardless of
which naming scheme is "correct."

Reproduced today (2026-07-29): local `checkpoints/` has `hbn--full-<hash>` x3
and `hopg--full-<hash>` x3 (profiles `sub_100keV` and
`hopg_hbn_microtrain_200fs`, the new microtrain longitudinal-bunch profile
from [[longitudinal-bunch-profiles]]-adjacent testing) -- all disabled in the
analysis app menu.

### Related, sharper bug found during investigation

`profiles.identity_from_stem` (`profiles.py:396-421`, used today only by
`slim.py::_grid_from_stem`) recovers identity from a hash stem by
**recomputing** `named_profile_identity` under each candidate `catalog_profile`
and matching digests live against the *current* catalog. This silently fails
once a named profile's definition changes after the run that produced the
stem: confirmed 2 of the 6 local stems (both `hopg_hbn_microtrain_200fs`, two
runs one minute apart, profile edited in between) no longer resolve via
`identity_from_stem`, even though each has a `meta.json` sidecar
(`_checkpoint_store.manifest_path`, `_checkpoint_store.py:36-37`) with the
correct `dataset_identity` recorded at write time (`run.py:217-257`).
`archive._dataset_identity` (`archive.py:158-169`) and
`_remote/lifecycle.py:1171-1224` already read that sidecar directly instead of
recomputing -- that's the robust, already-established pattern in this repo.
**Any fix here should read the sidecar, not re-derive via `identity_from_stem`.**

### Same-day hotfix (already landed directly on `main`, not this branch)

A minimal unblock shipped same-day: a "Profile" selector in
`notebooks/analysis_app.py` (mirrors the existing Face selector pattern,
`analyze.face_menu` / `analyze.py:170-187`), backed by a new
`analyze.profile_menu(material, checkpoint_dir)` that reads each variant
stem's `meta.json` `dataset_identity` directly (no re-hash) to decide which
stems belong to the selected material and what to label them. This task is
the **real fix**: reconcile the naming scheme with the locked plan (or
formally amend the plan to the shipped hash scheme) and decide how much of the
plan's remaining CLI surface (`cxr config`, `--preset`, `cxr checkpoints`) is
still wanted versus superseded by `catalog_profile`-based naming.

## Implementation path and likely owners

- `docs/scan-config-cli-plan.md` -- either (a) update to describe the
  catalog_profile/hash-stem scheme that shipped and retire the `@`-stem /
  `cxr config` design, or (b) if `@`-stems are still wanted, scope the
  migration of `variant_stem` to produce them and what happens to existing
  `--hash` checkpoints on disk (rename? dual-read?). **This is the primary
  open decision -- see below.**
- `src/cxr_mc/profiles.py` -- `variant_stem`, `named_profile_stem`,
  `identity_from_stem`, `_VARIANT_STEM_RE`: whichever naming direction is
  chosen, `identity_from_stem` should stop recomputing and instead read the
  stem's own `meta.json` sidecar (falling back to recompute only for stems
  that predate the sidecar, if any still matter).
- `src/cxr_mc/analyze.py` -- reconcile the hotfix's `profile_menu` with
  whatever `cxr checkpoints` listing (plan slice 4, doc lines 118-121) is
  decided; today's hotfix only touches the marimo app, not a standalone CLI
  listing command.
- `src/cxr_mc/_remote/lifecycle.py`, `_remote/scripts.py` -- already stem-
  and sidecar-aware for remote pull/prune; verify no assumptions collide with
  whatever naming change is chosen (`resolve_profile_stem`,
  `_list_checkpoint_dirs_command`, `_prune_checkpoint_stems_command`).
- Likely owner tier: **normal** (`implement-task`) -- multi-file but
  mechanical once the naming decision is made; escalate to `lead-task` only if
  the naming decision requires a disk-migration script for existing
  `--hash` checkpoints.

## Stepwise checklist

1. Resolve the open decision below (naming scheme: keep hash, adopt `@`-set,
   or hybrid) -- get explicit sign-off before touching code, this is a repeat
   of a decision already locked once and reversed in practice.
2. Update `docs/scan-config-cli-plan.md` status/content to match the decision
   (or mark it superseded and point to this task).
3. Fix `identity_from_stem` (or its call sites) to read `meta.json`
   `dataset_identity` directly instead of recomputing; add a regression test
   using a stem whose backing profile was edited after the run (mirrors the
   real failure found today).
4. Reconcile the hotfix's `analyze.profile_menu`/notebook wiring with the
   final decision -- if stems change shape, update the sidecar-read matching
   logic accordingly; if hash stems stay, promote the hotfix's approach to
   the "supported" implementation rather than a stopgap.
5. Decide and implement `cxr checkpoints` (plan slice 4) if still in scope,
   or explicitly drop it from the plan.
6. Update `docs/repo_map.md` / README CLI table / `docs/cli-reference.md` if
   CLI surface changes.
7. Full test suite + `cxr app analysis --smoke` for at least one canonical and
   one variant material; `scripts/dev.py verify`.

## Decisions and open questions

- **Naming scheme**: keep the shipped `<material>--<fidelity>-<hash>` (cheap,
  already deployed, already has remote/archive/prune support) and update the
  plan doc to match, OR migrate to `<material>@<profile>` as originally locked
  (more readable, matches user's original mental model, but requires a
  migration path for existing on-disk `--hash` checkpoints and touches
  `_remote/*` stem prediction). No implementation should start until this is
  picked.
- If migrating to `@`-stems: what happens to the two edited-profile stems
  that no longer even recompute correctly today -- do they need manual
  relabeling, or does the sidecar-read fix make them self-describing again
  regardless of stem text?
- Is the plan's `cxr config` / `cxr-overrides.toml` / `--preset` CLI surface
  (doc lines 34-108) still wanted, or superseded by the `catalog_profile`
  mechanism (`materials.toml` profiles) that shipped instead? These look like
  they solve the same problem two different ways.
- Should `cxr checkpoints` (plan slice 4) list *all* stems including
  non-resolvable ones (flagged "unknown"), or only sidecar-resolvable ones?

## Delegation slices and required skills

1. **Design reconciliation** (doc + decision) -- needs user sign-off, not a
   worker slice.
2. **`identity_from_stem` sidecar-read fix + regression test** -- bounded,
   `implement-task-lite` eligible; skills: `scientific-library` (owns
   `profiles.py`), `regression-testing`.
3. **Analyze/CLI consumer wiring** (`cxr checkpoints`, notebook consolidation
   with the hotfix) -- `implement-task`; skills: `cli-ui-ux`,
   `notebook-workflow`.
4. **Remote consumer verification** -- `implement-task-lite`; skill:
   `remote-gpu-jobs`.
5. **Docs sync** -- `documentation-maintenance`.

## Acceptance checks

- `docs/scan-config-cli-plan.md` accurately describes the shipped (or newly
  migrated) naming scheme; no doc/code contradiction remains.
- `identity_from_stem` (or its replacement) correctly resolves identity for a
  stem whose backing named profile was edited after the run, via sidecar
  read, with a regression test covering exactly that case.
- Every checkpoint stem from a named-profile or otherwise non-canonical run
  is browsable from `cxr app analysis` (or explicitly, intentionally listed
  as unsupported with a clear reason, if that's the decided scope).
- `scripts/dev.py test`, `lint`, `typecheck`, `verify` all pass; `marimo
  check notebooks/analysis_app.py` passes if the notebook changed further.
- `docs/repo_map.md`, README CLI table, `docs/cli-reference.md` regenerated
  if CLI surface changed.

## Resolution status (2026-07-29)

Decisions locked by user (do not re-litigate):
- **A** — migrate stems to `<material>@<label>-<digest>` (@-stems), retaining a
  12-hex digest for collision-safety; label = non-standard `catalog_profile`
  else fidelity.
- **B** — drop the `cxr config` / `--preset` / `cxr-overrides.toml` design;
  keep `catalog_profile` / `materials.toml`.

Migration path chosen: **dual-read** (non-destructive). `variant_stem` writes
@-stems; `_VARIANT_STEM_RE` + `identity_from_stem` resolve BOTH new `@` and
legacy `--<fidelity>-` stems. No rename/disk-touching migration; existing
on-disk `--hash` checkpoints keep resolving via their sidecar (authoritative)
or recompute (both stem shapes handled).

Checklist:
1. Naming scheme — DONE (Decision A).
2. Plan doc rewritten for @-stems, `cxr config`/overrides retired — DONE.
3. Sidecar-authoritative identity — LANDED in c6586c4; kept green.
4. `analyze.profile_menu`/`material_menu` promoted (sidecar-driven, @- and
   `--`-stem agnostic); `material_menu` type-narrowed. New @-stem browsability
   test added; verified live in `cxr app analysis --smoke`. — DONE.
5. `cxr checkpoints` — **DEFERRED** (recommendation recorded in plan doc):
   browsability delivered by the app; `cxr checkpoint list` already = archive
   shelf, so an active-stem listing needs a fresh cli-ui-ux naming decision
   (beyond the two locked decisions). Recommend follow-up `cxr checkpoint ls`,
   list all, flag unresolvable `unknown`.
6. `--coherent` help + `docs/cli-reference.md` + `tests/data/cli_contract.json`
   regenerated. `repo_map.md`/README unchanged (command set unchanged). — DONE.
7. Full suite (2133 pass; 3 PRE-EXISTING `attach`/`REMOTE_COMMANDS` failures
   unrelated to this task, confirmed at base 353e52f), lint clean, typecheck
   clean re: this work (6 optional-dep import errors are env-only on WSL),
   marimo check pass, smoke pass (canonical hopg + real legacy-variant hopg).

Remote consumers verified for both stem shapes: `_SHELL_TOKEN_RE` allows `@`;
`_split_profile_selector` treats a full @-stem as literal (not a query);
`_resolve_survey_stems` matches `@survey-`; `resolve_profile_stem`/prune resolve
via sidecar unchanged.

Follow-ups: `cxr checkpoint ls` (above); emission-token stem embedding is
cosmetic and owned nominally here but left to `feature/profile-emission-modes`
coordination (digest+sidecar already carry correctness).
