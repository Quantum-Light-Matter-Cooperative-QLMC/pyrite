# Checkpoint command surface rework

Branch: `feature/checkpoint-command-rework`
TODO scope: from `>user<` backlog prose, items 1/3/4 (split `cxr clear`, drop
`cxr checkpoint recompute`, rework `cxr checkpoint` broadly).

## Problem

Three related complaints about the checkpoint/clear command surface, bundled
because they're the same design space and any fix to one reshapes the others:

1. **`cxr clear` is remote-only and monolithic.** The only `clear` command is
   `cxr remote clear` (`clear_command`, `src/cxr_mc/_remote/cli.py:1121` →
   `clear_remote`, `src/cxr_mc/_remote/lifecycle.py:39`). It takes
   material(s), `--all`, or `--profile` and deletes the matching checkpoint
   stems on the box. There is no local equivalent and no per-scope split
   (e.g. `cxr profile clear <profile>` to drop just that profile's stems).
   User asks whether/how to split this into scope-specific commands, and how
   that interacts with shared checkpoints once cross-profile case reuse lands
   (deleting a profile's stem may be deleting cases other profiles still
   want).
2. **`cxr checkpoint recompute` is confusing under current profile
   conventions.** `cxr checkpoint recompute {brem,line}` (`recompute_command`,
   `src/cxr_mc/cli/checkpoint.py:47`) dispatches to `rebrem`/`reline`, which
   both resolve settings/sweeps through `recompute_defaults.py`. That module
   hardcodes a binary `PROFILE_NAMES = ("full", "survey")` fidelity scheme
   (`recompute_defaults.py:10`) with a `TypeError`-catching fallback path for
   when `default_settings`/`material_sweep` don't yet accept `profile=` — a
   sign this predates the richer named-`catalog_profile` scheme now used
   elsewhere (`profiles.py`). User wants this dropped as extraneous/confusing
   unless it can be reconciled with current profile conventions.
3. **`cxr checkpoint` overall needs a rework for shared checkpoints.**
   The group (`command`, `src/cxr_mc/cli/checkpoint.py:58`) exposes
   `slim`/`recompute`/`archive`/`restore`/`list`/`merge`/`prune` against
   single-stem, single-owner checkpoints. The landed shared per-material case
   store ([`docs/checkpoint-case-store.md`](../../../docs/checkpoint-case-store.md))
   makes checkpoint provenance/ownership/naming for these commands
   contested: whose checkpoint is `cxr checkpoint archive` archiving if cases
   are shared across profiles? User explicitly asks for advice on reworking
   the group and migrating any functionality worth keeping into other
   commands.

## Implementation path

This is design-first, not a mechanical slice — items 2 and 3 both explicitly
ask "please advise."

1. **Sequencing.** Items 1 and 3 are unblocked by the landed shared-checkpoint
   storage model in
   [`docs/checkpoint-case-store.md`](../../../docs/checkpoint-case-store.md).
   Design against its material CAS, per-profile `cases.json` manifests, and
   retained component checkpoints, plus the stem scheme from
   `feature/checkpoint-variant-naming`.
2. **`cxr checkpoint recompute`:** independent of the above — this is a
   profile-convention mismatch bug/design question. Read `_recompute_options`
   (`src/cxr_mc/_remote/cli.py:507`) and `recompute_defaults.py` fully, then
   decide: (a) delete `recompute_defaults.py`'s full/survey fallback and wire
   `rebrem`/`reline` through the current `catalog_profile` resolution path
   directly, or (b) drop the `cxr checkpoint recompute` subgroup entirely and
   fold `brem`/`line` recompute into `cxr run --recompute-brem-only` /
   `--recompute-line-only`-style flags on the main run path (check `slim.py`'s
   `brem_only`/`line_only` flags, `src/cxr_mc/slim.py:192`, for a naming
   precedent already in the codebase). Record the decision here before
   implementing.
3. **`cxr clear` split:** using the landed shared-checkpoint provenance,
   propose the local/remote command shape (e.g. `cxr checkpoint clear
   [materials|--profile|--all]` as a local counterpart to `cxr remote clear`,
   sharing the same guard logic against live-job/reservation protection that
   `clear_remote` already implements) and how a `--profile`-scoped clear
   should behave if the profile's cases are still referenced by another
   profile's stem.

## Decisions / open questions

- Does `cxr checkpoint recompute` get fixed or dropped? (open — see above)
- Does `cxr clear` gain a local counterpart, or does splitting mean adding
  profile/material-scoped subcommands under the existing remote `clear`?
- What happens to a shared case on `clear --profile X` once cross-profile
  reuse exists — refuse, warn, or clear only the profile-exclusive subset?
- Should any `cxr checkpoint` subcommands move elsewhere (e.g. under
  `cxr profile`) once ownership is profile-scoped rather than stem-scoped?

## Delegation

Design/investigation first (`lead-task` tier — touches the durable checkpoint
provenance model in
[`docs/checkpoint-case-store.md`](../../../docs/checkpoint-case-store.md) and
`feature/checkpoint-variant-naming`). `cxr checkpoint recompute`'s
profile-convention fix can split off as its own `implement-task` slice once
(a)/(b) above is decided, independent of the clear/rework sequencing.
Relevant skills: `cli-ui-ux`, `run-cxr-mc`.

## Acceptance

- Design phase: a written proposal here covering the three points above,
  cross-checked against the landed per-case store and
  `feature/checkpoint-variant-naming`'s stem scheme.
- Implementation phase (separate dispatch once designed):
  `docs/cli-reference.md` regenerated for any changed/removed commands;
  `uv run cxr-dev verify` passes; existing `tests/test_checkpoint_cli.py`
  and `tests/test_remote.py` clear/recompute tests updated to match.
