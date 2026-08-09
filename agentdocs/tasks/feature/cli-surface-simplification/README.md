# CLI surface simplification and clarification

Branch: `feature/cli-surface-simplification`

## Problem and scope

The CLI has accumulated inconsistent verbs, nested command shapes, and lifecycle
rules that are individually defensible but hard to predict together. This task
performs one contract-level sweep across four related areas while preserving
scripts through explicit aliases/deprecations where practical:

1. **Profile mutation vocabulary.** Range values use `profile set|add|remove`,
   but material membership uses `profile members set|add|remove|reset`. Decide
   whether membership should use the same primary verbs/options as other
   profile-owned parameters, and define the fate of the current `members`
   surface and compatibility aliases. Preserve the semantic distinction:
   `set` replaces supplied values; `add` unions/deduplicates; implicit
   membership means every in-use material until explicitly restricted.
2. **Completion lifecycle.** `cxr completion install` is the only current leaf.
   Decide whether to collapse it to `cxr completion`, or retain a verb pair and
   add idempotent removal. Define no-option behavior: current shell from
   `$SHELL`, every detected installed shell, or another explicit policy. Keep
   `--shell`, `--rc-file`, and `--dry-run` semantics unambiguous and safe.
3. **Checkpoint and cleanup vocabulary.** Rework local `cxr checkpoint` and the
   remote cleanup surface against the landed shared per-case CAS, manifests,
   and garbage-collection model documented in
   [`docs/checkpoint-case-store.md`](../../../../docs/checkpoint-case-store.md).
   Add safely scoped local deletion corresponding to `remote clear`; clarify
   the resources targeted by `remote prune` (obsolete checkpoint records),
   `remote prune-jobs` (terminal job directories), and `remote clear`
   (checkpoint trees). Fix or drop `cxr checkpoint recompute` under current
   named-profile conventions. Decide whether `slim`, `archive`, `restore`,
   `list`, `merge`, and `prune` remain checkpoint-owned once dataset manifests
   reference shared cases.
4. **Performance-log lifecycle and mode defaults.** Add a safe way to preview
   and remove stale local/remote performance logs; decide when completed logs
   auto-pull. Audit every reason `--chunk-minutes 0`, one material, or another
   constraint is required for `--perf`, `--perf-reps`, and `--nsys`; auto-set
   an unambiguous required value, otherwise accept the broader case or emit a
   precise warning/error. `-p/--perf` already implies the landed default
   `--no-cache`; retain and document that behavior. Decide whether catalog
   profiles may opt into performance mode persistently, and how an explicit
   command-line choice overrides that marker.

Out of scope: changing the landed shared-CAS storage format without a proven
command-contract need; compute-kernel optimization owned by
`feature/compute-performance-optimization`; broad unrelated command renames.

## Implementation path and likely owners

- `src/cxr_mc/cli/profile.py`, `src/cxr_mc/data/materials.toml`/catalog profile
  serialization, and profile CLI contract tests: profile mutation and optional
  persistent performance marker.
- `src/cxr_mc/cli/completion.py`, `tests/test_cli_completion.py`: completion
  install/remove/detection policy, exact target preview, idempotence, and
  compatibility dispatch.
- `src/cxr_mc/cli/checkpoint.py`, `src/cxr_mc/recompute_defaults.py`,
  `src/cxr_mc/{run,_checkpoint_store,archive,prune,rebrem,reline,slim}.py`, and
  `tests/test_checkpoint_cli.py`: checkpoint ownership, local cleanup,
  recompute disposition, shared-case reachability, and compatibility aliases.
- `src/cxr_mc/_remote/cli.py`, `_remote/lifecycle.py`, `_remote/scripts.py`,
  `tests/remote/test_remote.py`: cleanup nouns, performance-log pull/removal, mode
  validation, and automatic defaulting.
- `src/cxr_mc/scan.py`: only if profile-owned performance selection or local
  performance-log lifecycle needs shared resolution; do not alter measurement
  workload semantics without recorded evidence.
- `docs/cli-reference.md` and `tests/data/cli_contract.json`: regenerate after
  every accepted public-surface change.

Sequence after dispatch:

- [x] Inventory current commands, aliases, selectors, defaults, precedence,
      prompts, streams, exit codes, destructive targets, and tests for all four
      areas. Record a proposed old -> new compatibility table before code.
- [x] Lock primary nouns/verbs and migration policy. Explicitly resolve the
      open decisions below; record checkpoint resource ownership before
      changing shared local/remote cleanup paths.
- [x] Implement profile-mutation changes with replacement/union/implicit-all
      behavior frozen by regression tests.
- [x] Implement completion lifecycle changes with dry-run, repeated install,
      repeated removal, unknown-shell, explicit-shell, and custom-rc tests.
- [x] Lock checkpoint dataset-versus-case ownership and reachability semantics.
      Define local and remote clear/prune targets, shared-case garbage
      collection, live-job/reservation protection, and exact preview/confirmation
      behavior before implementing deletion.
- [x] Fix `cxr checkpoint recompute` through current named-profile resolution or
      remove/migrate it with a compatibility path. Reassess the remaining
      checkpoint subcommands against manifest and shared-CAS ownership.
- [x] Implement performance-log list/pull/prune flow. Preview exact destructive
      targets; revalidate before deletion; keep live/incomplete jobs fail-closed.
- [x] Audit performance-mode restrictions using representative remote command
      previews and existing telemetry provenance. Remove, warn, error, or
      auto-set each restriction with a documented reason.
- [x] Reconcile local and remote checkpoint cleanup through shared selectors and
      safety invariants where practical. Keep one implementation owner per path.
- [x] Regenerate CLI docs/contracts; run focused CLI tests, real help/dry-run
      probes, then full repository verification.

## Implementation evidence (2026-08-01)

- Profile membership, completion lifecycle, checkpoint clear/aliases/recompute,
  and local/remote performance lifecycle landed as task-local checkpoint commits.
- Performance checks: 59 local tests; 407 remote tests; combined focused suite
  418 passed. Ruff and Pyright passed before documentation regeneration.
- Generated `docs/cli-reference.md` and `tests/data/cli_contract.json`; updated
  repository ownership map. Generator checks and runtime probes passed.
- `cxr-dev verify`: lint/typecheck and 2,289 tests passed, 39 skipped; its sole
  sandbox failure was the known forkserver Unix-socket restriction. The failed
  CPU end-to-end test passed separately outside the sandbox (1 passed).

## Decisions

### Locked command contract (2026-08-01)

Compatibility paths remain callable because repository automation still uses
them. Each deprecated invocation emits exactly one diagnostic on stderr; result
stdout, side effects, and exit status remain those of the canonical handler.
Removal is not scheduled: delete a compatibility path only after repository
callers and operator scripts have migrated.

| Existing path | Canonical path | Compatibility policy |
|---|---|---|
| `profile members set NAME MATERIAL...` | `profile set NAME --materials MATERIAL,...` | Keep hidden/nested; warn. |
| `profile members add NAME MATERIAL...` | `profile add NAME --materials MATERIAL,...` | Keep hidden/nested; warn. |
| `profile members remove NAME MATERIAL...` | `profile remove NAME --materials MATERIAL,...` | Keep hidden/nested; warn. |
| `profile members reset NAME` | `profile set NAME --all-materials` | Keep hidden/nested; warn. |
| `profile add-material` / `remove-material` | matching `profile add` / `remove --materials` | Keep hidden; update warning target. |
| `profile analyze NAME` | `performance analyze NAME` | Keep hidden; warn. |
| `remote profile pull NAME` | `remote performance pull NAME` | Keep hidden; warn. |
| top-level `slim`, `rebrem`, `reline`, `archive`, `restore`, `archives`, `union`, `prune` | matching `checkpoint ...` path | Keep hidden; warn. |

Primary profile membership uses comma-separated `--materials`, matching
`profile create`. `set` replaces membership; `add` unions in catalog order and
reports duplicates; `remove` subtracts and reports non-members;
`set --all-materials` removes the explicit key and restores implicit all-in-use
membership. Membership and range options may be combined atomically.

Completion retains `completion install|remove`. With no `--shell`, only
`$SHELL` is targeted. Both commands preserve `--shell`, `--rc-file`, and
`--dry-run`; managed markers make current installs idempotent, and removal also
recognizes the historical comment-plus-source-line installation.

Checkpoint datasets and shared per-material CAS blobs are separate resources.
Active and archived manifests are CAS reachability roots. Keep grouped
`slim`, `archive`, `restore`, `list`, `merge`, `prune`, and `recompute`.
`checkpoint clear MATERIAL...|--profile NAME|--all [--yes]` previews selected
datasets and CAS blobs made unreachable by deleting them. Before mutation it
re-reads every retained active/archive manifest; malformed or changed
manifests abort. Remote deletion additionally revalidates jobs and reservations;
unknown remote state blocks.

Retain and repair `checkpoint recompute`. Resolve profile/dataset identity from
manifest metadata; legacy or ambiguous stems require explicit `--profile`.
Explicit grids/electron options override profile defaults. Rewrite component
content keys, CAS blobs, and manifests atomically.

Performance lifecycle is `performance list|analyze|prune` locally and
`remote performance list|pull|prune` remotely. Explicit user selection defines
stale logs; prune previews by default, and live/unknown remote state blocks.
An attached successful performance run auto-pulls performance artifacts unless
`--no-pull`; headless submission prints the explicit pull command.

No persistent profile performance marker. Explicit `--perf` retains implied
`--no-cache`; explicit `--recompute` overrides that default. Omitted
`--chunk-minutes` becomes `0` for `--perf-reps >1`; explicit nonzero conflicts.
Remote `--nsys` implies performance mode, one repetition, and zero chunking,
but requires one explicitly selected material. Local `--nsys` retains
full-profile support. Experimental chunk controls remain performance-only.

## Delegation slices and required skills

Lead/integration tier: cross-command public contract, destructive operations,
and overlap with two active branches require one owner to lock design and
sequence slices. No worker may edit checkpoint-owned paths before coordination.

1. Profile vocabulary/schema: `implement-task`; skills `cli-ui-ux`,
   `regression-testing`, `scientific-library` if catalog types change.
2. Completion lifecycle: bounded `implement-task-lite` after policy decision;
   skills `cli-ui-ux`, `regression-testing`.
3. Performance-log lifecycle/default audit: `implement-task`; skills
   `cli-ui-ux`, `performance`, `remote-gpu-jobs`, `regression-testing`.
4. Checkpoint command/reachability rework: `lead-task` owns the design;
   implementation may split only after resource ownership and compatibility are
   locked. Skills `cli-ui-ux`, `run-cxr-mc`, `regression-testing`,
   `documentation-maintenance`; add `remote-gpu-jobs` for remote deletion.
5. Docs/contracts: `documentation-maintenance`, after command decisions land.

## Acceptance checks

- Written old -> new command table covers compatibility aliases, warnings,
  removal timeline, selector precedence, side effects, stdout/stderr, and exit
  codes; user approves decisions before implementation.
- One predictable profile mutation vocabulary; `set` replacement, `add`
  union/deduplication, and implicit all-material membership tested.
- Completion install/removal policy is idempotent, dry-runnable, shell-safe,
  and never edits an inferred file without preview/explicit command authority.
- Performance logs can be listed, pulled, and preview-pruned locally/remotely;
  live or uncertain remote state blocks deletion.
- Local and remote checkpoint cleanup preview exact dataset and shared-case
  effects, preserve cases reachable from retained manifests, revalidate remote
  jobs/reservations before mutation, and default to non-destructive preview.
- `cxr checkpoint recompute` is either profile-correct or removed/migrated with
  frozen compatibility behavior; remaining checkpoint verbs name the resource
  they mutate.
- Every `--perf`/`--perf-reps`/`--nsys` material/chunk/cache constraint has a
  tested behavior and help-text rationale. `-p/--perf` continues to imply
  `--no-cache` unless explicitly superseded by the landed cache contract.
- `docs/cli-reference.md` and `tests/data/cli_contract.json` regenerated;
  focused completion/profile/checkpoint/remote tests and `cxr-dev verify` pass.
