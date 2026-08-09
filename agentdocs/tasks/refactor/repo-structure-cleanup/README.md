# Repository and source-package structure cleanup

## Problem and scope

Structural survey of the tree (2026-08-07) on top of the accepted
[`docs/package-structure-rfc.md`](../../../../docs/package-structure-rfc.md).
That RFC's P4/P5 landed, P1/P2 largely landed, P3 is sequenced with the CLI
verb collapse. This task collects what the RFC does **not** cover, plus concrete
shapes for the layout questions it left open.

One correctness defect (S1) and seven layout defects. Every slice below is a
pure move or deletion except S1, which changes path resolution semantics for an
installed distribution.

In scope: where code and workspace state *live*. Out of scope: CLI surface
(`docs/cli-redesign-rfc.md`), artifact/data model
(`docs/cli-artifact-model-rfc.md`), any physics or numerical behavior, and the
`tests/` layout rule (landed separately on 2026-08-07 via
`refactor/test-refactoring`, which reduced `tests/*.py` from 20 loose modules
to 1).

Adjacent owners, do not duplicate:

- `feature/uv-workspace-split` — member/dependency boundaries. S1 and S2 change
  *what is inside* `cxr_mc`; the workspace split decides *how it is
  distributed*. S1 should land first so the workspace design is not built on a
  package that cannot run outside its checkout.
- `feature/cli-verb-collapse-module-fold` — RFC P3 recompute/prune module fold.
- `refactor/energy-grid-rename` — RFC P2; S8's empty-directory removal is its
  leftover.

---

## S1 — Installed package resolves the developer checkout (defect)

Shipped code computes the repo root at import time and reads files that the
wheel does not contain.

Evidence:

- 14 `Path(__file__).resolve().parents[2]` sites in `src/cxr_mc/`:
  `run.py:43` (default checkpoint dir), `archive.py:43`,
  `checkpoint_cleanup.py:26`, `analyze.py:57`, `viewer.py:50`,
  `check.py:33` (`notebooks/validation_defaults.json`), `check.py:201`
  (the `checks/` directory), `_dev.py:44`, `_acp.py:12`, `cli/_completion.py`.
- `analyze.py:51` — `NOTEBOOK = "notebooks/analysis_app.py"`; `viewer.py` and
  `check.py` target `notebooks/trace_app.py` / `notebooks/validation_app.py`.
- `pyproject.toml` `[tool.hatch.build.targets.wheel] packages = ["src/cxr_mc"]`.

So an installed `cxr` has `app analysis|viewer|validation` targets that do not
exist, and resolves checkpoint/state paths into `site-packages/../..`.
`scripts/package_smoke.py` builds, installs, and imports, but never invokes a
command from a cwd outside the checkout, so this passes today.

Path:

- Add `cxr_mc/paths.py`: `data_dir()` (packaged; the existing `DATA_DIR`),
  `workspace_root()` (explicit argument → `CXR_HOME` → config store → cwd;
  never `__file__`), `state_dir()` (platform config/state dir, matching the
  Click platform dir the config store already uses). Replace all 14 anchors.
- Move the marimo apps into the distribution: `notebooks/{analysis,scan,trace,validation}_app.py`,
  `analysis_ui/` (4490 LOC), `_design.py`, `_widgets.py`, `layouts/`,
  `validation_defaults.json` → `cxr_mc/apps/`, added to packaged data.
  **Decided 2026-08-09 (user): `notebooks/` is retired on move, not kept as
  launch aliases.** The marimo working-directory and `watcher_on_save` workflow
  must still survive (`AGENTS.md` still requires `uv run marimo check <app.py>`
  after edits) — against `src/cxr_mc/apps/*_app.py` as the new target. 30 files
  outside `tasks/` reference `notebooks/`; `pyproject.toml:198`'s `B018` ruff
  per-file ignore must follow the apps to their new path or every bare-last-line
  marimo cell starts erroring.
- Anything `check.py` loads from `checks/` at runtime moves into the package;
  `checks/` becomes dev-only.
- Extend `scripts/package_smoke.py`: from a temp cwd outside the checkout, run
  `cxr app analysis launch --smoke`, `cxr app viewer launch --smoke`,
  `cxr run --help`, and assert the resolved checkpoint root is not inside
  `site-packages`.

Risk: highest of the slices. Default checkpoint-root resolution is
user-visible; a checkout-local `cwd` must keep resolving to the same
`./checkpoints` it does today, or every existing dataset appears to vanish.
Needs an explicit compatibility check, not just green tests.

## S2 — `src/cxr_mc/` top-level module sprawl

32 loose modules at the package root with no predictive rule. Retires TODO
Inbox #5. Proposed grouping:

| subpackage | absorbs |
|---|---|
| `checkpoints/` | `_checkpoint_io`, `_checkpoint_store`, `archive`, `slim`, `checkpoint_cleanup`, `recompute`, `recompute_defaults`, `campaign_lock` |
| `campaign/` | `config`, `profiles`, `sweep`, `transverse`, `longitudinal`, `beam_metrics` |
| `runs/` | `run`, `scan`, `blaze` |
| `apps/` | `analyze`, `check`, `viewer`, `export`, `_acp`, `_compile_nb` + the marimo files from S1 |
| `validation/` | `validation_oracles`, `validation_background`, `_zhai`, `check_config` |
| `perf/` | `performance_analysis`, `performance_profile` |

`remote.py` (242 lines) is a thin facade over `_remote/` (5.8k LOC): fold it in
and drop the underscore — `remote/` public, `remote/_cli.py` internal.

Constraint: `cxr_mc/__init__.py` exports stay byte-identical (frozen-export
guards), and no public domain import path moves without a re-export. RFC §4
non-goals still hold for `materials/`, `montecarlo/`, `results/`, `plots/`,
`detectors/` *contents*; S6 only regroups `plots/` internally.

Sequence after S1 so the app modules move exactly once.

## S3 — Delete the `cli/` alias shims — **done 2026-08-09**

`cli/{profile,sweep,checkpoint,energy_grid,material,app,performance,completion,backend_setup}.py`
were 9 three-line `sys.modules[__name__] = _implementation` swaps left from RFC
P1 (`feature/cli-command-home`).

They are internal wiring, not the `cxr` command surface, so the D7 two-minor
deprecation window in `docs/cli-deprecations.md` does not apply. The
`sys.modules` swap also defeats `ty`, ruff import resolution, and IDE
go-to-definition — `feature/cli-command-home` already had to repoint two
modules at `cli.commands.*` to clear ty errors caused by them.

Resolves RFC §6 open question 1 in favor of `cli/commands/`.

Outcome, correcting this section's original survey:

- 10 shims, not 9. `cli/beam.py` postdates the survey (added by
  `feature/named-beam-objects`). `rg -n "sys.modules\[__name__\]" src/` now
  returns nothing; that grep is the check for regrowth.
- "Zero references" was wrong. Six lazy-import target strings inside `cli/`
  pointed at the shim paths and had to be repointed at `cli.commands.*` first:
  `cli/commands/completion.py:193-194`, `cli/commands/checkpoint.py:11`,
  `cli/commands/app.py:14-16`. Because `cli/__init__` dispatches through
  `LazyGroup`, a stale target string does not fail at import or on the parent
  `--help` — only on the leaf invocation. Later slices that move CLI modules
  must grep dotted strings, not just `import` statements, and must exercise a
  leaf subcommand.
- Seven test modules imported the shim paths and were repointed.
- `docs/repo_map.md`'s alias note removed. Verified with `lint`, `typecheck`,
  `test-suite cli` (1123 passed), and `--help` on each affected group plus one
  leaf each.

## S4 — Two files misclassified as CLI wiring

- `cli/commands/profile.py` — 1461 lines / 56 KB, largest module in the CLI
  layer. Most is profile *domain* logic (resolution, membership, per-material
  overrides), not Click wiring. Push domain into the `campaign/profiles`
  owner from S2; leave the command thin, per the RFC P1 rule it nominally
  already follows.
- `cli/_dashboard.py` — 1106 lines / 42 KB. A live TUI, not dispatcher
  plumbing. Promote to `cli/dashboard/`, split render / state / poll.
  Coordinate with `feature/local-run-dashboard`.

## S5 — Monte Carlo mega-modules

`montecarlo/spectrum.py` 2026, `transport.py` 1788, `runner.py` 1709 lines.
Retires TODO Inbox #7, which asks only "evaluate"; the missing piece is a split
axis:

- `runner/` → `scheduling.py`, `chunking.py`, `pool.py`, `oom.py` (the
  OOM-retry work already wants its own home).
- `spectrum/` → `lines.py`, `brem.py`, `coherent.py`, with each
  `*_jit_kernel.py` filed next to its caller instead of six flat siblings.
- `transport.py` split axis to be determined during the slice; do not split it
  blind.

Physics-gated by `AGENTS.md`: pure move, so golden artifacts and every
`Validation: <id>` marker and ledger row must come out unchanged — that
invariance *is* the acceptance evidence. Sequence last; coordinate with
`feature/compute-performance-optimization` and
`feature/coherent-line-batching`, which are both live in these files.

## S6 — `plots/` backend interleave

12 flat modules with two parallel backend families: `altair_spectra`/`spectra`,
`altair_trajectories`/`trajectories`/`plotly_trajectories`,
`altair_detectors`/`detectors`, `altair_sweeps`/`sweeps`. Regroup as
`plots/{altair,mpl,plotly}/` over the shared `_common`/`_style`/`_frames`, so
"which backend renders X" is answerable from the tree. Public import paths
re-export.

## S7 — Working-tree hygiene — **done 2026-08-09**

- ~~`worktrees/` is **3.6 GB inside the repo directory**~~ — **done 2026-08-07.**
  Relocated to `../cxr-mc-worktrees/` with `git worktree move`; the empty
  in-repo `worktrees/` directory is removed. No skill edits were required:
  `triage`, `dispatch-task`, and `repo-cleanup` all discover worktrees through
  `git worktree list --porcelain` rather than a hardcoded path, and
  `.worktreeinclude` lists bare filenames. Task docs repointed. The
  `.gitignore` `worktrees/` entry is kept as a guard against re-creation.
- ~~Same argument, lower stakes: `performance-profiles/` (69 MB),
  `docs/_build` (41 MB), `checkpoints/` (93 MB).~~ — **no action, 2026-08-09.**
  The survey numbers no longer hold: in `/home/alex/dev/cxr-mc`,
  `performance-profiles/` and `docs/_build/` do not exist and `checkpoints/` is
  88 MB. All three are already gitignored, are regenerable, and are per-checkout
  by nature, so relocation buys nothing. The 4.5 GB checkout is 4.4 GB `.venv`
  (plus 25 MB `.serena`, 12 MB `.git`) — a tool-managed environment that must
  stay checkout-local. The `worktrees/` case was different only because 3.6 GB
  of *other checkouts* lived under a path the repo also walks.
- ~~`checkpoints/**/cxr.lock.json` is tracked for 5 datasets~~ — **untracked
  2026-08-09.** Not deliberate provenance: `git log --diff-filter=A` shows each
  of the 5 swept into an unrelated commit (`9e726d2` materials/trace_app,
  `be1bb63` JIT kernel rework, `f3bd000` docs). A tracked lock is actively
  wrong, not merely undocumented — `campaign_lock` locks are artifact-GC roots
  (`energy_grid/gc.py:186`), so a fresh clone inherits GC roots for `.pkl`
  payloads that `.gitignore` withheld. `git rm --cached` on the 5,
  `checkpoints/{*,**}/*.lock.json` added to `.gitignore` (the legacy sibling
  `<stem>.lock.json` form from `campaign_lock.py:24` needs both patterns), and
  the tracking rule written into `docs/checkpoint-case-store.md`. Files stay on
  disk; no test or script reads the repo-local `checkpoints/` tree.

## S8 — Leftovers and generated-doc drift

- ~~`src/cxr_mc/line_grid/` is empty (only `__pycache__`) after the P2 rename —
  delete.~~ **Done** — already absent by 2026-08-09; no action was needed.
- `docs/repo_map.md` is 40 KB of hand-maintained prose while
  `cxr-dev repo-map` generates only the top-level inventory. The
  dependency-layer DAG is the part that rots, and it is exactly the part a
  machine can derive. Generate the DAG section from an AST import scan
  (stdlib `ast`, no external indexer) into a marked region, keep the prose
  blurbs hand-written below it, and add a `--check` mode to the existing
  generator wired into `cxr-dev verify` alongside the CLI-reference freeze.
  Build the graph with `grimp` (a lint-group dependency), not a hand-rolled
  `ast` walk: relative imports, `from . import x`, submodule-vs-attribute
  disambiguation, and `__init__` re-exports are exactly where a hand-rolled
  walk gets the layering wrong. Stronger option, and the recommended one:
  encode the layer contract in `import-linter` (`layers`, `forbidden`,
  `independence`) so `cxr-dev verify` *enforces* the DAG instead of only
  reporting it, and generate the prose from the same graph. Neither an
  external code index nor an LSP server belongs in this path — it must be a
  pinned, offline, daemon-free dependency. Known blind spot for any static
  graph: the deliberate lazy seams (`cli/__init__` `LazyGroup`, the
  `scan`/`blaze` `__getattr__` re-exports, `materials`/`detectors`
  `__getattr__`, `montecarlo/_backend.py` `import_module` backend probing)
  must be annotated by hand or declared as contract exceptions.
- Three homes for executable non-app code: `checks/` (492 KB one-off physics
  scripts, read at runtime by `check.py:201`), `scripts/` (maintenance and
  generators), and `_dev.py` (549 lines) with `scripts/testing.py` overlapping
  it. After S1 removes the runtime dependency, give `checks/` a README mapping
  each script to its ledger id, and make `scripts/` modules importable by
  `_dev` rather than a parallel entry path — `cxr-dev` stays the sole runner.

---

## Checklist

- [ ] S1 `paths.py` + apps into the distribution + package-smoke assertions
- [ ] S2 subpackage grouping + `remote` fold, exports unchanged
- [x] S3 delete 10 `cli/` alias shims (9 from original task + `beam.py` postdating named-beam-objects)
- [ ] S4 `profile.py` domain extraction; `cli/dashboard/` promotion
- [ ] S5 `runner/` and `spectrum/` splits (golden + ledger invariant)
- [ ] S6 `plots/{altair,mpl,plotly}/`
- [x] S7 worktrees relocation (2026-08-07) + checkpoint lockfile tracking rule
      documented and the 5 stray locks untracked (2026-08-09); the
      `performance-profiles`/`docs/_build`/`checkpoints` size bullet closed as
      no-action
- [x] S8 empty-dir removal (`src/cxr_mc/line_grid/` confirmed deleted)
- [ ] S8 repo-map DAG generator, `checks/`/`scripts/` ownership
- [ ] `docs/repo_map.md` regenerated/updated per source-touching slice
- [ ] `docs/package-structure-rfc.md` amended with S1–S8 as P6+ (or this doc
      cited from it) so the RFC stays the durable record

### S1 checkpoint progress

- 2026-08-09: added `cxr_mc.paths` and migrated checkpoint roots, completion,
  developer tooling, ACP state, and analysis/viewer persisted defaults away
  from checkout-derived paths. `workspace.root` now follows explicit argument
  → `CXR_HOME` → config store → cwd; checkout-local cwd still resolves to the
  existing `./checkpoints`. Packaged app/default/export resources remain for
  the next S1 checkpoint, so S1 stays unchecked.

## Sequencing

1. ~~**S3 + S8 empty dir**~~ — done 2026-08-09.
2. ~~**S7 remainder**~~ — done 2026-08-09.
3. **S1** — next up; the only correctness defect. Before S2 so app modules move once,
   and before `feature/uv-workspace-split` designs members.
4. **S2**, then **S4**, **S6**, **S8 remainder**.
5. **S5** last: physics-gated, and two live branches are editing those files.

## Non-goals

- No CLI surface change (spelling, help, exit codes, output envelopes), with one
  carve-out forced by S1's `notebooks/` retirement: ~10 help strings name the
  app file by path ("Launch notebooks/analysis_app.py with marimo run or edit"),
  frozen in `docs/cli-reference.md` and `tests/data/cli_contract.json`. Those
  paths are already wrong for an installed wheel, so they are part of the S1
  defect, not collateral. Resolution pending — see S1.
- No physics or numerical behavior change; no ledger row edits.
- No `tests/` layout work — owned by `refactor/test-refactoring`.
- No public import-path removal; frozen-export guards stay green throughout.
- No artifact-store or checkpoint-format change.

## Acceptance evidence

- Per slice: `cxr-dev lint`, `typecheck`, and the narrowest relevant
  `test-suite` green; `docs/cli-reference.md` freeze test unchanged.
- S1: `scripts/package_smoke.py` exercises commands from an external cwd; a
  checkout-local `cxr run` still resolves `./checkpoints` to the same path as
  before the change.
- S2/S4/S6: `cxr_mc.__all__` and the export-freeze guards byte-identical.
- S5: golden artifacts bit-identical, `Validation:` marker set and ledger rows
  unchanged (`physics-ledger-auditor` clean).
- Full `cxr-dev verify` before the last slice of each group lands on `main`.

## Delegation

Each S-slice is an independently ownable branch off this task. Suggested
tiers: S3/S8-empty-dir → `implement-task-lite`; S2/S4/S6/S7 →
`implement-task`; S1 and S5 → `lead-task` (S1 for user-visible path semantics,
S5 for the physics gate).
