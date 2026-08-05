# Write next handoff to: /home/alexa/dev/cxr-mc/.remember/remember.md

## feature/cli-deprecation-substrate — worktree `worktrees/cli-rework`

Base commit `9b36337 WIP` (added `src/cxr_mc/cli/_deprecations.py` only). Everything
below is UNCOMMITTED in that worktree. Task = finish the RFC D7 deprecation harness.

### Goal (RFC D7, `docs/cli-redesign-rfc.md` L224-232)
Every renamed/retired `cxr` spelling: (a) keeps working for a published support
window, (b) emits ONE stderr warning naming its replacement, (c) appears in a
generated `docs/cli-deprecations.md`. `tests/test_cli_deprecations.py` holds
registry and live command tree in sync in both directions.

### Design as built
`src/cxr_mc/cli/_deprecations.py` owns everything:
- `SUPPORT_WINDOW_MINORS = 2`; deprecated in `0.1.0` -> removed in `0.3.0`.
- frozen `Deprecation(path, replacement, deprecated_in, remove_in, note)`;
  `DEPRECATIONS` keyed by command path MINUS the `cxr` prefix (25 rows).
- `message(path, replacement=None)` / `warn(path, replacement=None)`.
- `WARNED_META_KEY` in `ctx.meta` (shared across the whole context stack) gives
  exactly one diagnostic per invocation even when group AND leaf are deprecated.
- `invocation_path(ctx)` walks parents and drops the root, because the root's
  `info_name` varies by entry point (`cxr`, `cxr-remote`, or the callback name
  under `CliRunner`). A group that can itself be a program root supplies the
  prefix it would lose via `deprecation_prefix=`.
- `DeprecatingGroup(click.Group)` warns in `resolve_command`, NOT in callbacks:
  the diagnostic then precedes any output or prompt, fires even when the command
  exits early on a usage error, and stays out of `--help` (suppressed when an arg
  matches `get_help_option_names`).
- `SELF_WARNING = {"sweep show", "sweep set"}` — replacement is argument-
  dependent, so those callbacks call `warn(path, replacement=...)` themselves and
  `DeprecatingGroup` skips them rather than pre-empting with generic text.

Wired into: `cli/_core.py` (`LazyGroup(DeprecatingGroup)`; its ad-hoc
`resolve_command` DELETED), `cli/__init__.py` (`_DEPRECATED_COMMANDS` dict
DELETED), `cli/commands/profile.py` (`_ProfileGroup(DeprecatingGroup)` +
`deprecation_prefix="profile"`; members group `deprecation_prefix="profile
members"`; `_warn_compat` and its 7 call sites DELETED), `cli/commands/sweep.py`
(`_warn` deleted, dynamic warns restored in `show_command`/`set_command`),
`energy_grid/_command.py` (`deprecation_prefix="energy-grid"`, makes the four
hidden flat job-verb aliases warn), `_remote/cli.py` (root
`deprecation_prefix="remote"`; `profile` group now `cls=DeprecatingGroup`;
hand-written pull warning DELETED).

Registry corrections made vs the WIP file:
- `sweep` group row split into `sweep show` + `sweep set` leaf rows (different
  canonical replacements). Root `lazy_hidden` in `cli/__init__.py` is therefore
  deliberately NOT derived from the registry — deriving would un-hide `sweep` in
  root help and change `docs/cli-reference.md`.
- `profile members` split into 4 leaf rows (set/add/remove/reset each map to a
  different canonical spelling).
- Broke chained deprecations: `profile add-material`/`remove-material` pointed at
  `cxr profile members add|remove`, themselves deprecated. Now point at
  `cxr profile add|remove NAME --materials MATERIAL,...` per the locked contract.
- `remote profile` group row -> `remote profile pull` leaf row, so the warning
  names a runnable command not a group (`pull` is its only leaf).

### Deliberate output-contract change (call out in the PR)
Replacements are now generic templates, not argv-interpolated: the profile
warnings say `cxr profile set NAME --materials MATERIAL,...` where they used to
interpolate the typed profile name. Both are templates (`MATERIAL,...` was
already a placeholder); the generic text now matches the published doc table.
`tests/test_cli_profile.py` updated accordingly — 8 assertion lines, rewritten by
`(cxr profile set|add|remove) <name> --materials MATERIAL` -> `NAME`.
The two `sweep` paths KEPT argv precision on purpose: `cxr sweep show` with no
MATERIAL is an overview whose real replacement is `cxr profile list`, not
`cxr material show`. That is the entire reason `SELF_WARNING` exists.

### State of the test suite
Last full run BEFORE the fixes above: `test-suite cli` = 8 failed / 882 passed,
all 8 asserting old warning text. All 8 have now been addressed:
- 2 sweep ones via `SELF_WARNING` + restored dynamic replacements,
- 5 profile ones via the `NAME` rewrite,
- `test_remote_click.py::test_legacy_remote_profile_pull_warns_once` via the
  `remote profile pull` leaf row.
NOT RE-RUN YET. First command next session:
```
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run --package cxr-mc-tests cxr-dev test-suite cli
```
Expect 890 passed. `_remote/cli.py:468 "args is unused" (ty)` is pre-existing and
unrelated to this branch.

### `tests/test_cli_deprecations.py` DOES NOT EXIST — still to write
Subagent `af4f785cdfcf1d592` spent its budget on reconnaissance and wrote NO test
code. Its research handoff (worth reading, saves a re-walk):
`/tmp/claude-1000/-home-alexa-dev-cxr-mc-worktrees-cli-rework/3c564ec5-f3ab-45d8-81b4-adfd0ecfe0f1/scratchpad/test_cli_deprecations_handoff.md`
Useful bits it captured: `tests/cli_helpers.py` exposes `invoke` /
`assert_clean_result`; hidden root children come from `command.lazy_hidden`
(currently `{slim, rebrem, reline, archive, restore, archives, union, prune,
sweep, check, check-config}`) plus `child.hidden`, while deeper hidden nodes set
`.hidden=True` in their own modules.

Intended 7 tests: hidden-command coverage with a "group is covered when every
child is covered" rule (needed for `sweep` and `profile members`); registry rows
name live commands; replacements resolve and are not themselves deprecated;
exactly one warning per path; `--help` emits none; support-window arithmetic;
generated doc is current.

### REAL GAP the subagent confirmed — decide before writing test 4
`cxr sweep show --zzz-not-a-flag` exits 2 with ZERO "is deprecated" lines.
`SELF_WARNING` paths warn from their callback, and Click's argument parsing fails
before the callback runs — so they contradict `DeprecatingGroup`'s docstring
promise that the warning "fires even when the callback exits early on a usage
error". Two ways out:
- Cheap: special-case `SELF_WARNING` in test 4 (drive those two with valid args)
  and soften the docstring. Leaves the hole.
- Proper: in `DeprecatingGroup.invoke`, wrap the child's `make_context` in
  `try/except click.UsageError` and emit the generic registry warning (meta guard
  keeps it single) before re-raising. Then SELF_WARNING paths get generic text on
  a usage error and precise text on success, and test 4 needs no exception.
Recommend the proper fix — it is ~5 lines and restores the invariant the whole
design is sold on.

### Remaining work
1. Re-run `test-suite cli`; fix fallout.
2. Reconcile the subagent's `tests/test_cli_deprecations.py` — the SELF_WARNING
   bug above, plus `remote profile` is now a leaf row and `sweep set` gained a
   `note`.
3. Regenerate `docs/cli-deprecations.md` via `cxr-dev cli-deprecations` (new
   subcommand in `src/cxr_mc/_dev.py`, backed by new
   `scripts/generate_cli_deprecations.py`, `--write`/`--check` mirroring
   `generate_cli_reference.py`). The checked-in copy is STALE — generated before
   the `sweep set` note and the `remote profile pull` leaf-row change.
4. Add a pointer to `docs/cli-deprecations.md` in the prose header inside
   `build_reference()`, then regenerate `docs/cli-reference.md` (repo contract:
   CLI changes regenerate it).
5. `docs/repo_map.md`: ownership entry for `cli/_deprecations.py` and
   `cli/commands/`.
6. `tasks/feature/cli-deprecation-substrate/README.md` + `TODO.md` pointer, per
   the AGENTS.md task-doc contract.
7. Full verify: `cxr-dev lint`, `typecheck`, `test-suite core|cli|apps|packaging`.
   Watch `tests/test_cli_reference.py::test_root_help_warm_median_below_200_ms` —
   `LazyGroup` lazy dispatch must stay intact.

### Gotchas hit this session
- Read tool returned mangled / word-stripped source repeatedly. Fall back to
  `awk`/`cat` via Bash; confirm with `python3 -c "import ast; ast.parse(...)"`.
- zsh: quote globs — `grep -rn pat src --include='*.py'`.
- Prefix bash grep/rg with `TOKENSAVE_DISABLE_GREP_HOOK=1` for symbol patterns.
- `SendMessage` would not load via ToolSearch this session, so the running
  subagent could not be corrected mid-flight.
