
# Handoff: tests/test_cli_deprecations.py (NOT YET WRITTEN)

Status: research/reconnaissance done, zero lines of the target test file written yet.
Running low on tokens — stopping here per orchestrator instruction.

## Important: other agents are LIVE-editing sibling files right now

`git status --short` at handoff time showed these modified (uncommitted, WIP by
other agents on this same worktree — do NOT revert):
```
 M src/cxr_mc/_dev.py
 M src/cxr_mc/_remote/cli.py
 M src/cxr_mc/cli/__init__.py
 M src/cxr_mc/cli/_core.py
 M src/cxr_mc/cli/_deprecations.py
 M src/cxr_mc/cli/commands/profile.py
 M src/cxr_mc/cli/commands/sweep.py
 M src/cxr_mc/energy_grid/_command.py
?? docs/cli-deprecations.md
?? scripts/generate_cli_deprecations.py
```
The refactor direction: manual per-call `_warn_compat`/`_warn` echo calls are being
replaced by the central `DeprecatingGroup` resolve-time mechanism (registry-driven).
`_core.LazyGroup` now subclasses `DeprecatingGroup` (was `click.Group`); the old
`lazy_deprecated` mapping on `LazyGroup` was removed entirely — root-level flat
aliases (`slim`, `archive`, etc.) now warn purely via `DEPRECATIONS` + `lazy_hidden`
membership. `_ProfileGroup` and `members_command` group in `commands/profile.py` now
also subclass/pass `cls=DeprecatingGroup` with `deprecation_prefix=...`. `energy-grid`
group and `remote` group (`_remote/cli.py`) likewise now use
`cls=DeprecatingGroup, deprecation_prefix=...`.

**Before resuming**: re-`cat -n` every file below fresh (do not trust anything cached
in this doc's inline snippets below — verify against disk again; they may have moved
further). Also re-check `git status --short` to see if the other agents' edits have
settled (no longer changing between two reads a few seconds apart).

## Files to re-read at resume (absolute paths)

- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/cli/_deprecations.py` — registry, `message()`, `warn()`, `warn_command()`, `invocation_path()`, `DeprecatingGroup`, and a NEW `SELF_WARNING` frozenset (see defect below).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/cli/_core.py` — `LazyGroup(DeprecatingGroup)`, `lazy_hidden` frozenset attribute (currently: `{slim, rebrem, reline, archive, restore, archives, union, prune, sweep, check, check-config}`).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/cli/__init__.py` — root `command` (`LazyGroup` instance), `_COMMANDS` lazy import map, `_COMMAND_HELP`.
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/cli/commands/sweep.py` — hidden `sweep` group (`show`, `set` leaves); both leaves now call `_deprecations.warn(path, replacement=<computed>)` from their own callback body, and BOTH `"sweep show"` and `"sweep set"` are in `SELF_WARNING` (confirmed via `git diff` at handoff time — earlier read this session showed only `"sweep show"` in `SELF_WARNING`, so this was actively changing under me; reread it).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/cli/commands/profile.py` — `_ProfileGroup(DeprecatingGroup)`, `members_command` hidden group (`deprecation_prefix="profile members"`), hidden `add-material`/`remove-material`/`analyze` commands (`hidden=True`), all now warn purely via `DeprecatingGroup` resolution (old `_warn_compat` calls were deleted in this session's diff).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/energy_grid/_command.py` — `energy-grid` group now `cls=DeprecatingGroup, deprecation_prefix="energy-grid"`; hidden legacy `attach`/`logs`/`status`/`stop` aliased in via `command.add_command(_alias)` with `_alias.hidden = True` (search around line 330-336 as of this session; re-verify line numbers).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/src/cxr_mc/_remote/cli.py` — `remote` root command now `cls=DeprecatingGroup, deprecation_prefix="remote"`; hidden `profile` subgroup (~line 1005) and hidden `check` alias command (~line 1449-1453); old inline `emit_diagnostic("warning: ...")` in `profile_command`/pull path was deleted this session (now handled by DeprecatingGroup resolution).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/scripts/generate_cli_reference.py` — has the `_context`/`_walk` helpers to mirror (note: file has DEAD CODE after its first `return` in `build_reference()` — a second unreachable loop block at the bottom, lines ~132-151. Not my file to fix; irrelevant to my test but noted).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/scripts/generate_cli_deprecations.py` — NEW, untracked, mirrors `generate_cli_reference.py`'s `--write`/`--check` pattern for `docs/cli-deprecations.md`. Test 7 (`test_generated_doc_is_current`) should subprocess this with `--check docs/cli-deprecations.md`, mirroring `tests/test_cli_reference.py::test_checked_cli_reference_is_current` exactly.
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/docs/cli-deprecations.md` — NEW, untracked, generated table; 25 rows at last read.
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/tests/test_cli_reference.py` and `tests/test_cli_profile.py` — style mirrors (already read in full this session).
- `/home/alexa/dev/cxr-mc/worktrees/cli-rework/tests/cli_helpers.py` — `invoke()`/`assert_clean_result()` helpers (full text captured below).

## cli_helpers.py (small, captured verbatim, unlikely to change)

```python
"""Assertions shared by Click migration tests."""

from __future__ import annotations

from collections.abc import Sequence

import click
from click.testing import CliRunner, Result


def invoke(command: click.Command, argv: Sequence[str] = (), *, input: str | None = None) -> Result:
    return CliRunner().invoke(command, list(argv), input=input, catch_exceptions=False)


def assert_clean_result(
    result: Result,
    *,
    exit_code: int = 0,
    stdout: str | None = None,
    stderr: str = "",
) -> None:
    assert result.exit_code == exit_code
    if stdout is not None:
        assert result.stdout == stdout
    assert result.stderr == stderr
    assert "Traceback" not in result.output
```

## DEPRECATIONS registry keys as of handoff (25 rows, from docs/cli-deprecations.md)

```
archive, archives, check, check-config,
energy-grid attach, energy-grid logs, energy-grid status, energy-grid stop,
profile add-material, profile analyze, profile members add, profile members remove,
profile members reset, profile members set, profile remove-material,
prune, rebrem, reline, remote check, remote profile, restore, slim,
sweep set, sweep show, union
```
(sorted; re-generate/recount at resume — table is regenerated by the script, don't
hand-copy into the test, just assert against the live `DEPRECATIONS` dict.)

## CONFIRMED DEFECT — do not fix, report it

`SELF_WARNING = frozenset({"sweep show", "sweep set"})` (as of last diff read) makes
`warn_command()` skip the automatic `DeprecatingGroup.resolve_command()` warning for
these two paths, on the theory that `sweep show`/`sweep set`'s own callback body will
call `_deprecations.warn(path, replacement=<computed>)` itself with a more specific
message (naming the actual material/profile).

**But** the callback body only runs if Click's argument parsing for the leaf command
succeeds. Verified live:
```
$ cxr sweep show --zzz-not-a-flag
exit 2
stderr: "Usage: cxr sweep show [OPTIONS] MATERIAL\nTry 'cxr sweep show --help' for help.\nError: No such option '--zzz-not-a-flag'.\n"
```
— **zero** "is deprecated" lines. Compare to every other hidden command (e.g.
`cxr slim --zzz-not-a-flag`), which warns from `DeprecatingGroup.resolve_command`
*before* Click parses the leaf's own args, so the warning survives usage errors.
`sweep show`/`sweep set` lose that guarantee specifically because they're in
`SELF_WARNING`. This contradicts `DeprecatingGroup`'s own docstring promise: "fires
it even when the callback exits early on a usage error."

**Impact on my assigned test 4** (`test_each_deprecated_path_warns_exactly_once`,
using the bogus-trailing-flag `--zzz-not-a-flag` methodology dictated by the task):
this specific methodology will find ZERO warnings for the `"sweep show"` and
`"sweep set"` rows, not one — because both the group-level and the callback-level
warning paths are skipped for those two keys under a usage error.

### Resolution options for whoever resumes (pick one, don't silently paper over it):

1. **Preferred**: special-case those two paths in test 4 — instead of a bogus
   trailing flag, invoke with a *valid* MATERIAL argument (so the callback body
   actually runs and self-warns) or with no `--zzz-not-a-flag` at all, and assert the
   self-computed message instead of `message(key)` exactly (since `message(key)` with
   no `replacement=` gives the *generic* registry replacement, but the actual runtime
   warning for these two keys is always the callback's *computed* replacement — those
   two are NEVER equal to plain `message(key)`, by design, once `SELF_WARNING` exists).
   This means test 4's blanket "equals `message(key)`" assertion is WRONG for
   `"sweep show"`/`"sweep set"` specifically and needs a branch: skip the strict
   equality for keys in `_deprecations.SELF_WARNING`, and instead assert the message
   still matches the `warning: 'cxr sweep {show,set}' is deprecated and will be
   removed in {remove_in}; use '...'` shape with *some* non-empty replacement.
2. Alternatively (if this turns out to still be true after other agents' WIP
   settles): report this as a genuine defect in `_deprecations.py`/`sweep.py` per the
   task's "if you find a defect in src, report it, do not fix it" instruction, and
   write test 4 to assert current (possibly-buggy) behavior for the 23 non-SELF_WARNING
   rows, with a narrower/separate assertion (or an `xfail`-free explicit check) for the
   2 SELF_WARNING rows that documents the gap instead of hiding it.

**Do NOT weaken test 4 for the other 23 rows** — those were confirmed still using the
plain `DeprecatingGroup` resolve-time path with no `SELF_WARNING` interaction, and
should satisfy the original "equals `message(key)` exactly" assertion.

## Root tree walk mechanics (for test 1)

Root `command` (`cxr_mc.cli.command`) is `LazyGroup(DeprecatingGroup(click.Group))`.
- `command.lazy_hidden` — frozenset of hidden **root-level** child names (see above).
- A root child counts as hidden if `child.hidden` (Click's normal attribute after
  `get_command()` resolves it — `LazyGroup.get_command()` copies the command and
  forces `.hidden = True` for anything in `lazy_hidden`) **or** its name is in
  `command.lazy_hidden` directly (belt & suspenders — mirrors the task instruction).
- Below the root, hidden groups/commands are plain Click `.hidden` attributes set
  directly in their defining module (`sweep` group `hidden=True`; `members` group
  `hidden=True`; `add-material`/`remove-material`/`analyze` commands `hidden=True`;
  energy-grid job aliases `_alias.hidden = True`; remote `profile` group
  `hidden=True`; remote `check` alias `.hidden = True`).
- Mirror `_walk`/`_context` from `scripts/generate_cli_reference.py` (already
  captured above) but do **NOT** `continue` on `child.hidden` — descend into hidden
  nodes too, and record their hidden-ness, since the whole point of test 1 is to
  enumerate hidden paths and check DEPRECATIONS coverage.
- Coverage rule (as given in task spec): a hidden path is "covered" if its own path
  string (space-joined, no `cxr` prefix) is a `DEPRECATIONS` key, OR (for a hidden
  **group**) every one of its (recursively enumerable, live) child leaf paths is
  itself covered. This is how `sweep` (group, hidden, not itself a key) and
  `profile members` (group, hidden, not itself a key) are covered — via their leaves.

## Test file skeleton (not yet written — start here)

Target: `/home/alexa/dev/cxr-mc/worktrees/cli-rework/tests/test_cli_deprecations.py`
(new file, mine to create — nothing to read-before-edit).

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from click.testing import CliRunner

from cxr_mc.cli import _deprecations, command
from cxr_mc.cli._deprecations import DEPRECATIONS, SUPPORT_WINDOW_MINORS, message
from tests.cli_helpers import invoke

ROOT = Path(__file__).resolve().parents[1]

# ... 7 test functions per the task list; see task prompt text (not re-pasted here
# for space — the orchestrator/parent thread still has the full original task
# message with all 7 items verbatim). Re-fetch it from conversation history if this
# doc is being read standalone without that context.
```

**IMPORTANT**: the full verbatim task spec (all 7 test names + exact assertions) was
given in the ORIGINAL task prompt to this subagent, not reproduced in full here to
save space. Whoever resumes must pull that prompt from the parent/orchestrator
thread — it is authoritative for exact test names/behavior. This handoff only adds
the *investigation* findings layered on top of that spec (the SELF_WARNING defect,
current file locations/line numbers, and the walk-coverage semantics).

## Not yet done at all

- Zero lines of `tests/test_cli_deprecations.py` written.
- Have NOT run `cxr-dev test`, `cxr-dev lint`, or `cxr-dev typecheck`.
- Have NOT re-verified file stability (whether the other agents' concurrent edits
  have settled) — do this first on resume via two `git status --short` / `git diff`
  snapshots a few seconds apart before trusting any current file content.

## Commands

```bash
cd /home/alexa/dev/cxr-mc/worktrees/cli-rework
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/test_cli_deprecations.py
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
```
