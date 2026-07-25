#!/usr/bin/env python
"""PreToolUse hook: block a *local* ``cxr scan`` before it runs.

Wired from .claude/settings.json on Bash. Reads the hook payload (JSON on
stdin), pulls the shell command, and blocks the one heavy-compute footgun in
this repo: ``cxr scan`` run locally.

Why: ``cxr scan <material>`` runs a material's full Monte-Carlo sweep on the
local box. On WSL that reliably OOMs / crashes the session. The lab GPU box is
the canonical home for sweeps -- ``cxr remote scan`` syncs code up, submits to
SLURM, waits, and pulls the checkpoint back. ``cxr remote scan`` is *allowed*;
only the flat local ``cxr scan`` is blocked.

Blocking contract: exit 2 makes Claude Code treat stderr as a blocking reason
shown to the model, so it can re-issue as ``cxr remote scan``. Any other path
(no cxr, a different subcommand, --help, a read-only command that merely
mentions cxr) exits 0 and lets the call through -- fail-open, never wedge the
session on a parse edge case.
"""

import json
import os
import shlex
import sys

# Leading commands that only *read* text; if the line starts with one of these,
# a bare "cxr scan" inside it is an argument (grep pattern, echoed string), not
# an invocation -- do not block.
READ_ONLY_LEADERS = {
    "grep",
    "rg",
    "egrep",
    "fgrep",
    "echo",
    "printf",
    "cat",
    "less",
    "head",
    "tail",
    "sed",
    "awk",
    "ls",
    "find",
    "git",
}


def _local_scan(command: str) -> bool:
    """True iff ``command`` invokes the flat local ``cxr scan`` (not remote)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False  # unbalanced quotes -> fail open
    if not tokens:
        return False
    if os.path.basename(tokens[0]) in READ_ONLY_LEADERS:
        return False
    if any(t in ("-h", "--help") for t in tokens):
        return False

    for i, tok in enumerate(tokens):
        if os.path.basename(tok) != "cxr":
            continue
        # First real subcommand after `cxr`: skip options and KEY=VAL assigns.
        for nxt in tokens[i + 1 :]:
            if nxt.startswith("-") or "=" in nxt:
                continue
            return nxt == "scan"  # `remote` (or anything else) -> allowed
        return False
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if _local_scan(command):
        sys.stderr.write(
            "Blocked: `cxr scan` runs a full Monte-Carlo sweep locally and "
            "OOMs/crashes WSL. Route it to the lab GPU box instead:\n"
            "  cxr remote scan <material>   (syncs code, submits SLURM, pulls checkpoint)\n"
            "See the remote-gpu-jobs skill. If you truly must run locally, ask "
            "the user to run it themselves.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
