#!/usr/bin/env python
"""Block local ``pyrite run`` invocations before Claude runs them.

Exit 2 blocks; parse failures and unrelated commands fail open.

The block is an opt-in-overridable safety default for GPU-less/underprovisioned
WSL hosts. A box with a real accelerator declares itself a local compute node by
setting ``PYRITE_LOCAL_SWEEP_OK`` truthy -- ambient in the hook's environment (e.g.
`.claude/settings.local.json` env) or inline on the command
(``PYRITE_LOCAL_SWEEP_OK=1 pyrite run ...``). ``CXR_LOCAL_SWEEP_OK`` remains a
compatibility alias. Default stays a hard block so shared
contributors on unfit hosts remain protected.
"""

import json
import os
import re
import shlex
import sys

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
SHELL_SEPARATORS = set(";&|()")

# Truthy override values; everything else (including unset) keeps the block.
_FALSEY = {"", "0", "false", "no", "off"}
_OVERRIDE_VARS = ("PYRITE_LOCAL_SWEEP_OK", "CXR_LOCAL_SWEEP_OK")
_OVERRIDE_ASSIGN_RE = re.compile(
    rf"^(?P<name>{'|'.join(map(re.escape, _OVERRIDE_VARS))})=(?P<value>.*)$"
)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _FALSEY


def _segments(command: str) -> list[list[str]]:
    """Split shell text at control operators while respecting quotes."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token and set(token) <= SHELL_SEPARATORS:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _segment_has_local_run(tokens: list[str]) -> bool:
    if any(token in ("-h", "--help") for token in tokens):
        return False

    for i, tok in enumerate(tokens):
        if os.path.basename(tok) not in {"pyrite", "cxr"}:
            continue
        trailing = tokens[i + 1 :]
        for nxt in trailing:
            if nxt.startswith("-") or "=" in nxt:
                continue
            if nxt != "run":
                return False
            return not any(
                option == "--remote" or option.startswith("--remote=") or option.startswith("-R")
                for option in trailing
            )
        return False
    return False


def _local_scan(command: str) -> bool:
    """Return whether any shell segment invokes a local PyRITE run."""
    try:
        return any(_segment_has_local_run(tokens) for tokens in _segments(command))
    except ValueError:
        return False


def _override_active(command: str) -> bool:
    """Return whether this host opted this ``pyrite run`` out of the sweep block.

    The canonical override wins when both canonical and compatibility names
    are present, whether ambient or inline.
    """
    for name in _OVERRIDE_VARS:
        if name in os.environ:
            return _is_truthy(os.environ[name])
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    assignments: dict[str, str] = {}
    for token in tokens:
        match = _OVERRIDE_ASSIGN_RE.match(token)
        if match:
            assignments[match.group("name")] = match.group("value")
    for name in _OVERRIDE_VARS:
        if name in assignments:
            return _is_truthy(assignments[name])
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if _local_scan(command) and not _override_active(command):
        sys.stderr.write(
            "Blocked: `pyrite run` runs a full Monte-Carlo sweep locally and "
            "OOMs/crashes WSL. Route it to the lab GPU box instead:\n"
            "  pyrite run [PROFILE] [-m MATERIAL] --remote\n"
            "See the remote-gpu-jobs skill. On a host with a real accelerator, "
            "set PYRITE_LOCAL_SWEEP_OK=1 (ambient or inline) to run locally. If you "
            "truly must run locally without it, ask the user to run it themselves.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
