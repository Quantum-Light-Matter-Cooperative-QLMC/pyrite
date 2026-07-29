#!/usr/bin/env python
"""Block local ``cxr run`` invocations before Claude runs them.

Exit 2 blocks; parse failures and unrelated commands fail open.
"""

import json
import os
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


def _leader(tokens: list[str]) -> str:
    """Return effective command name through ``rtk env``/``env`` wrappers."""
    index = 0
    while index < len(tokens) and "=" in tokens[index]:
        index += 1
    if index < len(tokens) and os.path.basename(tokens[index]) == "rtk":
        index += 1
    if index < len(tokens) and os.path.basename(tokens[index]) == "env":
        index += 1
        while index < len(tokens) and (tokens[index].startswith("-") or "=" in tokens[index]):
            index += 1
    return os.path.basename(tokens[index]) if index < len(tokens) else ""


def _segment_has_local_run(tokens: list[str]) -> bool:
    if _leader(tokens) in READ_ONLY_LEADERS:
        return False
    if any(token in ("-h", "--help") for token in tokens):
        return False

    for i, tok in enumerate(tokens):
        if os.path.basename(tok) != "cxr":
            continue
        for nxt in tokens[i + 1 :]:
            if nxt.startswith("-") or "=" in nxt:
                continue
            return nxt == "run"
        return False
    return False


def _local_scan(command: str) -> bool:
    """Return whether any shell segment invokes flat local ``cxr run``."""
    try:
        return any(_segment_has_local_run(tokens) for tokens in _segments(command))
    except ValueError:
        return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if _local_scan(command):
        sys.stderr.write(
            "Blocked: `cxr run` runs a full Monte-Carlo sweep locally and "
            "OOMs/crashes WSL. Route it to the lab GPU box instead:\n"
            "  cxr remote run [PROFILE] [-m MATERIAL]\n"
            "See the remote-gpu-jobs skill. If you truly must run locally, ask "
            "the user to run it themselves.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
