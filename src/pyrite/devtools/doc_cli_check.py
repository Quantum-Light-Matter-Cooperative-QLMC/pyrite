"""Check documented ``bash`` command lines against the live CLI trees.

Guides show ``pyrite`` and ``pyrite-dev`` invocations as prose examples that
are never executed (they mutate workspaces, reach remotes, launch Monte Carlo
runs). This module instead resolves each documented command line against the
live Click tree (``pyrite``) and argparse tree (``pyrite-dev``, including its
two relocated Click sub-trees) and asserts every option flag it uses exists.
It never runs a command.

See ``docs/guides/*.md``, ``pyrite.devtools.doc_blocks``, and
``tests/dev/test_doc_blocks.py``.
"""

import argparse
import re
import shlex
from dataclasses import dataclass

import click

from pyrite.cli import command as pyrite_command
from pyrite.cli.commands.scan import performance_command as perf_command
from pyrite.devtools.cli_commands import energy_grid_command, performance_command
from pyrite.devtools.cli_reference import _walk as _click_walk
from pyrite.devtools.doc_blocks import FencedBlock

from .dev_cli import build_parser

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# pyrite-dev subcommands that hand their remaining argv to a relocated Click
# tree (`_run_relocated_click` in `pyrite._dev`) instead of parsing it with
# argparse. Their argparse subparser is declared ``add_help=False`` with a
# ``REMAINDER`` positional for exactly this reason.
_CLICK_DELEGATES: dict[tuple[str, ...], click.Command] = {
    ("pyrite-dev", "perf"): perf_command,
    ("pyrite-dev", "performance"): performance_command,
    ("pyrite-dev", "energy-grid"): energy_grid_command,
}


class DocCommandError(ValueError):
    """A documented command line could not be tokenized or resolved at all."""


def _click_tree(root: click.Command, root_path: tuple[str, ...]):
    return {path: (cmd, ctx) for path, cmd, ctx in _click_walk(root, root_path)}


def _click_option_strings(cmd: click.Command, ctx: click.Context) -> frozenset[str]:
    names: set[str] = set()
    for param in cmd.get_params(ctx):
        if isinstance(param, click.Option):
            names.update(param.opts)
            names.update(param.secondary_opts)
    return frozenset(names)


def _resolve_click(
    tree: dict[tuple[str, ...], tuple[click.Command, click.Context]],
    root_path: tuple[str, ...],
    tokens: list[str],
) -> tuple[tuple[str, ...], frozenset[str], list[str]]:
    path = root_path
    cmd, ctx = tree[path]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            break
        if not isinstance(cmd, click.Group):
            # Leaf command: remaining non-flag tokens are argument values.
            break
        candidate = (*path, token)
        if candidate not in tree:
            raise DocCommandError(f"unknown subcommand {token!r} under `{' '.join(path)}`")
        path = candidate
        cmd, ctx = tree[path]
        i += 1
    return path, _click_option_strings(cmd, ctx), tokens[i:]


def _subparsers_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:  # noqa: SLF001 -- argparse has no public accessor
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action
    return None


def _argparse_option_strings(parser: argparse.ArgumentParser) -> frozenset[str]:
    names: set[str] = set()
    for action in parser._actions:  # noqa: SLF001
        names.update(action.option_strings)
    return frozenset(names)


def _resolve_argparse(
    root_parser: argparse.ArgumentParser,
    root_path: tuple[str, ...],
    tokens: list[str],
) -> tuple[tuple[str, ...], argparse.ArgumentParser, list[str]]:
    path = root_path
    parser = root_parser
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            break
        sub_action = _subparsers_action(parser)
        if sub_action is None:
            # No nested subcommand at this level: remaining tokens are
            # positional argument values (e.g. `test-suite packaging`).
            break
        if token not in sub_action.choices:
            raise DocCommandError(f"unknown subcommand {token!r} under `{' '.join(path)}`")
        path = (*path, token)
        parser = sub_action.choices[token]
        i += 1
    return path, parser, tokens[i:]


def _pyrite_tree() -> dict[tuple[str, ...], tuple[click.Command, click.Context]]:
    return _click_tree(pyrite_command, ("pyrite",))


def resolve_command(tokens: list[str]) -> tuple[tuple[str, ...], frozenset[str], list[str]]:
    """Resolve a tokenized ``pyrite``/``pyrite-dev`` invocation.

    Returns ``(resolved_path, valid_option_strings, remaining_tokens)`` where
    ``remaining_tokens`` are the tokens after the resolved command path
    (option flags and positional values, unconsumed by resolution).

    Raises ``DocCommandError`` if ``tokens`` does not start with ``pyrite`` or
    ``pyrite-dev``, or if a subcommand token does not exist on the live tree.
    """
    if not tokens:
        raise DocCommandError("empty command line")
    if tokens[0] == "pyrite":
        return _resolve_click(_pyrite_tree(), ("pyrite",), tokens[1:])
    if tokens[0] == "pyrite-dev":
        path, parser, rest = _resolve_argparse(build_parser(), ("pyrite-dev",), tokens[1:])
        delegate = _CLICK_DELEGATES.get(path)
        if delegate is not None:
            return _resolve_click(_click_tree(delegate, path), path, rest)
        return path, _argparse_option_strings(parser), rest
    raise DocCommandError(f"unrecognized command root {tokens[0]!r}; expected pyrite/pyrite-dev")


def is_documented_invocation(tokens: list[str]) -> bool:
    """Return whether ``tokens`` (after prefix-stripping) invokes pyrite/pyrite-dev."""
    return bool(tokens) and tokens[0] in ("pyrite", "pyrite-dev")


def strip_line_prefixes(tokens: list[str]) -> list[str]:
    """Strip leading ``VAR=value`` env assignments and a leading ``uv run``."""
    while tokens and _ENV_ASSIGNMENT_RE.match(tokens[0]):
        tokens = tokens[1:]
    if len(tokens) >= 2 and tokens[0] == "uv" and tokens[1] == "run":
        tokens = tokens[2:]
        while tokens and (_ENV_ASSIGNMENT_RE.match(tokens[0]) or tokens[0].startswith("--extra")):
            tokens = tokens[2:] if tokens[0] == "--extra" else tokens[1:]
    return tokens


@dataclass(frozen=True)
class LogicalLine:
    """One shell command line from a bash block, continuations already joined."""

    source_line: int
    text: str


def _join_continuations(block: FencedBlock) -> list[LogicalLine]:
    raw_lines = block.body.split("\n")
    logical: list[LogicalLine] = []
    buffer = ""
    start_index: int | None = None
    for index, raw in enumerate(raw_lines):
        if start_index is None:
            start_index = index
        stripped = raw.rstrip()
        if stripped.endswith("\\") and not stripped.endswith("\\\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += raw
        logical.append(LogicalLine(source_line=block.line_at(start_index), text=buffer))
        buffer = ""
        start_index = None
    if buffer:
        logical.append(LogicalLine(source_line=block.line_at(start_index or 0), text=buffer))
    return logical


def check_bash_block(block: FencedBlock) -> list[str]:
    """Return error strings for one ``bash`` fenced block; empty means clean.

    A block carrying a ``skip_reason`` (an immediately preceding
    ``<!-- verify: skip (reason) -->`` marker) is exempt and always returns
    ``[]``. Otherwise every logical line is tokenized with ``shlex``; a
    tokenization failure is reported as an error, never silently skipped.
    Lines that do not resolve to a ``pyrite``/``pyrite-dev`` invocation
    (environment setup, other tools) are ignored. If the block ends with no
    ``pyrite``/``pyrite-dev`` invocation at all and no skip marker, that is
    reported as an error: every block must be either checked or explicitly
    opted out.
    """
    if block.skip_reason is not None:
        return []

    errors: list[str] = []
    checked_any = False
    for line in _join_continuations(block):
        try:
            tokens = shlex.split(line.text, comments=True)
        except ValueError as exc:
            errors.append(f"{block.path}:{line.source_line}: could not tokenize command: {exc}")
            continue
        if not tokens:
            continue
        tokens = strip_line_prefixes(tokens)
        if not is_documented_invocation(tokens):
            continue
        checked_any = True
        location = f"{block.path}:{line.source_line}"
        try:
            path, valid_options, rest = resolve_command(tokens)
        except DocCommandError as exc:
            errors.append(f"{location}: {exc}")
            continue
        command_str = " ".join(path)
        for token in rest:
            if not token.startswith("-"):
                continue
            flag = token.split("=", 1)[0]
            if flag not in valid_options:
                errors.append(f"{location}: unknown option {flag!r} for `{command_str}`")

    if not checked_any:
        errors.append(
            f"{block.location}: no pyrite/pyrite-dev command found and no "
            "`<!-- verify: skip (reason) -->` marker above the fence"
        )
    return errors
