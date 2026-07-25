"""Freeze argparse CLI behavior before Click migration.

Usage:
    python scripts/freeze_cli_contract.py --write tests/data/cli_contract.json
    python scripts/freeze_cli_contract.py --check tests/data/cli_contract.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from cxr_mc import cli

SCHEMA_VERSION = 1
INTENTIONAL_P0_CORRECTIONS = [
    {
        "id": "remote-input-validation",
        "paths": ["remote"],
        "contract": (
            "Reject unsafe host, remote directory, executable, SSH/SCP word, Bash payload, "
            "and SBATCH directive inputs before crossing process boundaries."
        ),
    },
    {
        "id": "line-grid-submit-forwarding",
        "paths": ["line-grid submit"],
        "contract": (
            "Forward tilts, azimuths, thickness, and set-default through every remote slice."
        ),
    },
    {
        "id": "line-grid-atomic-validation",
        "paths": ["line-grid apply", "line-grid set", "line-grid set-brem"],
        "contract": (
            "Validate proposed catalog and provenance before atomic replacement; failed writes "
            "leave both unchanged."
        ),
    },
    {
        "id": "line-grid-defaults-semantics",
        "paths": ["line-grid defaults"],
        "contract": (
            "Value flags require --set; numeric domains are validated; zero is preserved only "
            "where documented."
        ),
    },
    {
        "id": "line-grid-brem-step",
        "paths": ["line-grid derive"],
        "contract": "--brem-step controls derived bremsstrahlung grid spacing.",
    },
    {
        "id": "line-grid-stale-golden",
        "paths": ["line-grid apply", "line-grid set", "line-grid set-brem"],
        "contract": "Stale material-catalog golden emits stderr warning after successful mutation.",
    },
    {
        "id": "remote-failure-exits",
        "paths": ["remote logs", "remote pull"],
        "contract": (
            "Follow-log failures are nonzero, interruption is 130, and partial pull failures "
            "remain failures even when some items succeed."
        ),
    },
    {
        "id": "regen-golden-source-only",
        "paths": ["line-grid regen-golden"],
        "contract": "Command succeeds only from source checkout and fails clearly from installed wheel.",
    },
    {
        "id": "remote-presentation-safety",
        "paths": ["remote attach", "remote jobs", "remote status"],
        "contract": (
            "Human presentation sanitizes controls and machine framing uses versioned base64 fields."
        ),
    },
]


class _ParserCaptured(Exception):
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


def _capture_parse_args(
    parser: argparse.ArgumentParser,
    args=None,
    namespace=None,
):
    del args, namespace
    raise _ParserCaptured(parser)


def capture_root_parser() -> argparse.ArgumentParser:
    """Capture parser constructed by real ``cxr_mc.cli.main``."""
    with mock.patch.object(argparse.ArgumentParser, "parse_args", _capture_parse_args):
        try:
            cli.main([])
        except _ParserCaptured as exc:
            return exc.parser
    raise AssertionError("cxr_mc.cli.main did not parse arguments")


def _qualname(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    name = f"{module}.{qualname}"
    closure = getattr(value, "__closure__", None)
    if closure:
        targets = [
            cell.cell_contents
            for cell in closure
            if isinstance(cell.cell_contents, Callable) and cell.cell_contents is not value
        ]
        if targets:
            name += "[" + ",".join(_qualname(target) for target in targets) + "]"
    return name


def _json_value(value: object):
    if value is argparse.SUPPRESS:
        return "<SUPPRESS>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if callable(value):
        return _qualname(value)
    return repr(value)


def _dispatch_name(parser: argparse.ArgumentParser) -> str | None:
    func = parser.get_default("func")
    return _qualname(func) if callable(func) else None


def _action_contract(action: argparse.Action) -> dict[str, object]:
    return {
        "action": type(action).__name__,
        "choices": _json_value(action.choices),
        "const": _json_value(action.const),
        "default": _json_value(action.default),
        "dest": action.dest,
        "help": _json_value(action.help),
        "metavar": _json_value(action.metavar),
        "nargs": _json_value(action.nargs),
        "option_strings": list(action.option_strings),
        "required": action.required,
        "type": _qualname(action.type) if callable(action.type) else None,
    }


def _parser_contract(
    parser: argparse.ArgumentParser,
    path: tuple[str, ...] = (),
) -> dict[str, object]:
    subparsers_action = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    actions = [
        _action_contract(action)
        for action in parser._actions
        if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))
    ]
    groups = [
        {
            "required": group.required,
            "destinations": [action.dest for action in group._group_actions],
        }
        for group in parser._mutually_exclusive_groups
    ]
    old_columns = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "80"
    try:
        help_text = parser.format_help()
    finally:
        if old_columns is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = old_columns
    subcommands = []
    if subparsers_action is not None:
        subcommands = [
            _parser_contract(child, (*path, name))
            for name, child in subparsers_action.choices.items()
        ]
    return {
        "actions": actions,
        "description": parser.description,
        "dispatch": _dispatch_name(parser),
        "epilog": parser.epilog,
        "help": help_text,
        "mutually_exclusive_groups": groups,
        "path": " ".join(path),
        "prog": parser.prog,
        "subcommands": subcommands,
        "subparsers_required": (
            subparsers_action.required if subparsers_action is not None else None
        ),
    }


def build_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "intentional_p0_corrections": INTENTIONAL_P0_CORRECTIONS,
        "root": _parser_contract(capture_root_parser()),
    }


def _encoded_contract() -> str:
    return json.dumps(build_contract(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, metavar="PATH")
    mode.add_argument("--check", type=Path, metavar="PATH")
    args = parser.parse_args(argv)
    encoded = _encoded_contract()
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(encoded, encoding="utf-8")
        return 0
    current = args.check.read_text(encoding="utf-8")
    if current != encoded:
        parser.error(f"CLI contract changed: regenerate with --write {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
