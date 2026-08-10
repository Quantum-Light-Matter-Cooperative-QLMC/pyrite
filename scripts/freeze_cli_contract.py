"""Freeze current Click command/help behavior.

Usage:
    python scripts/freeze_cli_contract.py --write tests/data/cli_contract.json
    python scripts/freeze_cli_contract.py --check tests/data/cli_contract.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import click
from click.testing import CliRunner

from pyrite import cli

SCHEMA_VERSION = 1
INTENTIONAL_P0_CORRECTIONS = [
    {
        "id": "remote-profile-default-selection",
        "paths": ["remote pull"],
        "contract": (
            "Positional PROFILE and --profile resolve explicit membership or the in-use "
            "manifest; -m/--material narrows, and redundant --all warns then is ignored."
        ),
    },
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
        "paths": ["remote jobs", "remote status"],
        "contract": (
            "Human presentation sanitizes controls and machine framing uses versioned base64 fields."
        ),
    },
]


def _action_contract(parameter: click.Parameter) -> dict[str, object]:
    if isinstance(parameter, click.Option):
        option_strings = [*parameter.opts, *parameter.secondary_opts]
        help_text = "<SUPPRESS>" if parameter.hidden else parameter.help
    else:
        option_strings = []
        help_text = None
    return {
        "dest": parameter.name,
        "help": help_text,
        "option_strings": option_strings,
        "required": parameter.required,
    }


def _help(path: tuple[str, ...]) -> str:
    result = CliRunner().invoke(cli.command, [*path, "--help"], prog_name="pyrite")
    if result.exit_code != 0:
        raise RuntimeError(f"help failed for {' '.join(path) or 'root'}: {result.stderr}")
    return result.stdout


def _command_contract(
    command: click.Command,
    path: tuple[str, ...] = (),
) -> dict[str, object]:
    subcommands = []
    if isinstance(command, click.Group):
        context = click.Context(command, info_name=path[-1] if path else "pyrite")
        for name in command.list_commands(context):
            child = command.get_command(context, name)
            if child is not None:
                subcommands.append(_command_contract(child, (*path, name)))
    return {
        "actions": [_action_contract(parameter) for parameter in command.params],
        "help": _help(path),
        "hidden": command.hidden,
        "path": " ".join(path),
        "subcommands": subcommands,
    }


def build_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "intentional_p0_corrections": INTENTIONAL_P0_CORRECTIONS,
        "root": _command_contract(cli.command),
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
