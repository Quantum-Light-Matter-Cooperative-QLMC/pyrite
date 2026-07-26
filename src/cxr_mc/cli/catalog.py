"""Canonical grouped CLI for material-catalog operations."""

from __future__ import annotations

import click

from ._core import LazyGroup, run

_COMMANDS = {
    "validate": "cxr_mc.check_config.command",
}

_COMMAND_HELP = {
    "validate": "Validate a material catalog without starting simulation.",
}


@click.group(
    "catalog",
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=True,
)
def command() -> None:
    """Inspect and validate material-catalog configuration.

    \b
    Example:
      cxr catalog validate
      cxr catalog validate path/to/materials.toml
    """


def main(argv=None):
    return run(command, argv, prog_name="cxr-catalog")


if __name__ == "__main__":
    raise SystemExit(main())
