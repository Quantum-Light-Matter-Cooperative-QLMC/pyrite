"""Manage persistent CLI context defaults."""

from __future__ import annotations

from pathlib import Path

import click

from ..._remote.config import validate_remote_target
from .. import _config
from .._core import CLIError

_KEY = click.Choice(_config.keys(), case_sensitive=True)


def _validated(key: str, value: str) -> str:
    if key == "remote.target":
        try:
            return validate_remote_target(value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="VALUE") from exc
    if key == "workspace.root":
        return str(Path(value).expanduser().resolve())
    from ...materials import CATALOG

    if value not in CATALOG.profile_names:
        choices = ", ".join(CATALOG.profile_names)
        raise click.BadParameter(
            f"unknown profile {value!r}; choose one of: {choices}", param_hint="VALUE"
        )
    return value


@click.group("config")
def command() -> None:
    """Set and inspect environment-scoped CLI defaults.

    Values resolve in one order everywhere: per-call flag, CXR_* environment,
    config store, then built-in default.
    """


@command.command("set")
@click.argument("key", type=_KEY)
@click.argument("value")
def set_command(key: str, value: str) -> None:
    """Persist VALUE for KEY."""
    value = _validated(key, value)
    try:
        _config.set_stored(key, value)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    click.echo(f"{key} = {value}")


@command.command("get")
@click.argument("key", type=_KEY)
def get_command(key: str) -> None:
    """Print the effective value for KEY."""
    try:
        click.echo(_config.resolve(key).value)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc


@command.command("list")
def list_command() -> None:
    """List effective values and the winning precedence source."""
    try:
        rows = []
        for key in _config.keys():
            resolved = _config.resolve(key)
            rows.append((key, resolved.value, resolved.source))
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    click.echo("KEY\tVALUE\tSOURCE")
    for key, value, source in rows:
        click.echo(f"{key}\t{value}\t{source}")
