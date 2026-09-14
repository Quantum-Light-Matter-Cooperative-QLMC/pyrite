"""The lazy Click group that binds the command tree together.

Split out of `cli/_core.py` so the shared terminal primitives could move below
`cli/` (:mod:`pyrite.console.output`) while the group that resolves command
modules on demand -- and the deprecation machinery it extends -- stays in the
package that owns the command tree.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from copy import copy

import click

from ..console.output import CLIError, paint
from ._deprecations import DeprecatingGroup


class LazyGroup(DeprecatingGroup):
    """Click group whose command objects import only when resolved."""

    def __init__(
        self,
        *args,
        lazy_commands: Mapping[str, str] | None = None,
        lazy_help: Mapping[str, str] | None = None,
        lazy_hidden: Sequence[str] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lazy_commands = dict(lazy_commands or {})
        self.lazy_help = dict(lazy_help or {})
        self.lazy_hidden = frozenset(lazy_hidden)

    def list_commands(self, ctx: click.Context) -> list[str]:
        eager = super().list_commands(ctx)
        return [*eager, *(name for name in self.lazy_commands if name not in eager)]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        import_path = self.lazy_commands.get(cmd_name)
        if import_path is None:
            return super().get_command(ctx, cmd_name)
        module_name, object_name = import_path.rsplit(".", 1)
        command = getattr(importlib.import_module(module_name), object_name)
        if not isinstance(command, click.Command):
            raise CLIError(f"lazy command {cmd_name!r} resolved to non-command {import_path!r}")
        if cmd_name in self.lazy_hidden:
            command = copy(command)
            command.hidden = True
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render root summaries without importing lazy command modules."""
        rows = []
        for name in self.list_commands(ctx):
            if name in self.lazy_hidden:
                continue
            if name in self.lazy_commands and name in self.lazy_help:
                rows.append((name, self.lazy_help[name]))
                continue
            command = self.get_command(ctx, name)
            if command is None or command.hidden:
                continue
            rows.append((name, command.get_short_help_str()))
        if rows:
            with formatter.section(paint("Commands", "active")):
                formatter.write_dl(rows)

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render root option heading with same terminal-safe accent."""
        rows = [
            record
            for parameter in self.get_params(ctx)
            if (record := parameter.get_help_record(ctx)) is not None
        ]
        if rows:
            with formatter.section(paint("Options", "active")):
                formatter.write_dl(rows)
        self.format_commands(ctx, formatter)
