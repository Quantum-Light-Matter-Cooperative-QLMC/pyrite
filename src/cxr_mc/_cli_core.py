"""Shared Click primitives for staged argparse-to-Click migration."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import click


class CLIError(click.ClickException):
    """Runtime CLI failure: diagnostic on stderr, exit 1."""

    exit_code = 1


class ResumableCLIError(CLIError):
    """Requested work paused with resumable state."""

    exit_code = 75


class LazyGroup(click.Group):
    """Click group whose command objects import only when resolved."""

    def __init__(
        self,
        *args,
        lazy_commands: Mapping[str, str] | None = None,
        lazy_help: Mapping[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lazy_commands = dict(lazy_commands or {})
        self.lazy_help = dict(lazy_help or {})

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
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render root summaries without importing lazy command modules."""
        rows = []
        for name in self.list_commands(ctx):
            if name in self.lazy_commands and name in self.lazy_help:
                rows.append((name, self.lazy_help[name]))
                continue
            command = self.get_command(ctx, name)
            if command is None or command.hidden:
                continue
            rows.append((name, command.get_short_help_str()))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


class FiniteRange(click.ParamType):
    """Finite number with explicit inclusive/exclusive lower bound."""

    name = "number"

    def __init__(
        self,
        *,
        integer: bool,
        minimum: int | float,
        minimum_open: bool,
    ):
        self.integer = integer
        self.minimum = minimum
        self.minimum_open = minimum_open

    def convert(self, value, param, ctx):
        if self.integer and isinstance(value, int) and not isinstance(value, bool):
            converted = value
        elif not self.integer and isinstance(value, int | float) and not isinstance(value, bool):
            converted = float(value)
        else:
            label = "integer" if self.integer else "number"
            try:
                converted = int(value) if self.integer else float(value)
            except (TypeError, ValueError):
                self.fail(f"{value!r} is not a valid {label}", param, ctx)
        if not math.isfinite(converted):
            self.fail(f"{value!r} must be finite", param, ctx)
        valid = converted > self.minimum if self.minimum_open else converted >= self.minimum
        if not valid:
            relation = "greater than" if self.minimum_open else "at least"
            self.fail(f"{value!r} must be {relation} {self.minimum}", param, ctx)
        return converted


class FiniteFloat(click.ParamType):
    """Finite float without a sign or magnitude restriction."""

    name = "number"

    def convert(self, value, param, ctx):
        try:
            converted = float(value)
        except (TypeError, ValueError):
            self.fail(f"{value!r} is not a valid number", param, ctx)
        if not math.isfinite(converted):
            self.fail(f"{value!r} must be finite", param, ctx)
        return converted


class BeamUVW(click.ParamType):
    """Three integer zone-axis components, excluding zero vector."""

    name = "H K L"

    def convert(self, value, param, ctx):
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            self.fail(f"{value!r} must contain exactly three integers", param, ctx)
        if len(value) != 3:
            self.fail(f"{value!r} must contain exactly three integers", param, ctx)
        try:
            beam = tuple(int(component) for component in value)
        except (TypeError, ValueError):
            self.fail(f"{value!r} must contain exactly three integers", param, ctx)
        if beam == (0, 0, 0):
            self.fail("(0, 0, 0) is not a valid beam direction", param, ctx)
        return beam


POSITIVE_INT = FiniteRange(integer=True, minimum=0, minimum_open=True)
NONNEGATIVE_INT = FiniteRange(integer=True, minimum=0, minimum_open=False)
POSITIVE_FLOAT = FiniteRange(integer=False, minimum=0.0, minimum_open=True)
NONNEGATIVE_FLOAT = FiniteRange(integer=False, minimum=0.0, minimum_open=False)
FINITE_FLOAT = FiniteFloat()
BEAM_UVW = BeamUVW()


def json_option(function):
    """Add explicit machine-output switch shared by JSON-capable commands."""
    return click.option(
        "--json",
        "json_output",
        is_flag=True,
        help="Emit one versioned JSON object on stdout.",
    )(function)


def emit_result(message: str) -> None:
    click.echo(message)


def emit_diagnostic(message: str) -> None:
    click.echo(message, err=True)


def emit_json(
    schema: str,
    payload: object,
    *,
    errors: Sequence[Mapping[str, object]] = (),
) -> None:
    """Emit exactly one stable JSON envelope plus newline."""
    error_list = [dict(error) for error in errors]
    envelope = {
        "schema": schema,
        "schema_version": 1,
        "ok": not error_list,
        "payload": payload,
        "errors": error_list,
    }
    click.echo(json.dumps(envelope, separators=(",", ":"), sort_keys=True))


def emit_json_result(result: Any, *, failure_exit: int = 1) -> None:
    """Emit a ``cli_json.JsonResult`` and map structured failures to an exit."""
    emit_json(result.schema, result.payload, errors=result.errors)
    if result.errors:
        raise click.exceptions.Exit(failure_exit)


def invoke_legacy(function, /, **values):
    """Invoke a staged argparse handler under Click.

    String ``SystemExit`` values become runtime diagnostics. Integer exit
    statuses remain exact, including resumable status 75.
    """
    try:
        return function(SimpleNamespace(**values))
    except SystemExit as exc:
        if exc.code is None:
            return None
        if isinstance(exc.code, int):
            raise click.exceptions.Exit(exc.code) from None
        raise CLIError(str(exc.code)) from None


def run(command: click.Command, argv: Sequence[str] | None = None, *, prog_name: str) -> Any:
    """Invoke Click without framework-owned exits; preserve cxr exit contract."""
    try:
        return command.main(
            args=None if argv is None else list(argv),
            prog_name=prog_name,
            standalone_mode=False,
        )
    except click.Abort:
        emit_diagnostic("Aborted!")
        return 130
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except Exception as exc:
        if (
            type(exc).__name__ == "MaterialConfigError"
            and type(exc).__module__ == "cxr_mc.materials.catalog"
        ):
            emit_diagnostic(str(exc))
            return 1
        raise
