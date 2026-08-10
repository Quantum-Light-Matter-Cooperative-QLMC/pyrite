"""Shared Click primitives for staged argparse-to-Click migration."""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from copy import copy
from types import SimpleNamespace
from typing import Any

import click

from pyrite.cli._deprecations import DeprecatingGroup, RetiredOption

_COLOR_MODE: ContextVar[str] = ContextVar("cxr_cli_color_mode", default="auto")

COLORS = {
    "active": (92, 207, 230),
    "done": (170, 217, 76),
    "warning": (255, 213, 128),
    "failed": (240, 113, 120),
    "inactive": (127, 140, 152),
}


def _auto_color_enabled(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and isatty()


def color_enabled(stream=None) -> bool:
    """Return effective human-output color policy for ``stream``."""
    stream = sys.stdout if stream is None else stream
    mode = _COLOR_MODE.get()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return _auto_color_enabled(stream)


def paint(text: object, role: str, *, stream=None, enabled: bool | None = None) -> str:
    """Add redundant semantic color without changing plain-text content."""
    value = str(text)
    if enabled is None:
        enabled = color_enabled(stream)
    if not enabled:
        return value
    red, green, blue = COLORS[role]
    return f"\033[38;2;{red};{green};{blue}m{value}\033[0m"


def _set_color(ctx: click.Context, _param: click.Parameter, value: str) -> str:
    token = _COLOR_MODE.set(value)
    ctx.call_on_close(lambda: _COLOR_MODE.reset(token))
    if value == "always":
        ctx.color = True
    elif value == "never" or (
        value == "auto"
        and (os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb")
    ):
        ctx.color = False
    return value


def color_option(function):
    """Add root color policy without leaking ANSI into default piped output."""
    return click.option(
        "--color",
        type=click.Choice(("auto", "always", "never"), case_sensitive=False),
        default="auto",
        show_default=True,
        is_eager=True,
        expose_value=False,
        callback=_set_color,
        help="Color human output: auto for terminals, always, or never.",
    )(function)


class CLIError(click.ClickException):
    """Runtime CLI failure: diagnostic on stderr, exit 1."""

    exit_code = 1


class ResumableCLIError(CLIError):
    """Requested work paused with resumable state."""

    exit_code = 75


def hidden_alias(group: click.Group, command: click.Command, name: str) -> click.Command:
    """Register *command* under a retired *name* as a hidden alias on *group*.

    The alias is a copy, so hiding the retired spelling never hides the
    canonical one. Every alias registered here needs a matching row in
    :mod:`pyrite.cli._deprecations`; `tests/cli/test_deprecations.py` enforces
    that in both directions.
    """
    alias = copy(command)
    alias.name = name
    alias.hidden = True
    group.add_command(alias)
    return alias


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


def _floats(text):
    return [float(item) for item in text.split(",")] if text else None


def _expand_range(token, label, param, ctx, fail):
    """Expand one ``start:stop:step`` token; stop is an inclusive bound.

    Values land on ``start + i*step`` while staying on the way to ``stop``
    (MATLAB colon semantics): ``50:100:25`` gives 50, 75, 100; ``30:100:25``
    gives 30, 55, 80 because 100 is not hit exactly.
    """
    parts = token.split(":")
    if len(parts) != 3:
        fail(f"{label} range {token!r} must be start:stop:step", param, ctx)
    try:
        start, stop, step = (float(part) for part in parts)
    except ValueError:
        fail(f"{label} range {token!r} must be numeric start:stop:step", param, ctx)
    if not all(math.isfinite(item) for item in (start, stop, step)):
        fail(f"{label} range {token!r} values must be finite", param, ctx)
    if step == 0:
        fail(f"{label} range {token!r} step must be nonzero", param, ctx)
    span = stop - start
    if span == 0:
        return [start]
    if span * step < 0:
        fail(f"{label} range {token!r} step does not approach stop", param, ctx)
    count = int(math.floor(span / step + 1e-9))
    if count + 1 > 100000:
        fail(f"{label} range {token!r} expands to more than 100000 values", param, ctx)
    return [round(start + i * step, 9) for i in range(count + 1)]


class _CSV(click.ParamType):
    """Comma-separated finite floats with an optional numeric domain."""

    name = "numbers"

    def __init__(
        self,
        label,
        *,
        lower=None,
        lower_open=False,
        upper=None,
        upper_inclusive=True,
        preserve_text=False,
        ranges=False,
    ):
        self.label = label
        self.lower = lower
        self.lower_open = lower_open
        self.upper = upper
        self.upper_inclusive = upper_inclusive
        self.preserve_text = preserve_text
        self.ranges = ranges

    def convert(self, value, param, ctx):
        if self.ranges and value and ":" in value:
            values = []
            for token in value.split(","):
                if ":" in token:
                    values.extend(_expand_range(token, self.label, param, ctx, self.fail))
                else:
                    try:
                        values.append(float(token))
                    except (TypeError, ValueError):
                        self.fail(f"{self.label} must be comma-separated numbers", param, ctx)
        else:
            try:
                values = _floats(value)
            except (TypeError, ValueError):
                self.fail(f"{self.label} must be comma-separated numbers", param, ctx)
        if not values:
            self.fail(f"{self.label} requires at least one value", param, ctx)
        for item in values:
            if not math.isfinite(item):
                self.fail(f"{self.label} values must be finite", param, ctx)
            if self.lower is not None:
                lower_ok = item > self.lower if self.lower_open else item >= self.lower
                if not lower_ok:
                    self._fail_domain(param, ctx)
            if self.upper is not None:
                upper_ok = item <= self.upper if self.upper_inclusive else item < self.upper
                if not upper_ok:
                    self._fail_domain(param, ctx)
        return value if self.preserve_text else values

    def _fail_domain(self, param, ctx):
        if self.lower == 0 and self.lower_open and self.upper is None:
            self.fail(f"{self.label} values must be finite and positive", param, ctx)
        relation = "<=" if self.upper_inclusive else "<"
        self.fail(
            f"{self.label} values must satisfy {self.lower:g} <= value {relation} {self.upper:g}",
            param,
            ctx,
        )


POSITIVE_INT = FiniteRange(integer=True, minimum=0, minimum_open=True)
NONNEGATIVE_INT = FiniteRange(integer=True, minimum=0, minimum_open=False)
POSITIVE_FLOAT = FiniteRange(integer=False, minimum=0.0, minimum_open=True)
NONNEGATIVE_FLOAT = FiniteRange(integer=False, minimum=0.0, minimum_open=False)
FINITE_FLOAT = FiniteFloat()
BEAM_UVW = BeamUVW()


class _IntCSV(click.ParamType):
    """Comma-separated positive integers (e.g. electron-count grids)."""

    name = "counts"

    def __init__(self, label):
        self.label = label

    def convert(self, value, param, ctx):
        try:
            counts = [int(token) for token in str(value).split(",")]
        except (TypeError, ValueError):
            self.fail(f"{self.label} must be comma-separated positive integers", param, ctx)
        if not counts or any(count <= 0 for count in counts):
            self.fail(f"{self.label} must be comma-separated positive integers", param, ctx)
        return counts


COUNT_CSV = _IntCSV("counts")

# Shared geometry CSV param types. ``_TEXT`` variants preserve the raw string for
# argv relaying (``energy-grid derive``/``submit``); the plain variants return the
# parsed float list for direct catalog writes (``sweep set``, ``energy-grid defaults``).
ENERGY_CSV = _CSV("energy", lower=0, lower_open=True)
ENERGY_CSV_TEXT = _CSV("energy", lower=0, lower_open=True, preserve_text=True)
THICKNESS_CSV = _CSV("thickness", lower=0, lower_open=True)
THICKNESS_CSV_TEXT = _CSV("thickness", lower=0, lower_open=True, preserve_text=True)
TILT_CSV = _CSV("tilt", lower=0, upper=90, upper_inclusive=False)
TILT_CSV_TEXT = _CSV("tilt", lower=0, upper=90, upper_inclusive=False, preserve_text=True)
AZIMUTH_CSV = _CSV("azimuth", lower=90, upper=270, lower_open=True, upper_inclusive=False)
AZIMUTH_CSV_TEXT = _CSV(
    "azimuth", lower=90, upper=270, lower_open=True, upper_inclusive=False, preserve_text=True
)

# Range-capable variants for direct catalog writes (``sweep set``, ``cxr
# profile``): additionally accept ``start:stop:step`` tokens (stop-inclusive
# bound), mixable with plain CSV values. The ``_TEXT`` relay variants stay
# colon-free because they forward argv to legacy argparse handlers.
ENERGY_CSV_RANGE = _CSV("energy", lower=0, lower_open=True, ranges=True)
THICKNESS_CSV_RANGE = _CSV("thickness", lower=0, lower_open=True, ranges=True)
TILT_CSV_RANGE = _CSV("tilt", lower=0, upper=90, upper_inclusive=False, ranges=True)
AZIMUTH_CSV_RANGE = _CSV(
    "azimuth", lower=90, upper=270, lower_open=True, upper_inclusive=False, ranges=True
)


def flatten_option_values(values: Sequence[Sequence[float]]) -> list[float] | None:
    """Concatenate repeatable CSV/range option occurrences in argv order."""
    if not values:
        return None
    return [value for occurrence in values for value in occurrence]


def _stdin_is_tty() -> bool:
    stream = click.get_text_stream("stdin")
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and isatty()


def confirm_destructive(yes: bool, prompt: str) -> bool:
    """Authorize a previewed destructive action without prompting automation."""
    if yes:
        return True
    if not _stdin_is_tty():
        emit_result("preview only; re-run with -y/--yes to execute")
        return False
    return click.confirm(prompt, default=False, err=True)


OUTPUT_CHOICES = click.Choice(("table", "json", "wide"), case_sensitive=True)


def _json_selected(_ctx, _param, value: str) -> bool:
    return value == "json"


def output_option(function):
    """Add canonical human/machine output selection plus retired ``--json``."""
    function = click.option(
        "--json",
        cls=RetiredOption,
        dest="json_output",
        replacement="--output json",
        is_flag=True,
        flag_value=True,
    )(function)
    return click.option(
        "-o",
        "--output",
        "json_output",
        type=OUTPUT_CHOICES,
        default="table",
        show_default=True,
        is_eager=True,
        callback=_json_selected,
        help="Output format; only json is a stable automation contract.",
    )(function)


def _remote_target(ctx, param, value):
    if value in (None, "__configured__"):
        return value
    from ..remote.config import validate_remote_target

    try:
        return validate_remote_target(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


def remote_option(function):
    """Add the shared optional remote-target execution modifier."""
    return click.option(
        "-R",
        "--remote",
        "remote_target",
        is_flag=False,
        flag_value="__configured__",
        default=None,
        callback=_remote_target,
        metavar="[TARGET]",
        help="Run remotely; bare uses the configured target, =TARGET overrides it.",
    )(function)


FIDELITY_CHOICES = click.Choice(("full", "survey"), case_sensitive=True)
_DEFAULT_FIDELITY_HELP = "Named settings/grid-reduction policy. survey is provisional and reduced."


def fidelity_option(*, help: str = _DEFAULT_FIDELITY_HELP):
    """Add ``--fidelity`` option to a Click command."""

    def decorator(function):
        return click.option(
            "--fidelity",
            type=FIDELITY_CHOICES,
            default="full",
            show_default=True,
            help=help,
        )(function)

    return decorator


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
    """Invoke Click without framework-owned exits; preserve pyrite exit contract."""
    token = _COLOR_MODE.set("auto")
    try:
        return command.main(
            args=None if argv is None else list(argv),
            prog_name=prog_name,
            standalone_mode=False,
        )
    except click.Abort:
        click.echo(
            paint("Aborted!", "warning", stream=sys.stderr),
            err=True,
            color=color_enabled(sys.stderr),
        )
        return 130
    except click.ClickException as exc:
        stream = click.get_text_stream("stderr")
        context = getattr(exc, "ctx", None)
        color = (
            context.color
            if context is not None and context.color is not None
            else color_enabled(stream)
        )
        if isinstance(exc, click.UsageError) and context is not None:
            help_names = context.command.get_help_option_names(context)
            hint = ""
            if help_names:
                hint = f"Try '{context.command_path} {max(help_names, key=len)}' for help.\n"
            click.echo(f"{context.get_usage()}\n{hint}", file=stream, color=color)
        label = paint("Error:", "failed", stream=stream, enabled=bool(color))
        click.echo(f"{label} {exc.format_message()}", file=stream, color=color)
        return exc.exit_code
    except Exception as exc:
        if (
            type(exc).__name__ == "MaterialConfigError"
            and type(exc).__module__ == "pyrite.materials.catalog"
        ):
            emit_diagnostic(str(exc))
            return 1
        raise
    finally:
        _COLOR_MODE.reset(token)
