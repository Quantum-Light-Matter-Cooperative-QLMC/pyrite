"""CLI options that need a driver package the shared primitives cannot import.

`console/output.py` sits below every domain package, so it cannot validate a
remote target: that lives in `remote/`. The `-R/--remote` modifier therefore
lives here, in `cli/`, which is allowed to reach down into `remote`
(issue #64, finding 2).
"""

from __future__ import annotations

import click

from ..remote.config import validate_remote_target


def _remote_target(ctx, param, value):
    if value in (None, "__configured__"):
        return value
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
