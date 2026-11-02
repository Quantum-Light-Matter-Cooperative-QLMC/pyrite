"""CLI options that need a driver package the shared primitives cannot import.

`console/output.py` sits below every domain package, so it cannot validate a
remote target: that lives in `remote/`. The `-R/--remote` modifier therefore
lives here, in `cli/`, which is allowed to reach down into `remote`
(issue #64, finding 2).
"""

from functools import wraps

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
    return target_option(
        click.option(
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
    )


def target_option(function):
    """Scope named-target selection to one complete command invocation."""

    @wraps(function)
    def invoke(*args, **kwargs):
        from ..remote.targets import selected

        target = kwargs.pop("target", None)
        if target is not None and "remote_target" in kwargs:
            if kwargs["remote_target"] is None:
                raise click.UsageError("--target requires --remote")
            if kwargs["remote_target"] != "__configured__":
                raise click.UsageError(
                    "use --remote --target NAME, not --remote=HOST with --target"
                )
        try:
            with selected(target, jobid=kwargs.get("jobid")):
                return function(*args, **kwargs)
        except (ValueError, TypeError) as exc:
            raise click.BadParameter(str(exc), param_hint="--target") from exc

    return click.option(
        "--target", default=None, metavar="NAME", help="Use named remote compute target NAME."
    )(invoke)


def target_commands(group):
    """Add target selection to each remote-contacting leaf command."""
    for command in group.commands.values():
        if isinstance(command, click.Group):
            target_commands(command)
        else:
            # Applying an option to a Command adds its parameter directly.
            callback = target_option(command.callback)
            command.params.extend(callback.__click_params__[-1:])
            command.callback = callback
