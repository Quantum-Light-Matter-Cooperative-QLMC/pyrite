"""User configuration CRUD for remote compute destinations."""

import json
from dataclasses import asdict

import click

from ...console.output import confirm_destructive, emit_diagnostic, emit_result, output_option
from ...remote import targets


@click.group("target", no_args_is_help=True)
def command():
    """Configure named SSH/SLURM destinations in your user config."""


def _show(value, json_output):
    emit_result(json.dumps(value) if json_output else json.dumps(value, indent=2))


@command.command("list")
@output_option
def list_command(json_output):
    """List configured targets."""
    _show({name: asdict(value) for name, value in targets.named_targets().items()}, json_output)


@command.command("show")
@click.argument("name")
@output_option
def show_command(name, json_output):
    """Show one resolved named target."""
    _show(asdict(targets.resolve_target(name)), json_output)


def _fields(function):
    for field in reversed(targets._FIELDS):
        function = click.option(f"--{field.replace('_', '-')}", default=None)(function)
    return function


def _save(name, values, edit):
    target = targets.save_target(
        name, {k: v for k, v in values.items() if v is not None}, edit=edit
    )
    for other in targets.named_targets().values():
        if (
            other.name != name
            and (other.host, other.remote_dir) == (target.host, target.remote_dir)
            and other.gpu_vendor != target.gpu_vendor
        ):
            emit_diagnostic(
                f"warning: {name} and {other.name} share host and remote_dir with different GPU vendors; use separate remote directories"
            )
    emit_result(f"saved remote target {name}")


@command.command("add")
@click.argument("name")
@_fields
def add_command(name, **values):
    """Add a target; --host is required. Other fields use built-in defaults."""
    if values["host"] is None:
        raise click.UsageError("target add requires --host HOST")
    _save(name, values, False)


@command.command("edit")
@click.argument("name")
@_fields
def edit_command(name, **values):
    """Update only the supplied fields of an existing target."""
    _save(name, values, True)


@command.command("remove")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Delete the previewed target.")
def remove_command(name, yes):
    """Preview or remove a target; removing the default clears its selection."""
    targets.resolve_target(name)
    emit_result(f"would remove remote target: {name}")
    if confirm_destructive(yes, "Remove this remote target?"):
        targets.remove_target(name)


@command.command("use")
@click.argument("name")
def use_command(name):
    """Select the default named target."""
    targets.default_target(name)
    emit_result(f"default remote target: {name}")


def _errors(function):
    from functools import wraps

    @wraps(function)
    def invoke(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError) as exc:
            raise click.BadParameter(str(exc)) from exc

    return invoke


for _command in command.commands.values():
    _command.callback = _errors(_command.callback)
