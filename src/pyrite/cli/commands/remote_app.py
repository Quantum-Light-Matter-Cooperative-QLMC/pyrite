"""Remote SLURM marimo apps, with automatic authenticated SSH forwarding."""

import sys

import click

from ...remote import app as remote_app
from .._options import target_commands
from . import app_analysis, app_validation, app_viewer


@click.group("app", no_args_is_help=True)
def command() -> None:
    """Submit analysis/viewer/validation as SLURM apps, or reattach."""


def _options(function):
    function = click.option("--no-token", is_flag=True, help="Forbidden in remote mode.")(function)
    function = click.option("--batch", is_flag=True, help="Export HTML and pull it locally.")(
        function
    )
    function = click.option(
        "--time",
        "walltime",
        default="04:00:00",
        show_default=True,
        help="SLURM walltime (HH:MM:SS).",
    )(function)
    function = click.option("--gpu", is_flag=True, help="Request the configured GPU gres.")(
        function
    )
    return click.option("--edit", is_flag=True, help="Use marimo edit, not run.")(function)


def _invocation(app: str, material: str | None, *, edit: bool, batch: bool):
    """Use each app's typed local builder for its marimo arguments.

    Validation has no material parameter, unlike analysis and viewer. Calling a
    runtime-selected module's ``_command`` loses that distinction for ty: the
    inferred union includes validation's keyword-only signature.
    """
    if app == "analysis":
        argv = (
            app_analysis._smoke_command(material or "hopg", "app.html")
            if batch
            else app_analysis._command(material or "hopg", edit=edit)
        )
        return argv, app_analysis.NOTEBOOK, app_analysis.TUNNEL_PORT
    if app == "viewer":
        argv = (
            app_viewer._smoke_command(material or "hopg", "app.html")
            if batch
            else app_viewer._command(material or "hopg", edit=edit)
        )
        return argv, app_viewer.NOTEBOOK, app_viewer.TUNNEL_PORT
    if app == "validation":
        argv = (
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                app_validation.NOTEBOOK,
                "--output",
                "app.html",
                "--force",
            ]
            if batch
            else app_validation._command(edit=edit)
        )
        return argv, app_validation.NOTEBOOK, app_validation.TUNNEL_PORT
    raise ValueError(f"unknown remote app: {app}")


def _launch(
    app: str,
    material: str | None,
    edit: bool,
    gpu: bool,
    walltime: str,
    batch: bool,
    no_token: bool,
) -> None:
    # Validate locally, before sync, SSH, or sbatch: no insecure fallback.
    if no_token:
        raise click.UsageError("--no-token is forbidden for remote apps")
    if batch and edit:
        raise click.UsageError("--batch and --edit cannot be combined")
    try:
        argv, notebook, port = _invocation(app, material, edit=edit, batch=batch)
        jobid = remote_app.submit(
            app,
            material,
            argv=argv,
            notebook=notebook,
            base_port=port,
            edit=edit,
            gpu=gpu,
            walltime=walltime,
            batch=batch,
        )
        remote_app.attach(jobid)
    except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@command.command("analysis")
@click.option("-m", "--material", default="hopg", show_default=True, help="Initial material.")
@_options
def analysis_command(
    material: str, edit: bool, gpu: bool, walltime: str, batch: bool, no_token: bool
) -> None:
    """Run the analysis app remotely."""
    _launch("analysis", material, edit, gpu, walltime, batch, no_token)


@command.command("viewer")
@click.option("-m", "--material", default="hopg", show_default=True, help="Initial material.")
@_options
def viewer_command(
    material: str, edit: bool, gpu: bool, walltime: str, batch: bool, no_token: bool
) -> None:
    """Run the 3D viewer remotely."""
    _launch("viewer", material, edit, gpu, walltime, batch, no_token)


@command.command("validation")
@_options
def validation_command(edit: bool, gpu: bool, walltime: str, batch: bool, no_token: bool) -> None:
    """Run the validation app remotely."""
    _launch("validation", None, edit, gpu, walltime, batch, no_token)


@command.command("attach")
@click.argument("jobid")
def attach_command(jobid: str) -> None:
    """Reopen an app tunnel, or pull a completed batch export."""
    try:
        remote_app.attach(jobid)
    except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


target_commands(command)
