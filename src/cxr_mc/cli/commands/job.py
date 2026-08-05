"""Canonical lifecycle for every remote asynchronous job."""

from __future__ import annotations

import click

from ..._remote import cli as remote_cli
from ..._remote import lifecycle, viewer
from .. import _completion as _cli_completion
from .._core import confirm_destructive, emit_result


def _invoke(handler, **values):
    """Run an existing remote handler through the shared Click exit contract."""
    return remote_cli._invoke_click(handler, values)


@click.group("job", no_args_is_help=True)
def command() -> None:
    """List, inspect, follow, or stop asynchronous remote jobs."""


@command.command("list", help="List jobs with scheduler IDs, kinds, materials, and states.")
@click.option(
    "--kind",
    type=click.Choice(("run", "grid", "recompute", "validate")),
    default=None,
    help="Show only one submission kind.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit one versioned JSON object.")
def list_command(kind: str | None, json_output: bool):
    return _invoke(
        remote_cli._cli_jobs,
        remote_command="list",
        kind=kind,
        json_output=json_output,
    )


@command.command("status", help="Show one job snapshot; JOBID defaults to latest.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for case progress and recent logs.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit one versioned JSON object.")
def status_command(jobid: str | None, verbose: int, json_output: bool):
    return _invoke(
        remote_cli._cli_status,
        remote_command="status",
        jobid=jobid,
        verbose=verbose,
        attach=False,
        json_output=json_output,
    )


@command.command("logs", help="Show a job diagnostic log; JOBID defaults to latest.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option("-f", "--follow", is_flag=True, help="Stream until interrupted.")
def logs_command(jobid: str | None, follow: bool):
    return _invoke(
        remote_cli._cli_logs,
        remote_command="logs",
        jobid=jobid,
        follow=follow,
    )


@command.command("attach", help="Monitor one job until terminal or interrupted.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for case progress and recent logs.",
)
def attach_command(jobid: str | None, verbose: int):
    viewer.attach(jobid, verbose)


@command.command("stop", help="Preview or stop one job, one profile's jobs, or all jobs.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Stop every live job.")
@click.option(
    "--profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Stop live jobs submitted for profile NAME.",
)
@click.option("-y", "--yes", is_flag=True, help="Stop exact previewed jobs.")
def stop_command(jobid: str | None, all_: bool, profile: str | None, yes: bool):
    selectors = int(jobid is not None) + int(all_) + int(profile is not None)
    if selectors != 1:
        raise click.UsageError("job stop needs exactly one of JOBID, --profile NAME, or --all")
    if jobid is not None:
        emit_result(f"would cancel remote job: {jobid}")
        if not confirm_destructive(yes, "Cancel this remote job?"):
            return
        lifecycle._stop_jobid(jobid)
        return
    lifecycle.stop_jobs(all_jobs=all_, profile=profile, yes=yes)
