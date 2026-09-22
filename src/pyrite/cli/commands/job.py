"""Canonical lifecycle for every remote asynchronous job."""

import click

from ...console.output import confirm_destructive, emit_result, output_option
from ...remote import lifecycle, viewer
from .. import _completion as _cli_completion
from . import remote as remote_cli
from ._remote_actions import _cli_jobs, _cli_logs, _cli_status


@click.group("job", no_args_is_help=True)
def command() -> None:
    """List, inspect, follow, or stop asynchronous remote jobs."""
    remote_cli._ensure_utf8_stdio()


@command.command("list", help="List jobs with scheduler IDs, kinds, materials, and states.")
@click.option(
    "--kind",
    type=click.Choice(("run", "grid", "recompute", "validate")),
    default=None,
    help="Show only one submission kind.",
)
@output_option
def list_command(kind: str | None, json_output: bool):
    return remote_cli._invoke_action(
        _cli_jobs,
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
@output_option
def status_command(jobid: str | None, verbose: int, json_output: bool):
    return remote_cli._invoke_action(
        _cli_status,
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
    return remote_cli._invoke_action(
        _cli_logs,
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
