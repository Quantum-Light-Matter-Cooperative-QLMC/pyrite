"""Click wiring for ``pyrite checkpoint recompute {brem,line}``.

The drivers these call live in :mod:`pyrite.checkpoints.recompute`; this module owns only
the command surface, per the command-home rule in
`docs/adr/0004-package-and-repository-structure.md` P1. The retired top-level ``pyrite rebrem`` and
``pyrite reline`` spellings resolve to the same commands through the deprecation
registry in :mod:`pyrite.cli._deprecations`.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import click

from ...checkpoints import _checkpoint_store
from ...checkpoints import recompute as _recompute
from .. import _completion as _cli_completion
from .. import _core as _cli_core
from .. import json as cli_json


def _brem_cli(args):
    """CLI handler -- returns None so the dict never reaches sys.exit."""
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "pyrite rebrem: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    return _recompute.rebrem_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        ne_brem=args.ne_brem,
        brem_start_eV=getattr(args, "start", None),
        brem_stop_eV=getattr(args, "stop", None),
        brem_step_eV=args.step,
        fidelity=getattr(args, "fidelity", None),
        catalog_profile=getattr(args, "catalog_profile", None),
        require_identity=True,
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


def _brem_cli_json(args):
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "pyrite rebrem: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    started = time.monotonic()
    if args.material:
        requested = list(args.material)
    else:
        requested = _checkpoint_store.discover(args.checkpoint_dir)
    statuses = {}
    caught = None
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _recompute.rebrem_checkpoints(
                materials=args.material or None,
                checkpoint_dir=args.checkpoint_dir,
                ne_brem=args.ne_brem,
                brem_start_eV=getattr(args, "start", None),
                brem_stop_eV=getattr(args, "stop", None),
                brem_step_eV=args.step,
                fidelity=getattr(args, "fidelity", None),
                catalog_profile=getattr(args, "catalog_profile", None),
                require_identity=True,
                redo_all=args.redo_all,
                save_every=args.save_every,
                progress_file=args.progress_file,
                max_minutes=args.max_minutes,
                summary_status=statuses,
            )
    except (Exception, SystemExit) as exc:
        caught = exc
    completed = [
        material
        for material in requested
        if material in statuses and statuses[material].get("complete", True)
    ]
    failed = [material for material in requested if material not in completed]
    resumable = isinstance(caught, SystemExit) and caught.code == 75
    message = (
        "resumable work remains"
        if resumable
        else (str(caught) or type(caught).__name__ if caught is not None else "operation failed")
    )
    errors = {material: message for material in failed}
    result = cli_json.operation_summary(
        "rebrem",
        requested,
        completed,
        failed_materials=failed,
        checkpoints=[Path(args.checkpoint_dir) / material for material in requested],
        elapsed_seconds=time.monotonic() - started,
        resumable=resumable,
        material_errors=errors,
    )
    _cli_core.emit_json_result(result, failure_exit=75 if resumable else 1)


def _remote_controls(function):
    function = click.option(
        "--chunk-minutes",
        type=_cli_core.NONNEGATIVE_FLOAT,
        default=None,
        metavar="MINUTES",
        help="Remote self-resubmitting slice length; 0 uses one monolithic job.",
    )(function)
    function = click.option("--no-sync", is_flag=True, help="Skip remote code upload.")(function)
    function = click.option(
        "--dry-run", is_flag=True, help="Preview remote submission; do not connect."
    )(function)
    function = click.option("--wait", is_flag=True, help="Wait and pull remote results.")(function)
    function = click.option("--detach", is_flag=True, help="Return after remote submission.")(
        function
    )
    return _cli_core.remote_option(function)


def _remote_requested(
    ctx: click.Context,
    *,
    remote_target: str | None,
    wait: bool,
    detach: bool,
    chunk_minutes: float | None,
    no_sync: bool,
    dry_run: bool,
) -> bool:
    if wait and detach:
        raise click.UsageError("--wait and --detach are mutually exclusive")
    remote_only = {
        "wait": "--wait",
        "detach": "--detach",
        "chunk_minutes": "--chunk-minutes",
        "no_sync": "--no-sync",
        "dry_run": "--dry-run",
    }
    if remote_target is None:
        explicit = [
            flag
            for parameter, flag in remote_only.items()
            if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
        ]
        if explicit:
            raise click.UsageError(
                f"remote-only option(s) require -R/--remote: {', '.join(explicit)}"
            )
        return False
    return True


def _reject_remote_local_options(ctx: click.Context, json_output: bool) -> None:
    if json_output:
        raise click.UsageError("remote recompute does not yet support --output json")
    local_only = {
        "catalog_profile": "--profile",
        "checkpoint_dir": "--checkpoint-dir",
        "progress_file": "--progress-file",
        "max_minutes": "--max-minutes",
        "save_every": "--save-every",
    }
    explicit = [
        flag
        for parameter, flag in local_only.items()
        if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
    ]
    if explicit:
        raise click.UsageError(
            f"remote recompute does not support local-only option(s): {', '.join(explicit)}"
        )


@click.command(
    "rebrem",
    help=(
        "Recompute only brem backgrounds in existing checkpoints.\n\n"
        "Pass MATERIALS or --all, never both. Updates checkpoint files in place "
        "and skips records already at target unless --redo-all."
    ),
)
@click.argument(
    "materials",
    nargs=-1,
    shell_complete=_cli_completion.complete_checkpoint_stem,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Recompute every checkpoint.")
@click.option(
    "--fidelity",
    type=_cli_core.FIDELITY_CHOICES,
    default=None,
    help="Override dataset fidelity; defaults to checkpoint metadata or full for legacy data.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Catalog profile for legacy data; otherwise must match checkpoint metadata.",
)
@click.option(
    "--ne-brem",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Bremsstrahlung electron count; overrides profile default.",
)
@click.option(
    "--start",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Wide-bremsstrahlung lower bound in eV; overrides profile.",
)
@click.option(
    "--stop",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Wide-bremsstrahlung exclusive upper bound in eV; default follows beam energy.",
)
@click.option(
    "--step",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Wide-bremsstrahlung grid spacing in eV; overrides profile default.",
)
@click.option("--redo-all", is_flag=True, help="Recompute records already at target.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Root containing component checkpoint directories to update.",
)
@click.option("--progress-file", default=None, hidden=True)
@click.option("--max-minutes", type=_cli_core.POSITIVE_FLOAT, default=None, hidden=True)
@click.option(
    "--save-every",
    type=_cli_core.POSITIVE_INT,
    default=100,
    show_default=True,
    metavar="N",
    help="Atomically save after every N recomputed records.",
)
@_remote_controls
@click.pass_context
@_cli_core.output_option
def brem_command(
    ctx,
    materials,
    all_,
    fidelity,
    catalog_profile,
    ne_brem,
    start,
    stop,
    step,
    redo_all,
    checkpoint_dir,
    progress_file,
    max_minutes,
    save_every,
    json_output,
    remote_target,
    wait,
    detach,
    chunk_minutes,
    no_sync,
    dry_run,
):
    if all_ and materials:
        raise click.UsageError("rebrem --all does not take material names")
    if not all_ and not materials:
        raise click.UsageError("rebrem needs material name(s), or use --all")
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("rebrem --stop must be greater than --start")
    if _remote_requested(
        ctx,
        remote_target=remote_target,
        wait=wait,
        detach=detach,
        chunk_minutes=chunk_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
    ):
        _reject_remote_local_options(ctx, json_output)
        from ...remote import cli as remote_cli
        from ...remote import config as remote_config

        target = None if remote_target == "__configured__" else remote_target
        with remote_config.override_remote_host(target):
            return ctx.invoke(
                remote_cli.rebrem_command,
                material=materials,
                all_=all_,
                fidelity=fidelity or "full",
                redo_all=redo_all,
                dry_run=dry_run,
                no_sync=no_sync,
                chunk_minutes=10.0 if chunk_minutes is None else chunk_minutes,
                ne_brem=ne_brem,
                start=start,
                stop=stop,
                step=step,
                detach=detach,
            )
    handler = _brem_cli_json if json_output else _brem_cli
    return _cli_core.invoke_legacy(
        handler,
        material=list(materials),
        all=all_,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        ne_brem=ne_brem,
        start=start,
        stop=stop,
        step=step,
        redo_all=redo_all,
        checkpoint_dir=checkpoint_dir,
        progress_file=progress_file,
        max_minutes=max_minutes,
        save_every=save_every,
        json_output=json_output,
    )


def _line_cli(args):
    """CLI handler -- returns None so the dict never reaches sys.exit."""
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "pyrite reline: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    return _recompute.reline_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        line_ne=args.line_ne,
        line_start_eV=getattr(args, "start", None),
        line_stop_eV=getattr(args, "stop", None),
        line_step_eV=args.line_step,
        fidelity=getattr(args, "fidelity", None),
        catalog_profile=getattr(args, "catalog_profile", None),
        require_identity=True,
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


def _line_cli_json(args):
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "pyrite reline: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    started = time.monotonic()
    if args.material:
        requested = list(args.material)
    else:
        requested = _checkpoint_store.discover(args.checkpoint_dir)
    statuses = {}
    caught = None
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _recompute.reline_checkpoints(
                materials=args.material or None,
                checkpoint_dir=args.checkpoint_dir,
                line_ne=args.line_ne,
                line_start_eV=getattr(args, "start", None),
                line_stop_eV=getattr(args, "stop", None),
                line_step_eV=args.line_step,
                fidelity=getattr(args, "fidelity", None),
                catalog_profile=getattr(args, "catalog_profile", None),
                require_identity=True,
                redo_all=args.redo_all,
                save_every=args.save_every,
                progress_file=args.progress_file,
                max_minutes=args.max_minutes,
                summary_status=statuses,
            )
    except (Exception, SystemExit) as exc:
        caught = exc
    completed = [
        material
        for material in requested
        if material in statuses and statuses[material].get("complete", True)
    ]
    failed = [material for material in requested if material not in completed]
    resumable = isinstance(caught, SystemExit) and caught.code == 75
    message = (
        "resumable work remains"
        if resumable
        else (str(caught) or type(caught).__name__ if caught is not None else "operation failed")
    )
    errors = {material: message for material in failed}
    result = cli_json.operation_summary(
        "reline",
        requested,
        completed,
        failed_materials=failed,
        checkpoints=[Path(args.checkpoint_dir) / material for material in requested],
        elapsed_seconds=time.monotonic() - started,
        resumable=resumable,
        material_errors=errors,
    )
    _cli_core.emit_json_result(result, failure_exit=75 if resumable else 1)


@click.command(
    "reline",
    help=(
        "Recompute only line spectra in existing checkpoints.\n\n"
        "Pass MATERIALS or --all, never both. Updates checkpoint files in place "
        "and skips records already at target unless --redo-all."
    ),
)
@click.argument(
    "materials",
    nargs=-1,
    shell_complete=_cli_completion.complete_checkpoint_stem,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Recompute every checkpoint.")
@click.option(
    "--fidelity",
    type=_cli_core.FIDELITY_CHOICES,
    default=None,
    help="Override dataset fidelity; defaults to checkpoint metadata or full for legacy data.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Catalog profile for legacy data; otherwise must match checkpoint metadata.",
)
@click.option(
    "--line-ne",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Line-spectrum electron count; overrides profile default.",
)
@click.option(
    "--start",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Line-grid lower bound in eV; overrides profile.",
)
@click.option(
    "--stop",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Line-grid exclusive upper bound in eV; overrides profile.",
)
@click.option(
    "--line-step",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Uniform line-grid spacing in eV; overrides profile grid.",
)
@click.option("--redo-all", is_flag=True, help="Recompute records already at target.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Root containing component checkpoint directories to update.",
)
@click.option("--progress-file", default=None, hidden=True)
@click.option("--max-minutes", type=_cli_core.POSITIVE_FLOAT, default=None, hidden=True)
@click.option(
    "--save-every",
    type=_cli_core.POSITIVE_INT,
    default=100,
    show_default=True,
    metavar="N",
    help="Atomically save after every N recomputed records.",
)
@_remote_controls
@click.pass_context
@_cli_core.output_option
def line_command(
    ctx,
    materials,
    all_,
    fidelity,
    catalog_profile,
    line_ne,
    start,
    stop,
    line_step,
    redo_all,
    checkpoint_dir,
    progress_file,
    max_minutes,
    save_every,
    json_output,
    remote_target,
    wait,
    detach,
    chunk_minutes,
    no_sync,
    dry_run,
):
    if all_ and materials:
        raise click.UsageError("reline --all does not take material names")
    if not all_ and not materials:
        raise click.UsageError("reline needs material name(s), or use --all")
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("reline --stop must be greater than --start")
    if _remote_requested(
        ctx,
        remote_target=remote_target,
        wait=wait,
        detach=detach,
        chunk_minutes=chunk_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
    ):
        _reject_remote_local_options(ctx, json_output)
        from ...remote import cli as remote_cli
        from ...remote import config as remote_config

        target = None if remote_target == "__configured__" else remote_target
        with remote_config.override_remote_host(target):
            return ctx.invoke(
                remote_cli.reline_command,
                material=materials,
                all_=all_,
                fidelity=fidelity or "full",
                redo_all=redo_all,
                dry_run=dry_run,
                no_sync=no_sync,
                chunk_minutes=10.0 if chunk_minutes is None else chunk_minutes,
                line_ne=line_ne,
                start=start,
                stop=stop,
                line_step=line_step,
                detach=detach,
            )
    handler = _line_cli_json if json_output else _line_cli
    return _cli_core.invoke_legacy(
        handler,
        material=list(materials),
        all=all_,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        line_ne=line_ne,
        start=start,
        stop=stop,
        line_step=line_step,
        redo_all=redo_all,
        checkpoint_dir=checkpoint_dir,
        progress_file=progress_file,
        max_minutes=max_minutes,
        save_every=save_every,
        json_output=json_output,
    )
