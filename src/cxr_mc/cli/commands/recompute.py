"""Click wiring for ``cxr checkpoint recompute {brem,line}``.

The drivers these call live in :mod:`cxr_mc.recompute`; this module owns only
the command surface, per the command-home rule in
`docs/package-structure-rfc.md` P1. The retired top-level ``cxr rebrem`` and
``cxr reline`` spellings resolve to the same commands through the deprecation
registry in :mod:`cxr_mc.cli._deprecations`.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import click

from ... import _checkpoint_store
from ... import recompute as _recompute
from .. import _completion as _cli_completion
from .. import _core as _cli_core
from .. import json as cli_json


def _brem_cli(args):
    """CLI handler -- returns None so the dict never reaches sys.exit."""
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "cxr rebrem: give one or more materials, or -a/--all for every checkpoint "
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
            "cxr rebrem: give one or more materials, or -a/--all for every checkpoint "
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
@_cli_core.output_option
def brem_command(
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
):
    if all_ and materials:
        raise click.UsageError("rebrem --all does not take material names")
    if not all_ and not materials:
        raise click.UsageError("rebrem needs material name(s), or use --all")
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("rebrem --stop must be greater than --start")
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
            "cxr reline: give one or more materials, or -a/--all for every checkpoint "
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
            "cxr reline: give one or more materials, or -a/--all for every checkpoint "
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
@_cli_core.output_option
def line_command(
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
):
    if all_ and materials:
        raise click.UsageError("reline --all does not take material names")
    if not all_ and not materials:
        raise click.UsageError("reline needs material name(s), or use --all")
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("reline --stop must be greater than --start")
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
