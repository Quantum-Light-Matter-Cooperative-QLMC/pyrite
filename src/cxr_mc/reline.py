"""``cxr reline`` -- recompute checkpoint line spectra with new parameters.

Rewrites ONLY the line ``spec`` (and ``E_grid``) of existing per-material
checkpoints; the brem background (``brem_wide``/``E_grid_brem``) is untouched
(only re-interpolated onto the new line grid). Mirror of ``cxr rebrem``: use it
to re-run coherent lines on new bespoke ``E_grid_line_by_energy`` bounds, or to
bump the line electron count, without recomputing brem.

    cxr reline MoS2                     # re-run lines on the current-config grid
    cxr reline MoS2 --line-ne 40000     # bump line Ne, same grid
    cxr reline --all --line-step 5      # explicit 5 eV line grid, every checkpoint

Runs locally on ``checkpoints/``, on the GPU box via ``cxr remote reline``, or
by hand over ssh. Records already at the target grid + Ne are skipped, so a
crashed/OOM run resumes; ``--redo-all`` forces a full recompute.
"""

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import click

from . import _checkpoint_store
from .cli import _completion as _cli_completion
from .cli import _core as _cli_core
from .cli import json as cli_json


def reline_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    line_ne=None,
    line_start_eV=None,
    line_stop_eV=None,
    line_step_eV=None,
    fidelity=None,
    from_config=True,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
    summary_status=None,
):
    """Run :func:`cxr_mc.run.reline_checkpoint` over one or more materials.
    ``materials=None`` sweeps every ``*.pkl`` (excluding ``*.slim.pkl``).
    Returns ``{stem: results}``. ``progress_file`` (single material only) writes
    the same compact JSON progress record ``cxr rebrem`` does, feeding the remote
    status/attach dashboard.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    from .run import checkpoint_path_for, reline_checkpoint

    ckpt_dir = Path(checkpoint_dir)
    if materials:
        paths = [checkpoint_path_for(m, str(ckpt_dir)) for m in materials]
    else:
        paths = [
            checkpoint_path_for(stem, str(ckpt_dir))
            for stem in _checkpoint_store.discover(ckpt_dir)
        ]
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    if progress_file is not None and len(paths) != 1:
        raise SystemExit("progress_file needs exactly one material")
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60
    incomplete = False
    out = {}
    for path in paths:
        stem = Path(path).stem if str(path).endswith(".pkl") else Path(path).name
        print(f"== {stem} ==")
        resolved_ne = line_ne
        if fidelity is not None and resolved_ne is None:
            from .recompute_defaults import settings

            resolved_ne = settings(fidelity).n_electrons
        kw = {}
        if progress_file is not None:
            from .scan import _write_progress_record

            latest = {"total_cases": 0, "cached_cases": 0, "completed_new_cases": 0}

            def _on_progress(done, todo_total, skipped, _latest=latest, _stem=stem):
                _latest.update(
                    total_cases=todo_total + skipped,
                    cached_cases=skipped,
                    completed_new_cases=done,
                )
                _write_progress_record(progress_file, material=_stem, state="running", **_latest)

            kw["on_progress"] = _on_progress
            _write_progress_record(progress_file, material=stem, state="running", **latest)
        max_seconds = None if deadline is None else max(0.0, deadline - time.monotonic())
        status = {}
        recompute_options = {
            "line_ne": resolved_ne,
            "line_step_eV": line_step_eV,
            "from_config": from_config,
            "redo_all": redo_all,
        }
        if line_start_eV is not None:
            recompute_options["line_start_eV"] = line_start_eV
        if line_stop_eV is not None:
            recompute_options["line_stop_eV"] = line_stop_eV
        if fidelity is not None:
            recompute_options["profile"] = fidelity
        try:
            out[stem] = reline_checkpoint(
                path,
                material=stem,
                save_every=save_every,
                max_seconds=max_seconds,
                status=status,
                **recompute_options,
                **kw,
            )
        except BaseException as exc:
            status["complete"] = False
            status["error"] = str(exc) or type(exc).__name__
            if progress_file is not None:
                _write_progress_record(progress_file, material=stem, state="failed", **latest)
            raise
        finally:
            if summary_status is not None:
                summary_status[stem] = dict(status)
        if progress_file is not None:
            _write_progress_record(progress_file, material=stem, state="done", **latest)
        if not status.get("complete", True):
            incomplete = True
    if incomplete:
        raise SystemExit(75)
    return out


def _cli(args):
    """CLI handler -- returns None so the dict never reaches sys.exit."""
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "cxr reline: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    return reline_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        line_ne=args.line_ne,
        line_start_eV=getattr(args, "start", None),
        line_stop_eV=getattr(args, "stop", None),
        line_step_eV=args.line_step,
        fidelity=getattr(args, "fidelity", None),
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


def _cli_json(args):
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
            reline_checkpoints(
                materials=args.material or None,
                checkpoint_dir=args.checkpoint_dir,
                line_ne=args.line_ne,
                line_start_eV=getattr(args, "start", None),
                line_stop_eV=getattr(args, "stop", None),
                line_step_eV=args.line_step,
                fidelity=getattr(args, "fidelity", None),
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
@_cli_core.fidelity_option(help="Fidelity preset supplying omitted grid and electron defaults.")
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
@_cli_core.json_option
def command(
    materials,
    all_,
    fidelity,
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
    handler = _cli_json if json_output else _cli
    return _cli_core.invoke_legacy(
        handler,
        material=list(materials),
        all=all_,
        fidelity=fidelity,
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
