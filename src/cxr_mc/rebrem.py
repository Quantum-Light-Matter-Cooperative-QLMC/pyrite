"""``cxr rebrem`` -- recompute checkpoint bremsstrahlung with new parameters.

Rewrites ONLY the brem portion (``brem_wide``, ``brem``, ``E_grid_brem`` and the
case's ``Ne_brem`` / ``E_grid_brem``) of existing per-material checkpoints; the
expensive line ``spec`` is untouched, so this is orders of magnitude cheaper
than re-running the sweep. Typical use: bump ``Ne_brem`` (brem noise falls as
1/sqrt(Ne_brem) -- the 100-electron default gives a visibly quantized endpoint
staircase) or refine the wide-grid spacing, without invalidating the cached
line spectra.

    cxr rebrem MoS2 --ne-brem 1000          # one material
    cxr rebrem --all --ne-brem 1000         # every checkpoints/*.pkl
    cxr rebrem MoS2 W --ne-brem 1000 --step 25

Runs wherever the pickles live: locally on ``checkpoints/``, on the GPU box
via ``cxr remote rebrem`` (submits a SLURM job with the same ``status``/
``attach`` progress dashboard as a sweep, then pulls the updated checkpoints),
or by hand over ssh. Records already at the target parameters are skipped, so a
crashed/OOM'd run just resumes on re-invocation; ``--redo-all`` forces a full
recompute (e.g. same parameters, new physics).
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


def rebrem_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    ne_brem=None,
    brem_start_eV=None,
    brem_stop_eV=None,
    brem_step_eV=None,
    fidelity=None,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
    summary_status=None,
):
    """Run :func:`cxr_mc.run.repair_checkpoint` with new brem parameters over one
    or more materials. ``materials=None`` sweeps every ``*.pkl`` in
    ``checkpoint_dir`` (grooved checkpoints included; ``*.slim.pkl`` transfer
    copies excluded). Returns ``{stem: results}``.

    ``progress_file`` (single material only) atomically maintains the same
    compact JSON progress record a remote scan writes (see
    ``scan._write_progress_record``), so the remote rebrem queue feeds the
    ``cxr remote status``/``attach`` case-progress dashboard: repaired records
    count as new cases, already-at-target records as cached.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    from .run import checkpoint_path_for, repair_checkpoint

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
        resolved_ne = ne_brem
        resolved_start = brem_start_eV
        resolved_step = brem_step_eV
        if fidelity is not None:
            from .recompute_defaults import settings, sweep, uniform_bounds

            profile_settings = settings(fidelity)
            if resolved_ne is None:
                resolved_ne = profile_settings.n_electrons_brem
            try:
                profile_sweep = sweep(stem, fidelity)
                profile_start, _profile_stop, profile_step = uniform_bounds(
                    profile_sweep.E_grid_brem
                )
            except (KeyError, TypeError, ValueError):
                # Derived stems (for example ``*_blazed``) have no catalog row.
                # Keep their stored grid while still applying profile Ne/provenance.
                pass
            else:
                if resolved_start is None:
                    resolved_start = profile_start
                if resolved_step is None:
                    resolved_step = profile_step
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
        try:
            out[stem] = repair_checkpoint(
                path,
                save_every=save_every,
                only_nonfinite=not redo_all,
                ne_brem=resolved_ne,
                brem_start_eV=resolved_start,
                brem_stop_eV=brem_stop_eV,
                brem_step_eV=resolved_step,
                profile=fidelity,
                max_seconds=max_seconds,
                status=status,
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
            "cxr rebrem: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    return rebrem_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        ne_brem=args.ne_brem,
        brem_start_eV=getattr(args, "start", None),
        brem_stop_eV=getattr(args, "stop", None),
        brem_step_eV=args.step,
        fidelity=getattr(args, "fidelity", None),
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


def _cli_json(args):
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
            rebrem_checkpoints(
                materials=args.material or None,
                checkpoint_dir=args.checkpoint_dir,
                ne_brem=args.ne_brem,
                brem_start_eV=getattr(args, "start", None),
                brem_stop_eV=getattr(args, "stop", None),
                brem_step_eV=args.step,
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
@_cli_core.fidelity_option(help="Named sweep profile supplying omitted grid and electron defaults.")

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
@_cli_core.json_option
def command(
    materials,
    all_,
    fidelity,
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
    handler = _cli_json if json_output else _cli
    return _cli_core.invoke_legacy(
        handler,
        material=list(materials),
        all=all_,
        fidelity=fidelity,
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
