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

import time
from pathlib import Path

import click

from . import _cli_core


def rebrem_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    ne_brem=None,
    brem_step_eV=None,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
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
        paths = sorted(str(p) for p in ckpt_dir.glob("*.pkl") if not p.name.endswith(".slim.pkl"))
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    if progress_file is not None and len(paths) != 1:
        raise SystemExit("progress_file needs exactly one material")
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60
    incomplete = False
    out = {}
    for path in paths:
        stem = Path(path).stem
        print(f"== {stem} ==")
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
                ne_brem=ne_brem,
                brem_step_eV=brem_step_eV,
                max_seconds=max_seconds,
                status=status,
                **kw,
            )
        except BaseException:
            if progress_file is not None:
                _write_progress_record(progress_file, material=stem, state="failed", **latest)
            raise
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
    rebrem_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        ne_brem=args.ne_brem,
        brem_step_eV=args.step,
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


@click.command(
    "rebrem",
    help=(
        "Recompute only brem backgrounds in existing checkpoints.\n\n"
        "Pass MATERIALS or --all, never both. Updates checkpoint files in place "
        "and skips records already at target unless --redo-all."
    ),
)
@click.argument("materials", nargs=-1)
@click.option("-a", "--all", "all_", is_flag=True, help="Recompute every checkpoint.")
@click.option(
    "--ne-brem",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Bremsstrahlung electron count; omit to use checkpoint settings.",
)
@click.option(
    "--step",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="EV",
    help="Wide-bremsstrahlung grid spacing in eV; omit to use checkpoint settings.",
)
@click.option("--redo-all", is_flag=True, help="Recompute records already at target.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Directory containing checkpoint pickles to update.",
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
def command(
    materials,
    all_,
    ne_brem,
    step,
    redo_all,
    checkpoint_dir,
    progress_file,
    max_minutes,
    save_every,
):
    return _cli_core.invoke_legacy(
        _cli,
        material=list(materials),
        all=all_,
        ne_brem=ne_brem,
        step=step,
        redo_all=redo_all,
        checkpoint_dir=checkpoint_dir,
        progress_file=progress_file,
        max_minutes=max_minutes,
        save_every=save_every,
    )
