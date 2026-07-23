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

import argparse
import time
from pathlib import Path


def reline_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    line_ne=None,
    line_step_eV=None,
    from_config=True,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
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
            out[stem] = reline_checkpoint(
                path,
                material=stem,
                save_every=save_every,
                line_ne=line_ne,
                line_step_eV=line_step_eV,
                from_config=from_config,
                redo_all=redo_all,
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
            "cxr reline: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    reline_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        line_ne=args.line_ne,
        line_step_eV=args.line_step,
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
        max_minutes=args.max_minutes,
    )


def add_subparser(sub):
    """Register the ``reline`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser(
        "reline",
        help="recompute ONLY the line spectra of existing checkpoints "
        "(new line grid / Ne); brem untouched",
    )
    ap.add_argument(
        "material", nargs="*", help="material stems (checkpoints/<material>.pkl); or -a/--all"
    )
    ap.add_argument(
        "-a", "--all", action="store_true", help="recompute every *.pkl in the checkpoint dir"
    )
    ap.add_argument(
        "--line-ne",
        type=int,
        default=None,
        help="new line electron count (sweep default per material)",
    )
    ap.add_argument(
        "--line-step",
        type=float,
        default=None,
        help="explicit uniform line-grid spacing [eV]; default rebuilds each "
        "record's grid from the material's current E_grid_line_by_energy config",
    )
    ap.add_argument(
        "--redo-all", action="store_true", help="recompute every record even if already at target"
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-minutes", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--save-every", type=int, default=100, help="re-pickle every N relined records (crash-safe)"
    )
    ap.set_defaults(func=_cli)
    return ap
