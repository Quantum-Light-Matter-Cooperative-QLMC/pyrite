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

Runs wherever the pickles live: locally on ``checkpoints/``, or on the GPU box
for remote checkpoints (``ssh qlmc 'cd dev/cxr-mc && ... cxr rebrem ...'``),
then pull as usual. Records already at the target parameters are skipped, so a
crashed/OOM'd run just resumes on re-invocation; ``--redo-all`` forces a full
recompute (e.g. same parameters, new physics).
"""

from pathlib import Path


def rebrem_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    ne_brem=None,
    brem_step_eV=None,
    redo_all=False,
    save_every=100,
):
    """Run :func:`cxr_mc.run.repair_checkpoint` with new brem parameters over one
    or more materials. ``materials=None`` sweeps every ``*.pkl`` in
    ``checkpoint_dir`` (grooved checkpoints included; ``*.slim.pkl`` transfer
    copies excluded). Returns ``{stem: results}``."""
    from .run import checkpoint_path_for, repair_checkpoint

    ckpt_dir = Path(checkpoint_dir)
    if materials:
        paths = [checkpoint_path_for(m, str(ckpt_dir)) for m in materials]
    else:
        paths = sorted(str(p) for p in ckpt_dir.glob("*.pkl") if not p.name.endswith(".slim.pkl"))
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    out = {}
    for path in paths:
        stem = Path(path).stem
        print(f"== {stem} ==")
        out[stem] = repair_checkpoint(
            path,
            save_every=save_every,
            only_nonfinite=not redo_all,
            ne_brem=ne_brem,
            brem_step_eV=brem_step_eV,
        )
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
    )


def add_subparser(sub):
    """Register the ``rebrem`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser(
        "rebrem",
        help="recompute ONLY the brem background of existing checkpoints "
        "(new Ne_brem / grid spacing); line spectra untouched",
    )
    ap.add_argument(
        "material",
        nargs="*",
        help="material stems (checkpoints/<material>.pkl); or use -a/--all",
    )
    ap.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="recompute every *.pkl in the checkpoint dir",
    )
    ap.add_argument(
        "--ne-brem",
        type=int,
        default=None,
        help="new brem electron count (noise ~ 1/sqrt(Ne_brem); sweep default 100)",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=None,
        help="new uniform wide-brem grid spacing [eV] (sweep default 50); grid is "
        "rebuilt from each record's low cutoff up to its beam energy",
    )
    ap.add_argument(
        "--redo-all",
        action="store_true",
        help="recompute every record even if already at the target parameters "
        "(default skips those, making interrupted runs resumable)",
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="re-pickle the checkpoint every N repaired records (crash-safe)",
    )
    ap.set_defaults(func=_cli)
    return ap
