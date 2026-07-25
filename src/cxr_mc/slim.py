"""``cxr slim`` -- shrink a per-material checkpoint for transfer (TODO P2 #5).

The GPU box writes one pickle per material holding the full union of every swept
config at full resolution (``results.store_result``); pulling it to the viz
laptop is gigabyte-scale and mostly stale for any single plot. This command
writes a smaller pickle -- dropping the full-range bremsstrahlung arrays and/or
downcasting the spectra to float32, and gzip-compressing the output (TODO P2 #8,
see ``cxr_mc._checkpoint_io``) -- that still loads and plots exactly like the
full one via ``run.load_checkpoint``.

    cxr slim checkpoints/hopg.pkl --drop-wide-brem --downcast
    cxr slim checkpoints/hopg.pkl -o hopg.slim.pkl --downcast

Value-based config filtering (keep only some tilts/energies) is available
programmatically via ``results.slim_results(..., tilt_deg=..., E0_keV=...)``.
"""

import os

import click

from . import _checkpoint_io, _cli_core
from .results import slim_results


def _material_from_stem(in_path):
    """The material key for ``--grid`` filtering, inferred from the checkpoint
    stem (basename without ``.pkl``). ``--quick`` grids are defined inline in
    ``scan.py`` and are not reproducible from ``material_sweep(material)`` alone,
    so a ``_quick`` stem is rejected with a clear error."""
    stem = os.path.splitext(os.path.basename(in_path))[0]
    if stem.endswith("_quick"):
        raise SystemExit(
            f"quick grids aren't grid-filterable: {os.path.basename(in_path)} is a "
            "--quick checkpoint, whose grid isn't reproducible from "
            "material_sweep(material). Drop --grid for this stem."
        )
    # deferred import: config imports results at module load, so importing it at
    # slim.py's top would re-enter that cycle (same pattern as results/selection)
    from .config import material_grid

    try:
        material_grid(stem)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"--grid: {e}") from None
    return stem


def _pct_smaller(before, after):
    """Integer percent saved by the slim. int(round(...)) rather than an f-string
    ':.0f', which renders a tiny negative pct (slim output a hair larger than the
    input) as the '-0%' artifact."""
    pct = 100.0 * (1.0 - after / before) if before else 0.0
    return int(round(pct))


def slim_checkpoint(
    in_path,
    out_path=None,
    *,
    grid=False,
    drop_wide_brem=False,
    downcast=False,
    dataset=None,
    compresslevel=6,
    **constraints,
):
    """Load a checkpoint, slim it (:func:`results.slim_results`), write a smaller
    pickle (atomic temp+replace), and report the size saved. ``out_path`` defaults
    to ``<stem>.slim<ext>``. ``grid`` keeps only the material's current-grid
    configs, inferring the material from the checkpoint stem (rejecting a
    ``_quick`` stem). ``compresslevel`` is forwarded to
    ``_checkpoint_io.dump`` -- with no other trimming flag it makes this call a
    pure lossless recompress (e.g. gzip level 9 for a smaller `cxr remote pull`
    transfer, independent of the level-6 default used while a sweep is still
    writing checkpoints). Extra keyword args are case-field constraints passed
    straight to ``slim_results``. Returns the slim results dict."""
    # validate the stem before loading: the load is the expensive step, and a bad
    # --grid stem should fail in milliseconds, not after a gigabyte unpickle
    material = _material_from_stem(in_path) if grid else None
    results = _checkpoint_io.load(in_path)
    if dataset is not None:
        from .results import project_dataset

        results = project_dataset(results, dataset)
    slim = slim_results(
        results, grid=material, drop_wide_brem=drop_wide_brem, downcast=downcast, **constraints
    )
    if out_path is None:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}.slim{ext or '.pkl'}"
    tmp = out_path + ".tmp"
    _checkpoint_io.dump(slim, tmp, compresslevel=compresslevel)
    os.replace(tmp, out_path)  # atomic: never leave a half-written pickle
    before, after = os.path.getsize(in_path), os.path.getsize(out_path)
    n_in = sum(len(v) for v in results.values())
    n_out = sum(len(v) for v in slim.values())
    print(
        f"slimmed {in_path} ({before / 1e6:.1f} MB, {n_in} records) -> "
        f"{out_path} ({after / 1e6:.1f} MB, {n_out} records); "
        f"{_pct_smaller(before, after)}% smaller"
    )
    return slim


def _cli(args):
    """CLI handler -- runs slim_checkpoint and returns None (the dict it returns
    must not reach sys.exit via the console-script wrapper)."""
    slim_checkpoint(
        args.checkpoint,
        args.out,
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        dataset="brem" if args.brem_only else ("line" if args.line_only else None),
        compresslevel=args.compresslevel,
    )


@click.command("slim", help="Shrink a checkpoint pickle for transfer.")
@click.argument("checkpoint")
@click.option("-o", "--out", default=None, help="Output path (default: <stem>.slim.pkl).")
@click.option("--grid", is_flag=True, help="Keep only current-grid configs.")
@click.option("--drop-wide-brem", is_flag=True, help="Drop full-range brem arrays.")
@click.option("--downcast", is_flag=True, help="Store spectral arrays as float32.")
@click.option(
    "--compresslevel",
    type=click.IntRange(1, 9),
    default=6,
    show_default=True,
    metavar="1-9",
)
@click.option("--brem-only", is_flag=True, help="Keep only brem arrays.")
@click.option("--line-only", is_flag=True, help="Keep only line arrays.")
def command(
    checkpoint,
    out,
    grid,
    drop_wide_brem,
    downcast,
    compresslevel,
    brem_only,
    line_only,
):
    if brem_only and line_only:
        raise click.UsageError("--brem-only and --line-only are mutually exclusive")
    return _cli_core.invoke_legacy(
        _cli,
        checkpoint=checkpoint,
        out=out,
        grid=grid,
        drop_wide_brem=drop_wide_brem,
        downcast=downcast,
        compresslevel=compresslevel,
        brem_only=brem_only,
        line_only=line_only,
    )


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="cxr-slim")


if __name__ == "__main__":
    raise SystemExit(main())
