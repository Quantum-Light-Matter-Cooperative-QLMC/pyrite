"""``cxr slim`` -- shrink a per-material checkpoint for transfer (TODO P2 #5).

The GPU box writes one pickle per material holding the full union of every swept
config at full resolution (``results.store_result``); pulling it to the viz
laptop is gigabyte-scale and mostly stale for any single plot. This command
writes a smaller pickle -- dropping the full-range bremsstrahlung arrays and/or
downcasting the spectra to float32, and zstd-compressing the output (TODO P2 #8,
see ``cxr_mc.checkpoints._checkpoint_io``) -- that still loads and plots exactly like the
full one via ``run.load_checkpoint``.

    cxr slim checkpoints/hopg.pkl --drop-wide-brem --downcast
    cxr slim checkpoints/hopg.pkl -o hopg.slim.pkl --downcast
    cxr slim checkpoints/hopg -o -            # stream to stdout (report on stderr)

Value-based config filtering (keep only some tilts/energies) is available
programmatically via ``results.slim_results(..., tilt_deg=..., E0_keV=...)``.
"""

import contextlib
import io
import os
import sys

from ..cli import _core as _cli_core
from ..results import slim_results
from . import _checkpoint_io, _checkpoint_store


def _material_from_stem(in_path):
    """The material key for ``--grid`` filtering, inferred from the checkpoint
    stem (basename without ``.pkl``). ``--quick`` grids are defined inline in
    ``scan.py`` and are not reproducible from ``material_sweep(material)`` alone,
    so a ``_quick`` stem is rejected with a clear error."""
    stem = os.path.splitext(os.path.basename(os.path.normpath(in_path)))[0]
    if stem.endswith("_quick"):
        raise SystemExit(
            f"quick grids aren't grid-filterable: {os.path.basename(in_path)} is a "
            "--quick checkpoint, whose grid isn't reproducible from "
            "material_sweep(material). Drop --grid for this stem."
        )
    # deferred import: config imports results at module load, so importing it at
    # slim.py's top would re-enter that cycle (same pattern as results/selection)
    from ..campaign.config import material_grid

    try:
        material_grid(stem)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"--grid: {e}") from None
    return stem


def _grid_from_stem(in_path):
    """Return ``(material, fidelity, catalog_profile)`` for a canonical or
    named-profile stem (a canonical stem yields the bare material key)."""
    stem = os.path.splitext(os.path.basename(os.path.normpath(in_path)))[0]
    # ``@`` (current @-stems) or ``--`` (legacy stems) marks a resolved variant
    # whose identity comes from its sidecar/recompute; anything else is a bare
    # canonical (or ``_quick``) material key.
    if "--" not in stem and "@" not in stem:
        return _material_from_stem(in_path)
    from ..campaign.profiles import identity_from_stem

    # Pass the checkpoint's parent dir so identity_from_stem can read the stem's
    # meta.json sidecar (authoritative dataset_identity recorded at run time)
    # rather than recomputing against the possibly-edited live catalog.
    root = os.path.dirname(os.path.normpath(in_path))
    identity = identity_from_stem(stem, root)
    if identity is None:
        raise SystemExit(
            f"--grid: cannot resolve named-profile identity from checkpoint stem {stem!r}"
        )
    return identity["material"], identity["fidelity"], identity.get("catalog_profile", "standard")


def _pct_smaller(before, after):
    """Integer percent saved by the slim. int(round(...)) rather than an f-string
    ':.0f', which renders a tiny negative pct (slim output a hair larger than the
    input) as the '-0%' artifact."""
    pct = 100.0 * (1.0 - after / before) if before else 0.0
    return int(round(pct))


def slim_checkpoint(in_path, out_path=None, **kwargs):
    """Slim a checkpoint (see :func:`_slim_checkpoint`).

    ``out_path="-"`` is pipe mode: the encoded artifact becomes this process's
    stdout, so stdout is claimed here and every ordinary print in the slim path
    (this module's report, plus anything ``slim_results`` emits) is redirected
    to stderr for the whole call -- one stray print would otherwise corrupt the
    byte stream a `cxr remote pull` is reading.
    """
    if out_path != "-":
        return _slim_checkpoint(in_path, out_path, **kwargs)
    pipe = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        return _slim_checkpoint(in_path, out_path, pipe=pipe, **kwargs)


def _slim_checkpoint(
    in_path,
    out_path=None,
    *,
    pipe=None,
    grid=False,
    drop_wide_brem=False,
    downcast=False,
    dataset=None,
    compresslevel=None,
    **constraints,
):
    """Load a checkpoint, slim it (:func:`results.slim_results`), write a smaller
    pickle (atomic temp+replace), and report the size saved. ``out_path`` defaults
    to ``<stem>.slim<ext>``. ``grid`` keeps only the material's current-grid
    configs, inferring the material from the checkpoint stem (rejecting a
    ``_quick`` stem). With ``pipe`` set (``out_path="-"``, see
    :func:`slim_checkpoint`) the artifact streams into that binary stream
    instead of a file, with no temp. ``compresslevel`` is a zstd level forwarded to
    ``_checkpoint_io`` (``None`` = its default); with no other trimming flag it
    makes this call a pure lossless recompress (e.g. level 19 for a smaller
    `cxr remote pull` transfer, independent of the level-3 default a live sweep
    writes at). Extra keyword args are case-field constraints passed straight to
    ``slim_results``. Returns the slim results dict."""
    # validate the stem before loading: the load is the expensive step, and a bad
    # --grid stem should fail in milliseconds, not after a gigabyte unpickle
    material = _grid_from_stem(in_path) if grid else None
    if os.path.isdir(in_path):
        path = os.path.normpath(in_path)
        results = _checkpoint_store.load(os.path.basename(path), os.path.dirname(path))
        input_paths = [
            _checkpoint_store.component_path(
                os.path.basename(path), component, os.path.dirname(path)
            )
            for component in _checkpoint_store.COMPONENTS
        ]
        before = sum(item.stat().st_size for item in input_paths if item.is_file())
    else:
        results = _checkpoint_io.load(in_path)
        before = os.path.getsize(in_path)
    if dataset is not None:
        from ..results import project_dataset

        results = project_dataset(results, dataset)
    slim = slim_results(
        results, grid=material, drop_wide_brem=drop_wide_brem, downcast=downcast, **constraints
    )
    if out_path is None:
        if os.path.isdir(in_path):
            out_path = f"{os.path.normpath(in_path)}.slim.pkl"
        else:
            root, ext = os.path.splitext(in_path)
            out_path = f"{root}.slim{ext or '.pkl'}"
    if pipe is not None:
        after = _dump_to_pipe(slim, pipe, compresslevel)
    else:
        tmp = out_path + ".tmp"
        _checkpoint_io.dump(slim, tmp, compresslevel=compresslevel)
        os.replace(tmp, out_path)  # atomic: never leave a half-written pickle
        after = os.path.getsize(out_path)
    n_in = sum(len(v) for v in results.values())
    n_out = sum(len(v) for v in slim.values())
    print(
        f"slimmed {in_path} ({before / 1e6:.1f} MB, {n_in} records) -> "
        f"{out_path} ({after / 1e6:.1f} MB, {n_out} records); "
        f"{_pct_smaller(before, after)}% smaller"
    )
    return slim


class _CountingWriter(io.RawIOBase):
    """Pass-through binary sink that totals the bytes it forwards."""

    def __init__(self, stream):
        self._stream = stream
        self.total = 0

    def writable(self):
        return True

    def write(self, data):
        self._stream.write(data)
        self.total += len(data)
        return len(data)


def _dump_to_pipe(slim, pipe, compresslevel):
    """Stream the encoded artifact down ``pipe``; return the byte count."""
    counter = _CountingWriter(pipe)
    _checkpoint_io.dump_stream(slim, counter, compresslevel=compresslevel)
    pipe.flush()
    return counter.total


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


def main(argv=None):
    from ..cli.commands.slim import command

    return _cli_core.run(command, argv, prog_name="cxr-slim")


def __getattr__(name: str):
    if name == "command":
        from ..cli.commands.slim import command

        return command
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
