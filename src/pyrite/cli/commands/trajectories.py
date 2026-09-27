"""Click wiring for exporting captured transport trajectories."""

from pathlib import Path

import click

from ...console import output as _cli_core


def _artifacts(paths):
    """Expand directories to ``(artifact, path relative to its root)`` pairs.

    A directory's artifacts keep their layout below it (every case has an
    ``E0_<E>keV.h5`` file, so names alone collide); an explicit file is its
    own name.
    """
    found = []
    for path in paths:
        if path.is_dir():
            found.extend(
                (p, p.relative_to(path)) for p in sorted(path.rglob("*.h5")) if p.is_file()
            )
        else:
            found.append((path, Path(path.name)))
    return found


@click.command(
    "export-trajectories",
    help=(
        "Export captured trajectory segments to VTK PolyData (.vtp).\n\n"
        "ARTIFACT is an HDF5 file written by `pyrite run --trajectories`, or a "
        "directory searched recursively for them. Each artifact becomes one .vtp "
        "of two-point line cells in the slab frame [angstrom] with per-segment "
        "cell data, including electron_id; per-electron arrays, tallies, and "
        "metadata stay only in the HDF5 artifact. Opens in ParaView, VisIt, and "
        "PyVista."
    ),
)
@click.argument(
    "artifacts",
    nargs=-1,
    required=True,
    metavar="ARTIFACT...",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Write .vtp files under DIR, mirroring each ARTIFACT directory (default: beside each artifact).",
)
@click.option("--no-vacuum", is_flag=True, help="Omit grooved runs' vacuum legs.")
@click.option("--overwrite", is_flag=True, help="Replace existing .vtp outputs.")
def command(artifacts, out_dir, no_vacuum, overwrite):
    from ...montecarlo.trajectories import TrajectoryArtifactError
    from ...montecarlo.trajectory_export import export_segments_vtp

    paths = _artifacts(artifacts)
    if not paths:
        raise _cli_core.CLIError("no trajectory artifacts (*.h5) found")
    targets = [
        ((out_dir / rel if out_dir is not None else path).with_suffix(".vtp"), path)
        for path, rel in paths
    ]
    names = [target for target, _ in targets]
    if len(set(names)) != len(names):
        raise click.UsageError(
            "several artifacts map to the same output path; pass their common "
            "directory instead of individual files"
        )
    existing = [target for target in names if target.exists()]
    if existing and not overwrite:
        raise _cli_core.CLIError(
            f"{len(existing)} output(s) already exist, e.g. {existing[0]}; pass --overwrite"
        )
    for target, path in targets:
        try:
            summary = export_segments_vtp(path, target, include_vacuum=not no_vacuum)
        except TrajectoryArtifactError as error:
            raise _cli_core.CLIError(str(error)) from error
        click.echo(f"{target}  {summary['cells']} cells ({summary['vacuum_legs']} vacuum)")
