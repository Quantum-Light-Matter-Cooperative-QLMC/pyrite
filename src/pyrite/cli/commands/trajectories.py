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


@click.command(
    "score-trajectories",
    help=(
        "Score checkpoint records from captured trajectories without re-transporting.\n\n"
        "ARTIFACT is an HDF5 file written by `pyrite run --trajectories`, or a "
        "directory searched recursively for them. Each artifact's spectrum phase "
        "(line, characteristic, and bremsstrahlung spectra) is replayed from its "
        "stored segments and saved as a record of the checkpoint stem and run "
        "identity it was captured under, in DIR/<stem>. Records already present "
        "are kept unless --overwrite; records of cases without an artifact are "
        "never touched. Each new record stores its artifact's path and SHA-256. "
        "The shared per-case cache is neither read nor written.\n\n"
        "Every artifact is checked before anything is written: schema-1, "
        "incomplete, foreign, or mismatched artifacts, and a checkpoint written "
        "by a different run, are refused. Cases needing every segment at once "
        "(coherent emission, temporal profile) are refused; score those with "
        "`spectrum_from_artifact` from Python."
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
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    type=click.Path(file_okay=False, path_type=Path),
    help="Root containing component checkpoint directories to write.",
)
@click.option(
    "--max-segments",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help=(
        "Segment rows read per block; bounds host memory (whole electrons stay "
        "together, so one long history may exceed it). [default: 1048576]"
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Re-score cases that already have a record in the target checkpoint.",
)
def score_command(artifacts, checkpoint_dir, max_segments, overwrite):
    from ...checkpoints.trajectory_scoring import (
        check_targets,
        discover_artifacts,
        plan_stems,
        score_stem,
    )
    from ...montecarlo.runner.artifacts import STREAM_MAX_SEGMENTS
    from ...montecarlo.trajectories import TrajectoryArtifactError

    paths = discover_artifacts(artifacts)
    if not paths:
        raise _cli_core.CLIError("no trajectory artifacts (*.h5) found")

    def _scored(artifact, out):
        click.echo(
            f"{artifact.provenance['checkpoint_stem']}  {artifact.case['name']}  "
            f"E0={float(artifact.case['E0_keV']):g} keV  {out['n_segments']} segments"
        )

    try:
        plans = plan_stems(paths)
        check_targets(plans.values(), checkpoint_dir)
        for plan in plans.values():
            summary = score_stem(
                plan,
                checkpoint_dir,
                max_segments=max_segments or STREAM_MAX_SEGMENTS,
                overwrite=overwrite,
                on_case=_scored,
            )
            kept = f"; {summary.kept} already scored (--overwrite to redo)" if summary.kept else ""
            click.echo(f"{summary.checkpoint_path}: scored {summary.scored}{kept}")
    except TrajectoryArtifactError as error:
        raise _cli_core.CLIError(str(error)) from error
