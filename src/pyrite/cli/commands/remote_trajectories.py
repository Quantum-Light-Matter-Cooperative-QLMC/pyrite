"""Explicit remote capture inspection, transfer, and replay commands."""

from pathlib import Path, PurePosixPath

import click

from ...console.output import POSITIVE_FLOAT, POSITIVE_INT, CLIError
from ...console.outputs import output_dir
from ...remote import trajectories


def _stem(_ctx, param, value):
    if value is not None and (
        value in {".", ".."}
        or PurePosixPath(value).name != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise click.BadParameter("STEM must be one directory name", param=param)
    return value

def _filters(function):
    function = click.option(
        "--energy",
        "energies",
        multiple=True,
        type=POSITIVE_FLOAT,
        metavar="KEV",
        help="Select exact incident energy in keV; repeatable.",
    )(function)
    function = click.option(
        "--case", "cases", multiple=True, metavar="NAME", help="Select exact case name; repeatable."
    )(function)
    return click.argument("stem", required=False, metavar="[STEM]", callback=_stem)(function)


def _checked(handler, *args, **kwargs):
    from ...montecarlo.trajectories import TrajectoryArtifactError

    try:
        return handler(*args, **kwargs)
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise CLIError(str(error)) from error


@click.group(
    "trajectories",
    help="Inspect remote captures, pull selected cases, or export selected histories. No automatic capture transfer.",
)
@click.option(
    "--root",
    default=None,
    metavar="DIR",
    help="Remote capture root inside the checkout (default: pyrite-output/trajectories).",
)
@click.pass_context
def command(ctx, root):
    ctx.ensure_object(dict)
    ctx.obj["trajectory_root"] = root


def _inventory(ctx, stem, cases, energies, hashes=False):
    return _checked(
        trajectories.remote_inventory,
        ctx.obj["trajectory_root"],
        stem,
        cases,
        energies,
        hashes=hashes,
    )


def _print_inventory(records):
    if not records:
        click.echo("No matching complete trajectory artifacts.")
        return
    for record in records:
        click.echo(
            f"{record['path']}  {record['bytes']} bytes  {record['segments']} segments  {record['energy_keV']:g} keV"
        )
    click.echo(f"{len(records)} artifact(s), {sum(record['bytes'] for record in records)} bytes")


@command.command("ls", help="List complete capture files and sizes per stem; reads headers only.")
@_filters
@click.pass_context
def ls_command(ctx, stem, cases, energies):
    _print_inventory(_inventory(ctx, stem, cases, energies))


@command.command("info", help="Show capture headers and sizes as JSON; reads no segment arrays.")
@_filters
@click.pass_context
def info_command(ctx, stem, cases, energies):
    import json

    click.echo(json.dumps(_inventory(ctx, stem, cases, energies)))


@command.command(
    "pull",
    help="Preview selected complete HDF5 files and bytes. --yes transfers with rsync resume, SHA-256 and header verification. No history filtering; use export for histories.",
)
@_filters
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Local capture root (default: pyrite-output/trajectories).",
)
@click.option(
    "-y", "--yes", is_flag=True, help="Transfer the selected files; without this flag only preview."
)
@click.option(
    "--overwrite", is_flag=True, help="Replace differing local files after successful verification."
)
@click.pass_context
def pull_command(ctx, stem, cases, energies, out_dir, yes, overwrite):
    records = _inventory(ctx, stem, cases, energies, hashes=yes)
    _print_inventory(records)
    if not yes:
        click.echo("Preview only; pass --yes to transfer.")
        return
    if not records:
        raise CLIError("no matching complete trajectory artifacts")
    root = _checked(trajectories.capture_root, ctx.obj["trajectory_root"])
    paths = _checked(
        trajectories.pull_files,
        records,
        root,
        out_dir or output_dir("trajectories"),
        overwrite=overwrite,
    )
    for path in paths:
        click.echo(path)


@command.command(
    "export",
    help="Export selected whole histories/tracks on the host and pull only .vtp files. Requires --history, --track, --first, or --sample; captures remain remote.",
)
@_filters
@click.option("--history", "histories", type=click.IntRange(min=0), multiple=True, metavar="ID")
@click.option("--track", "tracks", type=click.IntRange(min=0), multiple=True, metavar="ID")
@click.option("--first", type=POSITIVE_INT, default=None, metavar="N")
@click.option("--sample", type=POSITIVE_INT, default=None, metavar="N")
@click.option("--seed", type=click.IntRange(min=0), default=None, metavar="S")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Local VTK root (default: pyrite-output/trajectories-exports).",
)
@click.option(
    "--overwrite", is_flag=True, help="Replace differing local VTK files after verification."
)
@click.pass_context
def export_command(
    ctx, stem, cases, energies, histories, tracks, first, sample, seed, out_dir, overwrite
):
    if sum(bool(value) for value in (histories, first, sample)) > 1:
        raise click.UsageError("--history, --first, and --sample are mutually exclusive")
    if seed is not None and sample is None:
        raise click.UsageError("--seed requires --sample")
    if not any((histories, tracks, first, sample)):
        raise click.UsageError("select --history, --track, --first, or --sample")
    request = dict(
        root=ctx.obj["trajectory_root"],
        stem=stem,
        cases=cases,
        energies=energies,
        selection=dict(
            histories=histories, tracks=tracks, first=first, sample=sample, seed=seed or 0
        ),
    )
    for path in _checked(
        trajectories.remote_export, request, out_dir or output_dir("trajectories-exports")
    ):
        click.echo(path)


@command.command(
    "score",
    help="Submit spectrum replay from selected captures to SLURM. Writes remote checkpoints; no transport or capture transfer. Requires one exact STEM. Uses already synced code.",
)
@_filters
@click.option(
    "--max-segments",
    type=POSITIVE_INT,
    default=None,
    metavar="N",
    help="Segment rows per scoring block; whole histories stay together.",
)
@click.option("--overwrite", is_flag=True, help="Re-score existing checkpoint records.")
@click.pass_context
def score_command(ctx, stem, cases, energies, max_segments, overwrite):
    if stem is None:
        raise click.UsageError("score requires one exact STEM")
    request = dict(
        root=ctx.obj["trajectory_root"],
        stem=stem,
        cases=cases,
        energies=energies,
        max_segments=max_segments,
        overwrite=overwrite,
    )
    jobid = _checked(trajectories.submit_score, request)
    click.echo(f"Submitted {jobid}; monitor with `pyrite job attach {jobid}`.")
