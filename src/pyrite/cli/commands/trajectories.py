"""Click wiring for listing, describing, exporting, and scoring trajectory captures.

``pyrite trajectories`` is the one home for artifacts written by
``pyrite run --trajectories`` (#397). ``pyrite trajectories export``
and ``score-trajectories`` stay as hidden deprecated aliases of ``export`` and
``score``.
"""

from pathlib import Path

import click

from ...console import output as _cli_core
from ...console.output import emit_diagnostic, emit_json, emit_result, output_option
from ...console.outputs import output_default, output_dir, output_label
from .. import _completion as _cli_completion
from .._groups import LazyGroup

_COMMANDS = {
    "ls": "pyrite.cli.commands.trajectories.ls_command",
    "info": "pyrite.cli.commands.trajectories.info_command",
    "export": "pyrite.cli.commands.trajectories.export_command",
    "score": "pyrite.cli.commands.trajectories.score_command",
}

_COMMAND_HELP = {
    "ls": "List captured artifacts, partial writes, and foreign files.",
    "info": "Describe one artifact's case, provenance, and fields.",
    "export": "Export captured trajectories to VTK PolyData.",
    "score": "Score checkpoint records from captured trajectories.",
}


@click.group(
    "trajectories",
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=True,
)
def command() -> None:
    """List, describe, export, and score saved trajectory captures.

    Captures are HDF5 artifacts written by `pyrite run --trajectories DIR`,
    one per case under DIR/<stem>/. They are not checkpoints; `score` turns
    them into checkpoint records.

    \b
    Examples:
      pyrite trajectories ls
      pyrite trajectories info hopg@trajectory_demo/<case>/E0_20keV.h5
      pyrite trajectories export hopg@trajectory_demo --first 20
      pyrite trajectories score hopg@trajectory_demo
    """


def _print_paraview_script(ctx, _param, value):
    if not value or ctx.resilient_parsing:
        return
    from importlib.resources import files

    click.echo(files("pyrite") / "data" / "paraview" / "pyrite_trajectories.py")
    ctx.exit()


def _discover(paths):
    """Complete artifacts under ``paths``; warn about every skipped file."""
    from ...montecarlo.trajectory_inventory import discover_artifacts

    found = discover_artifacts(paths)
    for skipped in found.skipped:
        emit_diagnostic(f"warning: skipping {skipped.status} file {skipped.path}: {skipped.reason}")
    if not found.artifacts:
        raise _cli_core.CLIError("no trajectory artifacts (*.h5) found")
    return found.artifacts


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _emit_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        emit_result(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


_LS_COLUMNS = (
    ("STEM", "stem"),
    ("CASE", "case"),
    ("E0_KEV", "E0_keV"),
    ("SCHEMA", "schema_version"),
    ("STATUS", "status"),
    ("SEGMENTS", "segments"),
    ("HISTORIES", "histories"),
    ("MAX_GEN", "max_generation"),
    ("BYTES", "bytes"),
    ("PATH", "path"),
)


@click.command(
    "ls",
    help=(
        "List trajectory artifacts under each DIR (default: the local capture "
        "root, pyrite-output/trajectories).\n\n"
        "One row per file: checkpoint stem, case name, incident energy E0 [keV], "
        "schema, status, segment rows, histories with segments, highest "
        "secondary generation, and size [bytes], then totals per stem. Stray "
        "interrupted writes (*.h5.partial), incomplete artifacts, and HDF5 files "
        "that are not trajectory artifacts are listed with their status instead "
        "of failing. History and generation counts scan only integer ID "
        "columns in bounded blocks; no segment array is loaded.\n\n"
        "DIR may also be a checkpoint selector naming its local capture "
        "directory, or an artifact file."
    ),
)
@click.argument(
    "dirs",
    nargs=-1,
    metavar="[DIR]...",
    type=_cli_completion.TRAJECTORY_ARTIFACT,
)
@output_option
def ls_command(dirs, json_output):
    from ...montecarlo.trajectories import TrajectoryArtifactError
    from ...montecarlo.trajectory_inventory import describe_artifact, discover_artifacts

    if not dirs:
        root = output_dir("trajectories")
        dirs = (root,) if root.is_dir() else ()
    found = discover_artifacts(dirs)
    rows = []
    for artifact in found.artifacts:
        try:
            info = describe_artifact(artifact.path)
        except (TrajectoryArtifactError, OSError) as error:
            rows.append(_skipped_row(artifact.path, "foreign", str(error)))
            continue
        rows.append(
            {
                "path": str(artifact.path),
                "stem": info["provenance"].get("checkpoint_stem"),
                "case": info["case"]["name"],
                "E0_keV": info["case"]["E0_keV"],
                "schema_version": info["schema_version"],
                "status": "complete",
                "segments": info["segments"],
                "histories": info.get("histories"),
                "max_generation": info.get("max_generation"),
                "bytes": info["bytes"],
            }
        )
    rows += [_skipped_row(item.path, item.status, item.reason) for item in found.skipped]
    rows.sort(key=lambda row: str(row["path"]))
    stems: dict[str, dict] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        total = stems.setdefault(
            row["stem"] or "-",
            {"stem": row["stem"], "artifacts": 0, "segments": 0, "bytes": 0},
        )
        total["artifacts"] += 1
        total["segments"] += row["segments"]
        total["bytes"] += row["bytes"]
    if json_output:
        emit_json(
            "pyrite.trajectories.ls.v1",
            {"artifacts": rows, "stems": list(stems.values())},
        )
        return
    if not rows:
        emit_result("No trajectory artifacts found.")
        return
    _emit_table(
        [title for title, _ in _LS_COLUMNS],
        [[_cell(row.get(key)) for _, key in _LS_COLUMNS] for row in rows],
    )
    for total in stems.values():
        emit_result(
            f"{_cell(total['stem'])}: {total['artifacts']} artifact(s), "
            f"{total['segments']} segments, {total['bytes']} bytes"
        )


def _skipped_row(path: Path, status: str, reason: str) -> dict:
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {
        "path": str(path),
        "stem": None,
        "case": None,
        "E0_keV": None,
        "schema_version": None,
        "status": status,
        "segments": None,
        "histories": None,
        "max_generation": None,
        "bytes": size,
        "reason": reason,
    }


@click.command(
    "info",
    help=(
        "Describe one trajectory artifact without loading its segment arrays.\n\n"
        "Prints the case, seed, provenance (checkpoint stem, profile, fidelity), "
        "whether a scene was recorded, schema and row order, then every "
        "/transport field with its role, dtype, shape, and units. Per-segment "
        "field names are the valid cell arrays of `pyrite trajectories export` "
        "and colouring names of the ParaView preset. History ID range, number "
        "of histories with segments, track count, and segments per secondary "
        "generation come from bounded scans of the integer ID columns.\n\n"
        "ARTIFACT is an HDF5 file written by `pyrite run --trajectories`; list "
        "candidates with `pyrite trajectories ls`."
    ),
)
@click.argument("artifact", metavar="ARTIFACT", type=_cli_completion.TRAJECTORY_ARTIFACT)
@output_option
def info_command(artifact, json_output):
    from ...montecarlo.trajectories import TrajectoryArtifactError
    from ...montecarlo.trajectory_inventory import describe_artifact

    if artifact.is_dir():
        raise click.BadParameter(
            f"{artifact} is a directory; pick one file from `pyrite trajectories ls {artifact}`",
            param_hint="ARTIFACT",
        )
    try:
        info = describe_artifact(artifact)
    except (TrajectoryArtifactError, OSError) as error:
        raise _cli_core.CLIError(str(error)) from error
    if json_output:
        emit_json("pyrite.trajectories.info.v1", info)
        return
    _print_info(info)


def _print_info(info: dict) -> None:
    case = info["case"]
    provenance = info["provenance"]
    lines = [
        ("path", info["path"]),
        ("case", case["name"]),
        ("E0 [keV]", case["E0_keV"]),
        ("crystal", case.get("crystal")),
        ("thickness [angstrom]", case.get("thickness_ang")),
        ("tilt [deg]", case.get("tilt_deg")),
        ("electrons (Ne)", case.get("Ne")),
        ("seed", info["seed"]),
        ("stem", provenance.get("checkpoint_stem")),
        ("profile", provenance.get("profile")),
        ("fidelity", provenance.get("fidelity")),
        ("scene", "recorded" if info["scene"] else "not recorded"),
        ("spectrum inputs", "recorded" if info["spectrum_inputs"] else "not recorded"),
        ("schema", info["schema_version"]),
        ("row order", info["row_order"]),
        ("pyrite version", info["pyrite_version"]),
        ("created (UTC)", info["created_utc"]),
        ("size [bytes]", info["bytes"]),
        ("segments", info["segments"]),
        ("sampled histories", info["sampled_histories"]),
        ("histories with segments", info.get("histories")),
    ]
    if info.get("history_id_range"):
        low, high = info["history_id_range"]
        lines.append(("history IDs", f"{low}..{high}"))
    if "tracks" in info:
        lines.append(("tracks", info["tracks"]))
    if "segments_by_generation" in info:
        lines.append(
            (
                "segments by generation",
                ", ".join(f"{g}: {n}" for g, n in info["segments_by_generation"].items()),
            )
        )
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        emit_result(f"{label.ljust(width)}  {_cell(value)}")
    for role, title in (
        ("segment", "Per-segment fields"),
        ("electron", "Per-electron fields"),
        ("nested", "Nested fields"),
        ("scalar", "Scalars"),
        ("other", "Other arrays"),
    ):
        records = [record for record in info["fields"] if record["role"] == role]
        if not records:
            continue
        emit_result("")
        emit_result(f"{title}:")
        _emit_table(
            ["  NAME", "DTYPE", "SHAPE", "UNITS"],
            [
                [
                    f"  {record['name']}",
                    record.get("dtype", record["kind"]),
                    "x".join(map(str, record.get("shape", []))) or "-",
                    record.get("units", "-"),
                ]
                for record in records
            ],
        )


@click.command(
    "export",
    help=(
        "Export captured trajectory segments to VTK PolyData (.vtp).\n\n"
        "ARTIFACT is an HDF5 file written by `pyrite run --trajectories`, a "
        "directory searched recursively for them (other files are skipped with "
        "a warning), or a checkpoint from "
        "`pyrite checkpoint list` (exact name, unique prefix, or unique "
        "fragment such as its hash) naming its local capture directory. "
        "Each artifact becomes one .vtp "
        "of two-point line cells in the slab frame [angstrom] with per-segment "
        "cell data, including electron_id; per-electron arrays, tallies, and "
        "complete transport metadata stay in the authoritative HDF5 artifact. "
        "Opens in ParaView, VisIt, and PyVista. --scene also exports lab-frame "
        "geometry as separate close-up and instrument scenes.\n\n"
        "--history, --track, --first, or --sample export only whole selected "
        "histories (primary electron_id with every secondary of its shower) or "
        "tracks, reading only their rows; the .vtp then adds a segment_id cell "
        "array and a FieldData record of the selection. Vacuum legs follow "
        "selected histories and are omitted from track selections.\n\n"
        "Examples:\n\n"
        "  pyrite trajectories export hopg@trajectory_demo --first 20\n\n"
        "  pyrite trajectories export case.h5 --sample 100 --seed 1 --scene\n\n"
        "ParaView preset (colouring, thresholds, scene views, headless PNG): "
        '`pvbatch "$(pyrite trajectories export --paraview-script)" '
        "case.vtp --screenshot case.png`, or import the script as a ParaView macro."
    ),
)
@click.argument(
    "artifacts",
    nargs=-1,
    required=True,
    metavar="ARTIFACT...",
    type=_cli_completion.TRAJECTORY_ARTIFACT,
)
@click.option(
    "--paraview-script",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_paraview_script,
    help="Print the path of the bundled ParaView preset script and exit.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Write .vtp files under DIR, mirroring each ARTIFACT directory (default: beside each artifact).",
)
@click.option("--no-vacuum", is_flag=True, help="Omit grooved runs' vacuum legs.")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace existing .vtp outputs and, with --scene, scene manifests.",
)
@click.option(
    "--scene",
    is_flag=True,
    help="Also write separate .vtm close-up [angstrom] and .instrument.vtm [mm] scenes with private sidecar files; missing recorded geometry is omitted.",
)
@click.option(
    "--history",
    "histories",
    multiple=True,
    type=click.IntRange(min=0),
    metavar="ID",
    help="Export this whole history (electron_id); repeatable.",
)
@click.option(
    "--track",
    "tracks",
    multiple=True,
    type=click.IntRange(min=0),
    metavar="ID",
    help="Export this whole track (track_id); repeatable; intersects history selection.",
)
@click.option(
    "--first",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Export the first N histories that have segments.",
)
@click.option(
    "--sample",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Export N histories drawn at random from those with segments (all if fewer).",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=None,
    metavar="S",
    help="Random seed of --sample. [default: 0]",
)
def export_command(
    artifacts, out_dir, no_vacuum, overwrite, scene, histories, tracks, first, sample, seed
):
    from ...montecarlo.trajectories import TrajectoryArtifactError
    from ...montecarlo.trajectory_export import export_segments_vtp
    from ...montecarlo.trajectory_scene import export_trajectory_scene, scene_output_paths
    from ...montecarlo.trajectory_selection import select_trajectories

    chosen = [
        name
        for name, value in (("--history", histories), ("--first", first), ("--sample", sample))
        if value
    ]
    if len(chosen) > 1:
        raise click.UsageError(f"{' and '.join(chosen)} are mutually exclusive")
    if seed is not None and sample is None:
        raise click.UsageError("--seed requires --sample")
    selecting = bool(histories or tracks or first or sample)

    paths = [(found.path, found.relative) for found in _discover(artifacts)]
    targets = [
        ((out_dir / rel if out_dir is not None else path).with_suffix(".vtp"), path)
        for path, rel in paths
    ]
    names = [target for target, _ in targets]
    if scene:
        names += [path for target, _ in targets for path in scene_output_paths(target)]
    if len(set(names)) != len(names):
        raise click.UsageError(
            "several artifacts map to the same output path; "
            + (
                "rename conflicting artifacts or export them to separate output directories"
                if scene
                else "pass their common directory instead of individual files"
            )
        )
    existing = [target for target in names if target.exists()]
    if existing and not overwrite:
        raise _cli_core.CLIError(
            f"{len(existing)} output(s) already exist, e.g. {existing[0]}; pass --overwrite"
        )
    # Resolve every selection before writing, so an empty one exports nothing.
    try:
        selections = [
            select_trajectories(
                path,
                histories=histories,
                tracks=tracks,
                first=first,
                sample=sample,
                seed=seed or 0,
            )
            if selecting
            else None
            for _, path in targets
        ]
    except (TrajectoryArtifactError, OSError) as error:
        raise _cli_core.CLIError(str(error)) from error
    for (target, path), selection in zip(targets, selections, strict=True):
        try:
            summary = export_segments_vtp(
                path, target, include_vacuum=not no_vacuum, selection=selection
            )
            manifests = (
                export_trajectory_scene(
                    path,
                    target,
                    include_vacuum=not no_vacuum,
                    overwrite=overwrite,
                    selection=selection,
                )
                if scene
                else ()
            )
        except (TrajectoryArtifactError, OSError, ValueError, KeyError) as error:
            raise _cli_core.CLIError(str(error)) from error
        chosen = f"; {selection.histories} histories selected" if selection is not None else ""
        click.echo(f"{target}  {summary['cells']} cells ({summary['vacuum_legs']} vacuum){chosen}")
        for manifest in manifests:
            click.echo(str(manifest))


@click.command(
    "score",
    help=(
        "Score checkpoint records from captured trajectories without re-transporting.\n\n"
        "ARTIFACT is an HDF5 file written by `pyrite run --trajectories`, a "
        "directory searched recursively for them (other files are skipped with "
        "a warning), or a checkpoint selector as for `pyrite trajectories "
        "export`. Each artifact's spectrum phase "
        "(line, characteristic, and bremsstrahlung spectra) is replayed from its "
        "stored segments and saved as a record of the checkpoint stem and run "
        "identity it was captured under, in DIR/<stem>. Records already present "
        "are kept unless --overwrite; records of cases without an artifact are "
        "never touched. Each new record stores its artifact's path and SHA-256. "
        "The shared per-case cache is neither read nor written.\n\n"
        "Every artifact is checked before anything is written: schema-1 or "
        "mismatched artifacts, incomplete or foreign files named explicitly, and "
        "a checkpoint written by a different run, are refused. Cases needing every segment at once "
        "(coherent emission, temporal profile) are refused; score those with "
        "`spectrum_from_artifact` from Python."
    ),
)
@click.argument(
    "artifacts",
    nargs=-1,
    required=True,
    metavar="ARTIFACT...",
    type=_cli_completion.TRAJECTORY_ARTIFACT,
)
@click.option(
    "--checkpoint-dir",
    default=output_default("checkpoints"),
    show_default=output_label("checkpoints"),
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
        plan_stems,
        score_stem,
    )
    from ...montecarlo.runner.artifacts import STREAM_MAX_SEGMENTS
    from ...montecarlo.trajectories import TrajectoryArtifactError

    paths = [found.path for found in _discover(artifacts)]

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
