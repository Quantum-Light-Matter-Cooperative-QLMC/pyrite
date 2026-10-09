"""Lazy saved HDF5 trajectory viewer launch and export contracts."""

from pathlib import Path

import click

from ...montecarlo.trajectories import TrajectoryArtifactError
from ...trajectory_viewer.selection import load_selection, plan_selection


def _selection_options(fn):
    for decorator in (
        click.argument("artifact", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option(
            "--history",
            "histories",
            multiple=True,
            type=click.IntRange(min=0),
            help="Complete history ID; repeatable. Intersects --track.",
        ),
        click.option(
            "--track",
            "tracks",
            multiple=True,
            type=click.IntRange(min=0),
            help="Complete captured track ID; repeatable.",
        ),
        click.option(
            "--attribute",
            "attributes",
            multiple=True,
            help="Additional numeric segment field; identity/control fields always included.",
        ),
        click.option(
            "--max-segments",
            default=1_000_000,
            type=click.IntRange(min=1),
            show_default=True,
            help="Admission limit; never truncates a track.",
        ),
        click.option(
            "--memory-mib",
            default=2048,
            type=click.IntRange(min=1),
            show_default=True,
            help="Estimated selection-memory budget (not a hard RSS limit).",
        ),
    ):
        fn = decorator(fn)
    return fn


def _plan(artifact, memory_mib, **selection):
    try:
        plan = plan_selection(artifact, **selection)
        click.echo(
            f"{plan.segments:,} segments; selected arrays {plan.array_bytes / 1024**2:.1f} MiB; estimated peak {plan.estimated_peak_bytes / 1024**2:.1f} MiB; budget {memory_mib} MiB",
            err=True,
        )
        return plan
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise click.ClickException(str(error)) from error


def _viewer(plan, memory_mib, off_screen):
    try:
        data = load_selection(plan, memory_budget_bytes=memory_mib * 1024**2)
        from ...trajectory_viewer.render import CaptureViewer

        return CaptureViewer(plan, data, off_screen=off_screen)
    except ImportError as error:
        raise click.ClickException(
            "optional viewer dependencies missing; install pyrite-xray[trajectory-viewer] (checkout: uv sync --extra trajectory-viewer)"
        ) from error
    except (OSError, ValueError, TrajectoryArtifactError, MemoryError) as error:
        raise click.ClickException(
            f"{error}; reduce --history/--track selection and retry"
        ) from error


@click.group("trajectories", no_args_is_help=True)
def command():
    """Inspect saved HDF5 showers with optional PyVista/trame; capture stays read-only."""


@command.command("inspect")
@_selection_options
def inspect_command(artifact, memory_mib, **selection):
    """Report bounded selection size/cost without loading rendering libraries."""
    plan = _plan(artifact, memory_mib, **selection)
    click.echo(f"{plan.segments} segments; fields: {', '.join(plan.fields)}")


@command.command("launch")
@_selection_options
@click.option(
    "--native",
    is_flag=True,
    help="Use native desktop controls instead of the image-streaming browser server.",
)
@click.option(
    "--port",
    default=2722,
    type=click.IntRange(1, 65535),
    show_default=True,
    help="Browser server port on 127.0.0.1; use an SSH tunnel remotely.",
)
@click.option(
    "--output-dir",
    default="trajectory-exports",
    type=click.Path(file_okay=False),
    show_default=True,
    help="Browser exports are unique PNG/GIF files on the server.",
)
def launch_command(artifact, memory_mib, native, port, output_dir, **selection):
    """Launch reactive server rendering, or a native desktop window."""
    plan = _plan(artifact, memory_mib, **selection)
    viewer = _viewer(plan, memory_mib, not native)
    try:
        if native:
            from ...trajectory_viewer.native import show

            show(viewer, output_dir=output_dir)
        else:
            from ...trajectory_viewer.server import build_server

            server = build_server(viewer, output_dir=output_dir)
            click.echo(
                f"http://127.0.0.1:{port}; remote: ssh -L {port}:127.0.0.1:{port} HOST", err=True
            )
            server.start(host="127.0.0.1", port=port, open_browser=False)
    except ImportError as error:
        raise click.ClickException(
            "install pyrite-xray[trajectory-viewer] for browser dependencies"
        ) from error
    finally:
        viewer.close()


@command.command("export")
@_selection_options
@click.argument("output", type=click.Path(path_type=Path))
@click.option(
    "--scale", type=click.Choice(["closeup", "instrument"]), default="closeup", show_default=True
)
@click.option(
    "--frames",
    default=30,
    type=click.IntRange(2, 120),
    show_default=True,
    help="GIF orbit frame bound; ignored for PNG.",
)
@click.option(
    "--fps", default=10, type=click.IntRange(1, 30), show_default=True, help="GIF frame rate."
)
@click.option(
    "--overwrite", is_flag=True, help="Replace the selected output and its provenance sidecar."
)
def export_command(artifact, output, memory_mib, scale, frames, fps, overwrite, **selection):
    """Write PNG or bounded orbit GIF plus <output>.json; no browser required."""
    if output.suffix.lower() not in {".png", ".gif"}:
        raise click.BadParameter("output must end in .png or .gif", param_hint="OUTPUT")
    for target in (output, output.with_suffix(output.suffix + ".json")):
        if target.exists() and not overwrite:
            raise click.ClickException(f"output exists: {target}; pass --overwrite")
    plan = _plan(artifact, memory_mib, **selection)
    viewer = _viewer(plan, memory_mib, True)
    try:
        viewer.show_scale(scale)
        if output.suffix.lower() == ".png":
            viewer.screenshot(output, overwrite=overwrite)
        else:
            viewer.movie(output, frames=frames, fps=fps, overwrite=overwrite)
        click.echo(str(output))
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        viewer.close()
