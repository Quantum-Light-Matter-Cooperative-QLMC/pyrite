"""Lazy ``pyrite app`` hierarchy for interactive marimo applications."""

from copy import copy
from importlib import import_module

import click

from .._groups import LazyGroup

_LEAVES = {
    "analysis": "pyrite.cli.commands.app.analysis_command",
    "pixels": "pyrite.cli.commands.app.pixels_command",
    "compare": "pyrite.cli.commands.app.compare_command",
    "viewer": "pyrite.cli.commands.app.viewer_command",
    "validation": "pyrite.cli.commands.app.validation_command",
}
_HELP = {
    "analysis": "Launch or export the analysis app.",
    "pixels": "Launch or export the pixel-detector observation app.",
    "compare": "Launch or export the case and cross-material comparison app.",
    "viewer": "Launch or export the 3D trajectory viewer.",
    "validation": "Launch or export cached validation figures.",
}


class AppGroup(LazyGroup):
    """Resolve leaf factories without importing app implementation modules for help."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        import_path = self.lazy_commands.get(cmd_name)
        if import_path is None:
            return super().get_command(ctx, cmd_name)
        loaded = _load(import_path)
        return loaded() if callable(loaded) and not isinstance(loaded, click.Command) else loaded


@click.command(cls=AppGroup, lazy_commands=_LEAVES, lazy_help=_HELP, no_args_is_help=True)
def command() -> None:
    """Launch or export interactive analysis notebooks."""


def _load(path: str):
    module_name, object_name = path.rsplit(".", 1)
    loaded = getattr(import_module(module_name), object_name)
    return loaded


def _launch_leaf(name: str, launch_path: str, export_path: str) -> click.Group:
    launch = _load(launch_path)
    if not isinstance(launch, click.Command):
        raise TypeError(f"{launch_path!r} did not resolve to a Click command")

    leaf = click.Group(name=name, help=launch.help, no_args_is_help=True)
    canonical_launch = copy(launch)
    canonical_launch.name = "launch"
    leaf.add_command(canonical_launch)
    export = _load(export_path)
    if not isinstance(export, click.Command):
        raise TypeError(f"{export_path!r} did not resolve to a Click command")
    leaf.add_command(export)
    return leaf


def analysis_command() -> click.Group:
    return _launch_leaf(
        "analysis",
        "pyrite.cli.commands.app_analysis.command",
        "pyrite.cli.commands.export.command",
    )


def pixels_command() -> click.Group:
    return _launch_leaf(
        "pixels",
        "pyrite.cli.commands.app_views.pixels_command",
        "pyrite.cli.commands.app_views.pixels_export_command",
    )


def compare_command() -> click.Group:
    return _launch_leaf(
        "compare",
        "pyrite.cli.commands.app_views.compare_command",
        "pyrite.cli.commands.app_views.compare_export_command",
    )


def viewer_command() -> click.Group:
    return _launch_leaf(
        "viewer",
        "pyrite.cli.commands.app_viewer.command",
        "pyrite.cli.commands.app_viewer.export_command",
    )


@click.group(
    "validation",
    no_args_is_help=True,
    help="Launch the interactive validation application, or export cached figures.",
)
def validation_command() -> None:
    """Launch or export cached validation figures."""


@click.command("launch", help="Launch the interactive validation application.")
@click.option("--watch", is_flag=True, help="Pass marimo's --watch.")
@click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
@click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
@click.option("--tunnel", is_flag=True, help="Use fixed port for SSH tunneling.")
@click.pass_context
def validation_launch_command(
    ctx: click.Context, watch: bool, edit: bool, acp: bool, tunnel: bool
) -> None:
    _launch_validation(ctx, watch=watch, edit=edit, acp=acp, tunnel=tunnel)


def _launch_validation(
    ctx: click.Context, *, watch: bool, edit: bool, acp: bool, tunnel: bool
) -> None:
    check = _load("pyrite.cli.commands.app_validation.command")
    ctx.invoke(
        check,
        watch=watch,
        edit=edit,
        acp=acp,
        tunnel=tunnel,
        export_=False,
        outdir="figures",
        ne=20_000,
        ne_brem=200,
        ne_supp=200,
    )


@click.command("export", help="Write cached validation figures; never starts marimo.")
@click.option("--outdir", type=click.Path(file_okay=False), default="figures", show_default=True)
@click.option("--ne", type=int, default=20_000, show_default=True)
@click.option("--ne-brem", type=int, default=200, show_default=True)
@click.option("--ne-supp", type=int, default=200, show_default=True)
@click.pass_context
def validation_export_command(
    ctx: click.Context, outdir: str, ne: int, ne_brem: int, ne_supp: int
) -> None:
    """Delegate cached-figure export to existing validation orchestration."""
    check = _load("pyrite.cli.commands.app_validation.command")
    ctx.invoke(
        check,
        watch=False,
        edit=False,
        acp=False,
        tunnel=False,
        export_=True,
        outdir=outdir,
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
    )


validation_command.add_command(validation_launch_command)
validation_command.add_command(validation_export_command)
