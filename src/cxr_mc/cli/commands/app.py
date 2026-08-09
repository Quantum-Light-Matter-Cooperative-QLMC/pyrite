"""Lazy ``cxr app`` hierarchy for interactive marimo applications."""

from __future__ import annotations

from copy import copy
from importlib import import_module

import click

from .._core import LazyGroup
from .._deprecations import warn_command

_LEAVES = {
    "analysis": "cxr_mc.cli.commands.app.analysis_command",
    "viewer": "cxr_mc.cli.commands.app.viewer_command",
    "validation": "cxr_mc.cli.commands.app.validation_command",
}
_HELP = {
    "analysis": "Launch or export the analysis app.",
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

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Warn only for an implicit launch, not the canonical launch/export leaf."""
        cmd_name, resolved, remaining = click.Group.resolve_command(self, ctx, args)
        help_requested = any(arg in self.get_help_option_names(ctx) for arg in remaining)
        explicit_leaf = (
            isinstance(resolved, click.Group)
            and bool(remaining)
            and remaining[0] in resolved.commands
        )
        if cmd_name is not None and not help_requested and not explicit_leaf:
            warn_command(ctx, cmd_name)
        return cmd_name, resolved, remaining


class LaunchLeafGroup(click.Group):
    """Treat an unrecognised first token as the wrapped launch command's argv."""

    def __init__(self, *args, launch: click.Command, **kwargs):
        self.launch = launch
        super().__init__(*args, params=[copy(param) for param in launch.params], **kwargs)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Keep wrapped launch parameters visible in help, but parse them downstream."""
        return [param for param in super().get_params(ctx) if param not in self.params]

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write_usage(ctx.command_path, "[OPTIONS] [MATERIAL] [COMMAND] [ARGS]...")

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            record
            for parameter in [*self.params, self.get_help_option(ctx)]
            if parameter is not None and (record := parameter.get_help_record(ctx)) is not None
        ]
        if rows:
            with formatter.section("Options"):
                formatter.write_dl(rows)
        self.format_commands(ctx, formatter)

    def invoke(self, ctx: click.Context):
        protected = ctx._protected_args
        launches = (protected and protected[0] not in self.commands) or (
            not protected and ctx.args and ctx.args[0].startswith("-")
        )
        if launches:
            ctx.args = [*protected, *ctx.args]
            ctx._protected_args = []
            callback = self.callback
            assert callback is not None
            return ctx.invoke(callback)
        return super().invoke(ctx)


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

    @click.pass_context
    def dispatch(ctx: click.Context) -> None:
        if ctx.invoked_subcommand is None:
            launch.main(args=ctx.args, prog_name=ctx.command_path, standalone_mode=False)

    leaf = LaunchLeafGroup(
        name=name,
        callback=dispatch,
        launch=launch,
        help=launch.help,
        invoke_without_command=True,
        no_args_is_help=False,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    canonical_launch = copy(launch)
    canonical_launch.name = "launch"
    leaf.add_command(canonical_launch)
    export = _load(export_path)
    if not isinstance(export, click.Command):
        raise TypeError(f"{export_path!r} did not resolve to a Click command")
    leaf.add_command(export)
    return leaf


def analysis_command() -> click.Group:
    return _launch_leaf("analysis", "cxr_mc.analyze.command", "cxr_mc.export.command")


def viewer_command() -> click.Group:
    return _launch_leaf("viewer", "cxr_mc.viewer.command", "cxr_mc.viewer.export_command")


@click.group(
    "validation",
    invoke_without_command=True,
    no_args_is_help=False,
    help="Launch notebooks/validation_app.py, or export cached validation figures.",
)
@click.option("--watch", is_flag=True, help="Pass marimo's --watch.")
@click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
@click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
@click.option("--tunnel", is_flag=True, help="Use fixed port for SSH tunneling.")
@click.pass_context
def validation_command(
    ctx: click.Context, watch: bool, edit: bool, acp: bool, tunnel: bool
) -> None:
    """Launch validation app when no nested command is selected."""
    if ctx.invoked_subcommand is None:
        _launch_validation(ctx, watch=watch, edit=edit, acp=acp, tunnel=tunnel)


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
    check = _load("cxr_mc.check.command")
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
    check = _load("cxr_mc.check.command")
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
