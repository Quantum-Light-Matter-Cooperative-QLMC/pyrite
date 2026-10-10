"""PyRITE Click command-line entry point."""

import os
from collections.abc import Sequence
from pathlib import Path

import click

from .. import __version__
from ..console.output import color_option, run
from ._groups import LazyGroup

_COMMANDS = {
    "run": "pyrite.cli.commands.scan.command",
    "app": "pyrite.cli.commands.app.command",
    "checkpoint": "pyrite.cli.commands.checkpoint.command",
    "config": "pyrite.cli.commands.config.command",
    "remote": "pyrite.cli.commands.remote.command",
    "job": "pyrite.cli.commands.job.command",
    "profile": "pyrite.cli.commands.profile.command",
    "material": "pyrite.cli.commands.material.command",
    "beam": "pyrite.cli.commands.beam.command",
    "detector": "pyrite.cli.commands.detector.command",
    "tables": "pyrite.cli.commands.tables.command",
}

_COMMAND_HELP = {
    "run": "Run a profile's MC sweeps and write checkpoints.",
    "app": "Launch or export interactive analysis notebooks.",
    "checkpoint": "Inspect, transform, recompute, archive, and reclaim checkpoints.",
    "config": "Set profile, catalog, workspace, and remote defaults.",
    "remote": "Run and manage MC sweeps on a remote GPU host.",
    "job": "List, inspect, follow, or stop asynchronous remote jobs.",
    "profile": "Manage named catalog campaigns and material membership.",
    "material": "Inspect, validate, edit, and blaze individual materials.",
    "beam": "Manage named beams, attachable to profiles by name.",
    "detector": "Manage named detector geometries.",
    "tables": "Inspect generated cross-section tables and external code trees.",
}


def _catalog_option(ctx: click.Context, _param: click.Parameter, catalog: Path | None) -> None:
    if catalog is None or ctx.resilient_parsing:
        return
    from ..materials.catalog import MaterialConfigError, load_material_catalog

    try:
        load_material_catalog(catalog)
    except MaterialConfigError as exc:
        raise click.BadParameter(str(exc), param_hint="--catalog") from exc
    previous = os.environ.get("PYRITE_CATALOG")
    os.environ["PYRITE_CATALOG"] = str(catalog.resolve())

    def restore_catalog() -> None:
        if previous is None:
            os.environ.pop("PYRITE_CATALOG", None)
        else:
            os.environ["PYRITE_CATALOG"] = previous

    ctx.call_on_close(restore_catalog)


@click.command(
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="PyRITE", message="PyRITE %(version)s")
@color_option
@click.option(
    "--catalog",
    type=click.Path(exists=True, path_type=Path),
    is_eager=True,
    callback=_catalog_option,
    expose_value=False,
    help=(
        "Use a complete catalog file or directory for this command; overrides "
        "PYRITE_CATALOG and catalog.path. Edits write there."
    ),
)
def command() -> None:
    """Manage coherent X-ray radiation simulation campaigns.

    Start with a named profile: it defines campaign ranges, workloads, and
    material membership. Then run campaigns, inspect checkpoints, or open analysis
    notebooks.

    Run ``pyrite COMMAND --help`` for command options, units, defaults, and side
    effects.

    \b
    Examples:
      pyrite config setup
      pyrite profile list
      pyrite profile show sub_100keV
      pyrite run sub_100keV -m hopg
      pyrite run sub_100keV --remote --dry-run
      pyrite app analysis launch
    """
    # Opt this process into moving legacy ./checkpoints-style outputs under
    # pyrite-output/ the first time a command resolves an output directory.
    from ..console.outputs import enable_legacy_migration

    enable_legacy_migration()


def main(argv: Sequence[str] | None = None):
    """Run canonical ``pyrite`` preserving exit-code and stream contracts."""
    result = run(command, argv, prog_name="pyrite")
    if isinstance(result, int):
        raise SystemExit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
