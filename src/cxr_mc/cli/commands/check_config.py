"""Click wiring for material-catalog validation."""

from pathlib import Path

import click

from ...validation import check_config as _check_config
from .. import _core as _cli_core


@click.command(
    "check-config",
    help=(
        "Validate bundled material catalog or an explicit full catalog TOML.\n\n"
        "With no MANIFEST, reloads packaged materials.toml. Performs no simulation, "
        "network access, or GPU probe."
    ),
)
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def command(manifest):
    return _cli_core.invoke_legacy(_check_config._run, manifest=manifest)
