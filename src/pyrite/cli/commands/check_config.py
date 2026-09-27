"""``pyrite material validate`` / ``pyrite check-config``: catalog validation.

The whole command lives here, wiring and body. It used to run from
``pyrite.validation.check_config``, which had to reach up into ``cli._core`` for
its output helpers and back into this module for the Click object -- both halves
of the ``validation`` <-> ``cli`` import cycle. Nothing about it is a validation
model: it loads the catalog and prints a one-line summary.
"""

from pathlib import Path
from types import SimpleNamespace

import click

from ...console import output as _cli_core
from ...materials import MaterialConfigError, load_material_catalog


def _run(args: SimpleNamespace) -> None:
    path = Path(args.manifest) if args.manifest is not None else None
    try:
        catalog = load_material_catalog(path)
    except MaterialConfigError as exc:
        raise SystemExit(str(exc)) from None

    source = str(path) if path is not None else "bundled catalog"
    print(
        f"{_cli_core.paint('valid', 'done')} material catalog: {source} "
        f"({len(catalog.materials)} materials, {len(catalog.crystals)} crystals, "
        f"{len(catalog.media)} media)"
    )


@click.command(
    "check-config",
    help=(
        "Validate bundled material catalog or an explicit full catalog.\n\n"
        "MANIFEST is a single catalog TOML file or a catalog directory. With no "
        "MANIFEST, reloads the packaged catalog. Performs no simulation, "
        "network access, or GPU probe."
    ),
)
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def command(manifest):
    return _cli_core.invoke_legacy(_run, manifest=manifest)


__all__ = ["_run", "command"]
