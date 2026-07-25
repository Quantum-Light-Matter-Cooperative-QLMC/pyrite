"""Validate the bundled material catalog or an explicit full catalog TOML."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click

from . import _cli_core
from .materials import MaterialConfigError, load_material_catalog


def _run(args: SimpleNamespace) -> None:
    path = Path(args.manifest) if args.manifest is not None else None
    try:
        catalog = load_material_catalog(path)
    except MaterialConfigError as exc:
        raise SystemExit(str(exc)) from None

    source = str(path) if path is not None else "bundled catalog"
    print(
        f"valid material catalog: {source} "
        f"({len(catalog.materials)} materials, {len(catalog.crystals)} crystals, "
        f"{len(catalog.media)} media)"
    )


@click.command(
    "check-config",
    help="Validate bundled material catalog or an explicit full catalog TOML.",
)
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def command(manifest):
    return _cli_core.invoke_legacy(_run, manifest=manifest)


__all__ = ["command"]
