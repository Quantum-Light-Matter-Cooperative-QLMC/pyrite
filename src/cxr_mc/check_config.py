"""Validate the bundled material catalog or an explicit full catalog TOML."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .cli import _core as _cli_core
from .materials import MaterialConfigError, load_material_catalog


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


def __getattr__(name: str):
    if name == "command":
        from .cli.commands.check_config import command

        return command
    raise AttributeError(name)


__all__ = ["_run"]
