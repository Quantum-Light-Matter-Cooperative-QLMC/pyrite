"""Validate the bundled material catalog or an explicit full catalog TOML."""

from __future__ import annotations

import argparse
from pathlib import Path

from .materials import MaterialConfigError, load_material_catalog


def _run(args: argparse.Namespace) -> None:
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


def add_subparser(sub):
    """Register the ``check-config`` subcommand."""
    parser = sub.add_parser(
        "check-config",
        help="validate the bundled material catalog or an explicit full catalog TOML",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        help="full material catalog TOML (default: reload the bundled catalog)",
    )
    parser.set_defaults(func=_run)
    return parser


__all__ = ["add_subparser"]
