"""Generate and pin SBETHE stopping tables for the runtime material catalog.

Maintainer-only: needs gfortran and the fetched SBETHE ``sdbase`` tree. The
generated ``.npz`` payloads and provenance manifests are copied into the
package's two-tier table store, where production transport resolves them by
material identity rather than by catalog label.

Usage::

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python scripts/release_sbethe_tables.py \
        --generate --pin
"""

import argparse
import shutil

from pyrite.materials import CATALOG
from pyrite.xsgen.sbethe import catalog_material, generate_material, resolve_catalog_table
from pyrite.xsgen.store import packaged_table_dir


def _catalog_keys() -> list[str]:
    """Return runnable materials plus amorphous media used in target stacks."""
    return sorted(set(CATALOG.materials) | set(CATALOG.media))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", action="append", default=None, help="restrict to key")
    parser.add_argument("--generate", action="store_true", help="generate absent user tables")
    parser.add_argument("--pin", action="store_true", help="copy tables into package data")
    args = parser.parse_args(argv)

    keys = _catalog_keys() if args.material is None else args.material
    destination = packaged_table_dir()
    if args.pin:
        destination.mkdir(parents=True, exist_ok=True)

    tables = {}
    for key in keys:
        if args.generate:
            material = catalog_material(key)
            generate_material(
                material.key,
                material.composition,
                density_g_cm3=material.density_g_cm3,
                mean_excitation_eV=material.mean_excitation_eV,
                band_gap_eV=material.band_gap_eV,
            )
        table = resolve_catalog_table(key)
        tables.setdefault(table.key, table)

    if args.pin:
        for table in tables.values():
            shutil.copyfile(table.path, destination / f"{table.key}.npz")
            shutil.copyfile(table.path.with_suffix(".json"), destination / f"{table.key}.json")

    print(f"catalog keys: {len(keys)}")
    print(f"unique tables: {len(tables)}")
    if args.pin:
        print(f"pinned: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
