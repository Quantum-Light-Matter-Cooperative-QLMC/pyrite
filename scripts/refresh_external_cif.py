"""Regenerate the cached external-database lattice fixtures.

Fetches each catalog crystal that carries a ``cod_id`` (and, when ``MP_API_KEY``
is set, each ``mp_id``) from the external structure database and records its
lattice parameters in ``tests/data/external_crystal_lattices.json``. That cached
file is what the offline cross-check (``tests/materials/test_crystal_external_db.py``)
diffs the local catalog against.

Usage::

    UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/refresh_external_cif.py

Requires network access. Run whenever a ``cod_id`` / ``mp_id`` is added or an
external record is known to have changed; commit the refreshed JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from tests.helpers.external_db_fixtures import (  # noqa: E402
    EXTERNAL_LATTICE_JSON,
    MPQueryError,
    fetch_external,
    iter_specs_sorted,
    lattice_tuple,
    load_cached_lattices,
    write_cached_lattices,
)


def main() -> int:
    entries = load_cached_lattices()
    written = 0
    skipped: list[str] = []
    failed: list[str] = []
    for key, cod_id, mp_id in iter_specs_sorted():
        try:
            crystal = fetch_external(cod_id, mp_id)
        except MPQueryError as exc:
            failed.append(key)
            print(f"  keep {key}: {exc}")
            continue
        if crystal is None:
            skipped.append(key)
            print(f"  skip {key}: mp-only, no MP_API_KEY set (keeping any cached entry)")
            continue
        a, b, c, alpha, beta, gamma = lattice_tuple(crystal)
        source = f"cod:{cod_id}" if cod_id is not None else f"mp:{mp_id}"
        entries[key] = {
            "source": source,
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
        print(
            f"  {key:20s} <- {source:16s} "
            f"({a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f})"
        )
        written += 1
    write_cached_lattices(entries)
    print(f"\nwrote {written} lattices -> {EXTERNAL_LATTICE_JSON}")
    if skipped:
        print(
            f"skipped {len(skipped)} mp-only entries (set MP_API_KEY to include): {', '.join(skipped)}"
        )
    if failed:
        print(f"failed {len(failed)} MP entries; cached values were retained: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
