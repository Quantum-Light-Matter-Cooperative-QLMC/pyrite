"""Shared helpers for the external structure-database cross-check.

Backs both the offline regression test (``test_crystal_external_db.py``) and the
regeneration script (``scripts/refresh_external_cif.py``). See
``docs/crystal-db-comparison.md`` for the design: this guards local crystal
*lattice geometry* against silent drift by diffing each catalog entry against a
pinned external record fetched once via the vendored ``crystals`` library.

The cached fixture stores the external **lattice parameters** only
(``tests/data/external_crystal_lattices.json``), not a full CIF: the comparison
is lattice-geometry-only, and ``crystals.Crystal.to_cif`` requires an spglib
symmetry pass that fails on several low-symmetry layered cells here. Storing the
six comparable numbers is deterministic and offline by construction.

It deliberately does NOT touch Debye-Waller / thermal parameters -- external
databases do not carry a reliable isotropic B, so ``issue_notes.md`` item #1
stays a separate concern.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crystals import Crystal

Lattice = tuple[float, float, float, float, float, float]

# Cached external lattices live beside the golden fixture.
EXTERNAL_LATTICE_JSON = Path(__file__).parent / "data" / "external_crystal_lattices.json"

# COD ships experimental cells, so an exact (<=1e-2 A / 0.1 deg) match is
# expected. Materials Project cells are DFT-relaxed and carry a systematic
# ~1-2 % offset, so they get a looser relative tolerance. Angles use the same
# absolute window for both sources.
TOL_COD_ANG = 1e-2
TOL_MP_REL = 2e-2
TOL_ANGLE_DEG = 0.1

# Materials Project needs an API key; read it from the environment so CI never
# hard-depends on a secret. Absent key -> MP entries are skipped, not failed.
MP_API_KEY_ENV = "MP_API_KEY"


def external_specs() -> list[tuple[str, int | None, str | None]]:
    """(crystal_key, cod_id, mp_id) for every catalog entry carrying an id."""
    from cxr_mc.materials import CATALOG

    specs: list[tuple[str, int | None, str | None]] = []
    for key, spec in CATALOG.crystals.items():
        if spec.cod_id is not None or spec.mp_id is not None:
            specs.append((key, spec.cod_id, spec.mp_id))
    return specs


def iter_specs_sorted() -> Iterator[tuple[str, int | None, str | None]]:
    """External specs sorted by crystal key for deterministic iteration."""
    yield from sorted(external_specs())


def lattice_tuple(crystal: Crystal) -> Lattice:
    """(a, b, c, alpha, beta, gamma) in Angstrom / degrees as plain floats."""
    a, b, c, alpha, beta, gamma = crystal.lattice_parameters
    return (float(a), float(b), float(c), float(alpha), float(beta), float(gamma))


def local_lattice_tuple(key: str) -> Lattice:
    """Local catalog lattice as (a, b, c, alpha, beta, gamma).

    Fills the right-angle / hexagonal defaults the catalog omits for
    high-symmetry systems so every entry compares on the full 6-tuple.
    """
    from cxr_mc.materials import CATALOG

    lat = dict(CATALOG.crystal(key).lattice)
    a = float(lat["a"])
    b = float(lat.get("b", a))
    c = float(lat.get("c", a))
    system = lat.get("system")
    default_gamma = 120.0 if system == "hexagonal" else 90.0
    alpha = float(lat.get("alpha", 90.0))
    beta = float(lat.get("beta", 90.0))
    gamma = float(lat.get("gamma", default_gamma))
    return (a, b, c, alpha, beta, gamma)


def fetch_external(cod_id: int | None, mp_id: str | None) -> Crystal | None:
    """Live-fetch a crystal from COD (preferred) or Materials Project.

    Returns ``None`` when only an ``mp_id`` is available and no API key is set,
    so callers can skip rather than fail. Requires network access.
    """
    from crystals import Crystal

    if cod_id is not None:
        return Crystal.from_cod(cod_id)
    if mp_id is not None:
        api_key = os.environ.get(MP_API_KEY_ENV)
        if not api_key:
            return None
        return Crystal.from_mp(mp_id, api_key=api_key)
    return None


def load_cached_lattices() -> dict[str, dict[str, object]]:
    """Committed external lattices keyed by crystal key (empty if absent)."""
    if not EXTERNAL_LATTICE_JSON.exists():
        return {}
    data = json.loads(EXTERNAL_LATTICE_JSON.read_text())
    return dict(data.get("crystals", {}))


def cached_lattice_tuple(entry: dict[str, object]) -> Lattice:
    return (
        float(entry["a"]),  # type: ignore[arg-type]
        float(entry["b"]),  # type: ignore[arg-type]
        float(entry["c"]),  # type: ignore[arg-type]
        float(entry["alpha"]),  # type: ignore[arg-type]
        float(entry["beta"]),  # type: ignore[arg-type]
        float(entry["gamma"]),  # type: ignore[arg-type]
    )


def write_cached_lattices(entries: dict[str, dict[str, object]]) -> None:
    payload = {
        "_note": (
            "External lattice references for the crystal-DB cross-check. "
            "Regenerate with scripts/refresh_external_cif.py. Geometry only; "
            "see docs/crystal-db-comparison.md."
        ),
        "crystals": {key: entries[key] for key in sorted(entries)},
    }
    EXTERNAL_LATTICE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
