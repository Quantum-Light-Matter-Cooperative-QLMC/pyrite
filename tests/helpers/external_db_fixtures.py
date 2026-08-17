"""Shared helpers for the external structure-database cross-check.

Backs both the offline regression test (``test_crystal_external_db.py``) and the
regeneration script (``scripts/refresh_external_cif.py``). See
``docs/validation/materials/crystal-db-comparison.md`` for the design: this guards local crystal
*lattice geometry* against silent drift by diffing each catalog entry against a
pinned external record fetched once through COD's vendored ``crystals`` adapter
or Materials Project's optional supported ``mp-api`` client.

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crystals import Crystal

Lattice = tuple[float, float, float, float, float, float]

# Cached external lattices live beside the golden fixture.
EXTERNAL_LATTICE_JSON = Path(__file__).parent.parent / "data" / "external_crystal_lattices.json"

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
_REPO_ROOT = Path(__file__).resolve().parents[1]


class MPQueryError(RuntimeError):
    """A configured Materials Project query could not be completed."""


def _dotenv_mp_api_key(path: Path) -> str | None:
    """Read ``MP_API_KEY`` from a local dotenv file without mutating ``os.environ``."""
    if not path.is_file():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != MP_API_KEY_ENV:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def resolve_mp_api_key(
    *, environ: Mapping[str, str] | None = None, dotenv_path: Path | None = None
) -> str | None:
    """Resolve a local MP key; a non-empty exported value wins over ``.env``."""
    environment = os.environ if environ is None else environ
    exported = environment.get(MP_API_KEY_ENV)
    if exported:
        return exported
    return _dotenv_mp_api_key(_REPO_ROOT / ".env" if dotenv_path is None else dotenv_path)


def _mp_rester(api_key: str) -> object:
    """Construct the optional, supported Materials Project client lazily."""
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise MPQueryError(
            "Materials Project lookup requires the external-db extra; "
            "run `uv sync --extra external-db` before retrying."
        ) from exc
    return MPRester(api_key)


def _structure_lattice_tuple(structure: object) -> Lattice:
    lattice = structure.lattice  # type: ignore[attr-defined]
    return tuple(float(value) for value in (*lattice.abc, *lattice.angles))  # type: ignore[return-value]


def fetch_mp_lattice(
    mp_id: str, api_key: str, *, rester_factory: Callable[[str], object] | None = None
) -> Lattice:
    """Fetch MP's final relaxed lattice through supported ``mp-api`` client."""
    factory = _mp_rester if rester_factory is None else rester_factory
    try:
        with factory(api_key) as rester:  # type: ignore[union-attr]
            structure = rester.get_structure_by_material_id(mp_id)  # type: ignore[attr-defined]
        return _structure_lattice_tuple(structure)
    except MPQueryError:
        raise
    except Exception as exc:
        raise MPQueryError(
            f"Materials Project query failed for {mp_id} ({type(exc).__name__}); "
            "verify MP_API_KEY access and retry."
        ) from exc


def fetch_mp_candidate_lattices(
    mp_id: str, api_key: str, *, rester_factory: Callable[[str], object] | None = None
) -> list[Lattice]:
    """Final relaxed lattice plus every pre-relaxation (initial) lattice.

    MP's initial structures are the experimental inputs a local CIF may
    legitimately derive from -- ``hfte2`` matches the mp-32887 pre-relaxation
    cell exactly -- so the live cross-check accepts a match against any of
    them rather than the final relaxed cell alone.
    """
    factory = _mp_rester if rester_factory is None else rester_factory
    try:
        with factory(api_key) as rester:  # type: ignore[union-attr]
            final = rester.get_structure_by_material_id(mp_id)  # type: ignore[attr-defined]
            initials = rester.get_structure_by_material_id(mp_id, final=False)  # type: ignore[attr-defined]
        if not isinstance(initials, list):
            initials = [initials]
        return [_structure_lattice_tuple(structure) for structure in (final, *initials)]
    except MPQueryError:
        raise
    except Exception as exc:
        raise MPQueryError(
            f"Materials Project query failed for {mp_id} ({type(exc).__name__}); "
            "verify MP_API_KEY access and retry."
        ) from exc


# MP-pinned entries whose ``mp_id`` is a provenance pointer only: their local
# geometry deliberately tracks experimental literature (``mose2``: Bronsema
# 1986; ``gese2``: Dittmar & Schaefer 1976 via the 2018 MP snapshot) or a
# historical MP relaxation (``res2``: 2018 snapshot), none of which current MP
# final or initial cells reproduce. See the "MP live audit" section of
# docs/validation/materials/crystal-db-comparison.md. The live cross-check
# skips the geometry assertion for these rather than failing on MP's
# re-relaxed cell.
MP_PROVENANCE_ONLY = frozenset({"gese2", "mose2", "res2"})


def external_specs() -> list[tuple[str, int | None, str | None]]:
    """(crystal_key, cod_id, mp_id) for every catalog entry carrying an id."""
    from pyrite.materials import CATALOG

    specs: list[tuple[str, int | None, str | None]] = []
    for key, spec in CATALOG.crystals.items():
        if spec.cod_id is not None or spec.mp_id is not None:
            specs.append((key, spec.cod_id, spec.mp_id))
    return specs


def iter_specs_sorted() -> Iterator[tuple[str, int | None, str | None]]:
    """External specs sorted by crystal key for deterministic iteration."""
    yield from sorted(external_specs())


def lattice_tuple(crystal: Crystal | Lattice) -> Lattice:
    """(a, b, c, alpha, beta, gamma) in Angstrom / degrees as plain floats."""
    if isinstance(crystal, tuple):
        return crystal
    a, b, c, alpha, beta, gamma = crystal.lattice_parameters
    return (float(a), float(b), float(c), float(alpha), float(beta), float(gamma))


def local_lattice_tuple(key: str) -> Lattice:
    """Local catalog lattice as (a, b, c, alpha, beta, gamma).

    Fills the right-angle / hexagonal defaults the catalog omits for
    high-symmetry systems so every entry compares on the full 6-tuple.
    """
    from pyrite.materials import CATALOG

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


def fetch_external(cod_id: int | None, mp_id: str | None) -> Crystal | Lattice | None:
    """Live-fetch lattice data from COD (preferred) or Materials Project.

    Returns ``None`` when only an ``mp_id`` is available and no API key is set,
    so callers can skip rather than fail. A configured MP query failure raises
    :class:`MPQueryError` rather than silently skipping. Requires network access.
    """
    from crystals import Crystal

    if cod_id is not None:
        return Crystal.from_cod(cod_id)
    if mp_id is not None:
        api_key = resolve_mp_api_key()
        if not api_key:
            return None
        return fetch_mp_lattice(mp_id, api_key)
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
            "see docs/validation/materials/crystal-db-comparison.md."
        ),
        "crystals": {key: entries[key] for key in sorted(entries)},
    }
    EXTERNAL_LATTICE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
