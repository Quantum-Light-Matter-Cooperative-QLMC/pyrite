"""Build validated material definitions, then explicitly register them in a catalog.

No build or save operation generates transport tables or downloads data.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from scipy.constants import Avogadro

from .. import _catalog_layout as layout
from ._catalog_decode import _Errors
from ._parse import _parse_crystals, _parse_media
from ._schema import MaterialConfigError, MediumSpec
from ._transport_data import TRANSPORT_ELEMENTS
from .catalog import (
    MaterialCatalog,
    _load_material_catalog_cached,
    _refresh_default_catalog,
    load_material_catalog,
)

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?")


@dataclass(frozen=True)
class MaterialDefinition:
    """Immutable, validated authoring result; not a persisted registration.

    ``composition`` contains atoms/angstrom³. ``identity`` hashes the normalized
    physical definition, excluding its registration name. ``record_toml`` is
    an implementation serialization, not the authoring input interface. CIF
    bytes are captured at build time, so save does not depend on the input file.
    """

    name: str
    kind: str
    composition: tuple[tuple[str, float], ...]
    density_g_cm3: float
    identity: str
    record_toml: str
    cif_bytes: bytes | None = None

    @property
    def medium(self) -> MediumSpec:
        """Return an inline medium for transport/attenuation, without registration."""
        return MediumSpec(self.name, self.composition)


def _name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            "name must start with a letter/digit and use letters, digits, '.', '_', '-'"
        )


def _positive(value: float, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be finite and positive") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _supported(elements: Mapping[str, float]) -> None:
    unsupported = sorted(set(elements) - set(TRANSPORT_ELEMENTS))
    if unsupported:
        raise ValueError(
            f"unsupported transport elements: {', '.join(unsupported)}; "
            f"supported: {', '.join(sorted(TRANSPORT_ELEMENTS))}"
        )


def _formula(formula: str) -> dict[str, float]:
    if not isinstance(formula, str) or not formula:
        raise ValueError("formula must be a nonempty ASCII formula, e.g. Si or Al2O3")
    position = 0
    counts: dict[str, float] = {}
    for match in _TOKEN.finditer(formula):
        if match.start() != position:
            break
        element, count = match.groups()
        counts[element] = counts.get(element, 0.0) + _positive(
            float(count) if count else 1.0, f"formula count for {element}"
        )
        position = match.end()
    if position != len(formula) or not counts:
        raise ValueError(
            f"invalid formula {formula!r}; use element symbols and positive counts "
            "(fractional counts allowed), without parentheses, charges, or isotope labels"
        )
    _supported(counts)
    return counts


def _weights(values: Mapping[str, float], field: str, *, fractions: bool) -> dict[str, float]:
    if not values:
        raise ValueError(f"{field} must be nonempty")
    _supported(values)
    out = {element: _positive(value, f"{field}[{element}]") for element, value in values.items()}
    if fractions and not math.isclose(math.fsum(out.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"{field} must sum to 1; use atom_counts for stoichiometric ratios")
    return out


def _density(composition: tuple[tuple[str, float], ...]) -> float:
    _supported(dict(composition))
    return math.fsum(n * TRANSPORT_ELEMENTS[e]["A"] for e, n in composition) / (Avogadro * 1e-24)


def _definition(
    name: str,
    kind: str,
    composition: tuple[tuple[str, float], ...],
    record: Mapping,
    cif_bytes: bytes | None = None,
) -> MaterialDefinition:
    import tomlkit

    physical = {"kind": kind, "record": record}
    # The managed CIF filename is a content hash, not a registration name.
    identity = hashlib.sha256(
        json.dumps(physical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return MaterialDefinition(
        name,
        kind,
        composition,
        _density(composition),
        identity,
        tomlkit.dumps(record),
        cif_bytes,
    )


def build_medium(
    name: str,
    *,
    density_g_cm3: float,
    formula: str | None = None,
    mass_fractions: Mapping[str, float] | None = None,
    atomic_fractions: Mapping[str, float] | None = None,
    atom_counts: Mapping[str, float] | None = None,
) -> MaterialDefinition:
    """Build an amorphous medium without writes.

    Supply exactly one composition representation and an explicit bulk density
    in g/cm³. Fractions must sum to one; atom counts are positive ratios. Simple
    ASCII formulas allow fractional counts. Native number densities use the
    existing transport atomic weights and Avogadro's constant. Supported
    elements are limited to ``TRANSPORT_ELEMENTS``; no isotopic enrichment.
    """
    _name(name)
    density = _positive(density_g_cm3, "density_g_cm3 (g/cm³)")
    if (
        sum(value is not None for value in (formula, mass_fractions, atomic_fractions, atom_counts))
        != 1
    ):
        raise ValueError(
            "supply exactly one of formula, mass_fractions, atomic_fractions, atom_counts"
        )
    if formula is not None:
        weights = _formula(formula)
    elif mass_fractions is not None:
        weights = _weights(mass_fractions, "mass_fractions", fractions=True)
        weights = {e: value / TRANSPORT_ELEMENTS[e]["A"] for e, value in weights.items()}
    elif atomic_fractions is not None:
        weights = _weights(atomic_fractions, "atomic_fractions", fractions=True)
    else:
        assert atom_counts is not None
        weights = _weights(atom_counts, "atom_counts", fractions=False)
    weights = dict(sorted(weights.items()))
    scale = max(weights.values())
    weights = {e: value / scale for e, value in weights.items()}
    mass = math.fsum(value * TRANSPORT_ELEMENTS[e]["A"] for e, value in weights.items())
    composition = tuple(
        (e, float(f"{density * Avogadro * 1e-24 * value / mass:.15g}"))
        for e, value in weights.items()
    )
    errors = _Errors()
    record = {"composition": dict(composition)}
    _parse_media({name: record}, errors)
    if errors.items:
        raise MaterialConfigError(tuple(errors.items))
    return _definition(name, "medium", composition, record)


def _safe_path(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError(f"unsafe traversal in path: {path}")
    path = path.absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"unsafe symlink: {component}")
    return path


def build_crystal(
    name: str,
    *,
    crystal: str | None = None,
    cif: Path | str | None = None,
    phase: str | None = None,
    formula: str | None = None,
    surface_hkl: tuple[int, int, int] | None = None,
    beam_uvw: tuple[int, int, int] | None = None,
    B_ang2: float | None = None,
    mosaic_fwhm_deg: float | None = None,
    density_g_cm3: float | None = None,
    catalog: Path | None = None,
) -> MaterialDefinition:
    """Build a crystal recipe from an existing crystal or a fully occupied CIF.

    Existing crystals reuse their catalog configuration; overrides are rejected.
    A new CIF requires formula, phase, exactly one orientation, nonnegative
    ``B_ang2``. Photon grids belong to numerics. Reflections use the existing
    automatic selection policy. Density is derived from the lattice; an optional
    supplied density must agree within 1e-6 relative tolerance. Partial occupancy
    and unsupported transport elements are rejected by existing validators.
    """
    _name(name)
    if (crystal is None) == (cif is None):
        raise ValueError("supply exactly one of crystal or cif")
    if crystal is not None:
        if any(
            value is not None
            for value in (
                phase,
                formula,
                surface_hkl,
                beam_uvw,
                B_ang2,
                mosaic_fwhm_deg,
            )
        ):
            raise ValueError("existing crystal configuration cannot be overridden; use a local CIF")
        loaded = load_material_catalog(catalog)
        if crystal not in loaded.crystals:
            raise ValueError(f"unknown crystal: {crystal}")
        spec = loaded.crystals[crystal]
        composition = tuple(sorted(spec.composition))
        definition = _definition(name, "crystal", composition, {"crystal": crystal})
        # Reference identity must include the resolved structure and configuration.
        sources = layout.read_raw(layout.selected_catalog() if catalog is None else catalog)
        resolved = dict(sources["crystals"][crystal])
        resolved.pop("E_grid", None)
        resolved["cif"] = f"cifs/{hashlib.sha256(spec.cif.read_bytes()).hexdigest()}.cif"
        identity = _definition(name, "cif", composition, resolved).identity
        definition = MaterialDefinition(
            name,
            definition.kind,
            composition,
            definition.density_g_cm3,
            identity,
            definition.record_toml,
        )
    else:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase is required for a CIF crystal")
        if formula is None:
            raise ValueError("formula is required for a CIF crystal")
        counts = _formula(formula)
        if B_ang2 is None or isinstance(B_ang2, bool) or not math.isfinite(B_ang2) or B_ang2 < 0:
            raise ValueError("B_ang2 must be explicitly supplied, finite and nonnegative")
        assert cif is not None
        source = _safe_path(Path(cif))
        if not source.is_file():
            raise ValueError(f"CIF is not a regular file: {source}")
        data = source.read_bytes()
        filename = f"cifs/{hashlib.sha256(data).hexdigest()}.cif"
        record = {
            "cif": filename,
            "validation_id": "user-supplied-unverified",
            "formula": formula,
            "phase": phase.strip(),
            "B_ang2": float(B_ang2),
        }
        if surface_hkl is not None:
            record["surface_hkl"] = list(surface_hkl)
        if beam_uvw is not None:
            record["beam_uvw"] = list(beam_uvw)
        if mosaic_fwhm_deg is not None:
            record["mosaic_fwhm_deg"] = mosaic_fwhm_deg
        with tempfile.TemporaryDirectory(prefix="pyrite-build-crystal-") as temporary:
            root = Path(temporary)
            (root / "cifs").mkdir()
            (root / filename).write_bytes(data)
            errors = _Errors()
            parsed = _parse_crystals({name: record}, errors, source=root / "catalog.toml")
            if errors.items:
                raise MaterialConfigError(tuple(errors.items))
            composition = tuple(sorted(parsed[name].composition))
        _supported(dict(composition))
        actual = dict(composition)
        if set(actual) != set(counts):
            raise ValueError("formula elements disagree with CIF sites")
        total_counts = math.fsum(counts.values())
        total_actual = math.fsum(actual.values())
        if any(
            not math.isclose(actual[e] / total_actual, counts[e] / total_counts, rel_tol=1e-6)
            for e in counts
        ):
            raise ValueError("formula stoichiometry disagrees with CIF sites")
        definition = _definition(name, "cif", composition, record, data)
    if density_g_cm3 is not None and not math.isclose(
        _positive(density_g_cm3, "density_g_cm3"),
        definition.density_g_cm3,
        rel_tol=1e-6,
    ):
        raise ValueError(f"density disagrees with CIF-derived {definition.density_g_cm3:g} g/cm³")
    return definition


def save_material(
    definition: MaterialDefinition, *, catalog: Path | None = None
) -> MaterialCatalog:
    """Explicitly register a definition and return a fresh immutable catalog.

    Bundled selection writes only to the user layer. Complete external catalogs
    write in place. Existing names are never replaced (case-insensitive).
    Validation precedes publication; failed writes roll back newly created
    records/assets. New CIFs use SHA-256 filenames and captured build-time bytes.
    Fresh ``load_material_catalog`` calls see the registration. Saving to the
    selected catalog refreshes PyRITE's default simulation conveniences in the
    current process. Previously held immutable catalog snapshots remain unchanged;
    the returned catalog is the new snapshot.
    """
    if not isinstance(definition, MaterialDefinition):
        raise TypeError("definition must be returned by build_medium or build_crystal")
    _name(definition.name)
    path = _safe_path(layout.selected_catalog() if catalog is None else Path(catalog))
    root = _safe_path(layout.catalog_root(path))
    root.mkdir(parents=True, exist_ok=True)
    with _authoring_lock(root):
        return _save_material_locked(definition, path, root)


@contextmanager
def _authoring_lock(root: Path) -> Iterator[None]:
    lock = root / ".material-authoring.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(
            f"catalog authoring is already in progress; retry after it finishes ({lock})"
        ) from None
    try:
        yield
    finally:
        lock.rmdir()


def _save_material_locked(
    definition: MaterialDefinition, path: Path, root: Path
) -> MaterialCatalog:
    import tomlkit

    raw = layout.read_raw(path)
    name = definition.name
    for table in ("media", "crystals", "materials"):
        if any(key.casefold() == name.casefold() for key in raw.get(table, {})):
            raise ValueError(f"name {name!r} already exists in {table}; choose a new name")
    record = tomlkit.parse(definition.record_toml).unwrap()
    if definition.kind == "crystal":
        rebuilt = build_crystal(name, crystal=record.get("crystal"), catalog=path)
        if rebuilt.identity != definition.identity:
            raise ValueError("referenced crystal changed since build; rebuild before saving")
    elif (
        _definition(name, definition.kind, definition.composition, record).identity
        != definition.identity
    ):
        raise ValueError("definition changed since build; rebuild before saving")
    if definition.kind == "medium":
        additions = {"media": {name: record}}
    elif definition.kind == "cif":
        additions = {"crystals": {name: record}, "materials": {name: {"crystal": name}}}
    elif definition.kind == "crystal":
        additions = {"materials": {name: record}}
    else:
        raise ValueError(f"unknown definition kind: {definition.kind}")
    original_text = layout.read_text(path)
    document = tomlkit.parse(original_text)
    for table, rows in additions.items():
        if table not in document:
            document[table] = tomlkit.table()
        document[table].update(rows)
    created: list[Path] = []
    single_file_published = False
    root.mkdir(parents=True, exist_ok=True)
    try:
        if definition.cif_bytes is not None:
            expected_cif = f"cifs/{hashlib.sha256(definition.cif_bytes).hexdigest()}.cif"
            if record.get("cif") != expected_cif:
                raise ValueError(
                    "CIF path must match its managed content hash; rebuild the definition"
                )
            asset = _safe_path(root / expected_cif)
            asset.parent.mkdir(parents=True, exist_ok=True)
            if asset.exists():
                if asset.read_bytes() != definition.cif_bytes:
                    raise ValueError(f"CIF content collision: {asset}")
            else:
                _exclusive_write(asset, definition.cif_bytes, created)
        fd, temporary = tempfile.mkstemp(dir=root, prefix=".", suffix=".toml")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(tomlkit.dumps(document))
            candidate = load_material_catalog(Path(temporary))
            resolved = (
                candidate.media[name].composition
                if definition.kind == "medium"
                else candidate.crystals[
                    name if definition.kind == "cif" else record["crystal"]
                ].composition
            )
            if tuple(sorted(resolved)) != definition.composition:
                raise ValueError(
                    "definition composition changed since build; rebuild before saving"
                )
        finally:
            Path(temporary).unlink(missing_ok=True)
        if path.is_dir():
            # Publish referenced objects first; the runnable recipe is last.
            for table, rows in additions.items():
                target = _safe_path(root / table / f"{name}.toml")
                target.parent.mkdir(parents=True, exist_ok=True)
                _exclusive_write(target, tomlkit.dumps(rows[name]).encode(), created)
        else:
            layout.write_text(path, tomlkit.dumps(document))
            single_file_published = True
        _load_material_catalog_cached.cache_clear()
        saved = load_material_catalog(path)
        if path.resolve() == layout.selected_catalog().resolve() and (
            layout.is_layered(path) or path.resolve() != layout.bundled_catalog().resolve()
        ):
            _refresh_default_catalog(saved)
        return saved
    except BaseException:
        if single_file_published:
            layout.write_text(path, original_text)
        for file in reversed(created):
            file.unlink(missing_ok=True)
        _load_material_catalog_cached.cache_clear()
        raise


def _exclusive_write(path: Path, data: bytes, created: list[Path]) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".pending")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.link(temporary, path)
        created.append(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
