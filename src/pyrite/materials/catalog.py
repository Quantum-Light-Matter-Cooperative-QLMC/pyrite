"""Immutable material catalog loaded from the packaged TOML schema.

The public module owns catalog loading, caching, and the default singleton. Schema
records and parsing helpers live in private sibling modules.
"""

import functools
import tomllib
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import cast

from .. import DATA_DIR
from ._beam_detector_parse import _parse_beams, _parse_detectors
from ._catalog_decode import LineGridByEnergy, _Errors, _grid
from ._parse import (
    _load_profile_artifacts,
    _parse_crystals,
    _parse_energy_grids,
    _parse_materials,
    _parse_media,
    _parse_profiles,
    _warn_missing_mott,
)
from ._schema import (
    _PROFILE_SCALAR_NUMERICS_KEYS,
    CrystalInfo,
    CrystalSpec,
    LayerSpec,
    MaterialCatalog,
    MaterialConfigError,
    MaterialSpec,
    MaterialValidationSpec,
    MediumSpec,
    ScanSpec,
)


def load_material_catalog(
    path: Path | None = None,
    *,
    profile: str = "standard",
) -> MaterialCatalog:
    """Load, validate, and deeply freeze a material catalog.

    Parameters
    ----------
    path
        TOML catalog path. ``None`` loads the packaged ``materials.toml``.
    profile
        Profile whose inherited scan grids and artifact references are resolved.

    Returns
    -------
    MaterialCatalog
        Validated immutable schema-version-1 catalog.

    Raises
    ------
    MaterialConfigError
        If the file cannot be read, parsed, or validated for ``profile``.
    """
    source = Path(DATA_DIR) / "materials.toml" if path is None else Path(path)

    try:
        content = source.read_bytes()
    except OSError as exc:
        raise MaterialConfigError((f"{source}: {exc}",)) from exc

    return _load_material_catalog_cached(source, content, profile)


@functools.lru_cache(maxsize=32)
def _load_material_catalog_cached(
    source: Path,
    content: bytes,
    profile: str,
) -> MaterialCatalog:
    errors = _Errors()

    try:
        raw = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise MaterialConfigError((f"{source}: {exc}",)) from exc

    if not isinstance(raw, Mapping):
        raise MaterialConfigError((f"{source}: root must be a table",))

    errors.keys(
        raw,
        "catalog",
        {
            "schema_version",
            "profiles",
            "crystals",
            "media",
            "materials",
            "energy_grids",
            "beams",
            "detectors",
        },
    )
    version = raw.get("schema_version")
    if type(version) is not int or version != 1:
        errors.add("schema_version", "must equal 1")
    for key in ("profiles", "crystals", "media", "materials"):
        if key not in raw:
            errors.add(key, "missing required table")
    profiles = _parse_profiles(raw.get("profiles"), errors)
    beams = _parse_beams(raw.get("beams", {}), errors)
    detectors, detector_labels = _parse_detectors(raw.get("detectors", {}), errors)
    profile_artifacts, profile_energy_grid_refs = _load_profile_artifacts(
        source, profiles, profile, errors
    )
    crystals = _parse_crystals(raw.get("crystals"), errors)
    media = _parse_media(raw.get("media"), errors)
    energy_grids = _parse_energy_grids(raw.get("energy_grids", {}), errors)
    resolved_energy_grid_refs: dict[str, str] = {}
    materials = _parse_materials(
        raw.get("materials"),
        crystals,
        media,
        energy_grids,
        errors,
        profiles,
        profile_artifacts,
        resolved_energy_grid_refs,
        profile_name=profile,
    )
    if errors.items:
        raise MaterialConfigError(errors.items, profile=profile)
    _warn_missing_mott(materials, crystals, media)
    profile_memberships = {
        name: tuple(cast("list[str]", row["materials"]))
        for name, row in profiles.items()
        if isinstance(row.get("materials"), list)
    }
    profile_beams: dict[str, Mapping[str, object]] = {}
    for name, row in profiles.items():
        beam_value = row.get("beam")
        if isinstance(beam_value, str):
            named_beam = beams.get(beam_value)
            if named_beam is None:
                errors.add(f"profiles.{name}.beam", f"unknown beam {beam_value!r}")
                continue
            # Resolve to values only -- the name itself must never reach a
            # consumer (decision 3), and `label` is display metadata that never
            # shaped the inline-block payload this replaces.
            profile_beams[name] = MappingProxyType(
                {k: v for k, v in named_beam.items() if k != "label"}
            )
        elif isinstance(beam_value, Mapping):
            profile_beams[name] = MappingProxyType(dict(cast("Mapping[str, object]", beam_value)))
    if errors.items:
        raise MaterialConfigError(errors.items, profile=profile)
    profile_detectors: dict[str, Mapping[str, object]] = {}
    for name, row in profiles.items():
        detector_value = row.get("detector")
        if isinstance(detector_value, str):
            named_detector = detectors.get(detector_value)
            if named_detector is None:
                errors.add(f"profiles.{name}.detector", f"unknown detector {detector_value!r}")
                continue
            profile_detectors[name] = named_detector
        elif isinstance(detector_value, Mapping):
            # _parse_profiles already replaced an inline block with its validated
            # spec; anything else with a detector key was rejected there.
            profile_detectors[name] = cast("Mapping[str, object]", detector_value)
    if errors.items:
        raise MaterialConfigError(errors.items, profile=profile)
    profile_emissions = {
        name: cast(str, row["emission"])
        for name, row in profiles.items()
        if isinstance(row.get("emission"), str)
    }
    profile_transport_numerics = {}
    for name, row in profiles.items():
        numerics = {key: row[key] for key in _PROFILE_SCALAR_NUMERICS_KEYS if key in row}
        for key in ("n_electrons", "n_electrons_brem"):
            if key not in row:
                continue
            grid = _grid(row[key], f"profiles.{name}.{key}", errors)
            if grid is not None:
                numerics[key] = tuple(int(value) for value in grid)
        if numerics:
            profile_transport_numerics[name] = MappingProxyType(numerics)
    if errors.items:
        raise MaterialConfigError(errors.items, profile=profile)
    profile_filters = {
        name: cast(tuple[Mapping[str, object], ...], row["filters"])
        for name, row in profiles.items()
        if isinstance(row.get("filters"), tuple)
    }
    profile_physical_detectors = {
        name: cast(Mapping[str, object], row["physical_detector"])
        for name, row in profiles.items()
        if isinstance(row.get("physical_detector"), Mapping)
    }
    return MaterialCatalog(
        schema_version=1,
        crystals=MappingProxyType(crystals),
        media=MappingProxyType(media),
        materials=MappingProxyType(materials),
        material_keys=tuple(materials),
        profile_names=tuple(profiles),
        profile_memberships=MappingProxyType(profile_memberships),
        profile_beams=MappingProxyType(profile_beams),
        profile_detectors=MappingProxyType(profile_detectors),
        profile_emissions=MappingProxyType(profile_emissions),
        profile_transport_numerics=MappingProxyType(profile_transport_numerics),
        profile_filters=MappingProxyType(profile_filters),
        profile_physical_detectors=MappingProxyType(profile_physical_detectors),
        profile_energy_grid_refs=MappingProxyType(profile_energy_grid_refs),
        resolved_energy_grid_refs=MappingProxyType(resolved_energy_grid_refs),
        beams=MappingProxyType(beams),
        beam_keys=tuple(beams),
        detectors=MappingProxyType(detectors),
        detector_labels=MappingProxyType(detector_labels),
        detector_keys=tuple(detectors),
    )


_DEFAULT_CATALOG: MaterialCatalog | None = None
_DEFAULT_CATALOG_LOCK = Lock()


# Test hook: clear the parse cache (e.g. after monkeypatching CIF loaders).
load_material_catalog.cache_clear = _load_material_catalog_cached.cache_clear  # ty: ignore[unresolved-attribute]


def _get_default_catalog() -> MaterialCatalog:
    """Load the bundled catalog once, on first package-level access."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        with _DEFAULT_CATALOG_LOCK:
            if _DEFAULT_CATALOG is None:
                _DEFAULT_CATALOG = load_material_catalog()
    return _DEFAULT_CATALOG


__all__ = [
    "CrystalInfo",
    "CrystalSpec",
    "LayerSpec",
    "LineGridByEnergy",
    "MaterialCatalog",
    "MaterialConfigError",
    "MaterialSpec",
    "MaterialValidationSpec",
    "MediumSpec",
    "ScanSpec",
    "load_material_catalog",
]
