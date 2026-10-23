"""Named physics and numerics presets: catalog schema and resolution (ADR-0021).

A ``[physics.NAME]`` object holds ``emission`` plus every profile key that is
a :class:`~pyrite._numerics.Physics` field. A ``[numerics.NAME]`` object holds
scalar trial counts, the remaining profile numerics keys, and optional
``precision`` and ``line_grid_policy`` tables. Both may carry a display
``description``. A profile attaches them with ``physics = "NAME"`` and
``numerics = "NAME"``; its own keys stay explicit overrides. The catalog folds
presets into each referencing profile row before any consumer reads it, so
identities see only effective values and never a preset name. This module
stays import-light because the material catalog parser uses it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._numerics import (
    CONVERGENCE_KEYS,
    PHYSICS_FIELDS,
    PROFILE_NUMERICS_KEYS,
    SAMPLING_KEYS,
    CxrReflections,
    Numerics,
    Physics,
    _emission_selection,
    validate_profile_numerics,
)

EMISSION_VALUES = ("incoherent", "coherent", "both")
METADATA_KEYS = ("description",)
#: Keys of one sampling policy: a profile setting any of them takes none of
#: them from its numerics preset, so fixed counts and adaptive precision never mix.
SAMPLING_POLICY_KEYS = (*SAMPLING_KEYS, "precision")


@dataclass(frozen=True)
class PresetKind:
    """One named-preset table and the profile keys it owns."""

    table: str
    #: Owned profile keys with scalar values, in profile-schema order.
    scalar_keys: tuple[str, ...]
    #: Owned profile keys holding nested tables.
    table_keys: tuple[str, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        return (*self.scalar_keys, *self.table_keys)

    @property
    def reference_key(self) -> str:
        return self.table


PHYSICS = PresetKind(
    "physics", ("emission", *(key for key in PROFILE_NUMERICS_KEYS if key in PHYSICS_FIELDS))
)
NUMERICS = PresetKind(
    "numerics",
    tuple(key for key in PROFILE_NUMERICS_KEYS if key not in PHYSICS_FIELDS),
    ("precision", "line_grid_policy"),
)
KINDS = (PHYSICS, NUMERICS)
TABLES = tuple(kind.table for kind in KINDS)


def preset_errors(kind: PresetKind, name: str, row: object) -> list[tuple[str, str]]:
    """Return ``(path, message)`` schema errors for one preset row.

    ``line_grid_policy`` tables are validated by the catalog parser, which
    owns that schema.
    """
    path = f"{kind.table}.{name}"
    if not isinstance(row, Mapping):
        return [(path, "must be a table")]
    errors = [
        (f"{path}.{key}", f"unknown key; {kind.table} presets hold {', '.join(kind.keys)}")
        for key in row
        if key not in kind.keys and key not in METADATA_KEYS
    ]
    description = row.get("description")
    if description is not None and not isinstance(description, str):
        errors.append((f"{path}.description", "must be a string"))
    if errors:
        return errors
    try:
        validate_values(kind, preset_values(kind, row))
    except (TypeError, ValueError) as exc:
        errors.append((path, str(exc)))
    return errors


def validate_values(kind: PresetKind, values: Mapping[str, Any]) -> None:
    """Validate one preset's configuration values through the typed objects."""
    unknown = sorted(set(values) - set(kind.keys))
    if unknown:
        raise ValueError(f"unknown {kind.table} preset field: {unknown[0]}")
    if kind is PHYSICS:
        emission = values.get("emission")
        if emission is not None and emission not in EMISSION_VALUES:
            raise ValueError(f"emission must be one of {EMISSION_VALUES}")
    for key in SAMPLING_KEYS:
        count = values.get(key)
        if count is not None and (type(count) is not int or count <= 0):
            raise ValueError(
                f"{key} must be one positive integer; sweep counts belong in profile grids"
            )
    if "precision" in values:
        if any(key in values for key in SAMPLING_KEYS):
            raise ValueError("precision cannot be combined with fixed trial counts")
        from ._precision import Precision

        Precision.from_dict(values["precision"])
    validate_profile_numerics(
        {key: values[key] for key in kind.scalar_keys if key in values and key != "emission"}
    )


def preset_values(kind: PresetKind, row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a preset row's configuration values without display metadata."""
    return {key: _plain(row[key]) for key in kind.keys if key in row}


def _profile_value(key: str, value: Any) -> Any:
    """Spell a preset value as its profile key: trial counts are value grids."""
    return {"values": [value]} if key in SAMPLING_KEYS else value


def fold_preset(kind: PresetKind, preset: Mapping[str, Any], row: Mapping[str, Any]) -> dict:
    """Return ``row`` with ``preset`` values merged below its own keys."""
    values = preset_values(kind, preset)
    if kind is NUMERICS and any(key in row for key in SAMPLING_POLICY_KEYS):
        values = {key: value for key, value in values.items() if key not in SAMPLING_POLICY_KEYS}
    merged = {key: _profile_value(key, value) for key, value in values.items()}
    merged.update((key, value) for key, value in row.items() if key != kind.reference_key)
    return merged


def resolve_profile_rows(
    profiles: object, presets: Mapping[PresetKind, Mapping[str, Mapping[str, Any]]]
) -> tuple[object, dict[str, dict[str, str]], list[tuple[str, str]]]:
    """Fold referenced presets under each profile's own fields.

    Returns ``(rows, references, errors)``: profile rows with reference keys
    removed and preset values merged below explicit keys, ``references[table]``
    as ``profile -> preset`` maps, and unknown-reference errors. Non-table
    input is returned unchanged for the profile parser to report.
    """
    references: dict[str, dict[str, str]] = {kind.table: {} for kind in KINDS}
    if not isinstance(profiles, Mapping):
        return profiles, references, []
    rows: dict[str, Any] = {}
    errors: list[tuple[str, str]] = []
    for name, row in profiles.items():
        for kind in KINDS:
            if not isinstance(row, Mapping) or kind.reference_key not in row:
                continue
            reference = row[kind.reference_key]
            path = f"profiles.{name}.{kind.reference_key}"
            if not isinstance(reference, str) or reference not in presets[kind]:
                errors.append((path, f"unknown {kind.table} preset {reference!r}"))
                row = {key: value for key, value in row.items() if key != kind.reference_key}
                continue
            references[kind.table][name] = reference
            row = fold_preset(kind, presets[kind][reference], row)
        rows[name] = row
    return rows, references, errors


def physics_from_values(values: Mapping[str, Any]) -> Physics:
    """Build the frozen :class:`Physics` that physics-preset values describe."""
    fields = {
        key: value for key, value in values.items() if key in PHYSICS_FIELDS and key != "emission"
    }
    return Physics(emission=_emission_selection(values.get("emission", "incoherent")), **fields)


def numerics_from_values(values: Mapping[str, Any]) -> Numerics:
    """Build the frozen :class:`Numerics` that numerics-preset values describe.

    ``line_grid_policy`` is a catalog campaign policy with no field on
    :class:`Numerics`; it applies only through profiles.
    """
    from ._precision import Precision

    reflections = CxrReflections(**{key: values[key] for key in CONVERGENCE_KEYS if key in values})
    renamed = {"n_electrons": "cxr_trials", "n_electrons_brem": "brem_trials"}
    fields = {
        renamed.get(key, key): value
        for key, value in values.items()
        if key in NUMERICS.scalar_keys and key not in CONVERGENCE_KEYS
    }
    precision = values.get("precision")
    return Numerics(
        **fields,
        cxr_reflections=reflections,
        precision=None if precision is None else Precision.from_dict(precision),
    )


def _load(kind: PresetKind, name: str, catalog: Any) -> Mapping[str, Any]:
    if catalog is None:
        from .materials import CATALOG as catalog
    presets = catalog.presets(kind.table)
    if name not in presets:
        raise KeyError(f"unknown {kind.table} preset {name!r}; have {sorted(presets)}")
    return presets[name]


def load_physics(name: str, *, catalog: Any = None) -> Physics:
    """Return the named physics preset as a frozen :class:`~pyrite.Physics`.

    Parameters
    ----------
    name
        A bundled or user ``[physics.NAME]`` catalog preset, e.g.
        ``"electron-standard"``. List them with ``pyrite physics preset list``.
    catalog
        A loaded :class:`~pyrite.materials.MaterialCatalog`; defaults to the
        selected catalog with its user layer.

    Returns
    -------
    Physics
        The preset's values over the built-in defaults, validated as for a
        profile. The object is independent of later catalog edits; adjust it
        with :func:`dataclasses.replace`.

    Raises
    ------
    KeyError
        ``name`` is not a physics preset in ``catalog``.
    """
    return physics_from_values(_load(PHYSICS, name, catalog))


def load_numerics(name: str, *, catalog: Any = None) -> Numerics:
    """Return the named numerics preset as a frozen :class:`~pyrite.Numerics`.

    Parameters
    ----------
    name
        A bundled or user ``[numerics.NAME]`` catalog preset. List them with
        ``pyrite numerics preset list``.
    catalog
        A loaded :class:`~pyrite.materials.MaterialCatalog`; defaults to the
        selected catalog with its user layer.

    Returns
    -------
    Numerics
        Trial counts (``n_electrons`` -> ``cxr_trials``, ``n_electrons_brem``
        -> ``brem_trials``), integration, reflection/mosaic and channeling-zone
        numerics and any adaptive ``precision`` over the built-in defaults.
        Energy bins and execution-only fields keep their defaults; a
        ``line_grid_policy`` applies only to catalog profiles. Adjust the
        result with :func:`dataclasses.replace`.

    Raises
    ------
    KeyError
        ``name`` is not a numerics preset in ``catalog``.
    """
    return numerics_from_values(_load(NUMERICS, name, catalog))


def preset_digest(kind: PresetKind, row: Mapping[str, Any]) -> str:
    """SHA-256 of a preset's canonical configuration values (no description)."""
    import hashlib
    import json

    encoded = json.dumps(_jsonable(preset_values(kind, row)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def preset_provenance(catalog: Any, profile: str) -> dict[str, dict[str, str]]:
    """Return ``{table: {"name", "sha256"}}`` for a profile's referenced presets.

    Run records carry this beside the dataset identity as a label: the digest
    identifies the preset definition a run resolved, but identity itself
    depends only on effective values (ADR-0021).
    """
    labels = {}
    for kind in KINDS:
        name = catalog.profile_preset(kind.table, profile)
        if name is not None:
            row = catalog.presets(kind.table)[name]
            labels[kind.table] = {"name": name, "sha256": preset_digest(kind, row)}
    return labels


def identity_labels(catalog: Any, profile: str) -> dict[str, Any]:
    """Return ``{"presets": ...}`` labels for a run identity, or ``{}`` without presets."""
    presets = preset_provenance(catalog, profile)
    return {"presets": presets} if presets else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def frozen_preset(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a deeply immutable copy of a validated preset row."""
    return _frozen(_plain(row))


def _plain(value: Any) -> Any:
    """Unwrap tomlkit items; TOML arrays stay lists as in profile rows."""
    return value.unwrap() if hasattr(value, "unwrap") else value


def _frozen(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    return value
