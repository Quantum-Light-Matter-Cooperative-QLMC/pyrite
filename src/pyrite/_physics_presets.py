"""Named physics presets: catalog schema and profile resolution (ADR-0021).

A ``[physics.NAME]`` catalog object holds data-only selections of existing
physics models and cutoffs: ``emission`` plus every profile key that is a
:class:`~pyrite._numerics.Physics` field, and an optional ``description``.
A profile attaches one with ``physics = "NAME"``; its own inline physics keys
stay explicit overrides. The catalog folds the preset into each referencing
profile row before any consumer reads it, so identities see only effective
values and never the preset name. This module stays import-light because the
material catalog parser uses it.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ._numerics import (
    PHYSICS_FIELDS,
    PROFILE_NUMERICS_KEYS,
    Physics,
    _emission_selection,
    validate_profile_numerics,
)

TABLE = "physics"
REFERENCE_KEY = "physics"
EMISSION_VALUES = ("incoherent", "coherent", "both")
#: Profile keys a physics preset may set, in profile-schema order.
PRESET_KEYS = ("emission", *(key for key in PROFILE_NUMERICS_KEYS if key in PHYSICS_FIELDS))
METADATA_KEYS = ("description",)


def preset_errors(name: str, row: object) -> list[tuple[str, str]]:
    """Return ``(path, message)`` schema errors for one ``[physics.NAME]`` row."""
    path = f"{TABLE}.{name}"
    if not isinstance(row, Mapping):
        return [(path, "must be a table")]
    errors = [
        (f"{path}.{key}", "unknown key; physics presets hold physics model and cutoff fields")
        for key in row
        if key not in PRESET_KEYS and key not in METADATA_KEYS
    ]
    description = row.get("description")
    if description is not None and not isinstance(description, str):
        errors.append((f"{path}.description", "must be a string"))
    emission = row.get("emission")
    if emission is not None and emission not in EMISSION_VALUES:
        errors.append((f"{path}.emission", f"must be one of {EMISSION_VALUES}"))
    if errors:
        return errors
    try:
        validate_profile_numerics(preset_values(row, emission=False))
    except (TypeError, ValueError) as exc:
        errors.append((path, str(exc)))
    return errors


def preset_values(row: Mapping[str, Any], *, emission: bool = True) -> dict[str, Any]:
    """Return a preset row's configuration values without display metadata."""
    keys = PRESET_KEYS if emission else PRESET_KEYS[1:]
    return {key: _plain(row[key]) for key in keys if key in row}


def resolve_profile_rows(
    profiles: object, presets: Mapping[str, Mapping[str, Any]]
) -> tuple[object, dict[str, str], list[tuple[str, str]]]:
    """Fold referenced presets under each profile's inline physics fields.

    Returns ``(rows, references, errors)``: copies of referencing rows with the
    reference key removed and preset values merged below explicit profile
    keys, the ``profile -> preset`` map, and unknown-reference errors.
    Non-table input is returned unchanged for the profile parser to report.
    """
    if not isinstance(profiles, Mapping):
        return profiles, {}, []
    rows: dict[str, Any] = {}
    references: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    for name, row in profiles.items():
        if not isinstance(row, Mapping) or REFERENCE_KEY not in row:
            rows[name] = row
            continue
        reference = row[REFERENCE_KEY]
        path = f"profiles.{name}.{REFERENCE_KEY}"
        explicit = {key: value for key, value in row.items() if key != REFERENCE_KEY}
        if not isinstance(reference, str):
            errors.append((path, "must be a physics preset name"))
        elif reference not in presets:
            errors.append((path, f"unknown physics preset {reference!r}"))
        else:
            references[name] = reference
            explicit = {**preset_values(presets[reference]), **explicit}
        rows[name] = explicit
    return rows, references, errors


def physics_from_values(values: Mapping[str, Any]) -> Physics:
    """Build the frozen :class:`Physics` a preset's values describe."""
    fields = {
        key: value for key, value in values.items() if key in PHYSICS_FIELDS and key != "emission"
    }
    return Physics(
        emission=_emission_selection(values.get("emission", "incoherent")),
        **fields,
    )


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
    if catalog is None:
        from .materials import CATALOG as catalog
    presets = catalog.physics_presets
    if name not in presets:
        raise KeyError(f"unknown physics preset {name!r}; have {sorted(presets)}")
    return physics_from_values(presets[name])


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
