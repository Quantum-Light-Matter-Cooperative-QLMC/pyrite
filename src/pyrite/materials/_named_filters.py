"""Bounded named downstream-filter presets (issue #665).

Presets are *values*, not scene actors or a new geometry hierarchy. An
instance's ordered location in ``profiles.NAME.filters`` owns its placement
and optional display name. Resolution happens before the existing filter
schema parser, so all downstream consumers receive the same validated rows
as legacy inline filters.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ._beam_detector_parse import _parse_filter_rows

if TYPE_CHECKING:
    from pyrite.instrument.model import FilterPlate

    from ._catalog_decode import _Errors
    from ._schema import MaterialCatalog

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def resolve_named_filters(
    raw: object, profiles: object, errors: _Errors
) -> tuple[
    Mapping[str, Mapping[str, object]],
    object,
    Mapping[str, tuple[str | None, ...]],
]:
    """Validate named plates and expand references into *value-only* profile rows.

    A definition is a complete finite plate without a name. Profile entries
    may specify ``ref`` with optional overrides of any plate field (including
    pose and the instance's ``name``); a bare inline plate remains valid. No
    preset name is copied into the resolved plate, observation, or identity.
    """
    table: Mapping[str, object]
    if raw is None:
        table = {}
    elif isinstance(raw, Mapping):
        table = raw
    else:
        errors.add("filters", "must be a table")
        table = {}

    definitions: dict[str, Mapping[str, object]] = {}
    templates: dict[str, dict[str, object]] = {}
    folded: dict[str, str] = {}
    for name, entry in table.items():
        path = f"filters.{name}"
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            errors.add(path, "invalid preset name: use letters, digits, '.', '_', '-'")
            continue
        previous = folded.setdefault(name.casefold(), name)
        if previous != name:
            errors.add(path, f"name differs from {previous!r} only by case")
            continue
        if not isinstance(entry, Mapping):
            errors.add(path, "must be a table")
            continue
        if "name" in entry:
            errors.add(f"{path}.name", "instance names belong to the profile placement")
        description = entry.get("description")
        if description is not None and (
            not isinstance(description, str) or not description.strip()
        ):
            errors.add(f"{path}.description", "must be a nonempty string")
        template = {key: value for key, value in entry.items() if key != "description"}
        # Reuse exactly the existing plate and unit/pose validator. Template
        # values must be complete; no unresolved refs or arbitrary bodies.
        decoded = _parse_filter_rows([template], path, errors)
        if decoded:
            definitions[name] = MappingProxyType(dict(decoded[0]))
            templates[name] = template

    if not isinstance(profiles, Mapping):
        return MappingProxyType(definitions), profiles, MappingProxyType({})

    expanded: dict[str, Any] = {}
    reference_labels: dict[str, tuple[str | None, ...]] = {}
    for profile_name, profile in profiles.items():
        if not isinstance(profile, Mapping) or not isinstance(profile.get("filters"), list):
            expanded[profile_name] = profile
            continue
        rows = []
        labels: list[str | None] = []
        for index, item in enumerate(profile["filters"]):
            path = f"profiles.{profile_name}.filters[{index}]"
            if not isinstance(item, Mapping) or "ref" not in item:
                rows.append(item)
                labels.append(None)
                continue
            name = item["ref"]
            if not isinstance(name, str) or name not in templates:
                errors.add(f"{path}.ref", f"unknown filter preset {name!r}")
                rows.append({key: value for key, value in item.items() if key != "ref"})
                labels.append(None)
                continue
            # Full definition plus per-placement overrides, in original order.
            # Description and reference are never included in resolved rows.
            rows.append({**templates[name], **{k: v for k, v in item.items() if k != "ref"}})
            labels.append(name)
        expanded[profile_name] = {**profile, "filters": rows}
        reference_labels[profile_name] = tuple(labels)

    return (
        MappingProxyType(definitions),
        expanded,
        MappingProxyType(reference_labels),
    )


def load_filter(name: str, *, catalog: MaterialCatalog | None = None) -> FilterPlate:
    """Load a named, validated :class:`~pyrite.geometry.FilterPlate` value.

    Placements can override any field by attaching ``ref = NAME`` in an
    ordered profile filter entry. This loader returns the unplaced definition's
    declared pose, without assigning the catalog name to the plate's name.
    """
    if catalog is None:
        from . import CATALOG as catalog
    try:
        definition = catalog.named_filters[name]
    except KeyError:
        raise KeyError(
            f"unknown filter preset {name!r}; have {sorted(catalog.named_filters)}"
        ) from None
    from ..campaign.observation import filter_from_config

    return filter_from_config(definition)
