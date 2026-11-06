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
    from ._catalog_decode import _Errors

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _parse_sequence(
    entry: Mapping[str, object], path: str, errors: _Errors
) -> tuple[Mapping[str, object] | None, list[dict[str, object]]]:
    """Validate a flat ordered sequence of complete inline plates."""
    errors.keys(entry, path, {"kind", "description", "plates"})
    plates = entry.get("plates")
    if not isinstance(plates, list) or not plates:
        errors.add(f"{path}.plates", "must be a nonempty array of complete plates")
        return None, []
    for index, plate in enumerate(plates):
        if isinstance(plate, Mapping) and "ref" in plate:
            errors.add(f"{path}.plates[{index}].ref", "nested filter references are not supported")
            return None, []
    decoded = _parse_filter_rows(plates, f"{path}.plates", errors)
    if decoded is None or len(decoded) != len(plates):
        return None, []
    definition = MappingProxyType(
        {"kind": "sequence", "plates": tuple(MappingProxyType(dict(row)) for row in decoded)}
    )
    return definition, [dict(plate) for plate in plates]


def _parse_definitions(
    raw: object, errors: _Errors
) -> tuple[dict[str, Mapping[str, object]], dict[str, Any]]:
    """Validate complete plate definitions independently of scene placements."""
    table: Mapping[str, object]
    if raw is None:
        table = {}
    elif isinstance(raw, Mapping):
        table = raw
    else:
        errors.add("filters", "must be a table")
        table = {}

    definitions: dict[str, Mapping[str, object]] = {}
    templates: dict[str, Any] = {}
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
        kind = entry.get("kind")
        if kind == "sequence":
            definition, plates = _parse_sequence(entry, path, errors)
            if definition is not None:
                definitions[name] = definition
                templates[name] = plates
            continue
        if kind is not None:
            errors.add(f"{path}.kind", "must be 'sequence'; omit it for a single plate")
            continue
        template = {key: value for key, value in entry.items() if key != "description"}
        # Reuse exactly the existing plate and unit/pose validator. Template
        # values must be complete; no unresolved refs or arbitrary bodies.
        decoded = _parse_filter_rows([template], path, errors)
        if decoded:
            definitions[name] = MappingProxyType(dict(decoded[0]))
            templates[name] = template

    return definitions, templates


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
    pose and the instance's ``name``); a bare inline plate remains valid. A
    ``kind = "sequence"`` reference expands in place to its complete plates and
    takes no overrides, so ``reference_labels`` aligns with expanded rows. No
    preset name is copied into the resolved plate, observation, or identity.
    """
    definitions, templates = _parse_definitions(raw, errors)

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
            template = templates[name]
            if isinstance(template, list):
                # Absolute member poses: group overrides would be ambiguous.
                extra = sorted(key for key in item if key != "ref")
                if extra:
                    errors.add(path, f"sequence placements accept only ref; got {extra}")
                rows.extend(dict(plate) for plate in template)
                labels.extend([name] * len(template))
                continue
            # Full definition plus per-placement overrides, in original order.
            # Description and reference are never included in resolved rows.
            rows.append({**template, **{k: v for k, v in item.items() if k != "ref"}})
            labels.append(name)
        expanded[profile_name] = {**profile, "filters": rows}
        reference_labels[profile_name] = tuple(labels)

    return (
        MappingProxyType(definitions),
        expanded,
        MappingProxyType(reference_labels),
    )
