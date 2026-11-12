"""Profile ``[channeling]`` table spelling and its flat lowering keys (#1017).

Import-light and package-neutral: the catalog parser (``pyrite.materials``),
profile editing (``pyrite.campaign``) and :mod:`pyrite.quantum._channeling`
share it without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: ``[channeling]`` profile-table key -> internal flat lowering field. The flat
#: names are the stored case/identity spelling and never public constructor
#: arguments.
CHANNELING_TABLE_KEYS: dict[str, str] = {
    "model": "channeling_model",
    "plane": "channeling_plane",
    "axial": "channeling_axial",
    "admission": "channeling_admission",
    "basis_size": "channeling_basis_size",
    "bloch_cells": "channeling_bloch_cells",
    "step_ang": "channeling_dz_ang",
    "zone_max_ang": "channeling_zone_max_ang",
}
#: Internal flat field -> ``[channeling]`` table key.
CHANNELING_FIELD_TABLE_KEYS = {flat: key for key, flat in CHANNELING_TABLE_KEYS.items()}
#: Flat lowering values of a run without channeling.
CHANNELING_OFF_FIELDS: dict[str, object] = {
    "channeling_model": "off",
    "channeling_plane": None,
    "channeling_axial": None,
    "channeling_admission": "admitted",
    "channeling_basis_size": 16,
    "channeling_bloch_cells": 16,
    "channeling_dz_ang": 250.0,
    "channeling_zone_max_ang": None,
}


def channeling_table_values(profile: Mapping[str, Any]) -> dict[str, object]:
    """Flat lowering fields of a profile's ``[channeling]`` table (#1017).

    Flat ``channeling_*`` profile keys were removed; they refuse with the
    table spelling.
    """
    flat = [key for key in CHANNELING_OFF_FIELDS if key in profile]
    if flat:
        key = flat[0]
        raise ValueError(
            f"profile key {key} was removed; use the [channeling] table key "
            f"{CHANNELING_FIELD_TABLE_KEYS[key]} (e.g. [profiles.NAME.channeling] "
            f"{CHANNELING_FIELD_TABLE_KEYS[key]} = ...)"
        )
    table = profile.get("channeling")
    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise ValueError("profile channeling must be a table")
    unknown = sorted(set(table) - set(CHANNELING_TABLE_KEYS))
    if unknown:
        raise ValueError(
            f"unknown [channeling] key: {unknown[0]}; "
            f"expected one of {', '.join(CHANNELING_TABLE_KEYS)}"
        )
    return {
        CHANNELING_TABLE_KEYS[key]: value.unwrap() if hasattr(value, "unwrap") else value
        for key, value in table.items()
    }
