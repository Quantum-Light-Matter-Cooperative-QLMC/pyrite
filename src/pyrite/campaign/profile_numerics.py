"""Profile numerical controls and coupled transport validation."""

import tomlkit

from pyrite._numerics import (
    CHANNELING_KEYS,
    PROFILE_NUMERICS_KEYS,
    SAMPLING_KEYS,
    validate_profile_numerics,
)
from pyrite.quantum._channeling import CHANNELING_FIELD_TABLE_KEYS, channeling_table_values

from ._profile_document import existing_profile, values_item
from .profile_cxr_grid import _check_cxr_grid_policy, profile_cxr_grid_values


def _set_field(target, key, value):
    """Write one flat field to its profile location: the channeling table or the row."""
    if key not in CHANNELING_KEYS:
        target[key] = value
        return
    table = target.get("channeling")
    if table is None:
        table = target["channeling"] = tomlkit.table()
    table[CHANNELING_FIELD_TABLE_KEYS[key]] = value


def _pop_field(target, key):
    """Remove one flat field from its profile location; drop an emptied table."""
    if key not in CHANNELING_KEYS:
        target.pop(key, None)
        return
    table = target.get("channeling")
    if table is None:
        return
    table.pop(CHANNELING_FIELD_TABLE_KEYS[key], None)
    if not table:
        target.pop("channeling", None)


def profile_numerics_values(profile):
    """Return explicit profile numerics as plain Python values."""
    values = channeling_table_values(profile)
    for key in PROFILE_NUMERICS_KEYS:
        if key not in profile or key in CHANNELING_KEYS:
            continue
        value = profile[key]
        if key in SAMPLING_KEYS:
            if not isinstance(value, dict) or not isinstance(value.get("values"), list):
                raise ValueError(f"{key} must be a values grid")
            value = list(value["values"])
        elif hasattr(value, "unwrap"):
            value = value.unwrap()
        values[key] = value
    return values


def set_numerics(document, name, updates):
    """Validate and atomically stage supplied result-affecting numerics."""
    target = existing_profile(document, name)
    if "precision" in target and any(key in updates for key in SAMPLING_KEYS):
        raise ValueError(
            f"profile {name!r} has an adaptive-sampling policy; select fixed counts with: "
            f"pyrite profile numerics adaptive-sampling disable {name} --cxr-histories N "
            "--brem-histories N"
        )
    merged = {**profile_numerics_values(target), **updates}
    validate_profile_numerics(merged)
    for key, value in updates.items():
        _set_field(target, key, values_item([value]) if key in SAMPLING_KEYS else value)
    _check_cxr_grid_policy(target, profile_cxr_grid_values(target))
    return tuple(updates)


def reset_numerics(document, name, fields=()):
    """Remove selected explicit numerics, or every explicit field when empty."""
    target = existing_profile(document, name)
    selected = tuple(fields) or PROFILE_NUMERICS_KEYS
    remaining = profile_numerics_values(target)
    removed = tuple(key for key in selected if key in remaining)
    for key in selected:
        remaining.pop(key, None)
    validate_profile_numerics(remaining)
    for key in removed:
        _pop_field(target, key)
    return removed


def _apply_transport_updates(target, updates):
    """Validate coupled transport controls, then write supplied values."""
    if not updates:
        return
    validate_profile_numerics({**profile_numerics_values(target), **updates})
    for key, value in updates.items():
        _set_field(target, key, value)


def profile_resolution(document, profile, **options):
    """Resolve ``profile``'s numerics over its referenced named presets, if any."""
    from .._config_presets import KINDS
    from .config_presets import profile_preset
    from .profiles import resolve_numerics

    named, sources = {}, {}
    for kind in KINDS:
        name, values = profile_preset(document, kind, profile)
        for key, value in values.items():
            if key in PROFILE_NUMERICS_KEYS:
                named[key] = value
                sources[key] = f"preset {name}"
    return resolve_numerics(
        profile_numerics_values(profile), named=named, named_sources=sources, **options
    )
