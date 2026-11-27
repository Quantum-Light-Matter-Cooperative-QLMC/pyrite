"""Profile numerical controls and coupled transport validation."""

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields

import tomlkit

from pyrite._fiber_average import FiberAverage
from pyrite._numerics import (
    CHANNELING_KEYS,
    PROFILE_NUMERICS_KEYS,
    SAMPLING_KEYS,
    validate_profile_numerics,
)
from pyrite._off_settings import normalize_off_values
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
    return normalize_off_values(values)


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


#: Fields of the profile's ``[cxr_fiber]`` table (``FiberAverage``, #1483).
FIBER_TABLE_KEYS = tuple(field.name for field in dataclass_fields(FiberAverage))
_FIBER_TABLE = "cxr_fiber"


def _fiber_values(name, table, owner="profile"):
    """``table`` as plain values; refuse a non-table or unknown entries."""
    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise ValueError(f"{owner} {name!r} cxr_fiber must be a table of {list(FIBER_TABLE_KEYS)}")
    values = dict(table.unwrap() if hasattr(table, "unwrap") else table)
    unknown = sorted(set(values) - set(FIBER_TABLE_KEYS))
    if unknown:
        raise ValueError(
            f"{owner} {name!r} cxr_fiber has unknown entry {unknown[0]!r}; "
            f"expected {list(FIBER_TABLE_KEYS)}"
        )
    return values


def _inherited_fiber(document, profile):
    """The attached numerics preset's ``cxr_fiber`` values, or ``{}``."""
    from .._config_presets import NUMERICS
    from .config_presets import profile_preset

    preset, values = profile_preset(document, NUMERICS, profile)
    return _fiber_values(preset, values.get(_FIBER_TABLE), owner="numerics preset")


def _checked_fiber(name, values):
    """An inline ``cxr_fiber`` table in ``FIBER_TABLE_KEYS`` order, cutoff required."""
    from ..montecarlo.runner.reflection_cutoff import F0_RANGE_INVANG

    cutoff = values.get("cutoff_invang")
    if isinstance(cutoff, int | float) and cutoff > F0_RANGE_INVANG:
        raise ValueError(
            f"cxr_fiber cutoff_invang {cutoff} exceeds the form-factor range "
            f"({F0_RANGE_INVANG:.1f} 1/angstrom)"
        )
    if "cutoff_invang" not in values:
        raise ValueError(
            f"profile {name!r} cxr_fiber needs cutoff_invang (no default: the full-fiber "
            f"domain is unconverged); pass --cxr-fiber-cutoff-invang, or remove the table "
            f"with: pyrite profile numerics reset {name} cxr-fiber"
        )
    table = tomlkit.inline_table()
    table.update({key: values[key] for key in FIBER_TABLE_KEYS if key in values})
    return table


def set_fiber(document, name, updates):
    """Patch PROFILE's ``[cxr_fiber]`` table with ``updates`` (``FIBER_TABLE_KEYS``).

    Unnamed fields keep their stored values; a profile without its own table
    starts from the attached numerics preset's. The result is validated with
    the other profile numerics before anything is written.
    """
    target = existing_profile(document, name)
    current = _fiber_values(name, target.get(_FIBER_TABLE))
    if _FIBER_TABLE not in target:
        current = _inherited_fiber(document, target)
    table = _checked_fiber(name, {**current, **updates})
    return set_numerics(document, name, {_FIBER_TABLE: table})


def reset_fiber(document, name, fields=(), *, explicit=True):
    """Remove selected ``[cxr_fiber]`` fields, or the whole table when empty.

    Only the profile's own table changes: a preset-supplied table stays in
    effect; naming it (``explicit``) on a profile without its own is refused.
    """
    target = existing_profile(document, name)
    if _FIBER_TABLE not in target:
        if explicit and _inherited_fiber(document, target):
            raise ValueError(
                f"profile {name!r} has no cxr_fiber table of its own; the full-fiber domain "
                "comes from its numerics preset (detach or edit the preset to remove it)"
            )
        return ()
    values = _fiber_values(name, target[_FIBER_TABLE])
    remaining = {key: value for key, value in values.items() if key not in fields}
    if not fields or not remaining:
        return reset_numerics(document, name, (_FIBER_TABLE,))
    set_numerics(document, name, {_FIBER_TABLE: _checked_fiber(name, remaining)})
    return (_FIBER_TABLE,)


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
