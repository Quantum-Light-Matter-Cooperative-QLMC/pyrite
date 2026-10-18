"""Profile numerical controls and coupled transport validation."""

from pyrite._numerics import PROFILE_NUMERICS_KEYS, SAMPLING_KEYS, validate_profile_numerics

from ._profile_document import existing_profile, values_item
from .profile_cxr_grid import _check_cxr_grid_policy, profile_cxr_grid_values


def profile_numerics_values(profile):
    """Return explicit profile numerics as plain Python values."""
    values = {}
    for key in PROFILE_NUMERICS_KEYS:
        if key not in profile:
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
            f"profile {name!r} has an adaptive precision policy; select fixed counts with: "
            f"pyrite numerics precision disable {name} --cxr-trials N --brem-trials N"
        )
    merged = {**profile_numerics_values(target), **updates}
    validate_profile_numerics(merged)
    for key, value in updates.items():
        target[key] = values_item([value]) if key in SAMPLING_KEYS else value
    _check_cxr_grid_policy(target, profile_cxr_grid_values(target))
    return tuple(updates)


def reset_numerics(document, name, fields=()):
    """Remove selected explicit numerics, or every explicit field when empty."""
    target = existing_profile(document, name)
    selected = tuple(fields) or PROFILE_NUMERICS_KEYS
    removed = tuple(key for key in selected if key in target)
    remaining = profile_numerics_values(target)
    for key in selected:
        remaining.pop(key, None)
    validate_profile_numerics(remaining)
    for key in removed:
        target.pop(key, None)
    return removed


def _apply_transport_updates(target, updates):
    """Validate coupled transport controls, then write supplied values."""
    if not updates:
        return
    validate_profile_numerics({**profile_numerics_values(target), **updates})
    for key, value in updates.items():
        target[key] = value
