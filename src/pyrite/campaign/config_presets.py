"""Edit named physics/numerics presets and profile references (ADR-0021).

Every operation takes a :class:`~pyrite._config_presets.PresetKind` and
mutates a tomlkit catalog document in memory; callers validate the rendered
catalog and write it atomically, so a rejected change writes nothing.
Bundled-object protection belongs to the catalog writer.
"""

import re

import tomlkit

from .._config_presets import (
    KINDS,
    NUMERICS,
    SAMPLING_POLICY_KEYS,
    fold_preset,
    preset_values,
    validate_values,
)
from .._numerics import SAMPLING_KEYS
from ._profile_document import existing_profile, profile_rows

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def preset_rows(document, kind, *, create=False):
    """Return the document's preset table for ``kind``, optionally creating it."""
    if kind.table not in document:
        if not create:
            return {}
        document[kind.table] = tomlkit.table(is_super_table=True)
    return document[kind.table]


def existing_preset(document, kind, name):
    """Return preset ``name`` or raise with the known names."""
    rows = preset_rows(document, kind)
    if name not in rows:
        known = ", ".join(rows) or "none"
        raise ValueError(
            f"unknown {kind.table} preset {name!r} (known: {known}); "
            f"create it with: pyrite {kind.table} preset create {name}"
        )
    return rows[name]


def require_new(document, kind, name):
    """Reject invalid names and reserve existing ones case-insensitively."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid {kind.table} preset name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )
    rows = preset_rows(document, kind)
    existing = next((key for key in rows if key.casefold() == name.casefold()), None)
    if existing is not None:
        raise ValueError(f"{kind.table} preset {name!r} already exists as {existing!r}")


def referencing_profiles(document, kind, name):
    """Return sorted names of profiles that reference preset ``name``."""
    return sorted(
        profile
        for profile, row in profile_rows(document).items()
        if isinstance(row, dict) and row.get(kind.reference_key) == name
    )


def _write_values(row, values):
    for key, value in values.items():
        row[key] = value


def create_preset(document, kind, name, *, source=None, values=None, description=None):
    """Add preset ``name``, cloned from ``source`` when given, then apply ``values``."""
    require_new(document, kind, name)
    origin = existing_preset(document, kind, source) if source is not None else {}
    merged = {**preset_values(kind, origin), **(values or {})}
    validate_values(kind, merged)
    row = tomlkit.table()
    description = description if description is not None else origin.get("description")
    if description is not None:
        row["description"] = description
    _write_values(row, {key: merged[key] for key in kind.keys if key in merged})
    preset_rows(document, kind, create=True)[name] = row
    return row


def set_preset(document, kind, name, *, values=None, description=None):
    """Validate and apply field updates to preset ``name``."""
    row = existing_preset(document, kind, name)
    updates = dict(values or {})
    validate_values(kind, {**preset_values(kind, row), **updates})
    _write_values(row, updates)
    if description is not None:
        row["description"] = description


def reset_preset(document, kind, name, fields=()):
    """Remove selected fields, or every configuration field, from preset ``name``."""
    row = existing_preset(document, kind, name)
    selected = tuple(fields) or kind.keys
    remaining = {k: v for k, v in preset_values(kind, row).items() if k not in selected}
    validate_values(kind, remaining)
    removed = tuple(key for key in selected if key in row)
    for key in removed:
        del row[key]
    return removed


def rename_preset(document, kind, name, new_name):
    """Rename preset ``name`` and every profile reference to it."""
    if new_name == name:
        raise ValueError(f"{kind.table} preset {name!r} already named {new_name!r}")
    require_new(document, kind, new_name)
    rows = preset_rows(document, kind)
    row = existing_preset(document, kind, name)
    del rows[name]
    rows[new_name] = row
    for profile in referencing_profiles(document, kind, name):
        profile_rows(document)[profile][kind.reference_key] = new_name


def delete_preset(document, kind, name):
    """Remove an unreferenced preset ``name``."""
    existing_preset(document, kind, name)
    referents = referencing_profiles(document, kind, name)
    if referents:
        raise ValueError(
            f"cannot delete {kind.table} preset {name!r}; still referenced by profiles: "
            + ", ".join(referents)
        )
    del preset_rows(document, kind)[name]


def _unwrap(value):
    return value.unwrap() if hasattr(value, "unwrap") else value


def inline_values(kind, profile):
    """Return a profile's own preset-owned fields as preset values.

    Single-value trial-count grids become scalar counts; multi-value grids are
    sweep axes and stay with the profile.
    """
    values = {}
    for key in kind.keys:
        if key not in profile:
            continue
        value = _unwrap(profile[key])
        if key in SAMPLING_KEYS:
            grid = value.get("values") if isinstance(value, dict) else None
            if not isinstance(grid, list) or len(grid) != 1:
                continue
            value = grid[0]
        values[key] = value
    return values


def attach_preset(document, kind, profile_name, name):
    """Reference preset ``name`` from ``profile_name``; own fields stay overrides."""
    profile = existing_profile(document, profile_name)
    existing_preset(document, kind, name)
    profile[kind.reference_key] = name


def detach_preset(document, kind, profile_name):
    """Remove ``profile_name``'s preset reference and return the detached name."""
    profile = existing_profile(document, profile_name)
    if kind.reference_key not in profile:
        raise ValueError(f"profile {profile_name} has no {kind.table} preset to remove")
    return str(profile.pop(kind.reference_key))


def adopt_profile(document, kind, profile_name, name, *, description=None):
    """Move a profile's effective preset-owned values into new preset ``name``.

    The new preset holds the current preset's values (if any) overlaid with
    the profile's own fields of this kind; the profile then references it and
    loses those fields, so every effective value is unchanged. Returns the
    moved field names.
    """
    profile = existing_profile(document, profile_name)
    current = profile.get(kind.reference_key)
    inline = inline_values(kind, profile)
    if current is None and not inline:
        raise ValueError(f"profile {profile_name} has no {kind.table} fields to adopt")
    base = preset_values(kind, existing_preset(document, kind, current)) if current else {}
    if kind is NUMERICS and any(key in profile for key in SAMPLING_POLICY_KEYS):
        # The profile's own sampling policy replaces the preset's whole group.
        base = {key: value for key, value in base.items() if key not in SAMPLING_POLICY_KEYS}
    create_preset(
        document,
        kind,
        name,
        values={**base, **inline},
        description=description or f"Adopted from profile {profile_name}",
    )
    for key in inline:
        del profile[key]
    profile[kind.reference_key] = name
    return tuple(inline)


def profile_preset(document, kind, profile):
    """Return ``(name, values)`` for a profile row's preset, or ``(None, {})``."""
    name = profile.get(kind.reference_key)
    if name is None:
        return None, {}
    values = preset_values(kind, existing_preset(document, kind, str(name)))
    if kind is NUMERICS and any(key in profile for key in SAMPLING_POLICY_KEYS):
        values = {key: value for key, value in values.items() if key not in SAMPLING_POLICY_KEYS}
    return str(name), values


def effective_profile(document, profile):
    """Return ``profile`` with its referenced presets folded in, as the catalog does.

    Unknown references are left for catalog validation to report.
    """
    row = dict(profile)
    for kind in KINDS:
        name = profile.get(kind.reference_key)
        rows = preset_rows(document, kind)
        if isinstance(name, str) and name in rows:
            row = fold_preset(kind, rows[name], row)
    return row
