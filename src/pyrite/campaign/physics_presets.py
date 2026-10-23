"""Edit named ``[physics.NAME]`` presets and profile references (ADR-0021).

Every operation mutates a tomlkit catalog document in memory; callers validate
the rendered catalog and write it atomically, so a rejected change writes
nothing. Bundled-object protection belongs to the catalog writer.
"""

import re

import tomlkit

from .._numerics import validate_profile_numerics
from .._physics_presets import (
    EMISSION_VALUES,
    PRESET_KEYS,
    REFERENCE_KEY,
    TABLE,
    preset_values,
)
from ._profile_document import existing_profile, profile_rows

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def preset_rows(document, *, create=False):
    """Return the document's ``[physics]`` table, optionally creating it."""
    if TABLE not in document:
        if not create:
            return {}
        document[TABLE] = tomlkit.table(is_super_table=True)
    return document[TABLE]


def existing_preset(document, name):
    """Return preset ``name`` or raise with the known names."""
    rows = preset_rows(document)
    if name not in rows:
        known = ", ".join(rows) or "none"
        raise ValueError(
            f"unknown physics preset {name!r} (known: {known}); "
            f"create it with: pyrite physics preset create {name}"
        )
    return rows[name]


def check_name(name):
    """Reject names that cannot be a catalog object file stem."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid physics preset name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


def require_new(document, name):
    """Reserve existing names case-insensitively, matching directory catalogs."""
    check_name(name)
    existing = next(
        (key for key in preset_rows(document) if key.casefold() == name.casefold()), None
    )
    if existing is not None:
        raise ValueError(f"physics preset {name!r} already exists as {existing!r}")


def referencing_profiles(document, name):
    """Return sorted names of profiles whose ``physics`` reference is ``name``."""
    return sorted(
        profile
        for profile, row in profile_rows(document).items()
        if isinstance(row, dict) and row.get(REFERENCE_KEY) == name
    )


def _validate(values):
    emission = values.get("emission")
    if emission is not None and emission not in EMISSION_VALUES:
        raise ValueError(f"emission must be one of {EMISSION_VALUES}")
    validate_profile_numerics({key: value for key, value in values.items() if key != "emission"})


def create_preset(document, name, *, source=None, values=None, description=None):
    """Add preset ``name``, cloned from ``source`` when given, then apply ``values``."""
    require_new(document, name)
    base = preset_values(existing_preset(document, source)) if source is not None else {}
    merged = {**base, **(values or {})}
    unknown = sorted(set(merged) - set(PRESET_KEYS))
    if unknown:
        raise ValueError(f"unknown physics preset field: {unknown[0]}")
    _validate(merged)
    row = tomlkit.table()
    if description is None and source is not None:
        description = existing_preset(document, source).get("description")
    if description is not None:
        row["description"] = description
    for key in PRESET_KEYS:
        if key in merged:
            row[key] = merged[key]
    preset_rows(document, create=True)[name] = row
    return row


def set_preset(document, name, *, values=None, description=None):
    """Validate and apply field updates to preset ``name``."""
    row = existing_preset(document, name)
    updates = dict(values or {})
    _validate({**preset_values(row), **updates})
    for key, value in updates.items():
        row[key] = value
    if description is not None:
        row["description"] = description


def reset_preset(document, name, fields=()):
    """Remove selected fields, or every configuration field, from preset ``name``."""
    row = existing_preset(document, name)
    selected = tuple(fields) or PRESET_KEYS
    remaining = {key: value for key, value in preset_values(row).items() if key not in selected}
    _validate(remaining)
    removed = tuple(key for key in selected if key in row)
    for key in removed:
        del row[key]
    return removed


def rename_preset(document, name, new_name):
    """Rename preset ``name`` and every profile reference to it."""
    if new_name == name:
        raise ValueError(f"physics preset {name!r} already named {new_name!r}")
    require_new(document, new_name)
    rows = preset_rows(document)
    row = existing_preset(document, name)
    del rows[name]
    rows[new_name] = row
    for profile in referencing_profiles(document, name):
        profile_rows(document)[profile][REFERENCE_KEY] = new_name


def delete_preset(document, name):
    """Remove an unreferenced preset ``name``."""
    existing_preset(document, name)
    referents = referencing_profiles(document, name)
    if referents:
        raise ValueError(
            f"cannot delete physics preset {name!r}; still referenced by profiles: "
            + ", ".join(referents)
        )
    del preset_rows(document)[name]


def inline_physics(profile):
    """Return a profile's own explicit physics fields as plain values."""
    return {
        key: profile[key].unwrap() if hasattr(profile[key], "unwrap") else profile[key]
        for key in PRESET_KEYS
        if key in profile
    }


def attach_preset(document, profile_name, name):
    """Reference preset ``name`` from ``profile_name``; inline fields stay overrides."""
    profile = existing_profile(document, profile_name)
    existing_preset(document, name)
    profile[REFERENCE_KEY] = name
    return tuple(inline_physics(profile))


def detach_preset(document, profile_name):
    """Remove ``profile_name``'s preset reference and return the detached name."""
    profile = existing_profile(document, profile_name)
    if REFERENCE_KEY not in profile:
        raise ValueError(f"profile {profile_name} has no physics preset to remove")
    return str(profile.pop(REFERENCE_KEY))


def adopt_profile(document, profile_name, name, *, description=None):
    """Move a profile's effective preset-owned values into new preset ``name``.

    The new preset holds the current preset's values (if any) overlaid with
    the profile's inline physics fields; the profile then references it and
    loses those inline fields, so every effective value is unchanged.
    """
    profile = existing_profile(document, profile_name)
    current = profile.get(REFERENCE_KEY)
    inline = inline_physics(profile)
    if current is None and not inline:
        raise ValueError(f"profile {profile_name} has no physics fields to adopt")
    base = preset_values(existing_preset(document, current)) if current is not None else {}
    create_preset(
        document,
        name,
        values={**base, **inline},
        description=description or f"Adopted from profile {profile_name}",
    )
    for key in inline:
        del profile[key]
    profile[REFERENCE_KEY] = name
    return tuple(inline)


def profile_preset(document, profile):
    """Return ``(name, values)`` for a profile row's preset, or ``(None, {})``."""
    name = profile.get(REFERENCE_KEY)
    if name is None:
        return None, {}
    return str(name), preset_values(existing_preset(document, str(name)))
