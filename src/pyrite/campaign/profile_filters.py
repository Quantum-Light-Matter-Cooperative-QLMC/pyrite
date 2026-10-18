"""Ordered profile filter queries and edits."""

import tomlkit

from ._profile_document import existing_profile


def filter_rows(profile):
    filters = profile.get("filters", [])
    if not isinstance(filters, list) or not all(isinstance(row, dict) for row in filters):
        raise ValueError("profile filters must be an array of tables")
    return filters


def add_filter(document, profile_name, row):
    profile = existing_profile(document, profile_name)
    filters = filter_rows(profile)
    name = row.get("name")
    if name is not None and any(existing.get("name") == name for existing in filters):
        raise ValueError(f"profile {profile_name!r} already has a filter named {name!r}")
    if "filters" not in profile:
        profile["filters"] = tomlkit.aot()
        filters = profile["filters"]
    filters.append(row)


def _filter_index(filters, profile_name, identifier):
    """Resolve a filter by one-based list index or display name."""
    if identifier.isdigit():
        candidate = int(identifier) - 1
        if 0 <= candidate < len(filters):
            return candidate
    index = next((i for i, row in enumerate(filters) if row.get("name") == identifier), None)
    if index is None:
        raise ValueError(
            f"profile {profile_name!r} has no filter {identifier!r}; use 'pyrite profile filter list {profile_name}'"
        )
    return index


def update_filter(document, profile_name, identifier, changes):
    """Merge ``changes`` into one filter in place, keeping its position.

    Returns the updated row as a plain mapping. A new ``name`` must stay unique
    within the profile; order is never changed, because it is part of the
    observation's identity.
    """
    profile = existing_profile(document, profile_name)
    filters = filter_rows(profile)
    index = _filter_index(filters, profile_name, identifier)
    row = filters[index]
    new_name = changes.get("name")
    if new_name is not None and any(
        other.get("name") == new_name for position, other in enumerate(filters) if position != index
    ):
        raise ValueError(f"profile {profile_name!r} already has a filter named {new_name!r}")
    for key, value in changes.items():
        row[key] = list(value) if isinstance(value, tuple) else value
    return dict(row.unwrap() if hasattr(row, "unwrap") else row)


def remove_filter(document, profile_name, identifier):
    profile = existing_profile(document, profile_name)
    filters = filter_rows(profile)
    index = _filter_index(filters, profile_name, identifier)
    removed = filters.pop(index)
    if not filters:
        profile.pop("filters", None)
    return dict(removed.unwrap() if hasattr(removed, "unwrap") else removed)
