"""Shared catalog document access, range values and TOML copying."""

import difflib

import tomlkit

RANGES = {
    "thickness": "thickness_ang",
    "energy": "energy_keV",
    "polar": "tilt_deg",
    "azimuth": "tilt_azim_deg",
}


EXTRA_RANGES = {"ne_line": "n_electrons", "ne_brem": "n_electrons_brem"}


def catalog_key(label):
    return RANGES.get(label) or EXTRA_RANGES[label]


def profile_rows(document):
    profiles = document.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("catalog profiles table is missing")
    return profiles


def material_rows(document):
    materials = document.get("materials", {})
    if not isinstance(materials, dict):
        raise ValueError("catalog materials table is missing")
    return materials


def beam_rows(document):
    beams = document.get("beams", {})
    if not isinstance(beams, dict):
        raise ValueError("catalog beams table must be a table")
    return beams


def detector_rows(document):
    detectors = document.get("detectors", {})
    if not isinstance(detectors, dict):
        raise ValueError("catalog detectors table must be a table")
    return detectors


def profile_overrides(profile):
    overrides = profile.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    return overrides


def range_values(row, key):
    value = row.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        raise ValueError(f"{key} must be a values grid")
    return [float(item) for item in value["values"]]


def values_item(values):
    item = tomlkit.inline_table()
    item["values"] = values
    return item


def display(values):
    return ", ".join(f"{value:g}" for value in values)


def unknown_profile(document, name):
    profiles = profile_rows(document)
    suggestions = difflib.get_close_matches(name, profiles, n=3, cutoff=0.5)
    message = f"unknown profile: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite profile create {name}"
    raise ValueError(message)


def existing_profile(document, name):
    """Return ``[profiles.NAME]`` or raise with suggestions."""
    profiles = profile_rows(document)
    if name not in profiles:
        unknown_profile(document, name)
    return profiles[name]


def energy_grid_refs(profile):
    """Return this profile's ``material -> artifact digest`` map, sorted."""
    refs = profile.get("energy_grid_refs")
    if not isinstance(refs, dict):
        return {}
    return {material: str(refs[material]) for material in sorted(refs)}


def clone_grid(value):
    """Deep-copy a grid descriptor as inline TOML."""
    plain = value.unwrap() if hasattr(value, "unwrap") else value
    if isinstance(plain, dict):
        table = tomlkit.inline_table()
        for key, item in plain.items():
            table[key] = clone_grid(item)
        return table
    return tomlkit.item(plain)


def unknown_beam(document, name):
    known = beam_rows(document)
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    message = f"unknown beam: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite beam create {name}"
    raise ValueError(message)


def unknown_detector(document, name):
    known = detector_rows(document)
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    message = f"unknown detector: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite detector create scalar {name}"
    raise ValueError(message)
