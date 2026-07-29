"""Shared TOML helpers for catalog-editing CLI commands."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import tomlkit

from cxr_mc.line_grid.apply import _MATERIALS_TOML
from cxr_mc.materials.catalog import load_material_catalog

RANGES = {
    "thickness": "thickness_ang",
    "energy": "energy_keV",
    "polar": "tilt_deg",
    "azimuth": "tilt_azim_deg",
}
DEFAULT_PROFILE = "standard"


def range_values(row, key):
    value = row.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        raise ValueError(f"{key} must be a values grid")
    return [float(item) for item in value["values"]]


def display(values):
    return ", ".join(f"{value:g}" for value in values)


def catalog_text(path=None):
    source = Path(_MATERIALS_TOML if path is None else path)
    text = source.read_text()
    return text, tomlkit.parse(text)


def validate(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        load_material_catalog(Path(temporary))
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def values_item(values):
    item = tomlkit.inline_table()
    item["values"] = values
    return item


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


def profile_overrides(profile):
    overrides = profile.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    return overrides


def material_override_table(profile, material):
    """Return writable ``[profiles.NAME.overrides.MATERIAL]`` table."""
    overrides = profile.get("overrides")
    if overrides is None:
        overrides = tomlkit.table()
        profile["overrides"] = overrides
    elif not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    target = overrides.get(material)
    if target is None:
        target = tomlkit.table()
        overrides[material] = target
    elif not isinstance(target, dict):
        raise ValueError(f"overrides.{material} must be a table")
    return target


def existing_profile(document, name):
    profiles = profile_rows(document)
    if name not in profiles:
        raise ValueError(f"unknown profile: {name}")
    return profiles[name]


def effective_ranges(document, material, profile_name=DEFAULT_PROFILE):
    materials = material_rows(document)
    if material not in materials:
        raise ValueError(f"unknown material: {material}")
    profile = existing_profile(document, profile_name)
    override = profile_overrides(profile).get(material, {})
    if not isinstance(override, dict):
        raise ValueError(f"overrides.{material} must be a table")
    return {
        label: (range_values(override if key in override else profile, key), key in override)
        for label, key in RANGES.items()
    }
