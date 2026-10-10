"""Shared TOML helpers for catalog-editing CLI commands."""

import os
import tempfile
from pathlib import Path

import tomlkit

from pyrite._catalog_layout import (
    BundledProfileError,
    bundled_catalog,
    catalog_root,
    read_text,
    write_text,
)
from pyrite.console.config import catalog_path
from pyrite.console.output import CLIError
from pyrite.energy_grid.apply import _CATALOG_PATH
from pyrite.materials.catalog import load_material_catalog

RANGES = {
    "thickness": "thickness_ang",
    "energy": "energy_keV",
    "polar": "tilt_deg",
    "azimuth": "tilt_azim_deg",
}
DEFAULT_PROFILE = "standard"


def active_catalog_path() -> Path:
    """Selected catalog, retaining the legacy test override for the default."""
    selected = catalog_path()
    return _CATALOG_PATH if selected == bundled_catalog().resolve() else selected


def range_values(row, key):
    value = row.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        raise ValueError(f"{key} must be a values grid")
    return [float(item) for item in value["values"]]


def display(values):
    return ", ".join(f"{value:g}" for value in values)


def current_text(path=None):
    """Return the catalog as one TOML text (a directory is assembled)."""
    return read_text(active_catalog_path() if path is None else path)


def catalog_text(path=None):
    text = current_text(path)
    return text, tomlkit.parse(text)


def validated_catalog(path, text, *, profile="standard"):
    """Parse proposed catalog ``text`` beside ``path`` and return the catalog.

    ``profile`` selects whose energy-grid references are resolved, as a run
    would.
    """
    root = catalog_root(path)
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".", suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        return load_material_catalog(Path(temporary), profile=profile)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate(path, text):
    """Load ``text`` as the catalog at ``path`` would, without touching it."""
    validated_catalog(path, text)


def atomic_write(path, text):
    """Store ``text``; a directory catalog rewrites only changed object files.

    Refusing to change a bundled demo profile is a runtime failure (exit 1)
    naming the ``create --from`` copy, not a traceback.
    """
    try:
        write_text(path, text)
    except BundledProfileError as exc:
        raise CLIError(str(exc)) from None


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


def beam_rows(document):
    """Return ``[beams.*]`` rows, or ``{}`` -- unlike materials/profiles,
    ``beams`` is an optional top-level table."""
    beams = document.get("beams", {})
    if not isinstance(beams, dict):
        raise ValueError("catalog beams table must be a table")
    return beams


def beams_table(document):
    """Return the writable top-level ``[beams]`` table, creating it if absent."""
    beams = document.get("beams")
    if beams is None:
        beams = tomlkit.table()
        document["beams"] = beams
    elif not isinstance(beams, dict):
        raise ValueError("catalog beams table must be a table")
    return beams


def detector_rows(document):
    """Return optional top-level ``[detectors.*]`` rows."""
    detectors = document.get("detectors", {})
    if not isinstance(detectors, dict):
        raise ValueError("catalog detectors table must be a table")
    return detectors


def detectors_table(document):
    """Return writable top-level ``[detectors]``, creating it if absent."""
    detectors = document.get("detectors")
    if detectors is None:
        detectors = tomlkit.table()
        document["detectors"] = detectors
    elif not isinstance(detectors, dict):
        raise ValueError("catalog detectors table must be a table")
    return detectors


def profile_overrides(profile):
    overrides = profile.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    return overrides


def filter_rows(profile):
    """Return ``[[profiles.NAME.filters]]`` rows, validating its TOML shape."""
    filters = profile.get("filters", [])
    if not isinstance(filters, list) or not all(isinstance(row, dict) for row in filters):
        raise ValueError("profile filters must be an array of tables")
    return filters


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
