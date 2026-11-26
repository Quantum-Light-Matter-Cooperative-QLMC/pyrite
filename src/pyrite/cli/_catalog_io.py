"""Shared TOML helpers for catalog-editing CLI commands.

Catalog document accessors (``RANGES``, ``*_rows``, ``range_values`` ...) are
owned by :mod:`pyrite.campaign._profile_document`; they are re-exported here so
CLI commands keep one import surface.
"""

from pathlib import Path

import tomlkit

from pyrite._catalog_layout import (
    BundledProfileError,
    bundled_catalog,
    read_text,
    write_text,
)
from pyrite.campaign._profile_document import (  # noqa: F401
    RANGES,
    beam_rows,
    detector_rows,
    display,
    material_rows,
    profile_overrides,
    profile_rows,
    range_values,
    values_item,
)
from pyrite.campaign.profile_filters import filter_rows  # noqa: F401
from pyrite.catalog_edit import validate as validate_draft
from pyrite.console.config import catalog_path
from pyrite.console.output import CLIError
from pyrite.energy_grid.apply import _CATALOG_PATH

DEFAULT_PROFILE = "standard"


def active_catalog_path() -> Path:
    """Selected catalog, retaining the legacy test override for the default."""
    selected = catalog_path()
    return _CATALOG_PATH if selected == bundled_catalog().resolve() else selected


def require_writable_object(table, name):
    """Reject bundled mutations before asking for confirmation."""
    from pyrite._catalog_layout import require_writable_object as check

    check(active_catalog_path(), table, name)


def require_new_object(rows, kind, name):
    """Reserve existing names case-insensitively, matching directory catalogs."""
    existing = next((key for key in rows if key.casefold() == name.casefold()), None)
    if existing is not None:
        raise ValueError(
            f"{kind} {name!r} already exists as {existing!r}; "
            f"choose a new name or edit it with: pyrite {kind} set {existing}"
        )


def current_text(path=None):
    """Return the catalog as one TOML text (a directory is assembled)."""
    return read_text(active_catalog_path() if path is None else path)


def catalog_text(path=None):
    text = current_text(path)
    from .._cxr_vocabulary import translate_catalog_text

    return text, tomlkit.parse(translate_catalog_text(text))


def is_unchanged(document, original):
    """Compare edits in the same vocabulary as the editable catalog view."""
    from .._cxr_vocabulary import translate_catalog_text

    return document.as_string() == tomlkit.parse(translate_catalog_text(original)).as_string()


def render_catalog(document):
    """Serialize an editable catalog using canonical names in the same order."""
    from .._cxr_vocabulary import translate_catalog_text

    return translate_catalog_text(document.as_string(), canonical=True)


def validated_catalog(path, text, *, profile="standard"):
    """Parse proposed catalog ``text`` beside ``path`` and return the catalog.

    ``profile`` selects whose energy-grid references are resolved, as a run
    would.
    """
    return validate_draft(path, text, profile=profile)


def validate(path, text):
    """Load ``text`` as the catalog at ``path`` would, without touching it."""
    validated_catalog(path, text)


def atomic_write(path, text):
    """Store ``text``; a directory catalog rewrites only changed object files.

    Refusing to change a bundled demo profile is a runtime failure (exit 1)
    naming the ``create --from`` copy, not a traceback.
    """
    from .._cxr_vocabulary import translate_catalog_text

    text = translate_catalog_text(text, canonical=True)
    try:
        write_text(path, text)
    except BundledProfileError as exc:
        raise CLIError(str(exc)) from None


def beams_table(document):
    """Return the writable top-level ``[beams]`` table, creating it if absent."""
    beams = document.get("beams")
    if beams is None:
        beams = tomlkit.table()
        document["beams"] = beams
    elif not isinstance(beams, dict):
        raise ValueError("catalog beams table must be a table")
    return beams


def detectors_table(document):
    """Return writable top-level ``[detectors]``, creating it if absent."""
    detectors = document.get("detectors")
    if detectors is None:
        detectors = tomlkit.table()
        document["detectors"] = detectors
    elif not isinstance(detectors, dict):
        raise ValueError("catalog detectors table must be a table")
    return detectors


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
