"""Profile detector attachments, legacy storage and explicit migration."""

import tomlkit

from pyrite.detectors.editing import _toml_value, merge_pixel_detector
from pyrite.detectors.spec import Detector

from ._profile_document import clone_grid, existing_profile

ACTIVE_DETECTOR_FIELDS = (
    ("observation_angle_deg", "observation angle", "deg"),
    ("polar_acceptance_deg", "polar acceptance (full span)", "deg"),
    ("solid_angle_sr", "solid angle", "sr"),
)


PHYSICAL_SECTIONS = ("scorer", "response", "acquisition")


def physical_detector_row(profile):
    """Return only the profile's own physical detector."""
    physical = profile.get("physical_detector")
    if physical is not None and not isinstance(physical, dict):
        raise ValueError("profile physical_detector must be a table")
    return physical


def _scalar_detector(values):
    """Build the scalar ``Detector`` a legacy detector row resolves to."""
    angle = values.get("observation_angle_deg")
    return Detector(
        **({} if angle is None else {"observation_angle_deg": angle}),
        _legacy_acceptance=(values.get("polar_acceptance_deg"), values.get("solid_angle_sr")),
    )


def own_physical_detector(profile):
    """Return the profile's own ``physical_detector`` table, or ``None``."""
    table = profile.get("physical_detector")
    if table is not None and not isinstance(table, dict):
        raise ValueError("profile physical_detector must be a table")
    return table


def set_physical_detector(
    document, name, *, geometry, scorer=None, response=None, acquisition=None
):
    """Merge updates into ``name``'s physical detector.

    ``response`` with a new ``kind`` replaces that section (parameters of
    another kind never leak into it); without one, parameters merge into the
    existing Timepix3 response. An
    acquisition axis (explicit edges, or min/max/bin width) replaces the other
    spelling, and ``mode = "expected"`` drops any realization seed.
    """
    profile = existing_profile(document, name)
    table = own_physical_detector(profile)
    if table is None:
        if "distance_mm" not in geometry:
            raise ValueError(
                f"profile {name!r} has no physical detector; creating one requires --distance-mm"
            )
        table = _toml_value({})
        profile["physical_detector"] = table
    merge_pixel_detector(
        table, geometry=geometry, scorer=scorer, response=response, acquisition=acquisition
    )


def reset_physical_detector(document, name, sections=()):
    """Remove ``name``'s physical detector, or only the named nested sections."""
    profile = existing_profile(document, name)
    table = own_physical_detector(profile)
    if table is None:
        raise ValueError(f"profile {name!r} has no physical detector")
    if not sections:
        profile.pop("physical_detector")
        return
    missing = [section for section in sections if section not in table]
    if missing:
        raise ValueError(f"profile {name!r} physical detector has no {', '.join(missing)} section")
    for section in sections:
        table.pop(section)


def detector_table(profile):
    """Return writable ``[profiles.NAME.detector]`` table."""
    detector = profile.get("detector")
    if detector is None:
        detector = tomlkit.table()
        profile["detector"] = detector
    elif isinstance(detector, str):
        raise ValueError(
            f"profile has detector = {detector!r} (a named reference); edit it with "
            f"'pyrite detector set {detector} ...', or replace the reference with "
            "--detector NAME"
        )
    elif not isinstance(detector, dict):
        raise ValueError("profile detector must be a table or named reference")
    return detector


def attach_detector_shortcut(profile, reference):
    """Replace an unambiguous attachment without discarding a collection."""
    collection = profile.get("detectors")
    if collection is None:
        profile["detector"] = reference
    elif len(collection) == 1:
        collection[next(iter(collection))] = reference
    else:
        raise ValueError(
            "multiple detector IDs; use: pyrite profile detector set PROFILE ID --detector REF"
        )


def profile_detector_entries(profile):
    """Return the existing detector IDs without converting legacy storage."""
    if "detectors" in profile:
        return profile["detectors"]
    entries = {}
    if "physical_detector" in profile:
        entries["physical"] = profile["physical_detector"]
    if "detector" in profile:
        entries["default"] = profile["detector"]
    return entries or {"default": {}}


def profile_detector_slot(profile, identifier, *, create_physical=False):
    """Return the owning table/key for one detector, preserving its representation."""
    if "detectors" in profile:
        if identifier not in profile["detectors"]:
            raise ValueError(f"unknown detector ID {identifier!r}")
        return profile["detectors"], identifier
    entries = profile_detector_entries(profile)
    if identifier not in entries and not (create_physical and identifier == "physical"):
        raise ValueError(f"unknown detector ID {identifier!r}; available: {', '.join(entries)}")
    return profile, "physical_detector" if identifier == "physical" else "detector"


def migrate_profile_detectors(document, name):
    """Explicitly convert legacy declarations, preserving order and resolved defaults."""
    profile = existing_profile(document, name)
    if "detectors" in profile:
        raise ValueError(f"profile {name!r} already uses a detector collection")
    collection = tomlkit.table()
    for identifier, entry in profile_detector_entries(profile).items():
        # An implicit scalar uses code defaults; collection entries must declare geometry.
        if not entry:
            entry = {
                key: getattr(Detector(), key)
                for key, _, _ in ACTIVE_DETECTOR_FIELDS
                if getattr(Detector(), key) is not None
            }
        collection[identifier] = clone_grid(entry)
    profile.pop("detector", None)
    profile.pop("physical_detector", None)
    profile["detectors"] = collection
    return list(collection)
