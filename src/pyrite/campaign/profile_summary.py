"""Read-only profile summary assembly for catalog consumers."""

from pyrite._config_presets import KINDS
from pyrite._numerics import TRANSPORT_KEYS

from ._profile_document import (
    EXTRA_RANGES,
    RANGES,
    detector_rows,
    energy_grid_refs,
    existing_profile,
    profile_overrides,
    range_values,
    unknown_detector,
)
from .config_presets import effective_profile
from .profile_cxr_grid import profile_cxr_grid_values
from .profile_detectors import (
    ACTIVE_DETECTOR_FIELDS,
    _scalar_detector,
    physical_detector_row,
    profile_detector_entries,
)
from .profile_filters import filter_rows
from .profile_precision import precision_resolution


def _acceptance(detector):
    return {key: getattr(detector, key) for key, _label, _unit in ACTIVE_DETECTOR_FIELDS}


def _legacy_detector_payload(document, profile):
    """Keep the legacy scalar summary even when a collection is declared."""
    raw_detector = profile.get("detector")
    detector_present = raw_detector is not None
    detector_ref = str(raw_detector) if isinstance(raw_detector, str) else None
    if detector_ref is not None:
        named = detector_rows(document).get(detector_ref)
        if not isinstance(named, dict):
            unknown_detector(document, detector_ref)
        raw_detector = named
    elif not isinstance(raw_detector, dict):
        raw_detector = {}
    detector = _scalar_detector(raw_detector)
    return {
        "detector_ref": detector_ref,
        "detector_present": detector_present,
        "detector": _acceptance(detector),
    }


def _detector_payload(document, entry):
    """Resolve one named or inline detector without rewriting its declaration."""
    reference = str(entry) if isinstance(entry, str) else None
    if reference is not None:
        entry = detector_rows(document).get(reference)
        if entry is None:
            unknown_detector(document, reference)
    values = dict(entry.unwrap() if hasattr(entry, "unwrap") else entry)
    if "distance_mm" not in values:
        acceptance = _acceptance(_scalar_detector(values))
        return {
            "kind": "scalar",
            "reference": reference,
            "settings": acceptance.copy(),
            "acceptance": acceptance,
        }
    from .observation import physical_detector_from_config

    geometry = {
        key: value
        for key, value in values.items()
        if key
        in {"distance_mm", "polar_deg", "azimuth_deg", "roll_deg", "offset_mm", "shape", "pitch_mm"}
    }
    projection = physical_detector_from_config(geometry).scalar_detector()
    return {
        "kind": "pixel",
        "reference": reference,
        "settings": values,
        "acceptance": _acceptance(projection),
    }


def detector_payloads(document, profile):
    """Summarize the same collection/legacy IDs used by edits and migration."""
    return {
        identifier: _detector_payload(document, entry)
        for identifier, entry in profile_detector_entries(profile).items()
    }


def profile_payload(document, name):
    profile = existing_profile(document, name)
    overrides = profile_overrides(profile)
    materials = profile.get("materials")
    range_keys = [*RANGES.items(), *EXTRA_RANGES.items()]
    beam = profile.get("beam")
    beam_ref = str(beam) if isinstance(beam, str) else None
    beam_payload = beam.unwrap() if beam_ref is None and hasattr(beam, "unwrap") else None
    legacy_detector = _legacy_detector_payload(document, profile)
    filters = [
        dict(row.unwrap() if hasattr(row, "unwrap") else row) for row in filter_rows(profile)
    ]
    physical = physical_detector_row(profile)
    presets = {kind.table: profile.get(kind.reference_key) for kind in KINDS}
    effective = effective_profile(document, profile)
    precision = dict(
        zip(
            ("explicit", "effective", "source", "fixed_reason"),
            precision_resolution(effective),
            strict=True,
        )
    )
    if precision["source"] == "profile" and "precision" not in profile:
        precision["source"] = f"preset {presets['numerics']}"
    return {
        "name": name,
        "ranges": [
            {"name": label, "catalog_key": key, "values": range_values(profile, key)}
            for label, key in range_keys
            if key in profile
        ],
        "materials": list(materials) if isinstance(materials, list) else None,
        "beam": beam_payload,
        "beam_ref": beam_ref,
        **legacy_detector,
        "detectors": detector_payloads(document, profile),
        "filters": filters,
        "physical_detector": (
            None
            if physical is None
            else dict(physical.unwrap() if hasattr(physical, "unwrap") else physical)
        ),
        "emission": profile.get("emission"),
        "effective_emission": effective.get("emission", "incoherent"),
        "presets": presets,
        "temporal_profile": profile.get("temporal_profile") is True,
        "line_grid_policy": profile_cxr_grid_values(profile),
        "precision": precision,
        "transport_numerics": {key: profile[key] for key in TRANSPORT_KEYS if key in profile},
        "overrides": {
            material: sorted(row)
            for material, row in overrides.items()
            if isinstance(row, dict) and row
        },
        "energy_grid_refs": energy_grid_refs(profile),
    }
