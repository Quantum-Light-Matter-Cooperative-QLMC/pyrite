"""Catalog-profile document queries and mutations.

This module owns profile semantics over an already parsed materials catalog.
CLI parsing, prompts, rendering, and atomic persistence remain in
``pyrite.cli.commands.profile``.
"""

from __future__ import annotations

import difflib

import tomlkit

from pyrite.detectors.spec import Detector

RANGES = {
    "thickness": "thickness_ang",
    "energy": "energy_keV",
    "polar": "tilt_deg",
    "azimuth": "tilt_azim_deg",
}
EXTRA_RANGES = {"ne_line": "n_electrons", "ne_brem": "n_electrons_brem"}
ACTIVE_DETECTOR_FIELDS = (
    ("observation_angle_deg", "observation angle", "deg"),
    ("polar_acceptance_deg", "polar acceptance (full span)", "deg"),
    ("solid_angle_sr", "solid angle", "sr"),
)
_EMISSION_DECODE = {
    "incoherent": frozenset({"incoherent"}),
    "coherent": frozenset({"coherent"}),
    "both": frozenset({"incoherent", "coherent"}),
}


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


def profile_payload(document, name):
    profile = existing_profile(document, name)
    profiles = profile_rows(document)
    overrides = profile_overrides(profile)
    materials = profile.get("materials")
    range_keys = [*RANGES.items(), *EXTRA_RANGES.items()]
    beam = profile.get("beam")
    beam_ref = str(beam) if isinstance(beam, str) else None
    beam_payload = beam.unwrap() if beam_ref is None and hasattr(beam, "unwrap") else None
    raw_detector = profile.get("detector")
    if not isinstance(raw_detector, dict):
        standard = profiles.get("standard", {})
        raw_detector = standard.get("detector", {}) if isinstance(standard, dict) else {}
    active_detector_keys = {key for key, _label, _unit in ACTIVE_DETECTOR_FIELDS}
    detector = Detector(
        **{key: value for key, value in raw_detector.items() if key in active_detector_keys}
    )
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
        "detector": {key: getattr(detector, key) for key, _label, _unit in ACTIVE_DETECTOR_FIELDS},
        "emission": profile.get("emission"),
        "transport_numerics": {
            key: profile[key]
            for key in ("straggling", "energy_model", "max_dE_frac")
            if key in profile
        },
        "overrides": {
            material: sorted(row)
            for material, row in overrides.items()
            if isinstance(row, dict) and row
        },
        "energy_grid_refs": energy_grid_refs(profile),
    }


def clone_grid(value):
    """Deep-copy a grid descriptor as inline TOML."""
    plain = value.unwrap() if hasattr(value, "unwrap") else value
    if isinstance(plain, dict):
        table = tomlkit.inline_table()
        for key, item in plain.items():
            table[key] = clone_grid(item)
        return table
    return tomlkit.item(plain)


def detector_table(profile):
    """Return writable ``[profiles.NAME.detector]`` table."""
    detector = profile.get("detector")
    if detector is None:
        detector = tomlkit.table()
        profile["detector"] = detector
    elif not isinstance(detector, dict):
        raise ValueError("profile detector must be a table")
    return detector


def unknown_beam(document, name):
    known = beam_rows(document)
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    message = f"unknown beam: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite beam create {name}"
    raise ValueError(message)


def apply_beam_updates(name, target, updates):
    """Apply inline beam fields without detaching a named beam reference."""
    if not updates:
        return
    existing = target.get("beam")
    if isinstance(existing, str):
        raise ValueError(
            f"profile {name} has beam = {existing!r} (a named reference); "
            f"edit it with 'pyrite beam set {existing} ...', or replace the "
            "reference with --beam NAME"
        )
    if existing is None:
        existing = tomlkit.table()
        target["beam"] = existing
    elif not isinstance(existing, dict):
        raise ValueError("profile beam must be a table or named reference")
    for key, value in updates.items():
        if key in ("longitudinal", "transverse"):
            policy = tomlkit.table()
            for policy_key, policy_value in value.items():
                policy[policy_key] = policy_value
            existing[key] = policy
        else:
            existing[key] = value
    if "transverse" in updates:
        for legacy in (
            "transverse_fwhm_mm",
            "transverse_fwhm_x_mm",
            "transverse_fwhm_y_mm",
        ):
            existing.pop(legacy, None)
    elif "transverse_fwhm_mm" in updates:
        existing.pop("transverse", None)


def merge_values(document, name, updates, *, add):
    """Mutate range values for ``profile add`` and ``profile remove``."""
    target = existing_profile(document, name)
    for label, values in updates.items():
        key = catalog_key(label)
        if key not in target:
            if not add:
                raise ValueError(f"profile {name} has no {key} grid to remove values from")
            existing = []
        else:
            existing = range_values(target, key)
        if add:
            merged = sorted(set(existing) | set(values))
        else:
            missing = [value for value in values if value not in existing]
            if missing:
                raise ValueError(
                    f"{label} values not present in profile {name}: {display(missing)}"
                )
            merged = sorted(set(existing) - set(values))
        target[key] = values_item(merged)


def _current_emission_modes(target):
    value = target.get("emission")
    return set(_EMISSION_DECODE[value]) if value is not None else set()


def _emission_label(modes):
    if modes == {"incoherent", "coherent"}:
        return "both"
    if modes == {"incoherent"}:
        return "incoherent"
    if modes == {"coherent"}:
        return "coherent"
    return None


def apply_emission_add(target, coherent, incoherent):
    current = _current_emission_modes(target)
    requested = {
        mode for mode, flag in (("coherent", coherent), ("incoherent", incoherent)) if flag
    }
    added = sorted(requested - current)
    if not added:
        return None
    new_modes = current | requested
    label = _emission_label(new_modes)
    auto_both = label == "both" and _emission_label(current) != "both"
    target["emission"] = label
    return label, added, auto_both


def apply_emission_remove(name, target, coherent, incoherent):
    current = _current_emission_modes(target)
    requested = {
        mode for mode, flag in (("coherent", coherent), ("incoherent", incoherent)) if flag
    }
    if not requested:
        return None
    missing = sorted(requested - current)
    if missing:
        raise ValueError(f"profile {name} emission does not include: {', '.join(missing)}")
    label = _emission_label(current - requested)
    if label is None:
        target.pop("emission", None)
    else:
        target["emission"] = label
    return label, sorted(requested)


def membership_target(document, name):
    """Return the profile's explicit ``materials`` list or raise with guidance."""
    target = existing_profile(document, name)
    materials = target.get("materials")
    if materials is None:
        raise ValueError(
            f"profile {name!r} has implicit all-catalog-materials membership; "
            "it already includes every material. To restrict it, use: "
            f"pyrite profile set {name} --material MATERIAL,..."
        )
    if not isinstance(materials, list):
        raise ValueError(f"profiles.{name}.materials must be an array of material keys")
    return target, materials


def csv_materials(material_csv):
    requested = [key.strip() for key in material_csv.split(",") if key.strip()]
    if not requested:
        raise ValueError("--material requires at least one material key")
    return requested


def validate_materials(document, requested):
    requested = set(requested)
    known = material_rows(document)
    unknown = sorted(requested - set(known))
    if unknown:
        raise ValueError(f"unknown material: {', '.join(unknown)}")
    return [key for key in known if key in requested]


def group_materials(document, requested, *, allow_unknown=False):
    """Validate requested membership and return catalog-ordered keys."""
    requested = list(requested)
    if not requested:
        raise ValueError("provide MATERIAL keys")
    if allow_unknown:
        known = material_rows(document)
        requested = list(dict.fromkeys(requested))
        return [key for key in known if key in requested] + [
            key for key in requested if key not in known
        ]
    return validate_materials(document, requested)


def add_membership(document, name, requested):
    target, membership = membership_target(document, name)
    requested = validate_materials(document, requested)
    added = [key for key in requested if key not in membership]
    target["materials"] = validate_materials(document, [*membership, *requested])
    return added, sorted(set(requested) - set(added))


def remove_membership(document, name, requested):
    target, membership = membership_target(document, name)
    requested = list(dict.fromkeys(requested))
    removed = [key for key in requested if key in membership]
    target["materials"] = [key for key in membership if key not in removed]
    return removed, sorted(set(requested) - set(removed))


def _apply_transport_updates(target, updates):
    """Validate coupled transport controls, then write supplied values."""
    if not updates:
        return
    energy_model = updates.get("energy_model", target.get("energy_model", "frozen"))
    max_dE_frac = updates.get("max_dE_frac", target.get("max_dE_frac", 0.0))
    if max_dE_frac > 0.0 and energy_model != "midpoint":
        raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")
    for key, value in updates.items():
        target[key] = value


def create_profile(
    document,
    name,
    source_name,
    *,
    updates,
    beam_name,
    beam_updates,
    detector_updates,
    transport_updates,
    materials,
):
    """Create a profile row from a source row and validated CLI values."""
    profiles = profile_rows(document)
    if name in profiles:
        raise ValueError(
            f"profile {name!r} already exists; edit it with: pyrite profile set {name}"
        )
    if beam_name is not None and beam_name not in beam_rows(document):
        unknown_beam(document, beam_name)
    if source_name not in profiles:
        raise ValueError(f"unknown source profile: {source_name}")
    target = tomlkit.table()
    for key, value in profiles[source_name].items():
        if key != "overrides":
            target[key] = clone_grid(value)
    for label, values in updates.items():
        target[catalog_key(label)] = values_item(values)
    if beam_name is not None:
        target["beam"] = beam_name
    else:
        apply_beam_updates(name, target, beam_updates)
    if detector_updates:
        detector = detector_table(target)
        for key, value in detector_updates.items():
            detector[key] = value
    _apply_transport_updates(target, transport_updates)
    if materials is not None:
        target["materials"] = validate_materials(document, csv_materials(materials))
    profiles[name] = target


def set_profile(
    document,
    name,
    *,
    updates,
    beam_name,
    beam_updates,
    detector_updates,
    transport_updates,
    materials,
    all_materials,
    emission,
):
    """Replace supplied profile fields and return overwritten field labels."""
    target = existing_profile(document, name)
    if beam_name is not None and beam_name not in beam_rows(document):
        unknown_beam(document, beam_name)
    material_keys = (
        validate_materials(document, csv_materials(materials)) if materials is not None else None
    )
    overwriting = [label for label in updates if catalog_key(label) in target]
    existing_detector = target.get("detector", {})
    detector_labels = [
        label for key, label, _unit in ACTIVE_DETECTOR_FIELDS if key in detector_updates
    ]
    overwriting.extend(
        label
        for key, label, _unit in ACTIVE_DETECTOR_FIELDS
        if key in detector_updates
        and isinstance(existing_detector, dict)
        and key in existing_detector
    )
    for label, values in updates.items():
        target[catalog_key(label)] = values_item(values)
    if beam_name is not None:
        target["beam"] = beam_name
    else:
        apply_beam_updates(name, target, beam_updates)
    if detector_updates:
        detector = detector_table(target)
        for key, value in detector_updates.items():
            detector[key] = value
    _apply_transport_updates(target, transport_updates)
    if material_keys is not None:
        target["materials"] = material_keys
    elif all_materials:
        target.pop("materials", None)
    if emission is not None:
        target["emission"] = emission
    return overwriting, detector_labels


def rename_profile(document, name, new_name):
    profiles = profile_rows(document)
    if new_name in profiles:
        raise ValueError(f"profile {new_name!r} already exists")
    target = existing_profile(document, name)
    del profiles[name]
    profiles[new_name] = target
    energy_grids = document.get("energy_grids")
    if isinstance(energy_grids, dict) and name in energy_grids:
        grid = energy_grids[name]
        del energy_grids[name]
        energy_grids[new_name] = grid


def delete_profile(document, name):
    """Validate references, remove a profile row, and return override count."""
    target = existing_profile(document, name)
    energy_grids = document.get("energy_grids", {})
    if isinstance(energy_grids, dict) and name in energy_grids:
        raise ValueError(
            f"cannot delete profile {name!r}; still referenced by:\n- "
            f"energy_grids.{name} (shared line-grid store; delete its rows with "
            f"pyrite energy-grid line delete {name} first)"
        )
    overrides = profile_overrides(target)
    n_overrides = sum(1 for row in overrides.values() if isinstance(row, dict) and row)
    del profile_rows(document)[name]
    return n_overrides
