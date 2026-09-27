"""Catalog-profile document queries and mutations.

This module owns profile semantics over an already parsed materials catalog.
CLI parsing, prompts, rendering, and atomic persistence remain in
``pyrite.cli.commands.profile``.
"""

import difflib

import tomlkit

from pyrite._numerics import (
    PROFILE_NUMERICS_KEYS,
    SAMPLING_KEYS,
    TRANSPORT_KEYS,
    validate_profile_numerics,
)
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


def profile_numerics_values(profile):
    """Return explicit profile numerics as plain Python values."""
    values = {}
    for key in PROFILE_NUMERICS_KEYS:
        if key not in profile:
            continue
        value = profile[key]
        if key in SAMPLING_KEYS:
            if not isinstance(value, dict) or not isinstance(value.get("values"), list):
                raise ValueError(f"{key} must be a values grid")
            value = list(value["values"])
        elif hasattr(value, "unwrap"):
            value = value.unwrap()
        values[key] = value
    return values


def set_numerics(document, name, updates):
    """Validate and atomically stage supplied result-affecting numerics."""
    target = existing_profile(document, name)
    merged = {**profile_numerics_values(target), **updates}
    validate_profile_numerics(merged)
    for key, value in updates.items():
        target[key] = values_item([value]) if key in SAMPLING_KEYS else value
    return tuple(updates)


def reset_numerics(document, name, fields=()):
    """Remove selected explicit numerics, or every explicit field when empty."""
    target = existing_profile(document, name)
    selected = tuple(fields) or PROFILE_NUMERICS_KEYS
    removed = tuple(key for key in selected if key in target)
    remaining = profile_numerics_values(target)
    for key in selected:
        remaining.pop(key, None)
    validate_profile_numerics(remaining)
    for key in removed:
        target.pop(key, None)
    return removed


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


def filter_rows(profile):
    filters = profile.get("filters", [])
    if not isinstance(filters, list) or not all(isinstance(row, dict) for row in filters):
        raise ValueError("profile filters must be an array of tables")
    return filters


def physical_detector_row(profile, profiles):
    """Return the selected physical detector, inheriting ``standard``."""
    physical = profile.get("physical_detector")
    if physical is None and profile is not profiles.get("standard"):
        standard = profiles.get("standard", {})
        physical = standard.get("physical_detector") if isinstance(standard, dict) else None
    if physical is not None and not isinstance(physical, dict):
        raise ValueError("profile physical_detector must be a table")
    return physical


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
    if raw_detector is None:
        standard = profiles.get("standard", {})
        raw_detector = standard.get("detector") if isinstance(standard, dict) else None
    detector_ref = str(raw_detector) if isinstance(raw_detector, str) else None
    if detector_ref is not None:
        named = detector_rows(document).get(detector_ref)
        if not isinstance(named, dict):
            unknown_detector(document, detector_ref)
        raw_detector = named
    elif not isinstance(raw_detector, dict):
        raw_detector = {}
    active_detector_keys = {key for key, _label, _unit in ACTIVE_DETECTOR_FIELDS}
    detector = Detector(
        **{key: value for key, value in raw_detector.items() if key in active_detector_keys}
    )
    filters = [
        dict(row.unwrap() if hasattr(row, "unwrap") else row) for row in filter_rows(profile)
    ]
    physical = physical_detector_row(profile, profiles)
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
        "detector_ref": detector_ref,
        "detector": {key: getattr(detector, key) for key, _label, _unit in ACTIVE_DETECTOR_FIELDS},
        "filters": filters,
        "physical_detector": (
            None
            if physical is None
            else dict(physical.unwrap() if hasattr(physical, "unwrap") else physical)
        ),
        "emission": profile.get("emission"),
        "transport_numerics": {key: profile[key] for key in TRANSPORT_KEYS if key in profile},
        "overrides": {
            material: sorted(row)
            for material, row in overrides.items()
            if isinstance(row, dict) and row
        },
        "energy_grid_refs": energy_grid_refs(profile),
    }


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


#: Nested ``physical_detector`` sections that ``reset`` can remove one at a time.
PHYSICAL_SECTIONS = ("scorer", "response", "acquisition")
_ACQUISITION_AXIS_KEYS = (
    "measured_edges_eV",
    "measured_min_eV",
    "measured_max_eV",
    "measured_bin_width_eV",
)


def _toml_value(value):
    if isinstance(value, dict):
        table = tomlkit.table()
        for key, item in value.items():
            table[key] = _toml_value(item)
        return table
    return list(value) if isinstance(value, tuple) else value


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

    Returns ``"created"`` for a new table, ``"copied"`` when the profile first
    received a local copy of ``standard``'s, else ``"updated"``.

    A profile inheriting ``standard``'s detector first receives a profile-local
    copy, so editing it never changes ``standard``. ``response`` with a new
    ``kind`` replaces that section (parameters of another kind never leak into
    it); without one, parameters merge into the existing Timepix3 response. An
    acquisition axis (explicit edges, or min/max/bin width) replaces the other
    spelling, and ``mode = "expected"`` drops any realization seed.
    """
    profile = existing_profile(document, name)
    table = own_physical_detector(profile)
    status = "updated"
    if table is None:
        inherited = physical_detector_row(profile, profile_rows(document))
        status = "created" if inherited is None else "copied"
        if inherited is None and "distance_mm" not in geometry:
            raise ValueError(
                f"profile {name!r} has no physical detector; creating one requires --distance-mm"
            )
        plain = {} if inherited is None else inherited.unwrap()
        table = _toml_value(plain)
        profile["physical_detector"] = table
    for key, value in geometry.items():
        table[key] = _toml_value(value)
    if scorer:
        section = table.setdefault("scorer", tomlkit.table())
        for key, value in scorer.items():
            section[key] = _toml_value(value)
    if response:
        current = table.get("response")
        kind = response.get("kind")
        parameters = {key: value for key, value in response.items() if key != "kind"}
        effective_kind = kind if kind is not None else (current or {}).get("kind")
        if parameters and effective_kind != "timepix3":
            raise ValueError("Timepix3 response options require --response timepix3")
        if kind is not None and (current is None or current.get("kind") != kind):
            table["response"] = _toml_value(dict(response))
        else:
            for key, value in parameters.items():
                current[key] = _toml_value(value)
    if acquisition:
        section = table.setdefault("acquisition", tomlkit.table())
        if any(key in acquisition for key in _ACQUISITION_AXIS_KEYS):
            for key in _ACQUISITION_AXIS_KEYS:
                section.pop(key, None)
        if acquisition.get("mode") == "expected":
            section.pop("seed", None)
        for key, value in acquisition.items():
            section[key] = _toml_value(value)
    return status


def reset_physical_detector(document, name, sections=()):
    """Remove ``name``'s physical detector, or only the named nested sections."""
    profile = existing_profile(document, name)
    table = own_physical_detector(profile)
    if table is None:
        if physical_detector_row(profile, profile_rows(document)) is not None:
            raise ValueError(
                f"profile {name!r} inherits the standard physical detector; reset 'standard' "
                f"or give {name!r} its own with 'pyrite profile physical-detector set'"
            )
        raise ValueError(f"profile {name!r} has no physical detector")
    if not sections:
        profile.pop("physical_detector")
        return
    missing = [section for section in sections if section not in table]
    if missing:
        raise ValueError(f"profile {name!r} physical detector has no {', '.join(missing)} section")
    for section in sections:
        table.pop(section)


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
    elif isinstance(detector, str):
        raise ValueError(
            f"profile has detector = {detector!r} (a named reference); edit it with "
            f"'pyrite detector set {detector} ...', or replace the reference with "
            "--detector NAME"
        )
    elif not isinstance(detector, dict):
        raise ValueError("profile detector must be a table or named reference")
    return detector


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
    message += f". Create it first with: pyrite detector create {name}"
    raise ValueError(message)


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
    validate_profile_numerics({**profile_numerics_values(target), **updates})
    for key, value in updates.items():
        target[key] = value


def create_profile(
    document,
    name,
    source_name,
    *,
    updates,
    beam_name,
    detector_name,
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
    if detector_name is not None and detector_name not in detector_rows(document):
        unknown_detector(document, detector_name)
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
    if detector_name is not None:
        target["detector"] = detector_name
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
    detector_name,
    transport_updates,
    materials,
    all_materials,
    emission,
):
    """Replace supplied profile fields and return overwritten field labels."""
    target = existing_profile(document, name)
    if beam_name is not None and beam_name not in beam_rows(document):
        unknown_beam(document, beam_name)
    if detector_name is not None and detector_name not in detector_rows(document):
        unknown_detector(document, detector_name)
    material_keys = (
        validate_materials(document, csv_materials(materials)) if materials is not None else None
    )
    overwriting = {
        label: range_values(target, catalog_key(label))
        for label in updates
        if catalog_key(label) in target
    }
    for label, values in updates.items():
        target[catalog_key(label)] = values_item(values)
    if beam_name is not None:
        target["beam"] = beam_name
    if detector_name is not None:
        target["detector"] = detector_name
    _apply_transport_updates(target, transport_updates)
    if material_keys is not None:
        target["materials"] = material_keys
    elif all_materials:
        target.pop("materials", None)
    if emission is not None:
        target["emission"] = emission
    return overwriting


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
