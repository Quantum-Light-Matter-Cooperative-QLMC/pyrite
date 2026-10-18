"""Profile creation, field updates, emission, rename and deletion."""

import tomlkit

from pyrite import DATA_DIR

from ._profile_document import (
    RANGES,
    beam_rows,
    catalog_key,
    clone_grid,
    detector_rows,
    display,
    existing_profile,
    material_rows,
    profile_overrides,
    profile_rows,
    range_values,
    unknown_beam,
    unknown_detector,
    values_item,
)
from .profile_cxr_grid import _check_cxr_grid_policy, profile_cxr_grid_values
from .profile_detectors import attach_detector_shortcut
from .profile_membership import csv_materials, validate_materials
from .profile_numerics import _apply_transport_updates

_EMISSION_DECODE = {
    "incoherent": frozenset({"incoherent"}),
    "coherent": frozenset({"coherent"}),
    "both": frozenset({"incoherent", "coherent"}),
}


_CREATION_KEYS = (*RANGES.values(), "E_grid_brem", "materials")


def creation_template(document):
    """Build a new profile from the immutable packaged standard sweep data.

    Override rows are not copied here; :func:`create_profile` adds them once
    membership is final (:func:`member_overrides`).
    """
    packaged = _packaged_standard()
    target = tomlkit.table()
    for key in _CREATION_KEYS:
        if key == "materials":
            # External catalogs may contain a smaller or different material set.
            selected = [name for name in packaged[key] if name in material_rows(document)]
            if selected:
                target[key] = selected
        elif key in packaged:
            target[key] = clone_grid(packaged[key])
    return target


def _packaged_standard():
    return tomlkit.parse((DATA_DIR / "catalog" / "profiles" / "standard.toml").read_text())


def _uniform_grid(value) -> bool:
    return isinstance(value, dict) and "arange" in value


def member_overrides(document, members):
    """Packaged ``standard`` override rows a new profile can actually read.

    Only rows for MEMBERS are copied: a non-member resolves against
    ``standard`` itself (``materials/_parse.py``), so its row would be dead.
    A uniform ``E_grid_brem`` row is dropped too: the case grid replaces its
    start with the medium floor and its stop with E0, so only ``step`` would
    survive, and the built-in default already carries that (issue #256).
    """
    known = material_rows(document)
    overrides = tomlkit.table()
    for material, row in _packaged_standard().get("overrides", {}).items():
        if material not in known or (members is not None and material not in members):
            continue
        entry = tomlkit.table()
        for key, value in row.items():
            if key == "E_grid_brem" and _uniform_grid(value):
                continue
            entry[key] = clone_grid(value)
        if entry:
            overrides[material] = entry
    return overrides


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
    _check_cxr_grid_policy(target, profile_cxr_grid_values(target))
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
    if source_name is not None and source_name not in profiles:
        raise ValueError(f"unknown source profile: {source_name}")
    target = creation_template(document) if source_name is None else tomlkit.table()
    if source_name is not None:
        for key, value in profiles[source_name].items():
            if key != "overrides":
                target[key] = clone_grid(value)
    for label, values in updates.items():
        target[catalog_key(label)] = values_item(values)
    if beam_name is not None:
        target["beam"] = beam_name
    if detector_name is not None:
        attach_detector_shortcut(target, detector_name)
    _apply_transport_updates(target, transport_updates)
    if materials is not None:
        target["materials"] = validate_materials(document, csv_materials(materials))
    if source_name is None:
        members = target.get("materials")
        overrides = member_overrides(document, None if members is None else set(members))
        if overrides:
            target["overrides"] = overrides
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
    temporal_profile=None,
):
    """Replace supplied profile fields and return overwritten field labels.

    ``temporal_profile`` True writes the opt-in key; False removes it (off is
    the absent default).
    """
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
        attach_detector_shortcut(target, detector_name)
    _apply_transport_updates(target, transport_updates)
    if material_keys is not None:
        target["materials"] = material_keys
    elif all_materials:
        target.pop("materials", None)
    if emission is not None:
        target["emission"] = emission
    if temporal_profile is True:
        target["temporal_profile"] = True
    elif temporal_profile is False:
        target.pop("temporal_profile", None)
    _check_cxr_grid_policy(target, profile_cxr_grid_values(target))
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
