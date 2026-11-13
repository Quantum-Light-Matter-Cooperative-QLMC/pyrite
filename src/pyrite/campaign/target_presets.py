"""Typed adapters for validated scalar target catalog values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.materials import MaterialCatalog, MaterialSpec

    from .geometry import Slab, Stack


def _typed_cut(values: dict[str, Any]) -> None:
    """Replace a decoded ``cut`` mapping by its :class:`CrystalCut` value."""
    cut = values.get("cut")
    if cut is not None:
        from .geometry import CrystalCut

        values["cut"] = CrystalCut(surface_hkl=tuple(cut["surface_hkl"]))


def target_from_preset(fields, catalog):
    """Lower recipe references through the same flat adapter used by campaigns."""
    from .flat_geometry import target_from_flat

    values = dict(fields)
    values["material"] = catalog.materials[values["material"]].crystal_key
    values["stack"] = values["stack"] or None
    _typed_cut(values)
    return target_from_flat(**values)


def load_target(name: str, *, catalog: MaterialCatalog | None = None) -> Slab | Stack:
    """Load a scalar named target as an existing immutable Slab or Stack.

    Definition thicknesses use angstroms, angles degrees, and transverse
    dimensions millimetres. Entrance recipe keys resolve to their crystal key;
    no catalog name or description enters the returned physical value.

    Raises
    ------
    KeyError
        If the preset does not exist in the selected catalog.
    """
    if catalog is None:
        from pyrite.materials import CATALOG as catalog
    try:
        fields = catalog.named_targets[name]
    except KeyError:
        raise KeyError(
            f"unknown target preset {name!r}; have {sorted(catalog.named_targets)}"
        ) from None
    return target_from_preset(fields, catalog)


def material_target(spec: MaterialSpec) -> Any:
    """Build the resolved catalog target without retaining reference metadata."""
    from .flat_geometry import target_from_flat

    if spec.target_fields is not None:
        values: dict[str, Any] = dict(spec.target_fields)
        values.update(
            {
                field: getattr(spec.scan, field)
                for field in ("thickness_ang", "tilt_deg", "tilt_azim_deg")
            }
        )
        values["material"] = spec.crystal_key
        values["stack"] = values["stack"] or None
        _typed_cut(values)
        return target_from_flat(**values)
    return target_from_flat(
        spec.crystal_key,
        thickness_ang=spec.scan.thickness_ang,
        tilt_deg=spec.scan.tilt_deg,
        tilt_azim_deg=spec.scan.tilt_azim_deg,
        substrate=spec.substrate,
        stack=spec.stack or None,
    )


def rename_target(document, name, new_name):
    """Rename an unsaved target and its profile references; caller owns saving."""
    from pyrite.materials._named_filters import _NAME

    rows = document.get("targets", {})
    if name not in rows:
        raise ValueError(f"unknown target preset {name!r}")
    if _NAME.fullmatch(new_name) is None:
        raise ValueError("invalid target preset name")
    if new_name.casefold() in {key.casefold() for key in rows}:
        raise ValueError(f"target preset {new_name!r} already exists")
    rows[new_name] = rows.pop(name)
    for row in document.get("profiles", {}).values():
        refs = row.get("targets", {})
        for material, target in refs.items():
            if target == name:
                refs[material] = new_name


def delete_target(document, name):
    """Remove an unreferenced unsaved target; catalog saving validates all rows."""
    rows = document.get("targets", {})
    if name not in rows:
        raise ValueError(f"unknown target preset {name!r}")
    users = [
        profile
        for profile, row in document.get("profiles", {}).items()
        if name in row.get("targets", {}).values()
    ]
    if users:
        raise ValueError(f"cannot delete referenced target preset {name!r}: {', '.join(users)}")
    del rows[name]


def target_entrance(row):
    """Return the entrance material recipe key of a raw target definition."""
    if row.get("kind") == "stack":
        layers = row.get("layers") or [{}]
        return layers[0].get("material")
    return row.get("material")


def target_reference_sites(document, name):
    """Return ``(profile, material)`` pairs referencing a target preset."""
    return [
        (profile, material)
        for profile, row in document.get("profiles", {}).items()
        for material, target in row.get("targets", {}).items()
        if target == name
    ]


def attach_target(document, profile_name, name):
    """Point a profile material at a target; returns the entrance material key."""
    import tomlkit

    from ._profile_document import existing_profile

    rows = document.get("targets", {})
    if name not in rows:
        raise ValueError(f"unknown target preset {name!r}; have {list(rows)}")
    profile = existing_profile(document, profile_name)
    material = target_entrance(rows[name])
    refs = profile.get("targets", {})
    if material in refs:
        raise ValueError(
            f"profile {profile_name!r} already maps {material!r} to target {refs[material]!r}; "
            "detach it first"
        )
    if "targets" not in profile:
        profile["targets"] = tomlkit.table()
    profile["targets"][material] = name
    return material


def detach_target(document, profile_name, material):
    """Remove a material's reference, restoring legacy profile/material geometry."""
    from ._profile_document import existing_profile

    profile = existing_profile(document, profile_name)
    refs = profile.get("targets", {})
    if material not in refs:
        raise ValueError(f"profile {profile_name!r} has no target for material {material!r}")
    name = str(refs[material])
    del refs[material]
    if not refs:
        del profile["targets"]
    return name
