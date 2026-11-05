"""Typed adapters for validated scalar target catalog values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.materials import MaterialCatalog, MaterialSpec

    from .geometry import Slab, Stack


def target_from_preset(fields, catalog):
    """Lower recipe references through the same flat adapter used by campaigns."""
    from .flat_geometry import target_from_flat

    values = dict(fields)
    values["material"] = catalog.materials[values["material"]].crystal_key
    values["stack"] = values["stack"] or None
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
