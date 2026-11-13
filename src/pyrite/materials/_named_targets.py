"""Data-only scalar target catalog codec; typed construction belongs to campaign."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np

from ._catalog_decode import _direction, _Errors, _number, _readonly
from ._named_filters import _NAME
from ._parse import TRANSPORT_ELEMENTS, _material_elements, _parse_layer
from ._schema import CrystalSpec, MaterialSpec, MediumSpec

GEOMETRY_KEYS = frozenset(
    {
        "thickness_ang",
        "thickness_layers",
        "tilt_deg",
        "tilt_azim_deg",
        "crystal_width_mm",
        "crystal_height_mm",
        "groove_spacing_ang",
        "mosaic",
        "allow_normal_incidence",
        "cut",
        "footprint",
        "substrate",
        "stack",
    }
)
#: ``footprint = "unbounded"``: a laterally unbounded slab (``Slab(width=None,
#: height=None)``). Omission keeps the historical finite 5 mm footprint.
UNBOUNDED_FOOTPRINT = "unbounded"


def _scalar(row, key, path, errors, *, default=None, positive=False):
    value = _number(row.get(key, default))
    if value is None or (positive and value <= 0):
        errors.add(
            f"{path}.{key}", "must be a finite scalar" + (" greater than zero" if positive else "")
        )
        return None
    return value


def _geometry(row, path, errors):
    values: dict[str, Any] = {}
    for key, default in (("tilt_deg", 30.0), ("tilt_azim_deg", 0.0)):
        values[key] = _scalar(row, key, path, errors, default=default)
    for key in ("allow_normal_incidence", "mosaic"):
        value = row.get(key, False)
        if type(value) is not bool:
            errors.add(f"{path}.{key}", "must be a boolean")
        values[key] = value
    tilt, azimuth = values["tilt_deg"], values["tilt_azim_deg"]
    if tilt is not None and (
        not 0 <= tilt < 90 or (not values["allow_normal_incidence"] and np.isclose(tilt, 0))
    ):
        errors.add(
            f"{path}.tilt_deg", "must satisfy 0 < tilt < 90 (zero requires allow_normal_incidence)"
        )
    if azimuth is not None and np.isclose(azimuth, 90.0):
        errors.add(f"{path}.tilt_azim_deg", "90 degrees is unsupported")
    # Omission means the historical finite 5 mm footprint; both keys must be
    # supplied together. ``footprint = "unbounded"`` replaces both (#624).
    widths = ("crystal_width_mm" in row, "crystal_height_mm" in row)
    if "footprint" in row:
        if row["footprint"] != UNBOUNDED_FOOTPRINT:
            errors.add(f"{path}.footprint", f"must be {UNBOUNDED_FOOTPRINT!r}")
        elif any(widths):
            errors.add(path, "footprint = 'unbounded' excludes crystal_width_mm/crystal_height_mm")
        values["crystal_width_mm"] = values["crystal_height_mm"] = None
        return values
    if widths[0] != widths[1]:
        errors.add(path, "crystal_width_mm and crystal_height_mm must be supplied together")
    for key in ("crystal_width_mm", "crystal_height_mm"):
        values[key] = _scalar(row, key, path, errors, default=5.0, positive=True)
    return values


def _cut(row, path, values, errors):
    """``cut = {surface_hkl = [h, k, l]}``: the slab-normal plane (``CrystalCut``)."""
    if "cut" not in row:
        return
    cut = row["cut"]
    if not isinstance(cut, Mapping):
        errors.add(f"{path}.cut", "must be a table such as {surface_hkl = [0, 0, 1]}")
        return
    errors.keys(cut, f"{path}.cut", {"surface_hkl"})
    if "surface_hkl" not in cut:
        errors.add(f"{path}.cut", "requires surface_hkl")
        return
    surface = _direction(cut["surface_hkl"], f"{path}.cut.surface_hkl", errors)
    if surface is not None:
        values["cut"] = MappingProxyType({"surface_hkl": surface})


def _body(row, kind, path, values, crystals, media, errors):
    if kind == "slab":
        material = row.get("material")
        values["thickness_ang"] = _scalar(row, "thickness_ang", path, errors, positive=True)
        layers = ()
        if "groove_spacing_ang" in row:
            values["groove_spacing_ang"] = _scalar(
                row, "groove_spacing_ang", path, errors, positive=True
            )
    else:
        raw_layers = row.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) < 2:
            errors.add(
                f"{path}.layers", "must contain entrance film and at least one underlying layer"
            )
            return None, ()
        entrance = raw_layers[0]
        if not isinstance(entrance, Mapping):
            errors.add(f"{path}.layers[0]", "must be a table")
            return None, ()
        errors.keys(entrance, f"{path}.layers[0]", {"material", "thickness_ang"})
        material = entrance.get("material")
        values["thickness_ang"] = _scalar(
            entrance, "thickness_ang", f"{path}.layers[0]", errors, positive=True
        )
        layers = tuple(
            layer
            for index, entry in enumerate(raw_layers[1:], 1)
            if (layer := _parse_layer(entry, f"{path}.layers[{index}]", crystals, media, errors))
            is not None
        )
    if "groove_spacing_ang" in values and (
        values["tilt_azim_deg"] is None or not np.isclose(values["tilt_azim_deg"], 180.0)
    ):
        errors.add(path, "blazed grooves require tilt_azim_deg == 180")
    if "groove_spacing_ang" in values and values["tilt_deg"] == 0:
        errors.add(path, "blazed grooves require tilt_deg > 0")
    return material, layers


def _definition(name, row, materials, crystals, media, errors):
    path = f"targets.{name}"
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        errors.add(path, "invalid target preset name")
        return None
    if not isinstance(row, Mapping):
        errors.add(path, "must be a table")
        return None
    kind = row.get("kind")
    if kind not in ("slab", "stack"):
        errors.add(f"{path}.kind", "must be 'slab' or 'stack'")
        return None
    allowed = {
        "kind",
        "description",
        "tilt_deg",
        "tilt_azim_deg",
        "crystal_width_mm",
        "crystal_height_mm",
        "allow_normal_incidence",
        "mosaic",
        "footprint",
    }
    errors.keys(
        row,
        path,
        allowed
        | (
            {"material", "thickness_ang", "groove_spacing_ang", "cut"}
            if kind == "slab"
            else {"layers"}
        ),
    )
    if "description" in row and (
        not isinstance(row["description"], str) or not row["description"].strip()
    ):
        errors.add(f"{path}.description", "must be a nonempty string")
    before = len(errors.items)
    values = _geometry(row, path, errors)
    _cut(row, path, values, errors)
    material, layers = _body(row, kind, path, values, crystals, media, errors)
    if not isinstance(material, str) or material not in materials:
        errors.add(f"{path}.material", "entrance must reference a runnable material recipe")
        return None
    spec = materials[material]
    # Recipes already declaring layers must not silently acquire a different body.
    if spec.substrate is not None or spec.stack:
        errors.add(
            path, "entrance recipe already declares substrate/stack; use a bare entrance recipe"
        )
    unsupported = _material_elements(
        replace(spec, substrate=None, stack=layers), crystals, media
    ) - set(TRANSPORT_ELEMENTS)
    if unsupported:
        errors.add(
            path, f"runnable composition has unsupported transport elements {sorted(unsupported)}"
        )
    if len(errors.items) != before:
        return None
    return MappingProxyType({"material": material, **values, "stack": layers})


def _references(profiles, definitions, materials, errors):
    references = {}
    for profile, row in profiles.items():
        refs = row.get("targets", {})
        path = f"profiles.{profile}.targets"
        if not isinstance(refs, Mapping):
            errors.add(path, "must be a table mapping material keys to target names")
            continue
        valid = {}
        for material, name in refs.items():
            target = definitions.get(name) if isinstance(name, str) else None
            if target is None:
                errors.add(f"{path}.{material}", f"unknown target preset {name!r}")
                continue
            if material not in materials or target["material"] != material:
                errors.add(f"{path}.{material}", "must match the target's entrance material recipe")
                continue
            if "materials" in row and material not in row["materials"]:
                errors.add(f"{path}.{material}", "material is outside profile membership")
            overrides = row.get("overrides", {}).get(material, {})
            conflicts = GEOMETRY_KEYS.intersection(overrides)
            if conflicts:
                errors.add(
                    f"{path}.{material}",
                    f"target reference conflicts with geometry overrides: {sorted(conflicts)}",
                )
            valid[material] = name
        references[profile] = MappingProxyType(valid)
    return MappingProxyType(references)


def resolve_named_targets(
    raw: object,
    profiles: Mapping[str, Mapping[str, object]],
    materials: dict[str, MaterialSpec],
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
    errors: _Errors,
):
    """Validate every definition/reference and replace only resolved target values."""
    if errors.items:
        return MappingProxyType({}), materials, MappingProxyType({})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        errors.add("targets", "must be a table")
        raw = {}
    definitions = {}
    folded = set()
    for name, row in raw.items():
        if isinstance(name, str) and name.casefold() in folded:
            errors.add(f"targets.{name}", "target names must differ by more than case")
            continue
        if isinstance(name, str):
            folded.add(name.casefold())
        target = _definition(name, row, materials, crystals, media, errors)
        if target is not None:
            definitions[name] = target
    refs = _references(profiles, definitions, materials, errors)
    resolved = dict(materials)
    for key, spec in materials.items():
        name = refs.get(spec.profile, {}).get(key)
        if name is None:
            continue
        target = definitions[name]
        scan = replace(
            spec.scan,
            thickness_layers=None,
            **{
                field: _readonly([target[field]])
                for field in ("thickness_ang", "tilt_deg", "tilt_azim_deg")
            },
        )
        resolved[key] = replace(
            spec, scan=scan, substrate=None, stack=target["stack"], target_fields=target
        )
    return MappingProxyType(definitions), resolved, refs
