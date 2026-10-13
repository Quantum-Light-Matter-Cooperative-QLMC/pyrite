"""Homogeneous substance lookup, independent of target recipes and solvers."""

import math
from typing import Literal

from ._schema import MaterialCatalog, MediumSpec

Composition = tuple[tuple[str, float], ...]
SubstanceNamespace = Literal["medium", "crystal"]


def resolve_composition(
    material: str | MediumSpec,
    *,
    catalog: MaterialCatalog,
    namespaces: tuple[SubstanceNamespace, ...] = ("medium", "crystal"),
) -> Composition:
    """Resolve ordered number densities [atoms/Å³] from an explicit catalog.

    Namespaces are searched in the supplied order, with exact key spelling.
    Inline media bypass catalog lookup. Runnable target recipes are never
    searched: a layered recipe is not a homogeneous substance. Missing keys
    raise ``KeyError``; callers retain their own contextual errors, case rules
    and validation. Composition order and duplicate entries remain unchanged
    for persisted identity compatibility.
    """
    if isinstance(material, MediumSpec):
        return material.composition
    if not isinstance(material, str):
        raise TypeError("material must be a catalog key or MediumSpec")
    for namespace in namespaces:
        if namespace == "medium":
            registry = catalog.media
        elif namespace == "crystal":
            registry = catalog.crystals
        else:
            raise ValueError(f"unsupported substance namespace {namespace!r}")
        if material in registry:
            return registry[material].composition
    raise KeyError(material)


def has_positive_number_densities(composition: Composition) -> bool:
    """Whether a non-empty composition has named, finite positive densities."""
    return bool(composition) and all(
        element and math.isfinite(float(density)) and float(density) > 0.0
        for element, density in composition
    )
