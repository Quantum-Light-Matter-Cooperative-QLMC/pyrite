"""Material identity, crystal configuration, and X-ray material physics.

The package-level surface contains registry conveniences and explicit authoring
operations. Import atomic, crystal, or attenuation details from their owners.
"""

from typing import TYPE_CHECKING

from ._identity import MaterialIdentity
from .catalog import (
    CrystalInfo,
    CrystalSpec,
    LayerSpec,
    MaterialCatalog,
    MaterialConfigError,
    MaterialSpec,
    MaterialValidationSpec,
    MediumSpec,
    ScanSpec,
    load_material_catalog,
)

if TYPE_CHECKING:
    from .authoring import (
        MaterialDefinition as MaterialDefinition,
    )
    from .authoring import (
        build_crystal as build_crystal,
    )
    from .authoring import (
        build_medium as build_medium,
    )
    from .authoring import (
        save_material as save_material,
    )

CATALOG: MaterialCatalog
CRYSTALS: dict[str, dict[str, object]]
MATERIAL_LABELS: dict[str, str]
MATERIALS: tuple[str, ...]


def __getattr__(name: str):
    """Lazily construct catalog-backed package conveniences."""
    if name == "CATALOG":
        from .catalog import _get_default_catalog

        value = _get_default_catalog()
    elif name == "CRYSTALS":
        from .crystal import CRYSTALS as value
    elif name == "MATERIAL_LABELS":
        catalog = __getattr__("CATALOG")
        value = {key: material.label for key, material in catalog.materials.items()}
    elif name == "MATERIALS":
        value = __getattr__("CATALOG").material_keys
    elif name in {"MaterialDefinition", "build_medium", "build_crystal", "save_material"}:
        from . import authoring

        value = getattr(authoring, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "MaterialDefinition",
    "build_medium",
    "build_crystal",
    "save_material",
    "CRYSTALS",
    "CATALOG",
    "MaterialCatalog",
    "CrystalInfo",
    "CrystalSpec",
    "MediumSpec",
    "MaterialIdentity",
    "MaterialSpec",
    "MaterialValidationSpec",
    "ScanSpec",
    "LayerSpec",
    "MaterialConfigError",
    "load_material_catalog",
    "MATERIAL_LABELS",
    "MATERIALS",
]
