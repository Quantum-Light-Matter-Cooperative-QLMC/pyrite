"""Material identity, crystal configuration, and X-ray material physics.

The package-level surface intentionally contains only scan-selection conveniences.
Import atomic, crystal, or attenuation details from their owning submodules.
"""

from .catalog import (
    CrystalInfo,
    CrystalSpec,
    LayerSpec,
    MaterialCatalog,
    MaterialConfigError,
    MaterialSpec,
    MediumSpec,
    ScanSpec,
    load_material_catalog,
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
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

__all__ = [
    "CRYSTALS",
    "CATALOG",
    "MaterialCatalog",
    "CrystalInfo",
    "CrystalSpec",
    "MediumSpec",
    "MaterialSpec",
    "ScanSpec",
    "LayerSpec",
    "MaterialConfigError",
    "load_material_catalog",
    "MATERIAL_LABELS",
    "MATERIALS",
]
