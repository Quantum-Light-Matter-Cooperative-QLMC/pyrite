"""Material identity, crystal configuration, and X-ray material physics.

The package-level surface intentionally contains only scan-selection conveniences.
Import atomic, crystal, or attenuation details from their owning submodules.
"""

from .catalog import (
    CATALOG,
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
from .crystal import CRYSTALS

MATERIAL_LABELS = {key: material.label for key, material in CATALOG.materials.items()}
MATERIALS = CATALOG.material_keys

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
