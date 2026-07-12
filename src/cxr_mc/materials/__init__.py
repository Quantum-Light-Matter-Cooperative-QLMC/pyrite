"""Material identity, crystal configuration, and X-ray material physics.

The package-level surface intentionally contains only scan-selection conveniences.
Import atomic, crystal, or attenuation details from their owning submodules.
"""

from .crystal import CRYSTALS
from .registry import (
    CRYSTAL_PARAMS,
    MATERIAL_CONFIGS,
    MATERIAL_GRIDS,
    MATERIAL_LABELS,
    MATERIALS,
    CrystalParamsGrid,
    Layer,
    MaterialConfig,
    MaterialGrid,
    ScalarOrSeq,
    crystal_config,
    material_crystal_key,
    material_scan_grid,
)

__all__ = [
    "CRYSTALS",
    "MATERIAL_CONFIGS",
    "MATERIAL_GRIDS",
    "CRYSTAL_PARAMS",
    "MATERIAL_LABELS",
    "MATERIALS",
    "Layer",
    "ScalarOrSeq",
    "MaterialConfig",
    "MaterialGrid",
    "CrystalParamsGrid",
    "material_crystal_key",
    "material_scan_grid",
    "crystal_config",
]
