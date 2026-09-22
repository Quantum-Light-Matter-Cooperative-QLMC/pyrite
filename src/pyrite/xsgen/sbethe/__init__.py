"""SBETHE input, output, and table-generation support."""

from .deck import OUTPUTS, PROJECTILES, SbetheDeck, material_token
from .catalog import CatalogMaterial, catalog_material, material_inputs_from_composition
from .generate import GenerationResult, generate_material, material_request
from .parse import (
    SbetheIntegratedCS,
    SbetheOscillator,
    SbetheStopping,
    parse_integrated,
    parse_oscillator,
    parse_stopping,
    table_arrays,
)

__all__ = [
    "OUTPUTS",
    "PROJECTILES",
    "GenerationResult",
    "CatalogMaterial",
    "SbetheDeck",
    "SbetheIntegratedCS",
    "SbetheOscillator",
    "SbetheStopping",
    "generate_material",
    "catalog_material",
    "material_inputs_from_composition",
    "material_request",
    "material_token",
    "parse_integrated",
    "parse_oscillator",
    "parse_stopping",
    "table_arrays",
]
