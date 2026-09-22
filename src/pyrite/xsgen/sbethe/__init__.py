"""SBETHE input, output, and table-generation support."""

from __future__ import annotations

from .deck import OUTPUTS, PROJECTILES, SbetheDeck, material_token
from .generate import GenerationResult, generate_material
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
    "SbetheDeck",
    "SbetheIntegratedCS",
    "SbetheOscillator",
    "SbetheStopping",
    "generate_material",
    "material_token",
    "parse_integrated",
    "parse_oscillator",
    "parse_stopping",
    "table_arrays",
]
