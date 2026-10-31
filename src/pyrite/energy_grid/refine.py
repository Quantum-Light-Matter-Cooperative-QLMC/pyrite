"""Node refinement of a photon-continuum grid near the structure it carries.

Public facade: the implementation lives in the package-root leaf
:mod:`pyrite._continuum_grid` so ``campaign`` can resolve each case's default
continuum grid without importing this driver package. See that module and
[Energy-grid semantics](../../docs/physics/radiation-physics/energy-grid-semantics.md)
for the derivation, assumptions, and limiting case.
"""

from .._continuum_grid import (
    CONTINUUM_GRID_MODEL,
    CONTINUUM_LOG_STEP,
    DETECTOR_PATH_ELEMENTS,
    EDGE_KIND,
    ENDPOINT_KIND,
    MERGE_FRACTION,
    RefinementMark,
    automatic_continuum_grid,
    case_endpoint_eV,
    continuum_refinement_marks,
    refine_continuum_nodes,
    refined_continuum_grid,
)

__all__ = [
    "CONTINUUM_GRID_MODEL",
    "CONTINUUM_LOG_STEP",
    "DETECTOR_PATH_ELEMENTS",
    "EDGE_KIND",
    "ENDPOINT_KIND",
    "MERGE_FRACTION",
    "RefinementMark",
    "automatic_continuum_grid",
    "case_endpoint_eV",
    "continuum_refinement_marks",
    "refine_continuum_nodes",
    "refined_continuum_grid",
]
