"""BremsLib library reading, conversion, and table generation."""

from .convert import QUANTITY, angular_integral, build_table, panel_of, shape_function
from .generate import GenerationResult, element_request, generate_element
from .read import (
    COMPLETE_T1_MAX_MEV,
    DdcsPanel,
    NodeFile,
    RatioTable,
    ddcs_filename,
    iter_node_files,
    library_root,
    library_version,
    parse_ddcs,
    parse_ratio_table,
)
from .release import catalogue_elements, catalogue_table, load_release_index

__all__ = [
    "COMPLETE_T1_MAX_MEV",
    "QUANTITY",
    "DdcsPanel",
    "GenerationResult",
    "NodeFile",
    "RatioTable",
    "angular_integral",
    "build_table",
    "catalogue_elements",
    "catalogue_table",
    "ddcs_filename",
    "element_request",
    "generate_element",
    "iter_node_files",
    "library_root",
    "library_version",
    "load_release_index",
    "panel_of",
    "parse_ddcs",
    "parse_ratio_table",
    "shape_function",
]
