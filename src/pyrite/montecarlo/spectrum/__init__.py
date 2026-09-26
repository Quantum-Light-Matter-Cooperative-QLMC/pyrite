# ruff: noqa: F401, I001
"""Line and bremsstrahlung spectrum implementations.

The package preserves the former :mod:`pyrite.montecarlo.spectrum` import
surface while filing renderer-sized implementations and CUDA kernels beside
their owning radiation path.
"""

from .brem import (
    BREM_ENDF_PARSERPY_VERSION,
    BREMSSTRAHLUNG_DATA_DIR,
    BREMSSTRAHLUNG_EEDL_FILENAME,
    BREMSSTRAHLUNG_EEDL_SHA256,
    BREMSSTRAHLUNG_MODEL,
    R_E_CM2,
    BremsstrahlungCrossSectionTable,
    EEDLBremsstrahlungDataUnavailable,
    _BREM_MC2_KEV,
    _USE_JIT_BREM_REDUCTION,
    _brem_dsigma_dk,
    _brem_dsigma_dk_core,
    _bremsstrahlung_dsigma_dk,
    _eedl_brem_dsigma_dk,
    load_bremsstrahlung_cross_sections,
    load_external_brem,
    mc_brem_spectrum,
)
from .characteristic import (
    CHARACTERISTIC_DATA_DIR,
    CHARACTERISTIC_EEDL_FILENAME,
    CHARACTERISTIC_EEDL_SHA256,
    CHARACTERISTIC_MODEL,
    CHARACTERISTIC_XRAYDB_VERSION,
    CharacteristicCrossSectionTable,
    load_characteristic_cross_sections,
    mc_characteristic_spectrum,
)
from .diagnostics import (
    DEFAULT_BREM_QUADRATURE_WARN,
    DEFAULT_RESONANCE_DRIFT_WARN,
    brem_endpoint_quadrature_error,
    cxr_endpoint_resonance_drift,
    subdivide_flights,
)
from .lines import (
    LINE_ESCAPE_MODEL,
    REAL,
    _INTERP_GATHER_LINE_TABLES_F32,
    _JIT_COHERENT_PAIR_TARGET,
    _JIT_LINE_BATCH_TARGET,
    _PREF_C1,
    _SEG_ARRAYS,
    _USE_JIT_COHERENT_REDUCTION,
    _USE_JIT_COHERENT_STREAM,
    _USE_JIT_LINE_REDUCTION,
    _clip_segments_to_cutoff,
    _dot3_core,
    _escape_length,
    _interp1,
    _interp_gather1d,
    _interp_gather2d,
    _interp_gather_line_tables,
    _interp_index,
    _line_amp_sq_core,
    _line_kin_core,
    _line_weight_core,
    _matvec3,
    _observation_direction,
    _polarization_pair,
    _rowdot3,
    _segment_escape_distance,
    _segments_in_layer,
    _segments_on_device,
    _sincsq_lineshape,
    _validate_groove_escape_direction,
    mc_spectrum,
    mc_spectrum_solid_angle,
    xp,
)

# Preserve the pre-split owner reported by introspection and frozen export
# guards. The package path itself is unchanged; only implementation files move.
for _function in (
    brem_endpoint_quadrature_error,
    cxr_endpoint_resonance_drift,
    load_external_brem,
    mc_brem_spectrum,
    mc_characteristic_spectrum,
    mc_spectrum,
    mc_spectrum_solid_angle,
    subdivide_flights,
):
    _function.__module__ = __name__

del _function
