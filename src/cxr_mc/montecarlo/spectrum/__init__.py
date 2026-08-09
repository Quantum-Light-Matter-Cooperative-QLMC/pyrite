# ruff: noqa: F401, I001
"""Line and bremsstrahlung spectrum implementations.

The package preserves the former :mod:`cxr_mc.montecarlo.spectrum` import
surface while filing renderer-sized implementations and CUDA kernels beside
their owning radiation path.
"""

from .brem import (
    R_E_CM2,
    _BREM_MC2_KEV,
    _USE_JIT_BREM_REDUCTION,
    _brem_dsigma_dk,
    _brem_dsigma_dk_core,
    load_external_brem,
    mc_brem_spectrum,
)
from .lines import (
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
for _function in (load_external_brem, mc_brem_spectrum, mc_spectrum, mc_spectrum_solid_angle):
    _function.__module__ = __name__

del _function
