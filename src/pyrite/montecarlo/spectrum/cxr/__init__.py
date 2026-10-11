# ruff: noqa: F401, I001
"""
montecarlo.spectrum.cxr

Radiation from the transported segments (Zhai SI S1): the finite-interaction-
time CXR (PXR + CBS) CXR spectrum ``mc_cxr_spectrum`` and its solid-angle-
integrated wrapper. The array-heavy inner loops run on the GPU backend (``xp``)
when available.

The implementation is split by phase, bottom-up:

``_policy``    the five device-kernel dispatch switches
``_kernels``   leaf array numerics: lineshape, interpolation, segment geometry
``_bin_quadrature``  closed-form sinc^2 bin masses (``line_quadrature="bin-mean"``)
``_setup``     ``SpectrumRequest`` and the one-shot ``_prepare_spectrum``
``_per_hkl``   accumulation one reflection at a time (the reference route)
``_batched``   accumulation over stacked reflection tables (the default route)
``_spectrum``  ``mc_cxr_spectrum``, route dispatch, finalization
``_temporal``  opt-in temporal intensity profile ``I(t)`` (per-hkl route)

This module re-exports the pre-split surface, so
``pyrite.montecarlo.spectrum.cxr.<name>`` still resolves for every consumer.
Note that re-exported names are bindings, not seams: a test that needs to
*replace* one patches the owning submodule (``_policy`` for the dispatch
switches, ``_kernels`` for ``chi_g``, ``_setup`` for ``xp``), because the
routes read them there.
"""

from ...._backend import REAL, _to_cpu, xp
from ....materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV, U_g, chi_g
from ._batched import (
    _accumulate_batched,
    _batched_block,
    _batched_coherent_block,
    _batched_coherent_finalize,
    _batched_incoherent_block,
    _batched_reflection_tables,
    _batched_tables,
    _BatchedBlock,
    _BatchedTables,
    _CxrBatch,
)
from ._kernels import (
    _INTERP_GATHER_CXR_TABLES_F32,
    _PREF_C1,
    _RESONANCE_ROOT_RTOL,
    _SEG_ARRAYS,
    _clip_segments_to_cutoff,
    _dot3_core,
    _elemental_log_mu_table,
    _escape_length,
    _flight_blocks,
    _in_medium_kinematics,
    _interp1,
    _interp_elemental_mu,
    _interp_gather1d,
    _interp_gather2d,
    _interp_gather_cxr_tables,
    _interp_index,
    _cxr_amp_sq_core,
    _cxr_kin_core,
    _cxr_emission_ceiling,
    _cxr_table_nan_ceiling,
    _cxr_table_nan_onsets,
    _cxr_tabulation_grid,
    _cxr_weight_core,
    _log_interp_fraction,
    _matvec3,
    _observation_direction,
    _polarization_pair,
    _reflection_tables,
    _rowdot3,
    _segment_escape_distance,
    _segments_in_layer,
    _segments_on_device,
    _sincsq_lineshape,
    _spliced_stopping_magnitude_xp,
    _validate_groove_escape_direction,
)
from ._per_hkl import (
    _accumulate_per_hkl,
    _accumulate_reflection,
    _accumulate_reflection_coherent,
    _coherent_electron_grouped_row,
    _coherent_jit_grouped_row,
    _row_decoherence_factor,
)
from ._policy import (
    _JIT_COHERENT_PAIR_TARGET,
    _JIT_CXR_BATCH_TARGET,
    _USE_JIT_COHERENT_REDUCTION,
    _USE_JIT_COHERENT_STREAM,
    _USE_JIT_CXR_REDUCTION,
)
from ._setup import CXR_ESCAPE_MODEL, SpectrumRequest, _prepare_spectrum, _SpectrumSetup
from ._temporal import TemporalProfile, temporal_profile_for
from ._spectrum import (
    _finalize_spectrum,
    _mc_spectrum,
    _needs_per_hkl_route,
    mc_cxr_spectrum,
    mc_cxr_spectrum_solid_angle,
)
