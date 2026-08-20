# ruff: noqa: F401, I001
"""
montecarlo.transport

Single-scattering electron transport (Zhai SI S2): element data, elastic
scattering models (Browning free paths + NIST-Mott-calibrated screened-
Rutherford angles, analytic SR fallback), Joy-Luo/Berger-Seltzer stopping
power, Urban energy-loss straggling, and Numba-compiled event-driven
transport cores. CPU-only; the spectrum phase consumes the segment arrays
:func:`simulate_trajectories` returns.

Split into submodules by concern (kinematics/RNG streams, elastic
scattering, stopping power, straggling, energy LUT, transport cores,
batching/dispatch, public API); this package re-exports every one of their
names so ``pyrite.montecarlo.transport.<name>`` (and
``from .transport import <name>``) keeps resolving exactly as it did when
this was one file. See docs/repo_map.md for the submodule breakdown.
"""

from .kinematics import (
    C_ANG_PER_FS,
    _sample_bunch_offsets,
    _beta_array,
    beta_from_keV,
    beta_from_keV_scalar,
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
    _SM64_S11,
    _SM64_ZERO,
    _SM64_ONE,
    _U53_SCALE,
    _splitmix64,
    _stream_key_scalar,
    _stream_uniform_scalar,
    stream_keys,
)

from .scattering import (
    MOTT_DIR,
    A0_SQ_CM2,
    _sigma_browning_cm2,
    _sigma_browning_cm2_scalar,
    _alpha_sr_joy,
    _alpha_sr_joy_scalar,
    _alpha_sr_joy_numba,
    _alpha_from_first_moment,
    _load_mott_transport,
    _mott_alpha_table,
    _NO_MOTT,
    _scatter_rates_mott_scalar,
    _scatter_rates_sr_scalar,
    _sample_cos_theta_sr_numba,
    _sample_cos_theta_from_alpha,
    _interp_mott_log_alpha_scalar,
    _sample_cos_theta_joy_scalar,
    _sample_cos_theta_mott_scalar,
    _sample_cos_theta,
)

from .stopping import (
    TRANSPORT_ELEMENTS,
    _dEds_keV_per_ang,
    _dEds_compound_scalar,
    _dEds_compound,
    _BS_PREFACTOR,
    _MC2_KEV,
    _LN2,
    _bs_tau_terms,
    _dEds_bs_keV_per_ang,
    _dEds_bs_compound_scalar,
    _dEds_bs_packed_scalar,
    _dEds_bs_compound,
    _bs_joy_luo_crossover_keV,
    _dEds_spliced_compound_scalar,
    _dEds_spliced_compound,
    _dEds_spliced_packed_scalar,
    _CROSSOVER_CACHE,
    _element_crossover_keV,
    spliced_stopping_keV_per_ang,
    sternheimer_delta,
    STOPPING_MODEL,
)

from .straggling import (
    _URBAN_STREAM_SALT,
    _urban_stream_key_scalar,
    _urban_flight_key_scalar,
    _URBAN_E0_KEV,
    _URBAN_E2_KEV_PER_Z2,
    _URBAN_RATE,
    _URBAN_POISSON_GAUSS_MIN,
    _urban_levels_scalar,
    _urban_channels_scalar,
    _urban_moments_element_scalar,
    _urban_poisson_scalar,
    _urban_ionisation_keV,
    _urban_sample_element_keV,
    _dEds_spliced_element_scalar,
    _urban_sample_compound_keV,
    urban_element_table,
    urban_loss_moments_keV,
)

from .lut import (
    TransportLUTConfig,
    DEFAULT_TRANSPORT_LUT_CONFIG,
    TransportEnergyLUT,
    _lut_index_frac_scalar,
    _lut_lerp_1d,
    _lut_lerp_2d,
    _lut_lerp_3d,
    build_transport_energy_lut,
)

from .cores import (
    _rotate_direction_scalar,
    _rotate_directions,
    _first_prism_exit_scalar,
    _transport_core_ungrooved,
    _transport_core_ungrooved_lut,
    _transport_core_grooved,
    EXIT_CUTOFF_STOPPED,
    EXIT_BACKSCATTERED,
    EXIT_TRANSMITTED,
    EXIT_SIDE,
    EXIT_STEP_LIMITED,
    EXIT_NOT_ENTERED,
    _searchsorted_right_scalar,
    _transport_core_ungrooved_perelectron,
    _transport_core_ungrooved_perelectron_lut,
)

from .batching import (
    pack_layer_tables,
    _percentile_summary,
    _flight_diagnostic_summary,
    PerElectronTransportConfig,
    DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    TRANSPORT_CORES,
    CUDA_TRANSPORT_MIN_ELECTRONS,
    _cuda_transport_available,
    resolve_transport_core,
    _SEG_SCRATCH_BYTES,
    _batch_size,
    _capacity_for,
    _batch_electrons,
    _run_per_electron_transport_lut,
    _run_per_electron_transport,
    _alloc_scratch,
)

from .api import (
    simulate_trajectories,
)

# Preserve the pre-split owner reported by introspection and frozen export
# guards. The package path itself is unchanged; only implementation files move.
for _function in (
    beta_from_keV,
    stream_keys,
    spliced_stopping_keV_per_ang,
    sternheimer_delta,
    urban_element_table,
    urban_loss_moments_keV,
    TransportLUTConfig,
    TransportEnergyLUT,
    build_transport_energy_lut,
    pack_layer_tables,
    PerElectronTransportConfig,
    resolve_transport_core,
    simulate_trajectories,
):
    _function.__module__ = __name__

del _function
