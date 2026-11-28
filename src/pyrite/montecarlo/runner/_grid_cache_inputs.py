"""Scientific inputs and revision markers for line-grid cache keys."""

# Case fields the automatic line-grid resolution depends on. Transport is a
# deterministic function of these plus the seed, so they content-address the
# resolved grid. Deliberately NOT the whole case: the photon grids and the
# emission-side switches do not move a single electron.
_RESOLUTION_INPUT_KEYS = (
    "crystal",
    "composition",
    "hkl_list",
    "E0_keV",
    "thickness_ang",
    "crystal_width_mm",
    "crystal_height_mm",
    "beam_fwhm_mm",
    "beam_fwhm_y_mm",
    "bunch_front_tilt",
    "energy_spread_frac",
    "longitudinal_distribution",
    "transverse_distribution",
    "theta_obs_rad",
    "tilt_deg",
    "tilt_azim_deg",
    "groove_spacing_ang",
    "straggling",
    "energy_model",
    "max_dE_frac",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "secondary_threshold_eV",
    "cascade_rng_model",
    "radiative_model",
    "radiative_moment_model",
    "radiative_cutoff_eV",
    "pair_production_model",
    "positron_transport",
    "atomic_electron_deflection",
    "elastic_model",
    "transport_mode",
    "mixed_c1",
    "mixed_c2",
    "mixed_max_step_um",
    "abs_layers",
    "Ne",
    "Ne_brem",
    "seed",
    "E_cut_lines_keV",
    "E_cut_brem_keV",
    # Split copies join the measured line population (#379); absent adds nothing.
    "cone_split",
)


# Additional case fields the kinematic window seeds read: the reflection rows
# depend on orientation and mosaic spread, which the sinc spacing does not.
_WINDOW_INPUT_KEYS = (
    "beam_uvw",
    "surface_hkl",
    "azimuth_rad",
    "recip_miscut_rad",
    "mosaic_mc_fwhm_rad",
    "mosaic_mc_nodes",
    "layer_radiators",
)
# The measured stop also reads the DBR claim (``diffracted_brem._in_claim``):
# a mosaic spread drops the Bragg-window coverage (#1243). A full-fiber case
# measures its domain's roll-averaged population, not ``hkl_list`` (#1489).
_WEIGHT_INPUT_KEYS = ("B_ang2", "mosaic_fwhm_rad", "cxr_fiber")


#: Bumped whenever the same inputs would give a different measured stop; it
#: keys the speed cache of ``resonance-population`` grids only. 1: the stop
#: covers every DBR/Kossel Bragg window (#1243); 2: the key reads
#: ``mosaic_fwhm_rad``, which the coverage depends on.
MEASURED_COVERAGE_REVISION = 2


#: Bumped whenever the same inputs would seed different coherent windows; it
#: keys the speed cache of coherent cases only.
COHERENT_WINDOW_REVISION = 11
