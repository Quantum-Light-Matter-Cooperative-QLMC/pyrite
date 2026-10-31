"""Public transport entry point: :func:`simulate_trajectories`.

Validation: sbethe-corrected-stopping
"""

import logging

from .arguments import TransportInputs
from .batching import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
from .beam_entry import check_electron_block
from .host_pass import run_transport_pass
from .lut import DEFAULT_TRANSPORT_LUT_CONFIG

logger = logging.getLogger(__name__)


def simulate_trajectories(
    E0_keV,
    Ne,
    thickness_ang,
    element=None,
    n_atoms_per_ang3=None,
    E_cut_keV=5.0,
    seed=0,
    max_steps=20000,
    elastic_model="elsepa",
    beam_dir=None,
    composition=None,
    layers=None,
    beam_fwhm_mm=None,
    beam_fwhm_y_mm=None,
    bunch_length_fs=None,
    long_shape="gaussian",
    long_offsets_fs=None,
    longitudinal_distribution=None,
    transverse_distribution=None,
    energy_spread_frac=None,
    crystal_width_mm=None,
    crystal_height_mm=None,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    groove=None,
    E_cut_by_electrons=None,
    transport_core="auto",
    per_electron_config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_segments_on_device=False,
    transport_lut_config=DEFAULT_TRANSPORT_LUT_CONFIG,
    collect_diagnostics=False,
    energy_model="midpoint",
    max_dE_frac=0.0,
    straggling=False,
    stopping_tables=None,
    inelastic_model="continuous",
    inelastic_cutoff_eV=None,
    inelastic_materials=None,
    elastic_tables=None,
    radiative_model="auto",
    radiative_cutoff_eV=None,
    bremslib_tables=None,
    gdf_source=None,
    secondary_threshold_eV=None,
    max_secondary_generations=64,
    max_secondary_tracks=None,
    *,
    radiative_moments="table",
    pair_production_model=None,
    positron_transport=False,
    atomic_electron_deflection=None,
    transport_precision="fp64",
    transport_progress=None,
    _secondary=None,
    _electron_start=0,
    _energy_range_keV=None,
    bunch_front_tilt=None,
    _complete_hard_photons=True,
):
    """
    Transport Ne electrons of energy E0_keV [keV] into a slab 0<=z<=thickness.
    Beam enters at the origin along +z. Electrons terminate when they exit
    either surface or drop below E_cut_keV (segments below the cutoff don't
    radiate in the spectral window of interest anyway).

    Full derivations, sources, and limiting-case checks for the physics below
    live in ``docs/physics/beam-transport/*.md``; validation status is recorded
    in ``docs/validation/``. This docstring states the parameter contract and BIT-FOR-BIT
    limiting case each one must preserve.

    elastic_model: "elsepa" (default; tabulated ELSEPA total cross sections
      and full angular distributions from ``elastic_tables``, resolved from
      the stored production tables when not passed; exact cores only, the
      transport LUT is bypassed), "sr" (analytic screened-Rutherford, no
      data files), or "mott" (deprecated; NIST SRD 64 Mott transport cross
      section via a Browning fit, read from the user-supplied
      ``mott.tables_dir``). See
      docs/physics/beam-transport/elastic-scattering.md.
      Validation: electron-transport
      Validation: elsepa-elastic-sampling

    elastic_tables: only accepted with ``elastic_model="elsepa"``, which
      resolves the stored production tables when omitted. One entry per layer, each a sequence of one mapping per
      element in composition order holding ``energy_eV``,
      ``total_elastic_cm2``, ``mu`` and ``dcs_cm2_sr`` on ELSEPA's shared
      angular grid. Every incident energy and cutoff must lie inside every
      table; extrapolation is rejected.

    beam_dir: initial electron direction in the SLAB frame (default +z,
    i.e. normal incidence). For a tilted sample use tilted_geometry().

    composition: for COMPOUNDS, [(element, number_density_1_per_Ang3), ...]
    overriding element/n_atoms_per_ang3. Free paths and stopping are additive
    over elements (Bragg's rule); the scattering element at each collision is
    chosen with probability n_i sigma_i / sum. See
    docs/physics/beam-transport/electron-transport.md.
    Validation: electron-transport

    layers: optional film-on-substrate stack
    [(z_top, z_bot, composition), ...] (top/entrance first, contiguous, deepest
    z_bot = total thickness). Each electron's free path/stopping/scattering
    element switches by the layer it is currently in; a flight is truncated at
    an internal boundary with no collision (the electron continues into the
    neighbor). None -> a single layer over [0, thickness_ang] (the old
    single-material transport, BIT-FOR-BIT). When given, thickness_ang is
    superseded by the stack's total thickness. See
    docs/validation/materials/multilayer-stack.md.
    Validation: multilayer-stack

    beam_fwhm_mm: transverse size of the incident electron beam -- an
    azimuthally-symmetric Gaussian spot of the given FULL WIDTH AT HALF
    MAXIMUM [mm] (same FWHM convention as mosaic_fwhm_rad / eds_fwhm_eV /
    aperture_fwhm_eV elsewhere), drawn per electron from an RNG stream
    independent of `seed`'s main stream and projected onto the (possibly
    tilted) sample entrance face via geometry.project_beam_entry. None
    (default) is a strict no-op -- the old point-source beam, BIT-FOR-BIT.
    Limiting case: beam_fwhm_mm -> 0 recovers the point source exactly. When
    both crystal_width_mm and crystal_height_mm are None, enabling it never
    perturbs the free-path/scattering-angle draws or any returned array
    except r_mid.

    NOT safe with mc_cxr_spectrum(coherent=True): the coherent phase reads r_mid
    directly, so the per-electron offset enters every cross-electron term and
    the transverse form factor that should average it away is unimplemented
    (`transverse-bunch-form-factor` discrepancy) -- see
    docs/physics/radiation-physics/coherent-emission.md before using a finite
    spot with emission="coherent"/"both". See
    docs/physics/beam-transport/beam-phase-space.md and
    docs/validation/geometry/finite-beam-size.md.
    Validation: finite-beam-size

    beam_fwhm_y_mm: y-plane spot FWHM [mm]; None preserves the isotropic path.

    transverse_distribution: Courant-Snyder entrance positions and directions,
    mutually exclusive with spot FWHMs. Unset preserves the shared beam_dir;
    eps_n -> 0 recovers the collimated point source. See
    docs/physics/beam-transport/beam-phase-space.md.
    Validation: beam-phase-space-injection

    energy_spread_frac: RMS relative spread; starts at E0_keV * (1 + f * u),
    u ~ Normal(0, 1), independent of arrival time. None/0 preserves the
    monoenergetic beam; non-positive drawn energies raise. See
    docs/physics/beam-transport/beam-phase-space.md.
    Validation: beam-energy-spread-injection

    Transverse, energy, and arrival sampling use independent child RNG streams.

    Longitudinal controls sample per-electron offsets via
    :func:`_sample_bunch_offsets`, returned as t0_ang separately from relative
    t_ang/clock. Absent controls preserve all-zero offsets. See
    docs/physics/beam-transport/longitudinal-structure.md.
    Validation: longitudinal-bunch-sampling

    Paired crystal dimensions define a rectangular prism centered at the beam
    origin. Missed entries count in n_missed (Ne unchanged); side exits count
    in n_side_exited. All-None preserves the infinite-slab path. See
    docs/validation/geometry/finite-transverse-crystal.md.
    Validation: finite-transverse-crystal

    Target tilt (Zhai convention) projects the spot onto the entrance face;
    zero angles preserve the identity projection. See
    docs/validation/geometry/grazing-beam-projection.md.
    Validation: grazing-beam-projection

    groove: optional :class:`~pyrite.montecarlo.groove.GrooveSpec` describing
    a blazed sawtooth material/vacuum boundary on the beam-entrance face.
    Initial rays enter through relief facets via entry_points; every later
    free flight is intersected with both periodic facet families and
    truncated at a material-to-vacuum crossing. A vacuum-side crossing
    advances the electron to its next re-entry point with no scattering,
    stopping, or radiation, then resumes material transport (resampling the
    elastic free path after re-entry is exact -- the collision-distance law
    is memoryless). Vacuum flights are returned as separate vacuum_*
    diagnostic arrays and never enter the material segment sum; a ray with no
    later re-entry is a permanent entrance-face exit. None is a strict
    no-op -- BIT-FOR-BIT identical to the ungrooved slab. See
    docs/validation/geometry/blazed-groove-geometry.md.
    Validation: blazed-groove-geometry

    stopping_tables: one stored SBETHE collision-stopping table per material
      layer, with native ``stopping_energy_eV`` and
      ``stopping_eV_per_angstrom`` arrays. Production case runners supply
      identity-matched tables. ``None`` keeps the Joy-Luo/Berger-Seltzer
      reference path for direct transport comparisons. Table energies must
      include every incident energy and cutoff; extrapolation is rejected.
      When supplied, the prepared tables are also carried on the returned
      segment mapping as ``stopping_tables`` so post-transport cutoff solves
      use the same model.
      Validation: sbethe-corrected-stopping

    inelastic_model: collision energy-loss scheme.
      "continuous" (default) -- the whole corrected stopping is a continuous
          loss (plus optional Urban straggling). BIT-FOR-BIT unchanged.
      "shell-soft-hard" -- opt-in PENELOPE-like mixed scheme: losses
          W <= inelastic_cutoff_eV stay continuous, larger ones are discrete
          hard events. Needs ``stopping_tables``, midpoint, and one catalog
          key per layer in ``inelastic_materials``; no CUDA LUT core. Adds
          ``hard_W_keV``/``hard_channel``/``inelastic`` to the result. See
          docs/physics/beam-transport/shell-soft-hard-transport.md.
          Validation: shell-soft-hard-transport

    atomic_electron_deflection: None (default, BIT-FOR-BIT) or "kawrakow" (Numerics
      default): elastic rate Z^2 -> Z(Z + xi). Validation: inelastic-angular-deflection

    transport_precision: "fp64" (default, BIT-FOR-BIT) or "mixed" (#838,
      opt-in): the exact CUDA kernel reads its energy-grid, ELSEPA, SBETHE,
      Mott and BremsLib tables as float32 and evaluates their lookups,
      interpolation, angle and photon sampling in FP32, while position,
      direction, energy, path, clock, optical depths and every rate sum stay
      FP64. Needs ``transport_core="cuda"`` and the exact core (no transport
      LUT); results change at rounding scale, so it is case identity. See
      ``_jit_precision``. Validation: gpu-mixed-precision-tables,
      gpu-mixed-precision-math

    radiative_moments: "table" (default) uses prepared float64 log-energy
      interpolation for frequent soft/rate totals; "exact" retains quadrature.
      Exact event totals and conditional photon sampling are unchanged.
      Validation: radiative-moment-interpolation (ensemble qualification incomplete).
    radiative_model: "auto" (default) couples when BremsLib tables are supplied,
      otherwise keeps post-hoc brem scoring and the existing tracks.
      "bremslib-soft-hard" (exact cores only, no LUT) adds soft BremsLib loss
      below ``radiative_cutoff_eV`` and explicit photons above it; see
      ``hard_radiative.validate_radiative_args`` and ``spectrum.brem_events``.
      Validation: bremslib-radiative-partition, bremslib-radiative-event-spectrum.

    secondary_threshold_eV: opt-in transport of hard-collision secondaries
      (shell-soft-hard only); one threshold [eV] is production cut and
      tracking cutoff. Generations are capped by ``max_secondary_generations``
      and ``max_secondary_tracks`` (default ``1000 * Ne``), which raise. Rows
      gain ``track_id``/``parent_id``/``generation``; ``electron_id`` stays the
      primary history. None (default) is BIT-FOR-BIT primary-only transport.
      ``pair_production_model`` adds pair conversion (``pair_production.py``);
      ``positron_transport=True`` with it also transports the positrons.
      Validation: shell-secondary-transport, photon-pair-first-interaction, bhabha-close

    transport_core: which ungrooved core runs the electrons. "auto" (default) --
    the CUDA core when this process has a CUDA device, the run is ungrooved,
    and Ne > CUDA_TRANSPORT_MIN_ELECTRONS; the lockstep core otherwise (see
    `resolve_transport_core`; pin with `PYRITE_MC_TRANSPORT_CORE`). "lockstep"
    -- the historical core, one shared Generator in step-major order,
    BIT-FOR-BIT unchanged. "per-electron" -- each electron on its own
    counter-addressed stream. "cuda" -- one CUDA thread per electron. The
    cores are NOT bit-for-bit with each other: they consume
    differently-ordered (and, for CUDA, differently-rounded) random streams,
    so each realizes a different sample of the SAME distribution --
    identical models, draw semantics, and aggregate agreement across seeds.
    Grooved transport always stays on the lockstep core. See
    docs/physics/beam-transport/electron-transport.md.
    Validation: gpu-transport-core

    per_electron_config: batching policy for the two new cores
    (:class:`PerElectronTransportConfig`). Segment capacity and scratch
    budget only bound memory and replay behavior; neither changes results.

    transport_progress: optional ``callable(electrons_done, Ne)`` per completed
    per-electron/CUDA batch, excluding capacity replays; lockstep and grooved
    cores report once, on completion.
    Each cascade generation reports its own ``Ne``. Host counters only: no extra
    device sync or draws; results stay BIT-FOR-BIT identical. None reports nothing.

    energy_model: how a physical flight's energy and clock advance along it.
      "frozen" -- the historical left-endpoint rule: stopping power
          and beta evaluated once at the flight's start energy and held
          constant over its whole length. BIT-FOR-BIT unchanged.
      "midpoint" -- explicit midpoint RK2:
          E_pred = E_start + (dE/ds)(E_start)*s, then
          E_end = E_start + (dE/ds)((E_start+E_pred)/2)*s, with
          the clock advanced by s/beta at that representative energy and the
          cutoff truncation distance solved for E_end == E_cut. Adds
          E_end_keV, t_end_ang, E_repr_keV = (E_start+E_end)/2 (the
          energy radiation kernels evaluate the row at), and event_kind --
          the row-end event code of `transport.events` -- to the returned rows.
          Elastic hazard stays frozen at the start energy within each row;
          `max_dE_frac` substeps re-evaluate it at the next row's start. The
          lockstep, per-electron, grooved, and CUDA cores all implement the
          midpoint schema; the unsupported CUDA-LUT/shell combination fails
          closed separately.

    See docs/validation/beam-transport/transport-midpoint-stopping.md.
    Validation: transport-midpoint-stopping

    max_dE_frac: numerical cap on one row's left-endpoint predicted mean loss
      fraction, ``|dE/ds|(E_start)*s/E_start``. The realized midpoint loss may
      be slightly larger, and an Urban draw is not bounded by this control.
      A binding cap splits a physical flight before any physical event. 0.0
      (default) disables substepping, leaving one row per flight.
      Requires energy_model="midpoint". The collision is drawn once per
      physical flight as an optical depth and consumed across its substeps
      at each substep's own hazard, so refining the cap never resamples the
      collision. Rows carry flight_id/substep_id; a substep keeps the
      flight's direction and identity and never scatters. See
      docs/validation/beam-transport/energy-controlled-propagation.md.

    Validation: transport-midpoint-stopping, energy-controlled-propagation

    straggling: sample the per-flight Urban energy-loss fluctuation.
      False (default) is BIT-FOR-BIT with every run before this parameter
      existed: the sampler is skipped entirely on every core, touching neither
      its RNG stream nor any output array. True samples the per-flight/-substep
      Urban compound loss on a counter-addressed stream disjoint from the
      free-path/scattering-angle draws -- keyed on this electron's own stream
      key via a salted rehash, so turning it on can never perturb *those* draws
      -- and returns the summed per-electron SAMPLED loss as
      ``result["straggle_dE_keV"]``.

      The sampled loss is applied on every host core and on both CUDA cores.
      The LUT cores (CPU and CUDA) interpolate the mean stopping, hazard and
      deterministic ``max_dE_frac`` cap from the energy LUT but sample the
      Urban loss per element from the exact packed tables, so the per-flight
      law is the exact core's at the LUT's own row length.
      ``E_keV``/``E_end_keV``, cutoff crossing and
      ``n_cutoff_stopped`` reflect the sampled loss. On a cutoff row the
      applied loss is ``E_start - E_cut``; the diagnostic retains the full
      sampled loss, including the discarded overshoot.

      Validation: energy-loss-straggling

    collect_diagnostics: opt in to fixed-size percentile summaries of the
    per-flight fractional energy loss, relative elastic-hazard change,
    left-endpoint versus midpoint clock estimate, and cutoff overshoot. Runs
    after transport, consumes no random draws, does not alter propagation,
    and retains no per-flight arrays. A device-resident run copies the four
    required segment arrays to the host only when explicitly requested.

    keep_segments_on_device: return the eight per-segment arrays as CUDA
    device arrays instead of copying them to the host (same dtypes, elements,
    and order -- only the payload's location moves). Requires
    transport_core="cuda"; every other returned array (incident phase-space
    diagnostics, groove-gap arrays) and every count stays NumPy. The caller's
    spectrum backend must then also be CUDA (PYRITE_MC_BACKEND resolving to
    CUDA) -- NumPy kernels refuse the implicit host conversion rather than
    performing it silently. Segments are joined with one concatenate (holds
    two copies transiently) and stay device-resident for as long as the
    caller keeps the dict.

    Parameters
    ----------
    E0_keV, Ne, thickness_ang
        Incident kinetic energy in keV, macro-electron count, and slab thickness
        in angstroms.
    element, n_atoms_per_ang3
        Elemental target symbol and number density. Superseded by ``composition``.
    E_cut_keV
        Electron kinetic-energy termination threshold in keV.
    seed, max_steps
        Random seed and maximum transport steps per electron.
    elastic_model
        ``"mott"`` tabulated scattering, analytic ``"sr"`` scattering, or
        ``"elsepa"`` sampling from ``elastic_tables``.
    beam_dir
        Mean incident direction in the slab frame; defaults to ``+z``.
    composition
        Compound ``(element, number_density)`` pairs in atoms per cubic angstrom.
    layers
        Optional contiguous film-first ``(z_top, z_bottom, composition)`` stack.
    beam_fwhm_mm, beam_fwhm_y_mm
        Gaussian entrance-spot FWHM values in mm.
    bunch_length_fs, long_shape, long_offsets_fs, longitudinal_distribution
        Longitudinal bunch sampling controls and resolved policy.
    bunch_front_tilt
        Linear lab-frame arrival-time slope as a BunchFrontTilt descriptor.
        Matching target tilt and azimuth makes the front parallel to its face.
    transverse_distribution
        Resolved Courant--Snyder entrance phase-space policy.
    energy_spread_frac
        RMS fractional incident-energy spread.
    crystal_width_mm, crystal_height_mm
        Paired full transverse prism dimensions in mm.
    tilt_polar_rad, tilt_azim_rad
        Target tilt used to project the finite entrance beam.
    groove
        Optional supported blazed-groove specification.
    E_cut_by_electrons
        Optional alternate cutoff assignment by electron group.
    transport_core
        ``"auto"``, ``"lockstep"``, ``"per-electron"``, or ``"cuda"``.
    per_electron_config, transport_lut_config
        Per-electron batching and transport lookup-table controls.
    keep_segments_on_device
        Return core segment arrays as CuPy arrays from CUDA transport.
    collect_diagnostics
        Collect fixed-size transport-error percentile summaries.
    energy_model, max_dE_frac
        Flight-energy integration rule and optional fractional-loss substep cap.
    straggling
        Enable stochastic Urban per-flight energy loss.

    gdf_source
        Internal validated GPT source descriptor produced by case construction;
        CPU-loaded records replace analytic initial phase space before numeric
        arrays enter the existing CPU/CUDA transport boundary.
    _electron_start, _energy_range_keV
        Internal electron-block offset and population table range (#361): every
        stream takes global indices ``[start, start + Ne)``; ``electron_id``
        stays block-local. See ``runner/block_transport.py`` and
        :func:`.beam_entry.check_electron_block`.
    _complete_hard_photons
        Internal (#864). ``False`` skips hard-photon direction completion and
        omits ``hard_radiative_direction``/``hard_radiative_target_momentum_eV_c``
        for live streamed scoring, whose estimators never read them. Completion
        draws from its own generator, so transport rows are unchanged bit for
        bit. Refused with secondary cascades, which consume the directions.

    Returns
    -------
    dict
        Per-segment arrays, incident phase-space diagnostics, non-radiating
        ``vacuum_*`` arrays, and scalar termination/count diagnostics. The
        complete schema is documented in ``transport-outputs.md``.

    Raises
    ------
    RuntimeError
        If any electron history remains incomplete.
    ValueError
        If physical inputs or mutually exclusive model controls are invalid.

    Validation: electron-transport, energy-loss-straggling, finite-beam-size,
    finite-transverse-crystal, grazing-beam-projection, multilayer-stack

    Validation: substep-radiation-invariance
    Validation: transverse-bunch-form-factor
    """
    check_electron_block(
        _electron_start,
        (secondary_threshold_eV, pair_production_model, positron_transport, _secondary, groove),
        (gdf_source, bunch_length_fs, long_offsets_fs, longitudinal_distribution),
    )
    if not _complete_hard_photons and (
        secondary_threshold_eV,
        pair_production_model,
        positron_transport,
    ) != (None, None, False):
        raise ValueError("secondary cascades need completed hard-photon directions")
    # Every parameter, named once: the cascade re-enters this function with
    # exactly these inputs, never host-side locals.
    inputs = TransportInputs(
        E0_keV=E0_keV,
        Ne=Ne,
        thickness_ang=thickness_ang,
        element=element,
        n_atoms_per_ang3=n_atoms_per_ang3,
        E_cut_keV=E_cut_keV,
        seed=seed,
        max_steps=max_steps,
        elastic_model=elastic_model,
        beam_dir=beam_dir,
        composition=composition,
        layers=layers,
        beam_fwhm_mm=beam_fwhm_mm,
        beam_fwhm_y_mm=beam_fwhm_y_mm,
        bunch_length_fs=bunch_length_fs,
        long_shape=long_shape,
        long_offsets_fs=long_offsets_fs,
        longitudinal_distribution=longitudinal_distribution,
        transverse_distribution=transverse_distribution,
        energy_spread_frac=energy_spread_frac,
        crystal_width_mm=crystal_width_mm,
        crystal_height_mm=crystal_height_mm,
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
        E_cut_by_electrons=E_cut_by_electrons,
        transport_core=transport_core,
        per_electron_config=per_electron_config,
        keep_segments_on_device=keep_segments_on_device,
        transport_precision=transport_precision,
        transport_lut_config=transport_lut_config,
        collect_diagnostics=collect_diagnostics,
        energy_model=energy_model,
        max_dE_frac=max_dE_frac,
        straggling=straggling,
        stopping_tables=stopping_tables,
        inelastic_model=inelastic_model,
        inelastic_cutoff_eV=inelastic_cutoff_eV,
        inelastic_materials=inelastic_materials,
        elastic_tables=elastic_tables,
        radiative_model=radiative_model,
        radiative_moments=radiative_moments,
        radiative_cutoff_eV=radiative_cutoff_eV,
        bremslib_tables=bremslib_tables,
        gdf_source=gdf_source,
        secondary_threshold_eV=secondary_threshold_eV,
        max_secondary_generations=max_secondary_generations,
        max_secondary_tracks=max_secondary_tracks,
        pair_production_model=pair_production_model,
        positron_transport=positron_transport,
        atomic_electron_deflection=atomic_electron_deflection,
        transport_progress=transport_progress,
        _secondary=_secondary,
        _electron_start=_electron_start,
        _energy_range_keV=_energy_range_keV,
        bunch_front_tilt=bunch_front_tilt,
        _complete_hard_photons=_complete_hard_photons,
    )
    if (secondary_threshold_eV, pair_production_model, positron_transport) != (None, None, False):
        from .secondaries import transport_secondary_cascade

        return transport_secondary_cascade(simulate_trajectories, inputs.as_kwargs())
    return run_transport_pass(inputs)
