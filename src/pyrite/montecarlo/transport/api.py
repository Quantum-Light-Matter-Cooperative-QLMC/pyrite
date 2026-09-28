"""Public transport entry point: :func:`simulate_trajectories`."""

import logging

import numpy as np

from ..._numerics import DEFAULT_RADIATIVE_CUTOFF_EV
from ...materials.attenuation import _normalize_composition
from ..geometry import beam_frame_basis, validate_transverse_dimensions
from .batching import (
    DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    _flight_diagnostic_summary,
    _run_per_electron_transport,
    _run_per_electron_transport_lut,
    pack_layer_tables,
    resolve_transport_core,
)
from .beam_entry import initial_beam_positions
from .cores import (
    _transport_core_grooved,
    _transport_core_ungrooved_lut,
    _transport_core_ungrooved_lut_inelastic,
    _transport_core_ungrooved_perelectron_lut,
    _transport_core_ungrooved_perelectron_lut_inelastic,
    exact_ungrooved_core,
)
from .hard_inelastic import hard_keys_from_stream_keys, hard_stream_keys, validate_inelastic_args
from .hard_radiative import validate_radiative_args
from .kinematics import _sample_bunch_offsets, stream_keys
from .layer_tables import build_layer_tables
from .lut import DEFAULT_TRANSPORT_LUT_CONFIG, build_transport_energy_lut
from .scattering import check_elsepa_coverage
from .stopping import (
    pack_sbethe_stopping_tables,
    prepare_sbethe_stopping_tables,
)

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
    elastic_model="mott",
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
    _secondary=None,
):
    """
    Transport Ne electrons of energy E0_keV [keV] into a slab 0<=z<=thickness.
    Beam enters at the origin along +z. Electrons terminate when they exit
    either surface or drop below E_cut_keV (segments below the cutoff don't
    radiate in the spectral window of interest anyway).

    Full derivations, sources, and limiting-case checks for the physics below
    live in ``docs/physics/beam-transport/*.md``; validation status is recorded
    in ``docs/validation/``. This docstring states only the
    parameter contract and the BIT-FOR-BIT
    limiting case each one must preserve.

    elastic_model: "mott" (default, NIST SRD 64 Mott transport cross section
      via a Browning fit), "sr" (analytic screened-Rutherford, no data
      files), or "elsepa" (tabulated ELSEPA total cross sections and full
      angular distributions from ``elastic_tables``; exact cores only, the
      transport LUT is bypassed). See
      docs/physics/beam-transport/elastic-scattering.md.
      Validation: electron-transport
      Validation: elsepa-elastic-sampling

    elastic_tables: required by, and only accepted with,
      ``elastic_model="elsepa"``. One entry per layer, each a sequence of one mapping per
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

    NOT safe with mc_spectrum(coherent=True): the coherent phase reads r_mid
    directly, so the per-electron offset enters every cross-electron term and
    the transverse form factor that should average it away is unimplemented
    (`transverse-bunch-form-factor` discrepancy) -- see
    docs/physics/radiation-physics/coherent-emission.md before using a finite
    spot with emission="coherent"/"both". See
    docs/physics/beam-transport/beam-phase-space.md and
    docs/validation/geometry/finite-beam-size.md.
    Validation: finite-beam-size

    beam_fwhm_y_mm: optional y-plane spot FWHM [mm] for an ELLIPTICAL beam.
    None -> equals beam_fwhm_mm (isotropic), which stays BIT-FOR-BIT with the
    historical scalar-spot path.

    transverse_distribution: the resolved Courant-Snyder policy, mutually
    exclusive with the spot FWHMs above (a spot fixes <x^2> alone; a Twiss
    triplet fixes <x^2>, <x x'>, and <x'^2>). Owns both the entry positions
    and the per-electron directions (via geometry.beam_frame_basis). Unset ->
    every direction is the shared beam_dir exactly, BIT-FOR-BIT. Limiting
    case: eps_n -> 0 recovers the collimated point source exactly. See
    docs/physics/beam-transport/beam-phase-space.md.
    Validation: beam-phase-space-injection

    energy_spread_frac: RMS *relative* energy spread. Each electron starts at
    E0_keV * (1 + f * u), u ~ Normal(0, 1), drawn uncorrelated with the
    arrival time. None/0 -> the monoenergetic beam, BIT-FOR-BIT. Raises
    rather than transporting a non-positive drawn energy. See
    docs/physics/beam-transport/beam-phase-space.md.
    Validation: beam-energy-spread-injection

    transverse_distribution and energy_spread_frac each use their own
    SeedSequence child stream, so enabling either never perturbs the
    free-path/scattering draws.

    bunch_length_fs, long_shape, long_offsets_fs, longitudinal_distribution:
    longitudinal bunch sampling via :func:`_sample_bunch_offsets`; each
    electron gets an arrival offset (its own independent RNG child stream)
    returned as per-segment t0_ang, kept SEPARATE from relative-age
    t_ang/clock. No legacy or resolved distribution -> all-zero offsets (the
    legacy point bunch, BIT-FOR-BIT). See
    docs/physics/beam-transport/longitudinal-structure.md.
    Validation: longitudinal-bunch-sampling

    crystal_width_mm, crystal_height_mm: optional full transverse dimensions
    [mm] of a rectangular prism centered at the beam origin; both or neither.
    An incident entry point outside that footprint is counted in n_missed
    (no segment, Ne unchanged); side-face exits are counted separately in
    n_side_exited. All-None -> the original laterally infinite slab, using
    its legacy free-flight path (BIT-FOR-BIT). See
    docs/validation/geometry/finite-transverse-crystal.md.
    Validation: finite-transverse-crystal

    tilt_polar_rad, tilt_azim_rad: sample tilt (Zhai convention, same angles
    passed to :func:`geometry.tilted_geometry`), used ONLY to project the
    beam_fwhm_mm Gaussian spot onto the tilted entrance face via
    :func:`geometry.project_beam_entry`. Both default to 0 (normal
    incidence), the identity projection, BIT-FOR-BIT; no effect without
    beam_fwhm_mm. See docs/validation/geometry/grazing-beam-projection.md.
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

    radiative_model: "auto" (default) couples when BremsLib tables are supplied,
      otherwise keeps post-hoc brem scoring and the existing tracks.
      "bremslib-soft-hard" (exact cores only, no LUT) adds
      soft BremsLib loss below ``radiative_cutoff_eV`` and explicit photons
      above it; see ``hard_radiative.validate_radiative_args`` for its
      requirements and ``spectrum.brem_events`` for scoring. Validation:
      bremslib-radiative-partition, bremslib-radiative-event-spectrum.

    secondary_threshold_eV: opt-in transport of hard-collision secondaries
      (shell-soft-hard only); one threshold [eV] is production cut and
      tracking cutoff. Generations are capped by ``max_secondary_generations``
      and ``max_secondary_tracks`` (default ``1000 * Ne``), which raise. Rows
      gain ``track_id``/``parent_id``/``generation``; ``electron_id`` stays the
      primary history. None (default) is BIT-FOR-BIT primary-only transport.
      See shell-soft-hard-transport.md. Validation: shell-secondary-transport

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
          midpoint schema; unsupported CUDA-LUT/straggling combinations fail
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

      The sampled loss is applied on every host core and the exact CUDA core.
      The CUDA LUT combination raises rather than silently returning an
      unstraggled result; production core selection falls back to exact CUDA
      when straggling is enabled. ``E_keV``/``E_end_keV``, cutoff crossing and
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
    """
    if secondary_threshold_eV is not None:
        arguments = dict(locals())
        from .secondaries import transport_secondary_cascade

        return transport_secondary_cascade(simulate_trajectories, arguments)
    # ``_secondary`` is the cascade's per-generation pass (secondaries.py).
    launch = None if _secondary is None else _secondary.launch
    if not np.isfinite(E0_keV) or E0_keV <= 0.0:
        raise ValueError("E0_keV must be finite and strictly positive")

    width_mm, height_mm = validate_transverse_dimensions(
        crystal_width_mm, crystal_height_mm, unit="mm"
    )
    width_ang = None if width_mm is None else width_mm * 1.0e7
    height_ang = None if height_mm is None else height_mm * 1.0e7
    finite_footprint = width_ang is not None
    max_segments = Ne * max_steps
    max_vac = Ne * max_steps

    # Build the layer stack: explicit `layers` (film-on-substrate) overrides;
    # else a single layer spanning the slab (bit-for-bit the old transport).
    if layers is None:
        layers = [
            (
                0.0,
                float(thickness_ang),
                _normalize_composition(element, n_atoms_per_ang3, composition),
            )
        ]

    if elastic_model not in ("mott", "sr", "elsepa"):
        raise ValueError("elastic_model must be 'mott', 'sr', or 'elsepa'")
    if (elastic_model == "elsepa") != (elastic_tables is not None):
        raise ValueError(
            "elastic_tables is required by, and only valid with, elastic_model='elsepa'"
        )
    if energy_model not in ("frozen", "midpoint"):
        raise ValueError("energy_model must be 'frozen' or 'midpoint'")
    shell_mode = validate_inelastic_args(
        inelastic_model,
        inelastic_cutoff_eV,
        inelastic_materials,
        energy_model=energy_model,
        groove=groove,
        stopping_tables=stopping_tables,
    )
    radiative_mode = validate_radiative_args(
        radiative_model,
        radiative_cutoff_eV,
        bremslib_tables,
        energy_model=energy_model,
        groove=groove,
        keep_segments_on_device=keep_segments_on_device,
    )
    if radiative_mode and radiative_cutoff_eV is None:
        radiative_cutoff_eV = DEFAULT_RADIATIVE_CUTOFF_EV

    requested_core = transport_core
    if launch is None:  # a launched generation names its per-electron core
        transport_core = resolve_transport_core(transport_core, Ne, groove)
    if transport_core != "lockstep" and groove is not None:
        raise ValueError("grooved transport is only implemented for the lockstep core")
    if keep_segments_on_device and transport_core != "cuda":
        raise ValueError(
            "keep_segments_on_device requires transport_core='cuda'; "
            f"{requested_core!r} resolved to {transport_core!r}"
        )
    max_dE_frac = float(max_dE_frac)
    if max_dE_frac < 0.0:
        raise ValueError("max_dE_frac must be non-negative")
    # Substepping a frozen flight is exactly the mis-phased configuration the
    # slice-E convergence study rejected: it multiplies rows without improving
    # the clock, so the two options are not independently selectable.
    if max_dE_frac > 0.0 and energy_model != "midpoint":
        raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")

    if E_cut_by_electrons is None:
        E_cut_by_electrons = np.full(
            Ne,
            float(E_cut_keV),
            dtype=np.float64,
        )
    else:
        E_cut_by_electrons = np.asarray(
            E_cut_by_electrons,
            dtype=np.float64,
        )

        if E_cut_by_electrons.shape != (Ne,):
            raise ValueError(
                f"E_cut_by_electrons must have shape ({Ne},), got {E_cut_by_electrons.shape}"
            )

    if not np.all(np.isfinite(E_cut_by_electrons)) or not np.all(E_cut_by_electrons > 0.0):
        raise ValueError("electron cutoff energies must be finite and strictly positive")

    # NVTX ranges split the host transport phase in a GPU capture; no-op off
    # the profiled path. Lazy import: runner imports this module.
    from ..runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.tables")
    z_total = float(layers[-1][1])
    n_layers = len(layers)
    prepared_stopping_tables = prepare_sbethe_stopping_tables(stopping_tables, n_layers)
    (
        L_Zs,
        L_Js,
        L_ncm3,
        L_ks,
        L_coeffs,
        L_E_cross,
        mott_tables,
        L_sr_rate_numer,
        L_mott_numer,
        L_mott_denom1,
        L_mott_denom2,
        L_sr_joy_numer,
        mott_group,
    ) = build_layer_tables(layers, elastic_model, elastic_tables)
    L_top = np.asarray([float(a) for (a, _, _) in layers], dtype=float)
    L_bot = np.asarray([float(b) for (_, b, _) in layers], dtype=float)
    internal_bounds = L_bot[:-1].copy()
    _nsys_pop()

    _nsys_push("cxr.transport.sample")
    rng = np.random.default_rng(seed)
    pos, transverse_slopes = initial_beam_positions(
        seed,
        Ne,
        transverse_distribution=transverse_distribution,
        beam_fwhm_mm=beam_fwhm_mm,
        beam_fwhm_y_mm=beam_fwhm_y_mm,
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
    )
    if beam_dir is None:
        beam_dir = np.array([0.0, 0.0, 1.0])
    beam_dir = np.asarray(beam_dir, dtype=float)
    if beam_dir.shape != (3,) or not np.all(np.isfinite(beam_dir)):
        raise ValueError("beam_dir must be a finite three-vector")
    beam_norm = np.linalg.norm(beam_dir)
    if not np.isfinite(beam_norm) or beam_norm == 0.0:
        raise ValueError("beam_dir must be nonzero")
    beam_dir = beam_dir / beam_norm
    if beam_dir[2] <= 1e-6:
        raise ValueError("beam_dir must point into the slab (z component > 0)")
    if transverse_slopes is None:
        dirs = np.tile(beam_dir, (Ne, 1))
    else:
        # Slopes are dx/dz, dy/dz about the beam axis, so the per-electron
        # direction is (x', y', 1) in the beam frame. Zero emittance gives
        # exactly `beam_dir` back, which is what makes the collimated limit
        # bit-for-bit rather than merely close.
        x_prime, y_prime = transverse_slopes
        basis = beam_frame_basis(beam_dir)
        dirs = x_prime[:, None] * basis[:, 0] + y_prime[:, None] * basis[:, 1] + beam_dir
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    E_keV = np.full(Ne, float(E0_keV))
    if energy_spread_frac:
        # RMS *relative* deviation, uncorrelated with arrival time (decision 3:
        # no chirp model). Its own child stream, spawn(6)[5], for the same
        # reason as the transverse draw above.
        spread_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[5])
        E_keV = E_keV * (1.0 + float(energy_spread_frac) * spread_rng.standard_normal(Ne))
        if not np.all(np.isfinite(E_keV)) or not np.all(E_keV > 0.0):
            raise ValueError(
                f"energy_spread_frac={energy_spread_frac} drew a non-finite or non-positive "
                "electron energy; the Gaussian spread model needs spread << 1"
            )
    gdf_t0 = None
    if gdf_source is not None:
        from ..gdf import load_gdf_beam

        if (
            groove is not None
            or beam_fwhm_mm
            or beam_fwhm_y_mm
            or transverse_distribution is not None
            or energy_spread_frac
            or bunch_length_fs is not None
            or long_offsets_fs is not None
            or longitudinal_distribution is not None
        ):
            raise ValueError(
                "gpt_gdf requires a flat entrance and no analytic phase-space settings"
            )
        gdf = load_gdf_beam(
            gdf_source["path"],
            gdf_source["time_s"],
            gdf_source["tolerance_s"],
            gdf_source["normalization"],
            screen_position_m=gdf_source.get("screen_position_m"),
            screen_tolerance_m=gdf_source.get("screen_tolerance_m", 1e-9),
        )
        if gdf.sha256 != gdf_source["sha256"]:
            raise ValueError("GDF file changed after case construction; rebuild the run")
        pos, dirs, E_keV, gdf_t0 = gdf.sample(
            Ne,
            seed,
            gdf_source["z_origin_m"],
            tilt_polar_rad,
            tilt_azim_rad,
            energy_keV=float(E0_keV) if gdf_source.get("shape_only", False) else None,
        )
    elif launch is not None:
        pos, dirs, E_keV = launch.r_ang.copy(), launch.v_hat.copy(), launch.E_keV.copy()
    if not np.all(E_cut_by_electrons < E_keV):
        raise ValueError("each electron cutoff energy must be below its initial energy")
    if prepared_stopping_tables is not None:
        for prepared in prepared_stopping_tables:
            lower = float(np.nextafter(np.exp(prepared[0][0]), -np.inf))
            upper = float(np.nextafter(np.exp(prepared[0][-1]), np.inf))
            if float(np.min(E_cut_by_electrons)) < lower or float(np.max(E_keV)) > upper:
                raise ValueError(
                    f"transport energy range must be within SBETHE table [{lower:g}, {upper:g}] keV"
                )
    check_elsepa_coverage(elastic_tables, float(np.min(E_cut_by_electrons)), float(np.max(E_keV)))
    shell_tables = None
    if shell_mode:
        # Lazy: the host shell model pulls in catalog and EEDL data.
        from .shell_transport import build_shell_inelastic_tables

        # Checked by validate_inelastic_args, which also requires stopping_tables.
        assert inelastic_materials is not None and inelastic_cutoff_eV is not None
        assert prepared_stopping_tables is not None
        _nsys_push("cxr.transport.inelastic")
        E_range_keV = (float(np.min(E_cut_by_electrons)), float(np.max(E_keV)))
        if _secondary is not None:  # one table set for every generation
            lo, hi = _secondary.table_range_keV
            E_range_keV = (min(lo, E_range_keV[0]), max(hi, E_range_keV[1]))
        shell_tables = build_shell_inelastic_tables(
            inelastic_materials, float(inelastic_cutoff_eV), prepared_stopping_tables, *E_range_keV
        )
        _nsys_pop()
        # The cores' continuous stopping is the soft share from here on.
        prepared_stopping_tables = list(shell_tables.soft_stopping_tables)
    radiative_args = None
    if radiative_mode:
        from .hard_radiative import radiative_core_args

        radiative_args = radiative_core_args(
            [layer[2] for layer in layers],
            bremslib_tables,
            radiative_cutoff_eV,
            E_cut_by_electrons,
            E_keV,
            seed,
            Ne,
        )
        if launch is not None:
            radiative_args = (launch.radiative_keys,) + radiative_args[1:]
    if finite_footprint:
        assert height_ang is not None
        alive = (np.abs(pos[:, 0]) <= width_ang / 2.0) & (np.abs(pos[:, 1]) <= height_ang / 2.0)
        n_missed = int((~alive).sum())
    else:
        alive = np.ones(Ne, dtype=bool)
        n_missed = 0
    n_back = n_trans = n_side = 0
    # per-electron clock: cumulative flight "time" sum(L/beta) [Ang, c=1], the
    # same unit as the radiation interaction time t_L. Recorded at each segment's
    # START so a segment carries (depth, energy, age) -- consumed by the
    # penetration / electron-lifetime plots (-> fs via c = 2997.92 Ang/fs).

    clock = np.zeros(Ne)
    # per-electron longitudinal bunch offset [Ang, c=1], kept SEPARATE from the
    # relative-age clock: the coherent sum later reads absolute time
    # t_abs = t_ang + t0_ang. None/None -> all-zero (point bunch, bit-for-bit),
    # and nothing reads it in the incoherent spectrum today.
    t0_electron = _sample_bunch_offsets(
        Ne,
        bunch_length_fs,
        long_shape,
        long_offsets_fs,
        seed,
        longitudinal_distribution,
    )
    if gdf_t0 is not None:
        t0_electron = gdf_t0
    elif launch is not None:
        clock, t0_electron = launch.t_ang.copy(), launch.t0_ang.copy()
    # Snapshot before transport mutates ``pos``, ``dirs``, and ``E``. These
    # arrays describe incident phase space, including particles that miss a
    # finite footprint.
    initial_r_ang = pos.copy()
    initial_v_hat = dirs.copy()
    initial_E_keV = E_keV.copy()
    _nsys_pop()

    elastic_model_code = {"sr": 0, "mott": 1, "elsepa": 2}[elastic_model]
    transport_lut = None
    # The LUT stores a screened-Rutherford screening parameter per energy,
    # which has no ELSEPA counterpart, so ELSEPA runs the exact cores; it
    # does not run the coupled radiative mode either.
    if (
        groove is None
        and transport_lut_config.enabled
        and not radiative_mode
        and elastic_model != "elsepa"
    ):
        _nsys_push("cxr.transport.lut")
        transport_lut = build_transport_energy_lut(
            float(np.min(E_cut_by_electrons)),
            float(np.max(E_keV)),
            elastic_model_code,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_sr_rate_numer,
            L_mott_numer,
            L_mott_denom1,
            L_mott_denom2,
            L_sr_joy_numer,
            mott_tables,
            config=transport_lut_config,
            stopping_tables=prepared_stopping_tables,
        )
        _nsys_pop()
    sbethe_group = pack_sbethe_stopping_tables(prepared_stopping_tables, n_layers)

    _nsys_push("cxr.transport.alloc")
    # Preallocate fixed-capacity output buffers.  ``max_steps`` is already a
    # conservative safety bound in ordinary runs; groove re-entry events can add
    # material segments without consuming it, so the compiled core raises a
    # clear buffer error if a pathological case exceeds this capacity.
    # A device-resident run fills none of them -- its batches are joined on the
    # device instead -- so it reserves no rows.
    n_rows = 0 if keep_segments_on_device else max_segments
    seg_mid = np.empty((n_rows, 3), dtype=float)
    seg_dir = np.empty((n_rows, 3), dtype=float)
    seg_len = np.empty(n_rows, dtype=float)
    seg_E = np.empty(n_rows, dtype=float)
    seg_t0 = np.empty(n_rows, dtype=float)
    seg_id = np.empty(n_rows, dtype=np.int64)
    seg_lay = np.empty(n_rows, dtype=np.int16)
    # Flight end state exists only under the controlled propagator; the frozen
    # rule keeps the historical seven-field row exactly.
    n_end_rows = n_rows if energy_model == "midpoint" else 0
    seg_E_end = np.empty(n_end_rows, dtype=float)
    seg_t_end = np.empty(n_end_rows, dtype=float)
    seg_flight = np.empty(n_end_rows, dtype=np.int64)
    seg_substep = np.empty(n_end_rows, dtype=np.int64)
    seg_event = np.empty(n_end_rows, dtype=np.int8)
    seg_hard_W = np.empty(n_end_rows if shell_mode else 0, dtype=float)
    seg_hard_ch = np.empty(seg_hard_W.size, dtype=np.int16)
    # Secondary launch directions (#94): rows only while a cascade runs.
    sec_on = shell_mode and _secondary is not None
    seg_hard_dir = np.empty((seg_hard_W.size if sec_on else 0, 3), dtype=float)
    seg_rad_k = np.empty(n_end_rows if radiative_mode else 0, dtype=float)
    seg_rad_Z = np.empty(seg_rad_k.size, dtype=np.int16)
    inelastic_args = None
    if shell_tables is not None:
        inelastic_args = shell_tables.core_args(
            hard_stream_keys(seed, Ne)
            if launch is None
            else hard_keys_from_stream_keys(launch.stream_keys)
        )
    _nsys_pop()

    # Where the segments end up living, and so which array module assembles the
    # output below. NumPy unless the run asked to keep them on the device.
    seg_xp = np
    dev_segs = None
    energy_model_code = 1 if energy_model == "midpoint" else 0

    # Straggling (slice D): False skips the Urban sampler on every core,
    # BIT-FOR-BIT. Lockstep/grooved cores key it by ``stragg_stream_keys``;
    # per-electron cores reuse their own stream keys. ``stragg_layer_tables``
    # gives the per-electron LUT core the per-element split the LUT lacks.
    layer_arrays = (
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_E_cross,
        L_ncm3,
        L_sr_rate_numer,
        L_mott_numer,
        L_mott_denom1,
        L_mott_denom2,
        L_sr_joy_numer,
    )
    straggle_on = bool(straggling)
    stragg_dE = np.zeros(Ne) if straggle_on else np.zeros(0)
    stragg_stream_keys = stream_keys(seed, Ne) if straggle_on else np.zeros(1, dtype=np.uint64)
    if straggle_on:
        _stragg_packed = pack_layer_tables(*layer_arrays)
        stragg_layer_tables = _stragg_packed[:5] + sbethe_group
    else:
        _stragg_dummy = np.zeros((1, 1), dtype=np.float64)
        stragg_layer_tables = (_stragg_dummy,) * 5 + sbethe_group

    # Grouped kernel argument tuples (issue #66): the lockstep calls below pass
    # them straight through to the cores; the per-electron drivers in
    # batching.py build their own after the device upload.
    width_or_zero = 0.0 if width_ang is None else float(width_ang)
    height_or_zero = 0.0 if height_ang is None else float(height_ang)
    control = (max_steps, max_segments, elastic_model_code, energy_model_code, max_dE_frac)
    state = (alive, clock, pos, dirs, E_keV, E_cut_by_electrons)
    segments = (
        seg_dir,
        seg_mid,
        seg_len,
        seg_E,
        seg_t0,
        seg_id,
        seg_lay,
        seg_E_end,
        seg_t_end,
        seg_flight,
        seg_substep,
        seg_event,
    )
    if shell_mode:
        segments += (seg_hard_W, seg_hard_ch, seg_hard_dir)
    if radiative_mode:
        segments += (seg_rad_k, seg_rad_Z)
    materials_ragged = layer_arrays + sbethe_group
    straggling_lockstep = (straggle_on, stragg_stream_keys, stragg_dE)
    # The exact lockstep cores index ragged rows and never read L_nel; it rides
    # in the geometry tuple only so every core shares one layout.
    L_nel = np.fromiter((zs.size for zs in L_Zs), dtype=np.int32, count=n_layers)
    geometry = (
        n_layers,
        internal_bounds,
        z_total,
        finite_footprint,
        width_or_zero,
        height_or_zero,
        L_nel,
        L_top,
        L_bot,
    )

    _nsys_push("cxr.transport.core")
    # Only the grooved core records vacuum crossings; it replaces these.
    nvac = 0
    vac_start = np.empty((0, 3), dtype=float)
    vac_end = np.empty((0, 3), dtype=float)
    vac_E = np.empty(0, dtype=float)
    vac_t0 = np.empty(0, dtype=float)
    vac_id = np.empty(0, dtype=np.int64)
    if groove is None and transport_lut is not None and transport_core != "lockstep":
        if transport_core == "cuda":
            if straggle_on:
                # Only the exact CUDA kernel carries the Urban sampler; the LUT
                # kernel has no per-element split to sample from. Raise rather
                # than return an unstraggled result.
                raise NotImplementedError(
                    "straggling=True is not implemented on the CUDA LUT core "
                    "(transport_core='cuda' with the LUT enabled); disable the "
                    "LUT (transport_lut_config=TransportLUTConfig(enabled=False)) "
                    "to reach the CUDA exact per-electron core, or run off CUDA "
                    "with transport_core='per-electron' or 'lockstep'"
                )
            if shell_mode:
                raise NotImplementedError(
                    "inelastic_model='shell-soft-hard' is not implemented on the CUDA "
                    "LUT core; disable the LUT to reach the exact CUDA core"
                )
            from ._jit_launch import make_cuda_transport_lut_core

            core, core_xp = make_cuda_transport_lut_core()
        elif shell_mode:
            core, core_xp = _transport_core_ungrooved_perelectron_lut_inelastic, np
        else:
            core, core_xp = _transport_core_ungrooved_perelectron_lut, np
        if keep_segments_on_device:
            seg_xp = core_xp

        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, dev_segs, stragg_dE = (
            _run_per_electron_transport_lut(
                core,
                core_xp,
                Ne,
                seed,
                max_steps,
                max_segments,
                n_layers,
                internal_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                0.0 if width_ang is None else float(width_ang),
                0.0 if height_ang is None else float(height_ang),
                alive,
                clock,
                pos,
                dirs,
                E_cut_by_electrons,
                transport_lut.n_el,
                L_top,
                L_bot,
                transport_lut,
                E_keV,
                seg_dir,
                seg_mid,
                seg_len,
                seg_E,
                seg_t0,
                seg_id,
                seg_lay,
                seg_E_end,
                seg_t_end,
                seg_flight,
                seg_substep,
                seg_event,
                stragg_layer_tables,
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
                inelastic=(
                    (inelastic_args, seg_hard_W, seg_hard_ch, seg_hard_dir, sec_on)
                    if shell_mode
                    else None
                ),
                keys=None if launch is None else launch.stream_keys,
            )
        )
    elif groove is None and transport_lut is not None:
        lut_core = (
            _transport_core_ungrooved_lut_inelastic if shell_mode else _transport_core_ungrooved_lut
        )
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = lut_core(
            Ne,
            rng,
            control,
            (
                n_layers,
                internal_bounds,
                z_total,
                finite_footprint,
                width_or_zero,
                height_or_zero,
                transport_lut.n_el,
                L_top,
                L_bot,
            ),
            (
                transport_lut.log_E_min,
                transport_lut.inv_dlogE,
                transport_lut.n_energy,
                transport_lut.total_rate,
                transport_lut.dEds,
                transport_lut.inv_beta,
                transport_lut.cdf,
                transport_lut.alpha,
            ),
            (L_Js, L_Zs, L_ks, L_coeffs, L_E_cross) + sbethe_group,
            state,
            segments,
            straggling_lockstep,
            *((inelastic_args,) if shell_mode else ()),
        )
    elif groove is None and transport_core != "lockstep":
        # Per-electron streams and run-to-completion ordering. Not bit-for-bit
        # with the lockstep core -- see `_transport_core_ungrooved_perelectron`.
        if transport_core == "cuda":
            from ._jit_launch import make_cuda_transport_core

            core, core_xp = make_cuda_transport_core()
        else:
            core = exact_ungrooved_core(
                per_electron=True, inelastic=shell_mode, radiative=radiative_mode
            )
            core_xp = np
        if keep_segments_on_device:
            seg_xp = core_xp

        packed_per_layer_tables = pack_layer_tables(*layer_arrays)
        packed_per_layer_tables += sbethe_group
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, dev_segs, stragg_dE = (
            _run_per_electron_transport(
                core,
                core_xp,
                Ne,
                seed,
                max_steps,
                max_segments,
                n_layers,
                internal_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                0.0 if width_ang is None else float(width_ang),
                0.0 if height_ang is None else float(height_ang),
                alive,
                clock,
                pos,
                dirs,
                E_cut_by_electrons,
                packed_per_layer_tables,
                L_top,
                L_bot,
                mott_group,
                E_keV,
                seg_dir,
                seg_mid,
                seg_len,
                seg_E,
                seg_t0,
                seg_id,
                seg_lay,
                seg_E_end,
                seg_t_end,
                seg_flight,
                seg_substep,
                seg_event,
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
                inelastic=(
                    (inelastic_args, seg_hard_W, seg_hard_ch, seg_hard_dir, sec_on)
                    if shell_mode
                    else None
                ),
                keys=None if launch is None else launch.stream_keys,
                radiative=((radiative_args, seg_rad_k, seg_rad_Z) if radiative_mode else None),
            )
        )
    elif groove is None:
        exact_core = exact_ungrooved_core(
            per_electron=False, inelastic=shell_mode, radiative=radiative_mode
        )
        extra_args = (inelastic_args,) if shell_mode else ()
        if radiative_mode:
            extra_args = (inelastic_args if shell_mode else (), radiative_args)
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = exact_core(
            Ne,
            rng,
            control,
            geometry,
            materials_ragged,
            mott_group,
            state,
            segments,
            straggling_lockstep,
            *extra_args,
        )
    else:
        vac_start_buf = np.empty((max_vac, 3), dtype=float)
        vac_end_buf = np.empty((max_vac, 3), dtype=float)
        vac_E_buf = np.empty(max_vac, dtype=float)
        vac_t0_buf = np.empty(max_vac, dtype=float)
        vac_id_buf = np.empty(max_vac, dtype=np.int64)

        tp = float(groove.tilt_polar_rad)
        groove_st = float(np.sin(tp))
        groove_ct = float(np.cos(tp))
        nseg, nvac, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_grooved(
            Ne,
            rng,
            control,
            geometry,
            (
                float(groove.spacing_ang),
                float(groove.depth_ang),
                groove_st,
                groove_ct,
                max_vac,
                vac_start_buf,
                vac_end_buf,
                vac_E_buf,
                vac_t0_buf,
                vac_id_buf,
            ),
            materials_ragged,
            mott_group,
            state,
            segments,
            straggling_lockstep,
        )
        vac_start = vac_start_buf[:nvac]
        vac_end = vac_end_buf[:nvac]
        vac_E = vac_E_buf[:nvac]
        vac_t0 = vac_t0_buf[:nvac]
        vac_id = vac_id_buf[:nvac]
    _nsys_pop()

    if n_step_limited:
        raise RuntimeError(
            "incomplete electron transport: "
            f"n_step_limited={n_step_limited}, Ne={Ne}, max_steps={max_steps}"
        )

    _nsys_push("cxr.transport.output")
    if dev_segs is None:
        r_mid = seg_mid[:nseg]
        v_hat = seg_dir[:nseg]
        L_ang = seg_len[:nseg]
        E_seg = seg_E[:nseg]
        t_ang = seg_t0[:nseg]
        elec_id = seg_id[:nseg]
        layer = seg_lay[:nseg]
    else:
        # Already sized to `nseg` by the join, in the scratch's field order.
        v_hat, r_mid, L_ang, E_seg, t_ang, elec_id, layer = dev_segs[:7]
        if energy_model == "midpoint":
            seg_E_end, seg_t_end, seg_flight, seg_substep, seg_event = dev_segs[7:12]
        if shell_mode:
            seg_hard_W, seg_hard_ch, seg_hard_dir = dev_segs[12:15]

    vacuum_start_ang = vac_start
    vacuum_end_ang = vac_end
    vacuum_E_keV = vac_E
    vacuum_t_ang = vac_t0
    vacuum_elec_id = vac_id

    # Per-segment, so it follows the segments: gathering on the device costs one
    # Ne-sized upload of `t0_electron` and saves an nseg-sized download.
    t0_by_electron = t0_electron if seg_xp is np else seg_xp.asarray(t0_electron)
    t0_ang = t0_by_electron[elec_id] if elec_id.size else seg_xp.empty(0, dtype=float)
    vacuum_t0_ang = t0_electron[vacuum_elec_id] if vacuum_elec_id.size else np.empty(0, dtype=float)
    _nsys_pop()

    result = {
        # Initial sampled phase space is diagnostic-only.  Keep per-electron
        # arrays (including missed entries), separate from per-segment arrays,
        # so beam metrics describe the incident bunch rather than its transport.
        "initial_r_ang": initial_r_ang,
        "initial_v_hat": initial_v_hat,
        "initial_E_keV": initial_E_keV,
        "initial_t0_ang": t0_electron.copy(),
        "r_mid": r_mid,
        "v_hat": v_hat,
        "L_ang": L_ang,
        # `E_keV`/`t_ang` are unscheduled compatibility aliases of the canonical
        # flight-start fields and never become midpoint/representative values.
        "E_keV": E_seg,
        "E_start_keV": E_seg,
        "t_ang": t_ang,  # segment-start age sum(L/beta) [Ang, c=1]
        "t_start_ang": t_ang,
        "t0_ang": t0_ang,  # per-electron longitudinal bunch offset [Ang, c=1]
        # `elec_id` is an unscheduled compatibility alias of `electron_id`.
        "elec_id": elec_id,  # emitting electron index in [0, Ne)
        "electron_id": elec_id,
        "layer": layer,  # emitting layer index in [0, n_layers)
        "vacuum_start_ang": vacuum_start_ang,
        "vacuum_end_ang": vacuum_end_ang,
        "vacuum_E_keV": vacuum_E_keV,
        "vacuum_t_ang": vacuum_t_ang,
        "vacuum_t0_ang": vacuum_t0_ang,
        "vacuum_elec_id": vacuum_elec_id,
        "n_backscattered": n_back,
        "n_transmitted": n_trans,
        "n_side_exited": n_side,
        "n_missed": n_missed,
        "n_cutoff_stopped": int(n_cutoff),
        "n_step_limited": 0,
        "n_stopped": int(n_cutoff),
        "Ne": Ne,
        "thickness_ang": z_total,
        "crystal_width_ang": width_ang,
        "crystal_height_ang": height_ang,
        "n_layers": n_layers,
    }
    if prepared_stopping_tables is not None:
        # Prepared (log-transformed) per-layer SBETHE tables, so post-transport
        # consumers that re-solve the cutoff (the spectrum cutoff clip) use the
        # same stopping model the transport ran with instead of the reference
        # splice. Metadata, not a per-row array: absent from _SEG_ARRAYS.
        result["stopping_tables"] = tuple(prepared_stopping_tables)
    if straggle_on:
        # The summed per-electron Urban-SAMPLED loss. Every host core and the
        # exact CUDA core apply it; the CUDA LUT combination fails closed.
        result["straggle_dE_keV"] = stragg_dE
    if energy_model == "midpoint":
        result["E_end_keV"] = seg_E_end[:nseg]
        result["t_end_ang"] = seg_t_end[:nseg]
        # Representative energy used by the clock and radiation quadrature.
        # The explicit RK2 stopping update itself uses (E_start + E_pred)/2.
        result["E_repr_keV"] = 0.5 * (E_seg + seg_E_end[:nseg])
        # `(electron_id, flight_id)` is the stable physical key; `substep_id`
        # indexes numerical rows inside one flight and is integration detail.
        result["flight_id"] = seg_flight[:nseg]
        result["substep_id"] = seg_substep[:nseg]
        # What ended each row; `transport.events` holds the codes and the
        # contract (`check_segment_event_contract`).
        result["event_kind"] = seg_event[:nseg]
    if shell_tables is not None:
        result.update(shell_tables.result_fields(seg_hard_W[:nseg], seg_hard_ch[:nseg]))
        if sec_on:
            result["hard_secondary_v_hat"] = seg_hard_dir[:nseg]
    if radiative_mode:
        from .hard_radiative import add_radiative_result_fields

        add_radiative_result_fields(
            result,
            seg_rad_k[:nseg],
            seg_rad_Z[:nseg],
            bremslib_tables,
            radiative_cutoff_eV,
            seed if _secondary is None else _secondary.photon_seed,
        )
    if collect_diagnostics:
        result["transport_diagnostics"] = _flight_diagnostic_summary(
            E_seg,
            L_ang,
            elec_id,
            layer,
            E_cut_by_electrons,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            elastic_model,
            mott_group[5:],
        )
    return result
