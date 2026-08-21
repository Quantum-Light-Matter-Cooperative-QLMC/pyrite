"""Public transport entry point: :func:`simulate_trajectories`."""

import logging

import numpy as np

from ...materials._transport_data import TRANSPORT_ELEMENTS
from ...materials.attenuation import _normalize_composition
from ...transverse import resolved_from_mapping, sample_transverse
from ..geometry import beam_frame_basis, project_beam_entry, validate_transverse_dimensions
from ..groove import entry_points
from .batching import (
    DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    _flight_diagnostic_summary,
    _run_per_electron_transport,
    _run_per_electron_transport_lut,
    pack_layer_tables,
    resolve_transport_core,
)
from .cores import (
    _transport_core_grooved,
    _transport_core_ungrooved,
    _transport_core_ungrooved_lut,
    _transport_core_ungrooved_perelectron,
    _transport_core_ungrooved_perelectron_lut,
)
from .kinematics import _sample_bunch_offsets, stream_keys
from .lut import DEFAULT_TRANSPORT_LUT_CONFIG, build_transport_energy_lut
from .scattering import _NO_MOTT, _mott_alpha_table
from .stopping import _element_crossover_keV

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
    energy_model="frozen",
    max_dE_frac=0.0,
    straggling=False,
):
    """
    Transport Ne electrons of energy E0_keV [keV] into a slab 0<=z<=thickness.
    Beam enters at the origin along +z. Electrons terminate when they exit
    either surface or drop below E_cut_keV (segments below the cutoff don't
    radiate in the spectral window of interest anyway).

    Full derivations, sources, and limiting-case checks for the physics below
    live in ``docs/physics/beam-transport/*.md`` and are independently verified
    in ``docs/validation/beam-transport/*.md`` and
    ``docs/validation/geometry/*.md``; this docstring states only the
    parameter contract and the BIT-FOR-BIT
    limiting case each one must preserve.

    elastic_model: "mott" (default, NIST SRD 64 Mott transport cross section
      via a Browning fit) or "sr" (analytic screened-Rutherford, no data
      files). See docs/physics/beam-transport/elastic-scattering.md.
      Validation: electron-transport

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

    transport_core: which ungrooved core runs the electrons. "auto" (default) --
    the CUDA core when this process has a CUDA device, the run is ungrooved,
    and Ne > CUDA_TRANSPORT_MIN_ELECTRONS; the lockstep core otherwise (see
    `resolve_transport_core`; pin with `CXR_MC_TRANSPORT_CORE`). "lockstep"
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
      "frozen" (default) -- the historical left-endpoint rule: stopping power
          and beta evaluated once at the flight's start energy and held
          constant over its whole length. BIT-FOR-BIT unchanged.
      "midpoint" -- second-order predictor-corrector for the implicit
          midpoint rule E_end = E_start + (dE/ds)((E_start+E_end)/2)*s, with
          the clock advanced by s/beta at that representative energy and the
          cutoff truncation distance solved for E_end == E_cut. Adds
          E_end_keV, t_end_ang, and E_repr_keV = (E_start+E_end)/2 (the
          energy radiation kernels evaluate the row at) to the returned rows.
          Elastic hazard stays frozen at the start energy; only stopping and
          the clock are controlled here. Lockstep core only -- any other
          core or a grooved run raises rather than returning the frozen
          schema.
    See docs/validation/beam-transport/transport-midpoint-stopping.md.
    Validation: transport-midpoint-stopping

    max_dE_frac: numerical cap on one row's fractional energy loss, splitting
      a physical flight into substeps when the cap binds before any physical
      event. 0.0 (default) disables substepping, leaving one row per flight.
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
    spectrum backend must then also be CUDA (CXR_MC_BACKEND resolving to
    CUDA) -- NumPy kernels refuse the implicit host conversion rather than
    performing it silently. Segments are joined with one concatenate (holds
    two copies transiently) and stay device-resident for as long as the
    caller keeps the dict.

    Returns dict of per-segment arrays (schema, including E_start_keV/
    t_start_ang canonical aliases and the energy_model="midpoint" additions:
    see docs/physics/beam-transport/transport-outputs.md), incident
    phase-space diagnostics (one row per sampled electron, including missed
    entries: initial_r_ang, initial_v_hat, initial_E_keV, initial_t0_ang),
    and non-radiating groove-gap "vacuum_*" arrays. Also returns the scalar
    diagnostics n_backscattered, n_transmitted, n_side_exited, n_missed,
    n_cutoff_stopped, n_step_limited, n_stopped (compatibility alias of
    n_cutoff_stopped), and n_layers. Incomplete histories raise RuntimeError
    rather than returning these arrays/counts.

    Validation: electron-transport, energy-loss-straggling, finite-beam-size,
    finite-transverse-crystal, grazing-beam-projection, multilayer-stack
    """
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

    if elastic_model not in ("mott", "sr"):
        raise ValueError("elastic_model must be 'mott' or 'sr'")
    if energy_model not in ("frozen", "midpoint"):
        raise ValueError("energy_model must be 'frozen' or 'midpoint'")

    requested_core = transport_core
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

    # NVTX ranges for the transport phase. Everything here is host-side, but a
    # CUDA run's wall clock is not: Round 4 left a 38-52% unattributed remainder
    # around the transport driver, and these are what a capture needs to split it
    # into table build / beam sampling / buffer reservation / core / output.
    # No-op off the profiled GPU path. Lazy import: runner imports this module.
    from ..runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.tables")
    z_total = float(layers[-1][1])
    n_layers = len(layers)
    L_Zs = []
    L_Js = []
    L_ncm3 = []
    L_ks = []
    L_coeffs = []
    L_E_cross = []
    mott_tables = []

    for _, _, lc in layers:
        elements = []
        ncm3_arr = []
        Z_arr = []
        J_arr = []
        k_arr = []
        coeff_arr = []
        E_cross_arr = []
        layer_mott_tables = []

        for el, n_i in lc:
            elements.append(el)
            params = TRANSPORT_ELEMENTS[el]
            Z_i = float(params["Z"])
            A_i = float(params["A"])
            J_i = float(params["J_keV"])
            k_i = 0.731 + 0.0688 * np.log10(Z_i)
            coeff_i = (n_i / 0.602214076) * Z_i

            ncm3_arr.append(n_i * 1e24)
            Z_arr.append(Z_i)
            J_arr.append(J_i)
            k_arr.append(k_i)
            coeff_arr.append(coeff_i)
            E_cross_arr.append(_element_crossover_keV(el, Z_i, A_i, J_i))

            table = None
            if elastic_model == "mott" and el not in _NO_MOTT:
                try:
                    table = _mott_alpha_table(el, Z_i)
                except FileNotFoundError:
                    logger.debug("No NIST Mott table for %s; using analytic SR angles", el)
                    _NO_MOTT.add(el)
            layer_mott_tables.append(table)

        L_Zs.append(np.asarray(Z_arr, dtype=float))
        L_Js.append(np.asarray(J_arr, dtype=float))
        L_ncm3.append(np.asarray(ncm3_arr, dtype=float))
        L_ks.append(np.asarray(k_arr, dtype=float))
        L_coeffs.append(np.asarray(coeff_arr, dtype=float))
        L_E_cross.append(np.asarray(E_cross_arr, dtype=float))
        mott_tables.append(layer_mott_tables)

    L_sr_rate_numer = []
    L_mott_numer = []
    L_mott_denom1 = []
    L_mott_denom2 = []
    L_sr_joy_numer = []

    for i, Z_i in enumerate(L_Zs):
        n_cm3_i = L_ncm3[i]
        # Rutherford Scattering coefficient hoisted out of hot loop
        L_sr_rate_numer.append(5.21e-21 * Z_i * Z_i * np.float64(4.0) * np.float64(np.pi) * n_cm3_i)

        # Browning fit coefficients to Mott scattering hoisted out of hot loop
        z17 = Z_i ** np.float64(1.7)
        L_mott_numer.append(np.float64(3.0e-18) * z17 * n_cm3_i)
        L_mott_denom1.append(np.float64(0.005) * z17)
        L_mott_denom2.append(np.float64(0.0007) * Z_i * Z_i)

        # Joy-Luo
        L_sr_joy_numer.append(np.float64(3.4e-3) * Z_i ** np.float64(0.67))

    L_top = np.asarray([float(a) for (a, _, _) in layers], dtype=float)
    L_bot = np.asarray([float(b) for (_, b, _) in layers], dtype=float)
    internal_bounds = L_bot[:-1].copy()

    # Numba only sees numeric Mott data. Tables are flattened because different
    # elements may have different grid lengths; start/length locate each table.
    max_elements = max(arr.size for arr in L_Zs)
    mott_has_table = np.zeros((n_layers, max_elements), dtype=np.bool_)
    mott_start = np.zeros((n_layers, max_elements), dtype=np.int64)
    mott_len = np.zeros((n_layers, max_elements), dtype=np.int64)
    mott_logE_chunks = []
    mott_logA_chunks = []
    offset = 0
    for L, layer_tables in enumerate(mott_tables):
        for i_el, table in enumerate(layer_tables):
            if table is None:
                continue
            logE, logA = table
            logE = np.asarray(logE, dtype=float)
            logA = np.asarray(logA, dtype=float)
            if logE.size != logA.size or logE.size == 0:
                raise ValueError("invalid Mott interpolation table")
            mott_has_table[L, i_el] = True
            mott_start[L, i_el] = offset
            mott_len[L, i_el] = logE.size
            mott_logE_chunks.append(logE)
            mott_logA_chunks.append(logA)
            offset += logE.size

    if mott_logE_chunks:
        mott_logE_flat = np.concatenate(mott_logE_chunks)
        mott_logA_flat = np.concatenate(mott_logA_chunks)
    else:
        mott_logE_flat = np.empty(0, dtype=float)
        mott_logA_flat = np.empty(0, dtype=float)
    _nsys_pop()

    _nsys_push("cxr.transport.sample")
    rng = np.random.default_rng(seed)
    pos = np.zeros((Ne, 3))
    transverse_slopes = None
    if transverse_distribution is not None:
        if beam_fwhm_mm or beam_fwhm_y_mm:
            raise ValueError("transverse_distribution is incompatible with the spot FWHM fields")
        # The Twiss policy owns BOTH transverse moments: positions here and the
        # correlated slopes fed to `dirs` below. Splitting them across two
        # sources would break the <x x'> correlation the emittance encodes.
        # Its own child stream (spawn(5)[4], the next index after the bunch's
        # spawn(4)[3]) keeps the main free-path / scattering draws untouched.
        MM_TO_ANG = 1.0e7
        resolved = resolved_from_mapping(transverse_distribution)
        transverse_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[4])
        x_mm, x_prime, y_mm, y_prime = sample_transverse(resolved, Ne, transverse_rng)
        transverse_slopes = (x_prime, y_prime)
        offsets = np.stack((x_mm * MM_TO_ANG, y_mm * MM_TO_ANG), axis=1)
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif beam_fwhm_mm or beam_fwhm_y_mm:
        # independent child stream: does not consume from `rng`, so the main
        # transport draws (free path, scattering angle) are untouched -- see
        # the beam_fwhm_mm docstring paragraph above for the invariance this
        # buys.
        MM_TO_ANG = 1.0e7
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # Per-plane elliptical spot (decision 8). y defaults to x, so an
        # isotropic beam draws sigma_x == sigma_y and stays bit-for-bit with the
        # historical scalar-spot path: normal(0,1,(Ne,2)) scaled by a scalar sigma
        # IS normal(0,sigma,(Ne,2)) element-for-element and consumes the stream
        # identically.
        fwhm_y = beam_fwhm_mm if beam_fwhm_y_mm is None else beam_fwhm_y_mm
        sigma_x = float(beam_fwhm_mm or 0.0) * MM_TO_ANG * fwhm_to_sigma
        sigma_y = float(fwhm_y or 0.0) * MM_TO_ANG * fwhm_to_sigma
        beam_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        offsets = beam_rng.normal(0.0, 1.0, size=(Ne, 2))
        offsets[:, 0] *= sigma_x
        offsets[:, 1] *= sigma_y
        # Project the lab-frame Gaussian spot onto the tilted sample entrance
        # face (grazing-incidence footprint elongation). tilt=0 -> (u, v)
        # bit-for-bit, so the untilted beam draw is unchanged.
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif groove is not None:
        # Groove effects depend on x mod spacing; a point source samples one
        # phase only. Draw the lateral phase uniformly over one period from an
        # independent child stream (same spawn pattern as beam_rng, so the
        # main free-path/scattering draws are untouched).
        phase_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        pos[:, 0] = phase_rng.uniform(0.0, groove.spacing_ang, size=Ne)
    if groove is not None:
        # Slide each ray along the beam to its relief-facet entry point.
        # Later flights use exact groove exit/re-entry events below.
        # Validation: blazed-groove-geometry
        x_e, z_e = entry_points(pos[:, 0], groove)
        pos[:, 0] = x_e
        pos[:, 2] = z_e
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
    if not np.all(E_cut_by_electrons < E_keV):
        raise ValueError("each electron cutoff energy must be below its initial energy")
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
    # Snapshot before transport mutates ``pos``, ``dirs``, and ``E``. These
    # arrays describe incident phase space, including particles that miss a
    # finite footprint.
    initial_r_ang = pos.copy()
    initial_v_hat = dirs.copy()
    initial_E_keV = E_keV.copy()
    _nsys_pop()

    elastic_model_code = 1 if elastic_model == "mott" else 0
    transport_lut = None
    if groove is None and transport_lut_config.enabled:
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
        )
        _nsys_pop()

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
    _nsys_pop()

    # Where the segments end up living, and so which array module assembles the
    # output below. NumPy unless the run asked to keep them on the device.
    seg_xp = np
    dev_segs = None
    energy_model_code = 1 if energy_model == "midpoint" else 0

    # Straggling (slice D). ``straggle_on`` is a plain bool -- every core
    # branches on it before touching the Urban sampler's own stream, so False
    # (the default) costs nothing beyond the Ne-sized zero allocations below
    # and is BIT-FOR-BIT with a run compiled before this parameter existed.
    # ``stragg_stream_keys`` feeds the lockstep/grooved cores, which have no
    # per-electron counter stream of their own (see the module-level "counter-
    # based per-electron RNG" comment); the per-electron cores instead reuse
    # their own ``stream_key``/``d_keys`` directly. ``stragg_layer_tables``
    # supplies the per-electron LUT core with the per-element split the LUT
    # itself does not carry (see `_transport_core_ungrooved_perelectron_lut`).
    straggle_on = bool(straggling)
    stragg_dE = np.zeros(Ne) if straggle_on else np.zeros(0)
    stragg_stream_keys = stream_keys(seed, Ne) if straggle_on else np.zeros(1, dtype=np.uint64)
    if straggle_on:
        _stragg_packed = pack_layer_tables(
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
        stragg_layer_tables = _stragg_packed[:5]
    else:
        _stragg_dummy = np.zeros((1, 1), dtype=np.float64)
        stragg_layer_tables = (
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
        )

    _nsys_push("cxr.transport.core")
    if groove is None and transport_lut is not None and transport_core != "lockstep":
        if transport_core == "cuda":
            if straggle_on:
                # Straggling is wired into the CUDA per-electron *exact* kernel
                # (_transport_kernel/run_transport_kernel) only: slice D
                # duplicated the sampler there and slice F applied the loss.
                # The LUT CUDA kernel (_transport_lut_kernel) has no
                # per-element split to sample from (see
                # _transport_core_ungrooved_perelectron_lut's docstring), no
                # duplicated sampler, and `run_transport_lut_kernel` does not
                # even accept the straggling parameters. Slice F deliberately
                # left it that way rather than adding a second ~150-line
                # transcription of the sampler to a kernel that cannot be
                # compiled or run on the machine writing it -- see the slice F
                # checklist entry. Raise rather than silently returning an
                # unstraggled result or an opaque TypeError. Use
                # transport_lut_config=TransportLUTConfig(enabled=False) to
                # reach the exact CUDA kernel instead, or transport_core=
                # "per-electron"/"lockstep" off CUDA.
                raise NotImplementedError(
                    "straggling=True is not implemented on the CUDA LUT core "
                    "(transport_core='cuda' with the LUT enabled); disable the "
                    "LUT (transport_lut_config=TransportLUTConfig(enabled=False)) "
                    "to reach the CUDA exact per-electron core, or run off CUDA "
                    "with transport_core='per-electron' or 'lockstep'"
                )
            from ..transport_jit_kernel import make_cuda_transport_lut_core

            core, core_xp = make_cuda_transport_lut_core()
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
                stragg_layer_tables,
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
            )
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None and transport_lut is not None:
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_ungrooved_lut(
            Ne,
            alive,
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
            clock,
            rng,
            pos,
            dirs,
            E_cut_by_electrons,
            transport_lut.n_el,
            L_top,
            L_bot,
            transport_lut.E_min_keV,
            transport_lut.inv_dE_keV,
            transport_lut.n_energy,
            transport_lut.total_rate,
            transport_lut.dEds,
            transport_lut.inv_beta,
            transport_lut.cdf,
            transport_lut.alpha,
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
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None and transport_core != "lockstep":
        # Per-electron streams and run-to-completion ordering. Not bit-for-bit
        # with the lockstep core -- see `_transport_core_ungrooved_perelectron`.
        if transport_core == "cuda":
            from ..transport_jit_kernel import make_cuda_transport_core

            core, core_xp = make_cuda_transport_core()
        else:
            core, core_xp = _transport_core_ungrooved_perelectron, np
        if keep_segments_on_device:
            seg_xp = core_xp

        packed_per_layer_tables = pack_layer_tables(
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
                (mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat),
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
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
            )
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None:
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_ungrooved(
            Ne,
            alive,
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
            clock,
            rng,
            pos,
            dirs,
            E_cut_by_electrons,
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
            L_top,
            L_bot,
            mott_has_table,
            mott_start,
            mott_len,
            mott_logE_flat,
            mott_logA_flat,
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
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
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
            alive,
            max_steps,
            max_segments,
            max_vac,
            n_layers,
            internal_bounds,
            elastic_model_code,
            energy_model_code,
            max_dE_frac,
            z_total,
            finite_footprint,
            0.0 if width_ang is None else float(width_ang),
            0.0 if height_ang is None else float(height_ang),
            float(groove.spacing_ang),
            float(groove.depth_ang),
            groove_st,
            groove_ct,
            clock,
            rng,
            pos,
            dirs,
            E_cut_by_electrons,
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
            L_top,
            L_bot,
            mott_has_table,
            mott_start,
            mott_len,
            mott_logE_flat,
            mott_logA_flat,
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
            vac_start_buf,
            vac_end_buf,
            vac_E_buf,
            vac_t0_buf,
            vac_id_buf,
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
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
            seg_E_end, seg_t_end, seg_flight, seg_substep = dev_segs[7:]

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
    if straggle_on:
        # The summed per-electron Urban-SAMPLED loss. Applied to the electron's
        # energy only on the ungrooved lockstep exact core (slice E); diagnostic
        # only on every other core until slice F. See the ``straggling``
        # paragraph in this function's docstring.
        result["straggle_dE_keV"] = stragg_dE
    if energy_model == "midpoint":
        result["E_end_keV"] = seg_E_end[:nseg]
        result["t_end_ang"] = seg_t_end[:nseg]
        # The propagator's own representative energy: the implicit midpoint rule
        # evaluates stopping and beta at (E_start + E_end)/2, so radiation and
        # quadrature consumers read that value instead of re-deriving one.
        result["E_repr_keV"] = 0.5 * (E_seg + seg_E_end[:nseg])
        # `(electron_id, flight_id)` is the stable physical key; `substep_id`
        # indexes numerical rows inside one flight and is integration detail.
        result["flight_id"] = seg_flight[:nseg]
        result["substep_id"] = seg_substep[:nseg]
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
        )
    return result
