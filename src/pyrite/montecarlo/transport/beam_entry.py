"""Initial electron positions on the sample entrance face.

Split out of :func:`pyrite.montecarlo.transport.api.simulate_trajectories`.
Every draw here lives in its own ``SeedSequence(seed)`` child namespace, so the
main free-path and scattering stream is untouched. Spot and Twiss draws are
counter-addressed per electron inside that namespace (#361).
"""

import numpy as np

from ..bunch_front import resolve_front_tilt
from ..geometry import (
    beam_entry_jacobian,
    beam_frame_basis,
    face_arrival_path_ang,
    project_beam_entry,
)
from ..groove import entry_points
from ..transverse import resolved_from_mapping, sample_transverse
from .kinematics import beta_from_keV, child_stream_root, counter_normals


def initial_beam_positions(
    seed,
    Ne,
    *,
    transverse_distribution,
    beam_fwhm_mm,
    beam_fwhm_y_mm,
    tilt_polar_rad,
    tilt_azim_rad,
    groove,
    start=0,
    E0_keV=None,
    energy_spread_frac=None,
    bunch_front_tilt=None,
):
    """Return ``(pos, transverse_slopes, beam_entry)`` for electrons ``[start, start + Ne)``.

    Spot and Twiss draws are counter-addressed per electron inside their child
    namespaces, so a block ``[start, stop)`` equals that slice of one larger
    draw. The groove phase stays on its sequential child generator; grooved
    transport is lockstep-only and never runs in electron blocks.

    ``transverse_slopes`` is ``(x', y')`` from a Twiss ``transverse_distribution``,
    else ``None``. ``beam_entry`` is :func:`beam_entry_record` for nominal
    energy ``E0_keV`` (``None`` when it is not given). See
    ``simulate_trajectories`` for the beam-spot, Twiss and groove-phase
    conventions.
    """
    pos = np.zeros((Ne, 3))
    transverse_slopes = None
    if transverse_distribution is not None:
        if beam_fwhm_mm or beam_fwhm_y_mm:
            raise ValueError("transverse_distribution is incompatible with the spot FWHM fields")
        # The Twiss policy owns BOTH transverse moments: positions here and the
        # correlated slopes the caller turns into directions. Splitting them across two
        # sources would break the <x x'> correlation the emittance encodes.
        # Its own child stream (spawn(5)[4], the next index after the bunch's
        # spawn(4)[3]) keeps the main free-path / scattering draws untouched.
        MM_TO_ANG = 1.0e7
        resolved = resolved_from_mapping(transverse_distribution)
        x_mm, x_prime, y_mm, y_prime = sample_transverse(resolved, Ne, seed, start=start)
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
        # Per-plane elliptical spot (decision 8). y defaults to x; both widths
        # scale the same per-electron standard normals (draw 0 -> x, 1 -> y),
        # so an isotropic beam is bit-for-bit the scalar-width spot.
        fwhm_y = beam_fwhm_mm if beam_fwhm_y_mm is None else beam_fwhm_y_mm
        sigma_x = float(beam_fwhm_mm or 0.0) * MM_TO_ANG * fwhm_to_sigma
        sigma_y = float(fwhm_y or 0.0) * MM_TO_ANG * fwhm_to_sigma
        offsets = counter_normals(child_stream_root(seed, 2, 1), Ne, 2, start=start)
        offsets[:, 0] *= sigma_x
        offsets[:, 1] *= sigma_y
        # Project the lab-frame Gaussian spot onto the tilted sample entrance
        # face (grazing-incidence footprint elongation). tilt=0 -> (u, v)
        # bit-for-bit, so the untilted beam draw is unchanged.
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif groove is not None:
        if start:
            raise ValueError("grooved beam entry does not support electron blocks")
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
    record = None
    if E0_keV is not None:
        record = beam_entry_record(
            transverse_distribution=transverse_distribution,
            beam_fwhm_mm=beam_fwhm_mm,
            beam_fwhm_y_mm=beam_fwhm_y_mm,
            tilt_polar_rad=tilt_polar_rad,
            tilt_azim_rad=tilt_azim_rad,
            E0_keV=E0_keV,
            energy_spread_frac=energy_spread_frac,
            groove=groove,
            bunch_front_tilt=bunch_front_tilt,
        )
    return pos, transverse_slopes, record


def beam_entry_record(
    *,
    transverse_distribution,
    beam_fwhm_mm,
    beam_fwhm_y_mm,
    tilt_polar_rad,
    tilt_azim_rad,
    E0_keV,
    energy_spread_frac,
    groove,
    bunch_front_tilt=None,
):
    """Plain-data description of the analytic incident spot, for coherent averages.

    ``spot_covariance_ang2`` is the lab ``(u, v)`` position covariance [Ang^2]
    of the sampled Gaussian spot (``diag(sigma_x^2, sigma_y^2)``; Twiss
    ``sigma^2 = eps beta``), or ``None`` without an analytic spot.
    ``face_arrival_delay`` marks that each electron's ``t0`` includes its
    vacuum flight ``s*/beta`` to the tilted face (:func:`face_arrival_delay_ang`).
    Tuples and scalars only: electron-block joins take metadata from the first
    block, and trajectory artifacts store it as JSON.

    Validation: transverse-bunch-form-factor
    """
    MM2_TO_ANG2 = 1.0e14
    covariance = None
    correlated = False
    if transverse_distribution is not None:
        resolved = resolved_from_mapping(transverse_distribution)
        covariance = (
            (resolved.x.sigma_position_mm**2 * MM2_TO_ANG2, 0.0),
            (0.0, resolved.y.sigma_position_mm**2 * MM2_TO_ANG2),
        )
        correlated = bool(resolved.x.alpha_twiss or resolved.y.alpha_twiss)
    elif beam_fwhm_mm or beam_fwhm_y_mm:
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        fwhm_y = beam_fwhm_mm if beam_fwhm_y_mm is None else beam_fwhm_y_mm
        sigma_x = float(beam_fwhm_mm or 0.0) * fwhm_to_sigma
        sigma_y = float(fwhm_y or 0.0) * fwhm_to_sigma
        covariance = ((sigma_x**2 * MM2_TO_ANG2, 0.0), (0.0, sigma_y**2 * MM2_TO_ANG2))
    record = {
        "tilt_polar_rad": float(tilt_polar_rad or 0.0),
        "tilt_azim_rad": float(tilt_azim_rad or 0.0),
        "E0_keV": float(E0_keV),
        "energy_spread": bool(energy_spread_frac),
        "spot_covariance_ang2": covariance,
        "position_slope_correlated": correlated,
        "face_arrival_delay": covariance is not None and groove is None,
    }
    front = resolve_front_tilt(bunch_front_tilt)
    if front is not None:
        record["bunch_front_gradient"] = tuple(front.gradient(E0_keV))
    return record


def bunch_front_delay_ang(record, initial_r_ang):
    """Imposed front delay c*t [Ang] at each parallel ray's incoming position.

    Source: equal-time plane kinematics, c*t=a.w, a=tan(theta)u/beta0.
    Invert the face projection p_xy=Jw; the geometric face-flight delay is
    added separately exactly once. Assumes monoenergetic parallel rays and
    the recorded lab gradient. No gradient/point spot gives exact zeros.
    Validation: linear-bunch-front-tilt
    """
    positions = np.asarray(initial_r_ang, dtype=float)
    if not record or "bunch_front_gradient" not in record:
        return np.zeros(positions.shape[0])
    J = beam_entry_jacobian(record["tilt_polar_rad"], record["tilt_azim_rad"])
    gradient_face = np.linalg.solve(J.T, np.asarray(record["bunch_front_gradient"]))
    return positions[:, :2] @ gradient_face


def face_arrival_delay_ang(record, initial_r_ang, initial_E_keV):
    """Per-electron vacuum flight time ``s*/beta`` [Ang, c=1] to the entry face.

    ``t0`` is the bunch time at the lab plane through the sample origin
    perpendicular to the beam; electron ``e`` reaches its tilted-face point
    ``s*_e / beta_e`` later (:func:`~pyrite.montecarlo.geometry.face_arrival_path_ang`).
    The pulse front is then perpendicular to the beam, not parallel to the
    face. Zero tilt, no analytic spot, or a groove give zeros, so those
    inputs keep their historical ``t0`` bit for bit. GDF beams already carry
    their own drift time.

    Source: ray-plane intersection plus constant-velocity vacuum flight.
    Validation: grazing-beam-projection, transverse-bunch-form-factor
    """
    initial_r_ang = np.asarray(initial_r_ang, dtype=float)
    n_electrons = initial_r_ang.shape[0]
    if not record or not record["face_arrival_delay"] or not record["tilt_polar_rad"]:
        return np.zeros(n_electrons)
    path = face_arrival_path_ang(
        initial_r_ang[:, :2], record["tilt_polar_rad"], record["tilt_azim_rad"]
    )
    return path / beta_from_keV(np.asarray(initial_E_keV, dtype=float))


def resolve_bunch_times(t0_ang, clock, record, pos, E_keV, *, gdf_t0=None, launch=None):
    """``(t0, clock, beam_entry)`` after GDF, secondary-launch or analytic-spot entry.

    GDF beams and secondary launches carry their own arrival times and no
    analytic spot. An analytic spot adds :func:`face_arrival_delay_ang` (exact
    zeros untilted), so a tilted face meets an oblique pulse front.

    Validation: transverse-bunch-form-factor
    """
    if gdf_t0 is not None:
        return gdf_t0, clock, None
    if launch is not None:
        return launch.t0_ang.copy(), launch.t_ang.copy(), None
    return (
        (t0_ang + bunch_front_delay_ang(record, pos) + face_arrival_delay_ang(record, pos, E_keV)),
        clock,
        record,
    )


def initial_energies_keV(E0_keV, Ne, seed, energy_spread_frac, *, start=0):
    """Incident kinetic energies [keV] of electrons ``[start, start + Ne)``.

    ``E_e = E0 (1 + f z_e)`` with ``z_e`` draw 0 of electron ``e``'s counter
    stream in the ``spawn(6)[5]`` child namespace -- RMS *relative* deviation,
    uncorrelated with arrival time (decision 3: no chirp model), and a pure
    function of ``(seed, e)``. ``f`` unset or 0 gives the monoenergetic beam
    bit for bit. Raises rather than returning a non-positive energy.

    Validation: beam-energy-spread-injection
    """
    E_keV = np.full(Ne, float(E0_keV))
    if energy_spread_frac:
        spread = counter_normals(child_stream_root(seed, 6, 5), Ne, 1, start=start)
        E_keV = E_keV * (1.0 + float(energy_spread_frac) * spread[:, 0])
        if not np.all(np.isfinite(E_keV)) or not np.all(E_keV > 0.0):
            raise ValueError(
                f"energy_spread_frac={energy_spread_frac} drew a non-finite or non-positive "
                "electron energy; the Gaussian spread model needs spread << 1"
            )
    return E_keV


def check_electron_block(start, cascade_or_geometry, beam_inputs):
    """Reject inputs an electron block (``start > 0``) cannot reproduce (#361).

    Secondary cascades, grooves and GDF beams are not block-addressed, and
    bunch offsets are centred on the realized population, so the block driver
    samples them once after the join.
    """
    if int(start) < 0:
        raise ValueError("_electron_start must be non-negative")
    threshold, pair_model, positrons, secondary, groove = cascade_or_geometry
    blocked = (threshold, pair_model, secondary, groove, *beam_inputs)
    if int(start) and (positrons or any(value is not None for value in blocked)):
        raise ValueError(
            "electron blocks support neither secondary cascades, grooves, GDF beams, "
            "nor in-block bunch sampling"
        )


def table_energy_range(E_cut_by_electrons, E_keV, override=None):
    """``(lo, hi)`` keV range the shell, radiative and LUT tables are built over.

    Defaults to this call's ``(min cutoff, max initial energy)``. A block passes
    its population's range so its tables -- and their floating-point
    interpolation -- are exactly those of the single ``[0, N)`` call.
    """
    lo, hi = float(np.min(E_cut_by_electrons)), float(np.max(E_keV))
    if override is None:
        return lo, hi
    range_lo, range_hi = (float(v) for v in override)
    if range_lo > lo or range_hi < hi:
        raise ValueError("_energy_range_keV must cover every transported electron")
    return range_lo, range_hi


def normalized_beam_dir(beam_dir):
    """Validated unit mean beam direction in the slab frame (``None``: ``+z``)."""
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
    return beam_dir


def primary_directions(beam_dir, transverse_slopes, Ne):
    """Per-electron incident directions ``(Ne, 3)`` about the unit ``beam_dir``."""
    if transverse_slopes is None:
        return np.tile(beam_dir, (Ne, 1))
    # Slopes are dx/dz, dy/dz about the beam axis, so the per-electron
    # direction is (x', y', 1) in the beam frame. Zero emittance gives
    # exactly `beam_dir` back, which is what makes the collimated limit
    # bit-for-bit rather than merely close.
    x_prime, y_prime = transverse_slopes
    basis = beam_frame_basis(beam_dir)
    dirs = x_prime[:, None] * basis[:, 0] + y_prime[:, None] * basis[:, 1] + beam_dir
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def sample_primary_entry(
    E0_keV,
    Ne,
    seed,
    *,
    beam_dir=None,
    beam_fwhm_mm=None,
    beam_fwhm_y_mm=None,
    bunch_length_fs=None,
    long_shape="gaussian",
    long_offsets_fs=None,
    longitudinal_distribution=None,
    transverse_distribution=None,
    energy_spread_frac=None,
    bunch_front_tilt=None,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
):
    """Primary entrance states exactly as an ungrooved analytic beam samples them.

    The same pure functions of ``seed`` that :func:`simulate_trajectories`
    calls, composed without transport, so a caller can launch the primaries
    later from an explicit state (the channeling zone, #506) with identical
    draws. Returns ``(r_ang, v_hat, E_keV, t_ang, t0_ang, beam_entry)``:
    entrance-face positions and directions in the slab frame, kinetic
    energies, start clock and bunch arrival offsets [Angstrom, c=1], and the
    incident phase-space record.
    """
    from ..bunch_front import validate_front_source
    from .kinematics import _sample_bunch_offsets

    bunch_front_tilt = validate_front_source(
        bunch_front_tilt,
        transverse=transverse_distribution,
        energy_spread=energy_spread_frac,
        source="analytic",
        longitudinal=longitudinal_distribution,
        offsets=long_offsets_fs,
        shape=long_shape,
        groove=None,
    )
    pos, slopes, record = initial_beam_positions(
        seed,
        Ne,
        transverse_distribution=transverse_distribution,
        beam_fwhm_mm=beam_fwhm_mm,
        beam_fwhm_y_mm=beam_fwhm_y_mm,
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=None,
        E0_keV=E0_keV,
        energy_spread_frac=energy_spread_frac,
        bunch_front_tilt=bunch_front_tilt,
    )
    dirs = primary_directions(normalized_beam_dir(beam_dir), slopes, Ne)
    E_keV = initial_energies_keV(E0_keV, Ne, seed, energy_spread_frac)
    t0 = _sample_bunch_offsets(
        Ne, bunch_length_fs, long_shape, long_offsets_fs, seed, longitudinal_distribution
    )
    t0, clock, record = resolve_bunch_times(t0, np.zeros(Ne), record, pos, E_keV)
    return pos, dirs, E_keV, clock, t0, record
