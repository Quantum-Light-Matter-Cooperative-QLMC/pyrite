"""Initial electron positions on the sample entrance face.

Split out of :func:`pyrite.montecarlo.transport.api.simulate_trajectories`.
Every draw here lives in its own ``SeedSequence(seed)`` child namespace, so the
main free-path and scattering stream is untouched. Spot and Twiss draws are
counter-addressed per electron inside that namespace (#361).
"""

import numpy as np

from ..geometry import project_beam_entry
from ..groove import entry_points
from ..transverse import resolved_from_mapping, sample_transverse
from .kinematics import child_stream_root, counter_normals


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
):
    """Return ``(pos, transverse_slopes)`` for electrons ``[start, start + Ne)``.

    Spot and Twiss draws are counter-addressed per electron inside their child
    namespaces, so a block ``[start, stop)`` equals that slice of one larger
    draw. The groove phase stays on its sequential child generator; grooved
    transport is lockstep-only and never runs in electron blocks.

    ``transverse_slopes`` is ``(x', y')`` from a Twiss ``transverse_distribution``,
    else ``None``. See ``simulate_trajectories`` for the beam-spot, Twiss and
    groove-phase conventions.
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
    return pos, transverse_slopes


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
