"""Initial electron positions on the sample entrance face.

Split out of :func:`pyrite.montecarlo.transport.api.simulate_trajectories`.
Every draw here uses its own ``SeedSequence(seed)`` child stream, so the main
free-path and scattering stream is untouched and results are bit-for-bit those
of the inline code.
"""

import numpy as np

from ..geometry import project_beam_entry
from ..groove import entry_points
from ..transverse import resolved_from_mapping, sample_transverse


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
):
    """Return ``(pos, transverse_slopes)`` for ``Ne`` electrons.

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
    return pos, transverse_slopes
