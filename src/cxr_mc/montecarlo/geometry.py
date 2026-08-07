"""
montecarlo.geometry

Lab/sample-frame geometry shared by transport, spectrum and detector: the
tilted-sample beam/detector directions, the finite-detector-face direction
grid, the crystal-orientation and small-tilt rotations, and the Gauss-Hermite
mosaic-orientation quadrature.

Tilt convention (matches Zhai SI): the sample tilt is a spherical (theta, phi)
pair. Positive polar tilt (``tilt_polar_rad`` / ``tilt_deg``) tilts the slab
normal -- and, by construction, the reciprocal vector g (g || n by default,
see :func:`_orientation_R`) -- TOWARD the detector. Positive azimuth
(``tilt_azim_rad`` / ``tilt_azim_deg``) is a CCW roll about the beam +z axis,
reported in [0 deg, 180 deg]; azimuth 0 places the tilt in the scattering x-z
plane (zero y-component).
"""

import numpy as np
from numba import jit as njit

from ..materials.crystal import _direct_lattice_vectors, _rotation_between, reciprocal_g_vector

X_MIN = 0
X_MAX = 1
Y_MIN = 2
Y_MAX = 3
Z_MIN = 4
Z_MAX = 5


def validate_transverse_dimensions(width, height, *, unit) -> tuple[float | None, float | None]:
    """Validate a matched pair of full transverse crystal dimensions.

    Dimensions are either both omitted, preserving the laterally infinite slab,
    or both finite and strictly positive. ``unit`` labels validation failures so
    callers can use this shared check for public mm inputs and internal Angstrom
    geometry without changing their units.
    """
    if width is None and height is None:
        return None, None
    if width is None or height is None:
        raise ValueError(f"width and height must both be specified in {unit}")

    width, height = float(width), float(height)
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError(f"width and height must be finite and strictly positive in {unit}")
    return width, height


@njit(cache=True)
def _first_prism_exit_numba(
    r,
    d,
    z_min_ang,
    z_max_ang,
    finite_xy,
    width_ang,
    height_ang,
):
    n = r.shape[0]

    distances = np.empty(n, dtype=r.dtype)
    faces = np.empty(n, dtype=np.int8)

    if finite_xy:
        x_min_ang = -width_ang / 2
        x_max_ang = width_ang / 2
        y_min_ang = -height_ang / 2
        y_max_ang = height_ang / 2

    for i in range(n):
        rx = r[i, 0]
        ry = r[i, 1]
        rz = r[i, 2]

        dx = d[i, 0]
        dy = d[i, 1]
        dz = d[i, 2]

        best_t = np.inf
        best_face = Z_MIN

        if finite_xy:
            if dx != 0.0:
                t = (x_min_ang - rx) / dx
                if t > 0.0 and t < best_t:
                    best_t = t
                    best_face = X_MIN

                t = (x_max_ang - rx) / dx
                if t > 0.0 and t < best_t:
                    best_t = t
                    best_face = X_MAX

            if dy != 0.0:
                t = (y_min_ang - ry) / dy
                if t > 0.0 and t < best_t:
                    best_t = t
                    best_face = Y_MIN

                t = (y_max_ang - ry) / dy
                if t > 0.0 and t < best_t:
                    best_t = t
                    best_face = Y_MAX

        if dz != 0.0:
            t = (z_min_ang - rz) / dz

            if t > 0.0 and t < best_t:
                best_t = t
                best_face = Z_MIN

            t = (z_max_ang - rz) / dz

            if t > 0.0 and t < best_t:
                best_t = t
                best_face = Z_MAX

        distances[i] = best_t
        faces[i] = best_face

    return distances, faces


def _face_distance(coord, direction, boundary, mask, xp):
    safe_direction = xp.where(mask, direction, 1.0)
    distance = (boundary - coord) / safe_direction
    return xp.where(mask, distance, xp.inf)


def first_prism_exit(
    r,
    d,
    *,
    z_min_ang,
    z_max_ang,
    width_ang=None,
    height_ang=None,
    xp=np,
):
    """Return each inside-origin ray's first forward rectangular-prism face exit.

    The sample-frame prism is ``[-width/2, width/2] x [-height/2, height/2] x
    [z_min_ang, z_max_ang]``. Origins must be inside that prism, and rays with
    a zero component never nominate the corresponding parallel faces. When both
    transverse dimensions are ``None``, the limiting geometry is the original
    z-only slab. Face ties resolve to the lowest face constant.

    Validation: finite-transverse-crystal
    """
    width_ang, height_ang = validate_transverse_dimensions(
        width_ang,
        height_ang,
        unit="Ang",
    )

    # Fast CPU path: compiled scalar/vector loop.
    if xp is np:
        r = np.asarray(r)
        d = np.asarray(d)

        # The xp path accepts a single shared direction (shape (3,)) broadcast
        # over all origins -- mc_self_absorption passes exactly that. The numba
        # kernel indexes d[i, k], so materialise the broadcast as a stride-0
        # view before handing it over.
        if d.ndim == 1:
            d = np.broadcast_to(d, r.shape)

        if width_ang is None:
            finite_xy = False
            width_numba = 0.0
            height_numba = 0.0
        else:
            finite_xy = True
            width_numba = width_ang
            height_numba = height_ang

        return _first_prism_exit_numba(
            r,
            d,
            z_min_ang,
            z_max_ang,
            finite_xy,
            width_numba,
            height_numba,
        )

    # GPU/backend-array path: stay entirely in xp.
    r = xp.asarray(r)
    d = xp.asarray(d)

    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    if d.ndim == 1:
        dx = d[0]
        dy = d[1]
        dz = d[2]
    else:
        dx = d[..., 0]
        dy = d[..., 1]
        dz = d[..., 2]

    # z faces always exist.
    s_zmin = _face_distance(z, dz, z_min_ang, dz < 0.0, xp)
    s_zmax = _face_distance(z, dz, z_max_ang, dz > 0.0, xp)

    if (width_ang is None) or (height_ang is None):
        # Only Z_MIN / Z_MAX are candidates.
        candidates = xp.stack(
            (
                s_zmin,
                s_zmax,
            ),
            axis=0,
        )

        which = xp.argmin(candidates, axis=0)
        exit_distance = xp.min(candidates, axis=0)

        # Candidate indices 0,1 correspond to face constants Z_MIN,Z_MAX.
        exit_face = xp.where(which == 0, Z_MIN, Z_MAX)

    else:
        x_min = -0.5 * width_ang
        x_max = +0.5 * width_ang
        y_min = -0.5 * height_ang
        y_max = +0.5 * height_ang

        s_xmin = _face_distance(x, dx, x_min, dx < 0.0, xp)
        s_xmax = _face_distance(x, dx, x_max, dx > 0.0, xp)
        s_ymin = _face_distance(y, dy, y_min, dy < 0.0, xp)
        s_ymax = _face_distance(y, dy, y_max, dy > 0.0, xp)

        candidates = xp.stack(
            (
                s_xmin,  # X_MIN = 0
                s_xmax,  # X_MAX = 1
                s_ymin,  # Y_MIN = 2
                s_ymax,  # Y_MAX = 3
                s_zmin,  # Z_MIN = 4
                s_zmax,  # Z_MAX = 5
            ),
            axis=0,
        )

        exit_face = xp.argmin(candidates, axis=0)
        exit_distance = xp.min(candidates, axis=0)

    return exit_distance, exit_face


def tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0):
    """
    Sample-frame beam and detector directions for a TILTED sample.

    Transport and radiation work in the sample frame (slab normal and crystal
    construction frame along +z, e.g. the HOPG c-axis). In the lab the beam
    is fixed along +z_lab and the detector sits at polar angle theta_obs_rad,
    azimuth 0. Tilting the sample so its normal points along
    (sin tp cos ta, sin tp sin ta, cos tp) in the lab (tp = tilt_polar_rad,
    ta = tilt_azim_rad) is equivalent to rotating the beam and detector into
    the sample frame.

    Convention (matches Zhai SI): POSITIVE tp tilts the slab normal -- and
    therefore the reciprocal vector g (g || n by default) -- TOWARD the
    detector. POSITIVE ta is a CCW roll about the beam +z axis, reported over
    [0 deg, 180 deg]; ta = 0 places the tilt in the scattering x-z plane, i.e.
    the normal has zero y-component.

    The starting production dispersion relation is
    ``omega = v0.g / (1 - v0.n_hat)`` (Zhai SI Eq. 10). For the beam-aligned,
    ``g || normal``, ``ta = 0`` default, both ``v0.n_hat`` and
    ``v0.g = beta |g| cos(tp)`` are even under ``tp -> -tp``. Thus neither
    zero-scattering scalar causes the observed full-model opposite-tilt
    intensity asymmetry; that can arise from direction-sensitive transport,
    escape, polarization/amplitude, or non-aligned reciprocal-vector effects.
    The reciprocal-harmonic sign mapping in the production numerator remains
    unresolved; this statement does not adjudicate it.

    Returns (beam_dir, n_hat) to pass to simulate_trajectories(beam_dir=...)
    and mc_spectrum(n_hat=...). For ta = 0 the detector's sample-frame polar
    angle is simply theta_obs - tilt_polar; the lab-frame quantity
    1 - v0.n_hat (hence the zero-scattering line energy denominator) is
    tilt-invariant. In the limiting case ``tilt_polar_rad = 0``, the untilted
    beam and detector directions are recovered.

    Validation: line-energy-dispersion
    """
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)
    beam_dir = R.T @ np.array([0.0, 0.0, 1.0])
    n_hat = R.T @ np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    return beam_dir, n_hat


def sample_to_lab_R(tilt_polar_rad, tilt_azim_rad=0.0):
    """Rotation ``R`` with ``v_lab = R @ v_sample``, the SAME rotation
    :func:`tilted_geometry` builds internally (sample construction ``+z`` ->
    the tilted slab normal ``normal_lab``). Exposed for callers -- e.g. the 3D
    trajectory viewer -- that need to express a full sample-frame point cloud
    in the lab frame, not just the single beam/detector directions
    ``tilted_geometry`` returns.

    Not new physics -- factors out ``tilted_geometry``'s existing rotation.
    ``tilted_geometry(theta_obs_rad, tp, ta)`` is exactly
    ``(R.T @ [0,0,1], R.T @ [sin theta_obs_rad, 0, cos theta_obs_rad])`` for
    ``R = sample_to_lab_R(tp, ta)``, bit-for-bit.

    Limiting case: ``tilt_polar_rad = 0`` gives ``normal_lab = +z``, so
    ``_rotation_between`` returns the identity and lab == sample, matching
    ``tilted_geometry``'s untilted limit.
    """
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    return _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)


def project_beam_entry(offsets_uv, tilt_polar_rad, tilt_azim_rad=0.0):
    """Sample-frame (x, y) entry points on the z=0 face for a COLLIMATED lab beam.

    ``offsets_uv`` is an ``(N, 2)`` array of transverse offsets ``(u, v)`` [Ang]
    of each electron in the LAB plane perpendicular to the fixed beam axis
    ``+z_lab`` (e.g. Gaussian beam-spot draws). The lab beam is a parallel
    bundle: ray i starts at ``o_lab = (u_i, v_i, 0)`` and travels along
    ``+z_lab``. Mapping into the sample frame uses the SAME tilt rotation
    ``R = _rotation_between(+z, normal_lab)`` as :func:`tilted_geometry`
    (``o_sample = R.T @ o_lab``; ``beam_dir = R.T @ +z_lab``), then intersects
    each ray with the sample entrance face ``z = 0``:

        s* = -o_sample_z / beam_dir_z,    p0 = o_sample + s* beam_dir   (p0_z = 0)

    Geometric content: the incident spot on the tilted face stretches by
    ``1 / cos(tilt_polar)`` along the tilt azimuth (the grazing-incidence
    footprint elongation), while the perpendicular extent is unchanged. Feeding
    the stretched entry points to the finite-footprint test in
    :func:`simulate_trajectories` makes an off-sample tail at grazing incidence
    register as ``n_missed`` -- the overlap loss that competes with the
    ``1/cos`` path-length yield enhancement.

    Source: elementary ray-plane intersection (no literature equation).
    Assumptions: the projection uses the NOMINAL beam axis, not each electron's
    own direction. With a finite emittance the two differ, but the offsets are
    specified at the entrance face itself (no drift length -- see
    ``docs/beam-phase-space.md``), so the residual is second order in the
    slope and negligible at mrad-scale divergence. The sample entrance face is
    the plane ``z = 0`` in the sample frame; an electron whose projected entry
    lands outside the transverse footprint misses the sample (handled by the
    caller, kept in ``Ne``).
    Limiting case: ``tilt_polar_rad = 0`` gives ``R = I``, so ``p0 = (u, v)``
    exactly -- the untilted isotropic entry, bit-for-bit.

    Validation: grazing-beam-projection
    """
    offsets_uv = np.asarray(offsets_uv, dtype=float)
    if not tilt_polar_rad:
        # R = I: the lab transverse plane IS the sample z=0 face; no projection.
        return offsets_uv.copy()
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    Rt = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab).T
    e_x, e_y, beam_dir = Rt[:, 0], Rt[:, 1], Rt[:, 2]
    # sample-frame origin of each lab ray = its transverse offset carried into
    # the sample frame (the ray then continues along beam_dir).
    o = offsets_uv[:, 0:1] * e_x + offsets_uv[:, 1:2] * e_y  # (N, 3)
    s_star = -o[:, 2] / beam_dir[2]
    p0 = o + s_star[:, None] * beam_dir
    # entrance face ``z = 0`` and return the (N, 2) sample-frame (x, y) entry
    # points. Solve for s* that zeroes the z-component, form p0, drop z.
    return p0[:, :2]


def detector_directions(
    theta_obs_rad,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    *,
    n_side=1,
    chip_mm=14.0,
    dist_mm=30.0,
    domega_sr,
):
    """
    Grid of detector directions {n_hat_i} (SAMPLE frame) tiling a flat square
    detector chip of side ``chip_mm`` at distance ``dist_mm`` facing the source,
    with their solid-angle weights -- the geometry input to the solid-angle-
    INTEGRATED spectrum (mc_spectrum_solid_angle). This replaces the single-n_hat
    + flat-Omega + analytic aperture_fwhm_eV approximation
    (docs/detector-solid-angle.md) with a first-principles tiling of the face.

    The central cell sits at polar angle ``theta_obs_rad`` (azimuth 0) in the lab
    and is mapped into the sample frame through the SAME tilt rotation as
    tilted_geometry() -- same convention: positive ``tilt_polar_rad`` tilts the
    slab normal (and g, by default) toward the detector; positive
    ``tilt_azim_rad`` is a CCW roll about the beam +z axis over [0, pi], with
    0 in the scattering x-z plane -- so n_side=1 returns exactly that single
    direction. The chip
    in-plane axes are chosen so one grid axis spreads in the scattering plane (the
    polar / Delta-theta direction) and the other out of plane (azimuth). The
    per-cell weight is the inverse-square + obliquity solid angle
    ``dOmega_i = dA_i cos(psi_i) / r_i^2`` (psi_i to the chip normal = central
    line of sight), then the whole set is rescaled so ``sum_i dOmega_i ==
    domega_sr``: the detector's known total solid angle is conserved and n_side=1
    reproduces today's ``spec * Omega`` exactly.

    Returns (n_hats, weights): n_hats is (N, 3) unit directions in the sample
    frame (N = n_side**2); weights is (N,) and sums to ``domega_sr``.
    """
    if n_side < 1:
        raise ValueError("n_side must be >= 1")
    # tilt rotation (same convention as tilted_geometry): sample normal -> lab
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)

    # central lab line of sight c, and chip in-plane axes (chip face _|_ c):
    #   w lies in the scattering (x-z) plane -> polar (Delta-theta) spread
    #   u is out of plane (+y)               -> azimuthal spread
    c = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    w = np.array([np.cos(theta_obs_rad), 0.0, -np.sin(theta_obs_rad)])
    u = np.array([0.0, 1.0, 0.0])

    step = chip_mm / n_side
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * step  # cell centres
    da = step**2  # cell area [mm^2]

    n_hats = np.empty((n_side * n_side, 3))
    weights = np.empty(n_side * n_side)
    i = 0
    for a in offs:  # out-of-plane (azimuth)
        for b in offs:  # in-plane (polar)
            P = dist_mm * c + a * u + b * w  # source -> cell vector [mm]
            r = float(np.linalg.norm(P))
            n_lab = P / r
            cos_psi = float(n_lab @ c)  # obliquity to the chip normal (= c)
            n_hats[i] = R.T @ n_lab
            weights[i] = da * cos_psi / r**2
            i += 1
    weights *= domega_sr / weights.sum()  # conserve the detector's total Omega
    return n_hats, weights


def _orientation_R(
    lattice,
    beam_uvw,
    azimuth_rad,
    recip_miscut_rad: tuple[float, float] | None = None,
    surface_hkl: tuple[int, int, int] | None = None,
):
    """Rotation applied to EVERY reciprocal vector: the minimal rotation taking the
    selected crystal direction onto +z (the slab normal), then a roll of
    `azimuth_rad` about +z. ``beam_uvw`` selects a direct-lattice direction;
    ``surface_hkl`` selects the reciprocal-lattice plane normal
    ``g_hkl = h*b1 + k*b2 + l*b3``. They are mutually exclusive. The reciprocal
    definition is exact for nonorthogonal cells; for an orthogonal cell a
    one-axis ``surface_hkl`` reduces to its parallel direct axis. Both paths use
    a proper minimal rotation, so handedness is preserved. Returns None for the
    construction-frame default (both orientations are None and azimuth_rad == 0).
    Shared by mc_spectrum and mosaic_psi_rad so the orientation convention lives
    in one place.

    Assumptions: Miller indices are expressed in the reciprocal basis owned by
    :func:`materials.crystal.reciprocal_g_vector`, and sample +z is the outward
    slab normal. A zero azimuth is the minimal-rotation in-plane convention;
    positive azimuth is a right-handed roll about sample +z.

    Validation: surface-hkl-orientation

    recip_miscut_rad: optional (polar_rad, azim_rad) EXTRA tilt of the
    reciprocal vector g away from the slab normal n -- a crystal miscut, where
    g no longer points along the physical normal (n stays whatever
    tilted_geometry() sets via tilt_polar_rad/tilt_azim_rad; ONLY g moves
    here). None (default) is a strict no-op: g stays aligned with n, today's
    behavior bit-for-bit. When given, uses the SAME spherical parametrization
    as tilted_geometry (Zhai convention: positive polar rotates g further from
    +z in the construction frame, azim CCW about +z), composed on top of the
    beam_uvw/azimuth rotation so it acts purely on g -- n_hat/beam_dir (set by
    tilted_geometry separately) are never touched. (0.0, anything) or
    (anything, 0.0-equivalent axis) collapses to the identity, so a caller
    that always passes recip_miscut_rad=(0.0, 0.0) gets the None behavior
    exactly. Not yet wired into any grid/study -- plumbed for a future
    asymmetric-reflection (g not || n) study."""
    if beam_uvw is not None and surface_hkl is not None:
        raise ValueError("beam_uvw and surface_hkl are mutually exclusive")

    R = None
    if beam_uvw is not None:
        u, v, w = np.asarray(beam_uvw, dtype=float)
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        axis = u * a1 + v * a2 + w * a3
        R = _rotation_between(axis / np.linalg.norm(axis), np.array([0.0, 0.0, 1.0]))
    elif surface_hkl is not None:
        axis, axis_norm = reciprocal_g_vector(surface_hkl, lattice)
        R = _rotation_between(axis / axis_norm, np.array([0.0, 0.0, 1.0]))
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        R = Rz if R is None else Rz @ R
    if recip_miscut_rad is not None:
        mp, ma = recip_miscut_rad
        if mp:
            st, ct = np.sin(mp), np.cos(mp)
            miscut_dir = np.array([st * np.cos(ma), st * np.sin(ma), ct])
            R_miscut = _rotation_between(np.array([0.0, 0.0, 1.0]), miscut_dir)
            R = R_miscut if R is None else R_miscut @ R
    return R


def _small_tilt_R(dx_rad, dy_rad):
    """Rotation that tilts the slab normal by the rotation vector (dx, dy, 0) [rad]
    via Rodrigues -- exact for any angle, exactly the identity at (0, 0). Used to
    misorient a mosaic crystallite's reciprocal vectors about the mean orientation."""
    ang = float(np.hypot(dx_rad, dy_rad))
    if ang < 1e-15:
        return np.eye(3)
    kx, ky = dx_rad / ang, dy_rad / ang
    K = np.array([[0.0, 0.0, ky], [0.0, 0.0, -kx], [-ky, kx, 0.0]])  # skew of axis
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def beam_frame_basis(beam_dir):
    """Rotation carrying the lab axes onto the beam's own frame.

    Columns 0 and 1 are the transverse basis vectors a per-electron slope pair
    ``(x', y')`` is measured against; column 2 is ``beam_dir`` itself. The
    rotation is the shortest arc from ``+z`` to ``beam_dir``, so an on-axis beam
    returns exactly the identity and the transverse planes coincide with the lab
    ``x`` / ``y`` the Gaussian spot already uses -- the two descriptions of the
    same beam then agree axis for axis.

    Limiting case: ``beam_dir = +z`` gives ``I`` bit-for-bit.

    Validation: beam-phase-space-injection
    """
    d = np.asarray(beam_dir, dtype=float)
    d = d / np.linalg.norm(d)
    sin_theta = float(np.hypot(d[0], d[1]))
    if sin_theta < 1e-15:
        return np.eye(3)
    # Rotate about the axis perpendicular to both z and d, by the angle between
    # them; _small_tilt_R takes that axis-angle as a rotation vector in the plane.
    theta = float(np.arctan2(sin_theta, d[2]))
    return _small_tilt_R(-d[1] * theta / sin_theta, d[0] * theta / sin_theta)


def _mosaic_quadrature(fwhm_rad, nodes):
    """Gauss-Hermite product quadrature over a 2-D Gaussian mosaic tilt of the
    crystallite normal (per-axis sigma = FWHM / 2.3548 -- the rocking curve is the
    1-D projection). Returns a list of (rotation matrix, weight) with weights
    summing to 1, used by mc_spectrum to INCOHERENTLY average a reflection's
    spectrum over crystallite orientations (the exact mosaic route,
    docs/crystal-mosaicity.md (2)).

    Returns None -- the perfect-crystal fast path, today's result bit-for-bit --
    when there is nothing to average: ``fwhm_rad`` falsy, or ``nodes`` <= 1 (the
    single Gauss-Hermite node sits at zero tilt, i.e. the identity, so the loop is
    skipped entirely).

    This is the DETERMINISTIC counterpart to drawing K random orientations: the
    integrand (spectrum vs crystallite tilt) is smooth, so product Gauss-Hermite
    converges in far fewer evaluations than random sampling and needs no RNG
    sub-stream. Cost is K = nodes**2 evaluations of the per-reflection block, a
    direct wall-clock multiplier on the (serial, with CuPy) GPU hot loop.

    Validation: mosaic-mc
    """
    if not fwhm_rad or nodes is None or nodes <= 1:
        return None
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    sigma = float(fwhm_rad) / 2.3548200450309493  # FWHM -> Gaussian sigma
    tilt = np.sqrt(2.0) * sigma * x  # quadrature nodes as tilt angles [rad]
    wt = w / np.sqrt(np.pi)  # per-axis weights, sum to 1
    return [
        (_small_tilt_R(tilt[j], tilt[k]), float(wt[j] * wt[k]))
        for j in range(len(tilt))
        for k in range(len(tilt))
    ]
