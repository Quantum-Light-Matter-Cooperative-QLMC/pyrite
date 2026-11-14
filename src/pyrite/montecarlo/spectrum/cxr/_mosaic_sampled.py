"""Per-history sampled crystallite orientations for the exact mosaic route.

A mosaic crystal radiates as an incoherent ensemble of crystallites, so the
observable is the single-orientation spectrum averaged over the tilt PDF
(``docs/physics/materials/crystal-mosaicity.md``). The Gauss--Hermite rule
(:func:`..geometry._mosaic_quadrature`) evaluates every row at ``nodes**2``
fixed tilts and resolves a broad mosaic line as discrete shifted replicas.
This module instead assigns each electron HISTORY its own crystallite: ``K``
tilts drawn from the mosaic Gaussian, held for every row of the history and
every reflection (one crystallite rotates every ``g``), each carrying weight
``1/K``. The estimator is unbiased for the same ensemble average and has no
replicas; its error is statistical and falls with the number of histories.

The ``K`` tilts of a history are a Cranley--Patterson-rotated rank-1 lattice
in the unit square, mapped through the Gaussian quantile per tilt axis. Each
point is marginally uniform, so each draw is an exact tilt sample; the
lattice stratifies a history's ``K`` draws against each other. ``K = 1`` is
plain Monte Carlo.

RNG: the shift of history ``e`` is counter draws 0 (x) and 1 (y) inside the
``SeedSequence(seed).spawn(8)[7]`` child (``docs/computation/random-streams.md``),
keyed by the history's ``elec_id``. It never touches the transport generator
and depends only on ``(seed, elec_id)`` -- not on blocking, chunking or the
backend; the tilts are formed on the host in float64 for every backend.

Validation: mosaic-mc-sampled
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ...transport.kinematics import child_stream_root, counter_uniforms

#: ``SeedSequence(seed).spawn(MOSAIC_CHILD[0])[MOSAIC_CHILD[1]]`` owns the draws.
MOSAIC_CHILD = (8, 7)
#: Rocking-curve FWHM to Gaussian sigma per tilt axis.
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
#: Open-interval guard for the quantile after the modular shift.
_U_EDGE = 2.0**-53


def _lattice_generator(draws):
    """Korobov generator ``a`` of the ``draws``-point rank-1 lattice.

    The nearest integer to ``draws (sqrt5 - 1)/2`` coprime to ``draws`` (the
    Fibonacci lattice when ``draws`` is a Fibonacci number). Any generator
    keeps the rotated points exact uniform samples; this one spreads them.
    """
    if draws <= 2:
        return 1
    target = draws * (np.sqrt(5.0) - 1.0) / 2.0
    for step in range(draws):
        for a in (int(round(target)) + step, int(round(target)) - step):
            if 0 < a < draws and np.gcd(a, draws) == 1:
                return a
    return 1


def lattice_points(draws):
    """``(draws, 2)`` rank-1 lattice ``(i/K, frac(i a/K))`` in ``[0, 1)^2``."""
    draws = int(draws)
    i = np.arange(draws, dtype=np.float64)
    a = _lattice_generator(draws)
    return np.stack((i / draws, np.mod(i * a, draws) / draws), axis=1)


def history_tilts(seed, elec_ids, fwhm_rad, draws):
    """Tilt rotation vectors ``(dx, dy)`` [rad], shape ``(n_ids, draws, 2)``.

    Draw ``k`` of history ``e`` is ``sigma Phi^-1((p_k + s_e) mod 1)`` per
    axis, with ``p_k`` the rank-1 lattice and ``s_e`` the history's counter
    shift; ``sigma = fwhm_rad / 2.3548``.

    Validation: mosaic-mc-sampled
    """
    ids = np.asarray(elec_ids, dtype=np.int64)
    draws = int(draws)
    if draws < 1:
        raise ValueError("mosaic draws per history must be a positive integer")
    if ids.size == 0:
        return np.zeros((0, draws, 2))
    if int(ids.min()) < 0:
        raise ValueError("mosaic history ids must be non-negative")
    root = child_stream_root(int(seed), *MOSAIC_CHILD)
    lo = int(ids.min())
    shift = counter_uniforms(root, int(ids.max()) - lo + 1, 2, start=lo)[ids - lo]
    u = np.mod(lattice_points(draws)[None, :, :] + shift[:, None, :], 1.0)
    u = np.clip(u, _U_EDGE, 1.0 - _U_EDGE)
    return float(fwhm_rad) * _FWHM_TO_SIGMA * ndtri(u)


def tilt_rotate(g, dx, dy):
    """Rows of ``_small_tilt_R(dx, dy) @ g`` for arrays ``dx``, ``dy`` [rad].

    Rodrigues rotation about the in-plane axis ``(dx, dy, 0)/|t|`` by ``|t|``;
    exactly ``g`` where ``|t| < 1e-15``, as :func:`..geometry._small_tilt_R`.
    ``g`` is ``(3,)``; returns ``(n, 3)`` float64.
    """
    g = np.asarray(g, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    ang = np.hypot(dx, dy)
    small = ang < 1e-15
    safe = np.where(small, 1.0, ang)
    kx, ky = dx / safe, dy / safe
    # K g and K (K g) for the skew matrix [[0,0,ky],[0,0,-kx],[-ky,kx,0]].
    kg = np.stack((ky * g[2], -kx * g[2], -ky * g[0] + kx * g[1]), axis=1)
    kkg = np.stack((ky * kg[:, 2], -kx * kg[:, 2], -ky * kg[:, 0] + kx * kg[:, 1]), axis=1)
    out = g[None, :] + np.sin(ang)[:, None] * kg + (1.0 - np.cos(ang))[:, None] * kkg
    out[small] = g
    return out


def polarization_pairs(k_hat, g_rows):
    """Row-wise :func:`._kernels._polarization_pair` for ``g_rows`` ``(n, 3)``."""
    k_hat = np.asarray(k_hat, dtype=np.float64)
    n_plane = np.cross(k_hat[None, :], g_rows)
    npl = np.linalg.norm(n_plane, axis=1)
    degenerate = npl < 1e-12
    if degenerate.any():
        fallback = np.cross(k_hat, np.array([0.0, 1.0, 0.0]))
        n_plane[degenerate] = fallback
        npl[degenerate] = np.linalg.norm(fallback)
    e_s = n_plane / npl[:, None]
    e_p = np.cross(e_s, k_hat[None, :])
    return e_s, e_p / np.linalg.norm(e_p, axis=1)[:, None]


@dataclass(frozen=True)
class SampledMosaic:
    """Per-history orientation state of one spectrum call.

    ``row_history`` maps every segment row to its history's index in
    ``tilts`` (``(n_histories, draws, 2)``, host float64).
    """

    row_history: object
    tilts: np.ndarray

    @property
    def draws(self):
        return int(self.tilts.shape[1])

    def orientation_rows(self, g_vec, k_hat, draw):
        """Per-history ``(g, e_s, e_p)`` host rows of one draw, each ``(n_h, 3)``."""
        tilt = self.tilts[:, int(draw), :]
        g_rows = tilt_rotate(g_vec, tilt[:, 0], tilt[:, 1])
        e_s, e_p = polarization_pairs(k_hat, g_rows)
        return g_rows, e_s, e_p


def sampled_mosaic(seg_elec_id_host, seed, fwhm_rad, draws):
    """Build :class:`SampledMosaic` from the host segment ``elec_id`` column."""
    ids = np.asarray(seg_elec_id_host, dtype=np.int64)
    uniq, row_history = np.unique(ids, return_inverse=True)
    return SampledMosaic(
        row_history=row_history.astype(np.int64),
        tilts=history_tilts(seed, uniq, fwhm_rad, draws),
    )


def dot_g(m, vec):
    """``m . vec`` per row: ``vec`` one ``(3,)`` vector or per-row ``(n, 3)``."""
    from ._kernels import _matvec3, _rowdot3

    return _rowdot3(m, vec) if vec.ndim == 2 else _matvec3(m, vec)


def kept_orientation(idx, g_vec, *per_row):
    """``(g, *per_row)`` restricted to rows ``idx`` when ``g`` is per row.

    A single crystallite (``g.ndim == 1``) passes through unchanged, so the
    quadrature and perfect-crystal paths stay bit-for-bit.
    """
    if g_vec.ndim != 2:
        return (g_vec, *per_row)
    return tuple(a[idx] for a in (g_vec, *per_row))


def accumulate_sampled_mosaic(st, tables, accumulate_reflection):
    """Per-history sampled mosaic: one ``accumulate_reflection`` pass per
    (reflection, draw), with ``g`` and its polarization pair gathered per
    segment row from the row's history crystallite, weight ``1/K``.

    The rotated rows are formed per HISTORY on the host in float64 and
    gathered to segments on the device, so every backend sees the same
    crystallite. ``|g|`` is rotation-invariant.
    Validation: mosaic-mc-sampled
    """
    from ...._backend import REAL, xp
    from ....materials.crystal import reciprocal_g_vector
    from ._kernels import _matvec3, _rowdot3

    sampled = st.mosaic_sampled
    chi_re, chi_im, u_re, u_im = tables
    rows = xp.asarray(sampled.row_history)
    wm = 1.0 / sampled.draws
    for i_hkl, hkl in enumerate(st.request.hkl_list):
        g_vec, _g = reciprocal_g_vector(hkl, st.info["lattice"])
        if st.R_orient is not None:
            g_vec = st.R_orient @ g_vec
        g2 = REAL(float(g_vec @ g_vec))
        for draw in range(sampled.draws):
            g_h, es_h, ep_h = sampled.orientation_rows(g_vec, st.n_hat, draw)
            G, ES, EP = (xp.asarray(a, dtype=REAL)[rows] for a in (g_h, es_h, ep_h))
            accumulate_reflection(
                st,
                G,
                ES,
                EP,
                g2,
                _matvec3(G, st.n_hat_d),
                _rowdot3(G, ES),
                _rowdot3(G, EP),
                chi_re[i_hkl],
                chi_im[i_hkl],
                u_re[i_hkl],
                u_im[i_hkl],
                wm,
            )


def mosaic_orientations(request, seg_elec_id_host):
    """``(mosaic_quad, mosaic_sampled)`` of one spectrum call; at most one set.

    Validation: mosaic-mc, mosaic-mc-sampled
    """
    from ...geometry import _mosaic_quadrature

    if request.mosaic_sampling == "history" and request.mosaic_fwhm_rad:
        sampled = sampled_mosaic(
            seg_elec_id_host, request.mosaic_seed, request.mosaic_fwhm_rad, request.mosaic_nodes
        )
        return None, sampled
    return _mosaic_quadrature(request.mosaic_fwhm_rad, request.mosaic_nodes), None
