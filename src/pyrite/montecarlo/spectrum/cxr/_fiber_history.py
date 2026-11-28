"""Experimental per-primary fiber roll on the existing per-hkl line route.

One supplied offset per incident primary, indexed by the retained global
``elec_id``. No random state is consumed here. Coupling tables are independent
of roll; only the reciprocal vector and polarization pair vary by history.
This is a fixed-grid research seam, not a persisted campaign sampling mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

SECTOR_RAD = math.pi / 3.0
# A dedicated SeedSequence spawn path, outside the small integer children used
# by transport, beam and mosaic. Versioned by fiber.FIBER_HISTORY_RNG.
FIBER_ROLL_STREAM = (0xF1527,)


@dataclass(frozen=True)
class HistoryRollSeed:
    """Transient seed for a sparse, global-primary-addressed sector roll."""

    seed: int

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("history fiber roll seed must be a nonnegative integer")

    def for_ids(self, ids):
        """Counter 0 of each global primary; allocate only represented IDs."""
        from ...transport.kinematics import (
            _SM64_GOLDEN,
            _SM64_ONE,
            _SM64_S12,
            _U52_SCALE,
            _splitmix64_array,
        )

        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("history fiber rolls require nonnegative integer global IDs")
        root = np.random.SeedSequence(int(self.seed), spawn_key=FIBER_ROLL_STREAM)
        key = root.generate_state(1, dtype=np.uint64)[0]
        # Same two unsigned hashes and open-interval 52-bit uniform as
        # counter_uniforms(root, ..., 1); sparse rather than a dense ID range.
        keys = _splitmix64_array(key + _SM64_GOLDEN * (ids.astype(np.uint64) + _SM64_ONE))
        draws = _splitmix64_array(keys + _SM64_GOLDEN)
        u = ((draws >> _SM64_S12).astype(np.float64) + 0.5) * _U52_SCALE
        return SECTOR_RAD * u


def validate_history_roll_request(request):
    """Fail closed outside the fixed-grid HOPG research envelope."""
    rolls = request._fiber_history_rolls
    if rolls is None:
        return
    if (
        request.crystal != "hopg"
        or request.coherent
        or request.line_quadrature != "bin-mean"
        or request.truncation_audit is not None
        or request.temporal is not None
        or request.coefficient_capture is not None
        or request.layers is not None
        or request.groove is not None
        or request.sinc_cutoff is not None
        or request.mosaic_sampling not in ("quadrature", "history")
        or request.phase_retention is not None
    ):
        raise ValueError("history fiber rolls require fixed-grid incoherent HOPG scoring")
    _validate_history_mosaic(request, rolls)
    if request.recip_miscut_rad is not None and request.recip_miscut_rad[0] != 0:
        raise ValueError("history fiber rolls do not yet support reciprocal miscut")
    for axis in (request.beam_uvw, request.surface_hkl):
        if axis is not None and (tuple(axis[:2]) != (0, 0) or axis[2] <= 0):
            raise ValueError("history fiber rolls require the positive c axis along sample +z")
    if any(tuple(hkl[:2]) == (0, 0) for hkl in request.hkl_list):
        raise ValueError("score basal reflections separately from history fiber rolls")
    if request.segments.get("cone_split") is not None:
        raise ValueError("history fiber rolls do not support detector cone splitting")
    _validate_primary_rolls(request, rolls)


def _validate_history_mosaic(request, rolls):
    """Require a common primary seed and valid sampled mosaic width."""
    if request.mosaic_sampling == "history" and (
        not isinstance(rolls, HistoryRollSeed)
        or request.mosaic_seed != rolls.seed
        or not request.mosaic_fwhm_rad
        or not math.isfinite(request.mosaic_fwhm_rad)
        or request.mosaic_fwhm_rad <= 0
    ):
        raise ValueError(
            "history fiber rolls with sampled mosaic require a common case seed and positive FWHM"
        )


def _validate_primary_rolls(request, rolls):
    """Validate global primary identity or an explicit sector-roll array."""
    if isinstance(rolls, HistoryRollSeed):
        # Ordinary transport has a shared global ID in both columns. Refuse
        # lineage/renumbering routes rather than assigning the wrong roll.
        from ...._backend import xp

        ids = xp.asarray(request.segments["elec_id"])
        root_ids = request.segments.get("electron_id")
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or bool(xp.any(ids < 0)):
            raise ValueError("history fiber rolls require global integer primary IDs")
        if root_ids is not None and not bool(xp.array_equal(ids, xp.asarray(root_ids))):
            raise ValueError("history fiber rolls require matching global primary ID columns")
        return
    angles = np.asarray(rolls, dtype=np.float64)
    ne = int(
        request.electron_limit if request.electron_limit is not None else request.segments["Ne"]
    )
    if (
        ne < 1
        or angles.shape != (ne,)
        or not np.isfinite(angles).all()
        or np.any((angles < 0.0) | (angles >= SECTOR_RAD))
    ):
        raise ValueError("expected one finite sector roll in [0, pi/3) per incident primary")


def primary_roll_rows(ids, rolls, *, xp):
    """Unique represented primary angles and inverse row map, on the backend.

    All escape pieces/substeps of a primary retrieve the same angle. IDs need
    not be contiguous or sorted. Missing primaries retain their denominator
    through the ordinary spectrum normalization.
    """
    ids = xp.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("history fiber rolls require integer primary IDs")
    if isinstance(rolls, HistoryRollSeed):
        if ids.size and bool(xp.any(ids < 0)):
            raise ValueError("history fiber rolls require nonnegative global primary IDs")
        unique, inverse = xp.unique(ids, return_inverse=True)
        host_ids = unique if xp is np else xp.asnumpy(unique)
        return xp.asarray(rolls.for_ids(host_ids)), inverse.reshape(-1)
    if ids.size and bool(xp.any((ids < 0) | (ids >= len(rolls)))):
        raise ValueError("history fiber rolls require global retained primary IDs")
    unique, inverse = xp.unique(ids, return_inverse=True)
    return xp.asarray(rolls, dtype=np.float64)[unique], inverse.reshape(-1)


def roll_vectors(g, cos_phi, sin_phi, *, xp, mosaic=None):
    """Rotate a sample-frame reciprocal vector about +z, then mosaic tilt.

    The order is R_m @ R_z(phi) @ g, exactly the scalar orientation route.
    Float64 orientation arithmetic precedes conversion to the line REAL dtype.
    """
    g = xp.asarray(g, dtype=np.float64)
    out = xp.stack(
        (
            cos_phi * g[0] - sin_phi * g[1],
            sin_phi * g[0] + cos_phi * g[1],
            xp.broadcast_to(g[2], cos_phi.shape),
        ),
        axis=1,
    )
    if mosaic is not None:
        out = out @ xp.asarray(mosaic, dtype=np.float64).T
    return out


def polarization_rows(n_hat, g_rows, *, xp):
    """Row version of the scalar sigma/pi basis, on CPU or GPU."""
    n = xp.asarray(n_hat, dtype=np.float64)
    normal = xp.cross(n, g_rows)
    length = xp.linalg.norm(normal, axis=1)
    fallback = xp.cross(n, xp.asarray([0.0, 1.0, 0.0]))
    # The scalar fallback uses y. If n itself is y, use x instead to avoid
    # a zero polarization basis in that exceptional geometry.
    fallback = xp.where(
        xp.linalg.norm(fallback) < 1e-12, xp.cross(n, xp.asarray([1.0, 0.0, 0.0])), fallback
    )
    normal = xp.where((length < 1e-12)[:, None], fallback, normal)
    es = normal / xp.linalg.norm(normal, axis=1)[:, None]
    ep = xp.cross(es, n)
    return es, ep / xp.linalg.norm(ep, axis=1)[:, None]


def history_coupling_tables(st, build):
    """Reuse only orientation-independent coupling rows across scoring blocks.

    Unlike the batched geometry cache, the key is based on the content of
    the actual tabulation mesh, not the identity of the requested output
    axis. Never cache per-primary orientations or any scored contribution.
    The runner supplies its existing bounded, OOM-evictable fiber cache.
    """
    from hashlib import sha256

    from ...._backend import REAL, xp

    req = st.request
    args = (req.crystal, req.hkl_list, st.E_tab, req.B_ang2, req.use_henke)
    cache = req._table_cache
    if cache is None:
        return build(*args)
    hkls = tuple(tuple(int(x) for x in hkl) for hkl in req.hkl_list)
    # emission_coupling_tables evaluates the thermal callable on -hkl.
    b = req.B_ang2
    thermal = (
        tuple(float(b(tuple(-x for x in hkl))) for hkl in hkls)
        if callable(b)
        else None
        if b is None
        else float(b)
    )
    mesh = np.ascontiguousarray(st.E_tab, dtype=np.float64)
    device = None if xp is np else int(xp.cuda.Device().id)
    key = (
        "fiber-history-couplings-v1",
        xp.__name__,
        device,
        np.dtype(REAL).str,
        req.crystal,
        hkls,
        thermal,
        bool(req.use_henke),
        mesh.shape,
        sha256(mesh.tobytes()).digest(),
    )
    tables = cache.get(key)
    if tables is None:
        tables = build(*args)
        cache[key] = tables
    return tables


def tilt_terms(tilts, *, xp):
    """One draw's Rodrigues coefficients, shared by all reflections on device."""
    dx, dy = xp.asarray(tilts, dtype=np.float64).T
    angle = xp.hypot(dx, dy)
    small = angle < 1e-15
    safe = xp.where(small, 1.0, angle)
    return (
        xp.where(small, 0.0, dx / safe),
        xp.where(small, 0.0, dy / safe),
        xp.sin(angle),
        1.0 - xp.cos(angle),
    )


def apply_history_tilt(g_rows, terms, *, xp):
    """Row-wise R_m(dx,dy) @ g, after roll; no dense rotation matrices."""
    kx, ky, sin_a, one_minus_cos = terms
    gx, gy, gz = g_rows.T
    kg = xp.stack((ky * gz, -kx * gz, -ky * gx + kx * gy), axis=1)
    kkg = xp.stack((ky * kg[:, 2], -kx * kg[:, 2], -ky * kg[:, 0] + kx * kg[:, 1]), axis=1)
    return g_rows + sin_a[:, None] * kg + one_minus_cos[:, None] * kkg


def _history_orientation_rows(st, c, s, *, xp):
    """Yield reflection index, invariant g², oriented history rows and weight."""
    from ....materials.crystal import reciprocal_g_vector

    reflections = []
    for hkl in st.request.hkl_list:
        g, _ = reciprocal_g_vector(hkl, st.info["lattice"])
        if st.R_orient is not None:
            g = st.R_orient @ g
        reflections.append((g, float(g @ g)))
    sampled = st.mosaic_sampled
    if sampled is not None:
        # Both sampled_mosaic and primary_roll_rows use the sorted unique
        # IDs of the SAME clipped/expanded segment rows. Their history axes
        # therefore agree, including for missing or step-major primaries.
        # Tilt draws are shared with the basal scorer through its existing
        # mosaic stream; the independent fiber roll stream is unchanged.
        if sampled.tilts.shape[0] != c.size:
            raise RuntimeError("sampled mosaic and fiber history populations disagree")
        for draw in range(sampled.draws):
            terms = tilt_terms(sampled.tilts[:, draw, :], xp=xp)
            for i, (g, g2) in enumerate(reflections):
                gh = roll_vectors(g, c, s, xp=xp)
                yield i, g2, apply_history_tilt(gh, terms, xp=xp), 1.0 / sampled.draws
        return
    orientations = ((None, 1.0),) if st.mosaic_quad is None else st.mosaic_quad
    orientations = tuple(
        (None if m is None else xp.asarray(m, dtype=np.float64), w) for m, w in orientations
    )
    # Keep the existing deterministic accumulation order unchanged.
    for i, (g, g2) in enumerate(reflections):
        for mosaic, weight in orientations:
            yield i, g2, roll_vectors(g, c, s, xp=xp, mosaic=mosaic), weight


def accumulate_history_rolls(st, tables, accumulate_reflection, mu_tables):
    """One coupling row per reflection, one roll per represented primary.

    Mosaic tilts are either deterministic quadrature or K counter-addressed
    draws of the existing sampled-mosaic route. No history x reflection x
    energy tensor is materialized. Each draw retains weight 1/K.
    """
    from ...._backend import REAL, xp
    from ._kernels import _matvec3, _rowdot3

    rolls, rows = primary_roll_rows(st.seg_elec_id, st.request._fiber_history_rolls, xp=xp)
    if not rows.size:
        return
    c, s = xp.cos(rolls), xp.sin(rolls)
    chi_re, chi_im, u_re, u_im = tables
    for i, g2, gh, weight in _history_orientation_rows(st, c, s, xp=xp):
        es, ep = polarization_rows(st.n_hat, gh, xp=xp)
        G, ES, EP = (a.astype(REAL, copy=False)[rows] for a in (gh, es, ep))
        accumulate_reflection(
            st,
            G,
            ES,
            EP,
            REAL(g2),
            _matvec3(G, st.n_hat_d),
            _rowdot3(G, ES),
            _rowdot3(G, EP),
            chi_re[i],
            chi_im[i],
            u_re[i],
            u_im[i],
            weight,
            mu_tables,
        )
