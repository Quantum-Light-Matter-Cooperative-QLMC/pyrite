"""Grid-independent optical-domain screening of the actual line population.

Counts describe escape-piece/reflection/mosaic pairs, never photon yield.
The vacuum seed is available even when the production root cannot be evaluated
because its optical data are missing. No extrapolation is used to certify it.
"""

import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ...._env import env_value
from ....materials.atomic import _chantler_bounds
from ....materials.crystal import _EDGE_PRONE, HBARC_EV_ANG, reciprocal_g_vector
from ....materials.photon_cross_sections import EPDL_E_MIN_EV
from ...geometry import _mosaic_quadrature
from ..reflection_rows import MIN_RESONANCE_EV
from ._kernels import (
    _cxr_emission_ceiling,
    _in_medium_kinematics,
)
from ._recoil import _line_recoil_terms, _recoil_line_allowed


class LineOpticalDataWarning(UserWarning):
    """Transported line pairs reach unavailable optical data."""


#: Opt-in switch for the screen: any nonempty value other than ``0`` enables it.
LINE_DATA_COVERAGE_ENV = "PYRITE_LINE_DATA_COVERAGE"


def line_data_coverage_enabled():
    """Whether this process runs the opt-in line optical-data screen.

    Off by default: the screen is diagnostic only (counts, never yield) and
    cost up to 95 % of the line phase before #286 vectorized it.
    """
    return env_value(LINE_DATA_COVERAGE_ENV, "") not in ("", "0")


_COLLECTOR: ContextVar[list | None] = ContextVar("line_data_coverage", default=None)


def _warn(records):
    missing = sum(
        r["counts"]["vacuum_seed_outside_optical_data"] + r["counts"]["root_missing_optical_data"]
        for r in records
    )
    if missing:
        total = sum(r["counts"]["pairs"] for r in records)
        warnings.warn(
            f"{missing}/{total} line escape-piece/reflection/mosaic pairs reach "
            "unavailable optical data (vacuum-seed screen or root lookup). "
            "High-energy PXR/CBS lines may be omitted, even on an automatically "
            "clipped photon grid. Counts are not lost photon yield. "
            "Inspect line_data_coverage; see issue #249.",
            LineOpticalDataWarning,
            stacklevel=3,
        )


@contextmanager
def collect_line_data_coverage():
    """Collect successful spectrum calls transactionally, including OOM retries.

    Nested collectors publish to their parent only on successful exit. A failed
    electron-block attempt therefore cannot inflate a later successful audit.
    """
    parent = _COLLECTOR.get()
    records = []
    token = _COLLECTOR.set(records)
    try:
        yield records
    except BaseException:
        raise
    else:
        if parent is None:
            _warn(records)
        else:
            parent.extend(records)
    finally:
        _COLLECTOR.reset(token)


def publish_line_data_coverage(record):
    """Publish after spectrum success; direct low-level calls still warn."""
    collector = _COLLECTOR.get()
    if collector is None:
        _warn([record])
    else:
        collector.append(record)


def line_data_coverage(st):
    """Screen all eligible pairs before the output-grid/finite-value masks.

    Reuses the production exact-recoil vacuum seed and Snell solver
    (``line-recoil-in-medium-root``), with the same optical
    table and backend precision. A seed outside the *physical data domain* is
    reported separately: it is not a solved in-medium resonance. Roots outside
    the output tabulation can use the production endpoint clamp and are flagged
    explicitly; they do not certify a root over the full optical domain.

    No new resonance equation, weighting or emission model is introduced.
    Counts are unweighted, depend on segmentation/mosaic quadrature, and cannot
    estimate missing yield or the probability of an unsupported electron.
    Every reflection/orientation row of a segment block is screened in one
    (segment, row) pass; counts are additive, so blocking never changes them.
    """
    req = st.request
    elements = {element for element, _ in st.info["basis"]}
    anomalous = elements if req.use_henke else elements & _EDGE_PRONE
    lower = max([EPDL_E_MIN_EV, *(_chantler_bounds(el)[0] for el in anomalous)])
    upper = _cxr_emission_ceiling(st.info, st.abs_comp, req.use_henke)
    counts = xp.zeros(len(_COUNT_KEYS), dtype=xp.int64)
    rows = _reflection_rows(st)
    if rows.shape[0]:
        domain = (lower, upper, *np.asarray(_to_cpu(st.E_grid))[[0, -1]])
        normal = np.asarray(_to_cpu(st.n_hat), dtype=np.float64)
        rows_dev = xp.asarray(rows, dtype=REAL)
        normal = xp.asarray(normal / np.linalg.norm(normal))
        # Bound diagnostic temporaries by (segment, row) pairs, independently
        # of the photon-axis size and the reflection count.
        chunk = max(1, min(40000, int(req.chunk), _PAIR_BUDGET // rows.shape[0]))
        for start in range(0, st.v_all.shape[0], chunk):
            block = slice(start, start + chunk)
            counts += _block_counts(st, block, rows_dev, normal, domain)
    return {
        "schema": "line-data-coverage.v3",
        "population": "unweighted-eligible-escape-piece/reflection/mosaic-pairs",
        "crystal": str(req.crystal),
        "coherent": bool(req.coherent),
        "direction": [float(v) for v in st.n_hat],
        "optical_domain_eV": [float(lower), float(upper)],
        "counts": dict(zip(_COUNT_KEYS, (int(v) for v in _to_cpu(counts)), strict=True)),
    }


_COUNT_KEYS = (
    "pairs",
    "invalid_vacuum_seed",
    "no_positive_vacuum_seed",
    "vacuum_seed_outside_optical_data",
    "root_missing_optical_data",
    "critical_angle_rejected",
    "root_nonconverged",
    "below_line_floor",
    "finite_root",
    "finite_root_outside_grid",
    "vacuum_seed_outside_tabulation",
)
# Segment-row pairs per screening pass: a few hundred MiB of temporaries.
_PAIR_BUDGET = 1 << 19


def _screening_orientations(st):
    """Orientations the coverage screen counts against.

    The per-history sampled mosaic has no fixed orientation set; it screens
    against the Gauss-Hermite nodes of the same width and ``mosaic_nodes``,
    which span the same tilt cone. Counts stay unweighted diagnostics.
    """
    if st.mosaic_sampled is not None:
        req = st.request
        return _mosaic_quadrature(req.mosaic_fwhm_rad, max(int(req.mosaic_nodes), 2))
    return st.mosaic_quad or [(None, 1.0)]


def _reflection_rows(st):
    """Float64 lab-frame g of every requested reflection and weighted orientation."""
    rows = []
    for hkl in st.request.hkl_list:
        g, _ = reciprocal_g_vector(hkl, st.info["lattice"])
        if st.R_orient is not None:
            g = st.R_orient @ g
        for rotation, weight in _screening_orientations(st):
            if weight > 0:
                rows.append(np.asarray(g if rotation is None else rotation @ g, dtype=np.float64))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _block_counts(st, block, rows_dev, normal, domain):
    """Outcome counts of one segment block against every reflection row."""
    lower, upper, grid_lo, grid_hi = domain
    rows = rows_dev
    velocity = st.v_all[block]
    gamma = st.gamma_all[block][:, None]
    r_num, r_den, inv_E = _line_recoil_terms(
        gamma,
        xp.sum(rows * rows, axis=1)[None, :],
        (rows @ xp.asarray(normal, dtype=REAL))[None, :],
    )
    denom_vac = st.denom_all[block][:, None] + r_den
    vg = xp.sum(velocity[:, None, :] * rows[None, :, :], axis=2) - r_num
    gradient = st.escape_gradient[block]
    vg_escape = xp.sum(velocity * gradient, axis=1)[:, None]
    grad2 = xp.sum(gradient * gradient, axis=1)[:, None]
    eligible = st.cxr_electron[block] & xp.isfinite(st.t_L_all[block])
    eligible &= st.t_L_all[block] > 0
    eligible = xp.broadcast_to(eligible[:, None], vg.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        seed = HBARC_EV_ANG * vg / denom_vac
        denom, delta = _in_medium_kinematics(
            denom_vac, vg, st.delta_tab_g, st.E_tab_g, vg_escape, grad2
        )
        root = HBARC_EV_ANG * vg / denom
        # A forbidden exact vacuum root (negative-energy branch) is no
        # positive seed; a forbidden in-medium root is a rejected root.
        allowed = _recoil_line_allowed(denom_vac, vg / denom_vac, inv_E, gamma)
        seed = xp.where(allowed | ~xp.isfinite(seed), seed, 0.0)
        root = xp.where(_recoil_line_allowed(denom, vg / denom, inv_E, gamma), root, xp.nan)
    positive = eligible & xp.isfinite(seed) & (seed > 0)
    seed_miss = positive & ((seed <= lower) | (seed >= upper))
    pending = positive & ~seed_miss
    optical_miss = pending & (
        ~xp.isfinite(delta) | (xp.isfinite(root) & ((root <= lower) | (root >= upper)))
    )
    critical = pending & ~optical_miss & (2.0 * xp.abs(delta) * grad2 >= 1)
    rejected = pending & ~optical_miss & ~critical & ~xp.isfinite(root)
    solved = pending & ~optical_miss & ~critical & ~rejected
    finite = solved & (root > MIN_RESONANCE_EV)
    masks = (
        eligible,
        eligible & ~xp.isfinite(seed),
        eligible & xp.isfinite(seed) & (seed <= 0),
        seed_miss,
        optical_miss,
        critical,
        rejected,
        solved & ~finite,
        finite,
        finite & ((root < grid_lo) | (root > grid_hi)),
        positive & ((seed < st.E_tab_g[0]) | (seed > st.E_tab_g[-1])),
    )
    return xp.stack([xp.count_nonzero(mask) for mask in masks])
