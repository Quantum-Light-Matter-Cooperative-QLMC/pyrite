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
from ....materials.atomic import _chantler_bounds
from ....materials.crystal import _EDGE_PRONE, HBARC_EV_ANG, reciprocal_g_vector
from ....materials.photon_cross_sections import EPDL_E_MIN_EV
from ....materials.recoil import recoil_shift_fraction
from ..reflection_rows import MIN_RESONANCE_EV
from ._kernels import _cxr_emission_ceiling, _in_medium_kinematics


class LineOpticalDataWarning(UserWarning):
    """Transported line pairs reach unavailable optical data."""


# Screening policy, not a physical constant: a 1 % vacuum line shift is
# comparable to typical spectrometer resolution at the affected energies.
# Record-only: ordinary 10-100 keV HOPG lines exceed it through the
# longitudinal lattice-recoil term, so a warning would fire on routine cases
# before the in-crystal applicability is settled (issues #249, #527).
RECOIL_SHIFT_LIMIT = 0.01


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

    Reuses the production vacuum seed and Snell solver, with the same optical
    table and backend precision. A seed outside the *physical data domain* is
    reported separately: it is not a solved in-medium resonance. Roots outside
    the output tabulation can use the production endpoint clamp and are flagged
    explicitly; they do not certify a root over the full optical domain.

    No new resonance equation, weighting or emission model is introduced.
    Counts are unweighted, depend on segmentation/mosaic quadrature, and cannot
    estimate missing yield or the probability of an unsupported electron.
    """
    req = st.request
    elements = {element for element, _ in st.info["basis"]}
    anomalous = elements if req.use_henke else elements & _EDGE_PRONE
    lower = max([EPDL_E_MIN_EV, *(_chantler_bounds(el)[0] for el in anomalous)])
    upper = _cxr_emission_ceiling(st.info, st.abs_comp, req.use_henke)
    keys = (
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
        "recoil_shift_exceeds_limit",
    )
    counts = xp.zeros(len(keys), dtype=xp.int64)
    grid = np.asarray(_to_cpu(st.E_grid))
    orientations = st.mosaic_quad or [(None, 1.0)]
    chunk = max(1, min(40000, int(req.chunk)))
    for hkl in req.hkl_list:
        g, _ = reciprocal_g_vector(hkl, st.info["lattice"])
        if st.R_orient is not None:
            g = st.R_orient @ g
        for rotation, weight in orientations:
            if weight <= 0:
                continue
            g_row = np.asarray(g if rotation is None else rotation @ g, dtype=np.float64)
            row = xp.asarray(g_row, dtype=REAL)
            # Bound diagnostic temporaries independently of the photon-axis size.
            for start in range(0, st.v_all.shape[0], chunk):
                block = slice(start, start + chunk)
                velocity = st.v_all[block]
                denom_vac = st.denom_all[block]
                vg = xp.sum(velocity * row, axis=1)
                gradient = st.escape_gradient[block]
                vg_escape = xp.sum(velocity * gradient, axis=1)
                grad2 = xp.sum(gradient * gradient, axis=1)
                eligible = st.cxr_electron[block] & xp.isfinite(st.t_L_all[block])
                eligible &= st.t_L_all[block] > 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    seed = HBARC_EV_ANG * vg / denom_vac
                    denom, delta = _in_medium_kinematics(
                        denom_vac, vg, st.delta_tab_g, st.E_tab_g, vg_escape, grad2
                    )
                    root = HBARC_EV_ANG * vg / denom
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
                    finite & ((root < grid[0]) | (root > grid[-1])),
                    positive & ((seed < st.E_tab_g[0]) | (seed > st.E_tab_g[-1])),
                )
                counts += xp.stack(
                    [xp.count_nonzero(mask) for mask in masks]
                    + [_recoil_count(st, block, positive, g_row)]
                )
    return {
        "schema": "line-data-coverage.v2",
        "population": "unweighted-eligible-escape-piece/reflection/mosaic-pairs",
        "crystal": str(req.crystal),
        "coherent": bool(req.coherent),
        "direction": [float(v) for v in st.n_hat],
        "optical_domain_eV": [float(lower), float(upper)],
        "recoil_shift_limit": RECOIL_SHIFT_LIMIT,
        "counts": dict(zip(keys, (int(v) for v in _to_cpu(counts)), strict=True)),
    }


def _recoil_count(st, block, positive, g_row):
    """Positive vacuum seeds whose exact vacuum recoil shift exceeds the limit.

    Float64 host evaluation of the exact/classical vacuum roots for each
    positive seed, from the transported kinetic energy and direction rather
    than the float32 speed, which rounds to one near 1.5 GeV. The energy field
    is the one the line kernels use. A segment whose classical root is not
    positive in float64 is not counted. Optical data are not needed, so seeds
    outside the optical domain are screened too.

    Validation: line-recoil-validity-screen
    """
    selected = np.asarray(_to_cpu(positive), dtype=bool)
    segments = st.segments
    field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    kinetic = 1.0e3 * np.asarray(_to_cpu(xp.asarray(segments[field])), dtype=np.float64)[block]
    direction = np.asarray(_to_cpu(xp.asarray(segments["v_hat"])), dtype=np.float64)[block]
    norm = np.linalg.norm(direction, axis=1)
    # The reference raises on degenerate input; such rows are not screened.
    selected &= np.isfinite(kinetic) & (kinetic > 0) & np.isfinite(norm) & (norm > 0)
    if not selected.any():
        return xp.asarray(0, dtype=xp.int64)
    kinetic = kinetic[selected]
    direction = direction[selected] / norm[selected, None]
    normal = np.asarray(_to_cpu(st.n_hat), dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    fraction = recoil_shift_fraction(kinetic, direction, g_row, normal)
    return xp.asarray(np.count_nonzero(np.abs(fraction) > RECOIL_SHIFT_LIMIT), dtype=xp.int64)
