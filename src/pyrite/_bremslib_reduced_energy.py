"""BremsLib cross sections between the library's ``k/T1 = 0`` and 0.1 nodes.

The library tabulates 13 ``k/T1`` values, ten equidistant from 0 to 0.9, and
the first interval is where the cross section curves most. Staging inserts
:data:`SMALL_REDUCED_ENERGY_NODES` there, and runtime interpolates linearly
on the refined grid. The sub-node values follow the k/T1 scheme that
BremsLib's own interpolation program prescribes (Poškus, *Interpolate_DCS
v1.0.7 User's Manual*, Sec. 2, p. 3; defaults from ``Default_parameters.inp``
and the ``k_T1_knots*.txt`` files shipped with it), at each library ``T1``:

- ``T1 < 30 keV`` (``T1_thr``): linear in ``k/T1``, as the program does when
  no knot file is set for that range.
- ``30 keV <= T1 <= 0.999 MeV`` (``T1_min_useRef``): a weighted
  least-squares cubic spline of the SDCS over all 13 grid points, with knots
  at ``k/T1 = 0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.975`` and the top node
  (``k_T1_knots1.txt``) and free end slopes. Residuals are relative
  (weight ``1/SDCS**2``).
- ``T1 > 0.999 MeV``: the SDCS divided by the screened Born SDCS is fitted
  the same way over ``k/T1 = 0 .. 0.9`` with knots ``0, 0.2, 0.4, 0.5, 0.7,
  0.8, 0.9`` (``k_T1_knots_ref_ratio.txt``), and multiplied by the Born SDCS.
  The Born SDCS comes from the library's 20-node Born table (10 extra
  geometric nodes below 0.1). It is interpolated by a not-a-knot cubic
  spline: that is the program's fit with its 18 knots and free end slopes
  on 20 points, which leaves no residual.

The program also down-weights grid points whose relative uncertainty
exceeds 1 %. Released tables drop the uncertainties, so it is not
reproduced. For Z = 1, 3, 5, 6, 29 and 79 it moves the sub-node values by
under 1e-4 relative. Across all Z (library uncertainties up to 17 %) it moves
them by up to 1 % (Z = 39, 30 MeV).
The fit misses the ``k/T1 = 0`` and 0.1 nodes by about 1e-4 relative. The
library node values are served unchanged, so the sub-nodes add the linear
interpolant of those two residuals, which keeps the curve continuous at the
nodes.

A non-positive or non-finite Born value (BremsLib v2.0.8 has a zero at
``Z = 5``, ``T1 = 3 MeV``, ``k/T1 = 0.0046``) is replaced as the program
does (``exclude_zero``): that ``k/T1`` column's ``ln B`` is interpolated in
``ln T1`` by a not-a-knot cubic spline through the column's valid rows,
when valid rows lie on both sides of it.

Rows above 0.999 MeV with no usable Born row use the sub-1 MeV SDCS spline.
These are the extension rows above 30 MeV, tables staged without Born data,
and a Born value not bracketed by valid rows of its column, which is not
extrapolated and warns. At the sub-nodes this fallback is within 3.0 % of the Born-ratio
scheme over all Z at 1-30 MeV (1.5 % below 25 MeV).

Validation: bremslib-electron-electron
"""

import warnings
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline, make_lsq_spline

from ._physics_fallback import PhysicsFallbackWarning

#: ``k/T1`` sub-nodes inserted between the library's 0 and 0.1 nodes.
SMALL_REDUCED_ENERGY_NODES = (0.004, 0.01, 0.02, 0.035, 0.06)

#: Below this ``T1`` (MeV) the library scheme is linear in ``k/T1``.
SPLINE_T1_MIN_MEV = 0.03

#: Above this ``T1`` (MeV) the library scheme fits the ratio to the Born SDCS.
BORN_T1_MIN_MEV = 0.999

#: Library ``k/T1`` column indices of the SDCS spline knots (0, 0.1, 0.3,
#: 0.5, 0.6, 0.7, 0.9, 0.975 and the top node).
_SDCS_KNOT_COLUMNS = (0, 1, 3, 5, 6, 7, 9, 11, 12)

#: Columns of the ``k/T1 = 0 .. 0.9`` fit to the Born ratio, and the knot
#: columns among them (0, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9).
_RATIO_COLUMNS = 10
_RATIO_KNOT_COLUMNS = (0, 2, 4, 5, 7, 8, 9)


def refine_reduced_energy(t1_MeV, reduced, sdcs, shape, born=None):
    """Insert :data:`SMALL_REDUCED_ENERGY_NODES` at every library ``T1`` row.

    Parameters
    ----------
    t1_MeV
        Library incident energies, shape ``(n,)``.
    reduced
        Actual ``k/T1`` grid per row, shape ``(n, 13)``. The last column is
        the top node, slightly below 1.
    sdcs
        Scaled SDCS in mb, shape ``(n, 13)``.
    shape
        Unit-integral angular shapes, shape ``(n, 13, n_theta)``.
    born
        Optional Born table: ``born_t1_MeV`` ``(m,)``, ``born_k_over_t1``
        ``(20,)`` and ``born_sdcs_mb`` ``(m, 20)``, as
        :func:`pyrite.xsgen.bremslib.convert.build_table` stores them.

    Returns
    -------
    tuple
        ``(sdcs, shape)`` with 18 columns. A sub-node's shape is the linear
        mix of its neighbours' shapes, so it still has unit integral, and
        its DDCS integrates to its SDCS.

    Validation: bremslib-electron-electron
    """
    inserted = np.asarray(SMALL_REDUCED_ENERGY_NODES, dtype=np.float64)
    w = inserted / reduced[0, 1]
    born = _repair_born(born)
    sdcs_sub = np.empty((sdcs.shape[0], inserted.size), dtype=np.float64)
    for row, t1 in enumerate(np.asarray(t1_MeV, dtype=np.float64)):
        sdcs_sub[row] = _sub_node_sdcs(
            float(t1), reduced[row], sdcs[row], _born_row(born, float(t1)), inserted
        )
    shape_sub = (1.0 - w)[:, None] * shape[:, :1] + w[:, None] * shape[:, 1:2]
    return (
        np.concatenate([sdcs[:, :1], sdcs_sub, sdcs[:, 1:]], axis=1),
        np.concatenate([shape[:, :1], shape_sub, shape[:, 1:]], axis=1),
    )


def _repair_born(born: Mapping[str, np.ndarray] | None):
    """Interpolate non-positive Born values in ``ln T1``, per ``k/T1`` column.

    Interpolate_DCS excludes a non-positive Born value from its column's
    not-a-knot spline of ``ln B`` against ``ln T1``, so the value is read off
    the spline through the column's other rows. Only values bracketed by
    valid rows are repaired: a cubic extrapolated past the column's ends is
    unbounded, and a staging ``T1`` cap can make an interior row the last.
    Unrepaired values stay bad, and :func:`_born_row` falls back for their
    rows with a warning.

    Validation: bremslib-electron-electron
    """
    if born is None:
        return None
    values = np.asarray(born["born_sdcs_mb"], dtype=np.float64)
    bad = ~(np.isfinite(values) & (values > 0.0))
    if not bad.any():
        return born
    log_t1 = np.log(np.asarray(born["born_t1_MeV"], dtype=np.float64))
    repaired = values.copy()
    for column in np.flatnonzero(bad.any(axis=0)):
        good = ~bad[:, column]
        valid = np.flatnonzero(good)
        if valid.size < 2:
            continue
        inside = bad[:, column].copy()
        inside[: valid[0]] = inside[valid[-1] + 1 :] = False
        if not inside.any():
            continue
        spline = CubicSpline(log_t1[good], np.log(values[good, column]), bc_type="not-a-knot")
        repaired[inside, column] = np.exp(spline(log_t1[inside]))
    return {**born, "born_sdcs_mb": repaired}


def _born_row(born: Mapping[str, np.ndarray] | None, t1_MeV: float):
    """Return ``(k/T1, Born SDCS)`` at ``t1_MeV``, or ``None`` if it is not usable."""
    if born is None or t1_MeV <= BORN_T1_MIN_MEV:
        return None
    born_t1 = np.asarray(born["born_t1_MeV"], dtype=np.float64)
    match = np.flatnonzero(np.isclose(born_t1, t1_MeV, rtol=1e-9, atol=0.0))
    if match.size != 1:
        return None
    values = np.asarray(born["born_sdcs_mb"], dtype=np.float64)[match[0]]
    if not np.all(np.isfinite(values) & (values > 0.0)):
        warnings.warn(
            f"BremsLib Born SDCS at T1 = {t1_MeV:g} MeV has a non-positive value that "
            "cannot be interpolated in T1; its k/T1 sub-nodes use the plain SDCS spline "
            "instead of the Born ratio",
            PhysicsFallbackWarning,
            stacklevel=4,
        )
        return None
    return np.asarray(born["born_k_over_t1"], dtype=np.float64), values


def _sub_node_sdcs(t1_MeV, x, y, born, inserted):
    """SDCS at ``inserted`` for one library row, anchored to its first two nodes."""
    w = inserted / x[1]
    linear = (1.0 - w) * y[0] + w * y[1]
    if t1_MeV < SPLINE_T1_MIN_MEV:
        return linear
    if born is None:
        fit = _lsq_spline(x, y, _SDCS_KNOT_COLUMNS)
    else:
        born_spline = make_interp_spline(born[0], born[1], k=3)
        head = slice(0, _RATIO_COLUMNS)
        ratio = _lsq_spline(x[head], y[head] / born_spline(x[head]), _RATIO_KNOT_COLUMNS)

        def fit(points):
            return ratio(points) * born_spline(points)

    ends = fit(x[:2])
    return fit(inserted) + (1.0 - w) * (y[0] - ends[0]) + w * (y[1] - ends[1])


def _lsq_spline(x, y, knot_columns):
    """Cubic spline minimizing ``sum(((s(x) - y) / y)**2)`` with knots at ``x[knot_columns]``.

    Clamped knot vector, so both end slopes are free parameters, as in
    BremsLib's ``Spline_fit``.
    """
    knots = x[list(knot_columns)]
    t = np.r_[[knots[0]] * 3, knots, [knots[-1]] * 3]
    return make_lsq_spline(x, y, t, k=3, w=1.0 / np.abs(y))
