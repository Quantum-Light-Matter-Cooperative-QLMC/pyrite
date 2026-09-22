"""Convert the BremsLib library into one native-grid table per element.

What a consumer needs from BremsLib is the single differential cross section
and, per ``(T1, k)`` node, the normalized angular shape function (D6, D7).
Both are built here on the library's own grids: the electron energy grid is
configurable in PyRITE, so a table resampled onto it would go stale whenever
that grid changed, and resampling belongs at load.

The angular grids are ragged -- 181, 221 or 441 points depending on ``T1`` --
so the per-node angular data are stored flat with an offset index rather than
as one rectangular block (D8).

The shape function itself is not stored. It is the DDCS divided by its own
angular integral, and both of those are, so storing it would repeat about a
third of the table's bytes to save one division. :func:`shape_function`
performs that division for a consumer.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from .read import (
    COMPLETE_T1_MAX_MEV,
    RATIO_COUNT,
    DdcsPanel,
    ddcs_integral_path,
    iter_node_files,
    node_index,
    parse_ddcs,
    parse_ratio_table,
    sdcs_path,
)

#: What :func:`build_table` produces, recorded in the table key so a consumer
#: asking for bremsstrahlung does not resolve an elastic table.
QUANTITY = "bremsstrahlung_sdcs_ddcs"


def angular_integral(theta_deg: np.ndarray, ddcs: np.ndarray) -> float:
    """Return the solid-angle integral of an azimuthally symmetric DDCS.

    For an unpolarized beam on an unoriented target the DDCS is independent
    of the azimuth, so ``dOmega = 2*pi*sin(theta)*dtheta`` and

    ``integral(DDCS dOmega) = 2*pi*integral_0^pi DDCS(theta)*sin(theta)*dtheta``.

    Integrated by composite Simpson on the library's own angular grid, which
    is non-uniform -- refined towards the forward peak -- and always has an
    even number of intervals, so the rule applies across the whole range.
    Upstream integrates the same grid with a 10th-order Newton-Cotes formula
    and publishes the result in ``DDCS_int_<Z>.txt``; Simpson reproduces those
    values to better than 2e-5 relative, against 1e-4 for the trapezoid, which
    is why the extra dependency on ``scipy`` is taken here.

    Limiting cases: the integrand vanishes at both endpoints, and the integral
    equals the SDCS at the same node up to the vendor's own reported
    deviation.

    Validation: bremslib-library-reference

    Raises
    ------
    ValueError
        If the integral is not positive and finite, which would make the
        shape function meaningless.
    """
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    values = np.asarray(ddcs, dtype=np.float64)
    integral = 2.0 * np.pi * float(simpson(values * np.sin(theta), x=theta))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("BremsLib DDCS node has no positive finite angular integral")
    return integral


def shape_function(theta_deg: np.ndarray, ddcs: np.ndarray) -> np.ndarray:
    """Return the DDCS normalized to unit solid-angle integral, in 1/sr.

    ``shape(theta) = DDCS(theta) / integral(DDCS dOmega)``, so
    ``integral(shape dOmega) = 1`` by construction. The upstream ``k / Z**2``
    scaling factor cancels in the ratio, which is why it is left in place on
    both stored cross sections.

    Validation: bremslib-library-reference
    """
    values = np.asarray(ddcs, dtype=np.float64)
    return values / angular_integral(theta_deg, values)


def build_table(
    library: Path, z: int, *, t1_max_MeV: float = COMPLETE_T1_MAX_MEV
) -> dict[str, np.ndarray]:
    """Read element ``z`` from ``library`` and return the stored arrays.

    Parameters
    ----------
    library
        Library data directory, from :func:`pyrite.xsgen.bremslib.read.library_root`.
    z
        Atomic number, 1 to 100.
    t1_max_MeV
        Highest incident energy to include. Defaults to
        :data:`~pyrite.xsgen.bremslib.read.COMPLETE_T1_MAX_MEV`, above which
        the library holds only ``k = 0`` and writes zeros for everything else.
        Lowering it is how a caller bounds the table's size.

    Returns
    -------
    dict of str to ndarray
        ``t1_MeV`` and ``k_over_t1`` are the SDCS grid; ``sdcs_mb``,
        ``sdcs_rel_err`` and ``sdcs_point_finite`` are indexed
        ``[T1, k/T1]``. The angular data are flat, indexed by
        ``node_offset[i]:node_offset[i+1]``, with ``node_t1_index`` and
        ``node_k_index`` placing each node on the SDCS grid.
        ``node_angular_integral_mb`` is PyRITE's own Simpson integral of each
        node and ``node_vendor_integral_mb`` is the value upstream published
        for it, so the two are comparable after the fact rather than only at
        generation time.

    Raises
    ------
    ValueError
        If the library's own structure does not hold: a retained energy
        without its full set of ``k/T1`` nodes, a node whose ratio is not a
        grid value, or two nodes claiming the same grid position.
    """
    nodes = [node for node in iter_node_files(library, z) if node.t1_MeV <= t1_max_MeV]
    if not nodes:
        raise ValueError(f"BremsLib holds no Z={int(z)} nodes at or below T1 = {t1_max_MeV!r} MeV")

    sdcs = parse_ratio_table(sdcs_path(library, z).read_text(encoding="ascii"))
    vendor = parse_ratio_table(ddcs_integral_path(library, z).read_text(encoding="ascii"))
    keep = np.flatnonzero(sdcs.t1_MeV <= t1_max_MeV)
    t1_grid = sdcs.t1_MeV[keep]
    # The vendor's angular-integral file carries its own, shorter energy grid,
    # so each retained energy is looked up in it by value rather than reused
    # from the SDCS row index.
    vendor_row = {row: vendor.row(float(energy)) for row, energy in enumerate(t1_grid)}

    t1_index: list[int] = []
    k_index: list[int] = []
    k_mev: list[float] = []
    offsets: list[int] = [0]
    theta_parts: list[np.ndarray] = []
    ddcs_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    ratio_parts: list[np.ndarray] = []
    integrals: list[float] = []
    vendor_integrals: list[float] = []
    finite_nucleus: list[bool] = []
    seen: set[tuple[int, int]] = set()

    rows_by_energy = {float(energy): row for row, energy in enumerate(t1_grid)}
    for node in nodes:
        row = _matching_row(rows_by_energy, node.t1_MeV)
        if row is None:
            # A node above the retained SDCS grid, or at an energy the SDCS
            # file does not carry. The SDCS is what a consumer normalizes
            # against, so a node without one is not usable.
            continue
        column = node_index(sdcs.k_over_t1, node.k_MeV / node.t1_MeV)
        if (row, column) in seen:
            raise ValueError(
                f"BremsLib Z={int(z)} has two nodes at T1 index {row}, k/T1 index {column}"
            )
        seen.add((row, column))
        panel = parse_ddcs(node.path.read_text(encoding="ascii"))
        t1_index.append(row)
        k_index.append(column)
        k_mev.append(node.k_MeV)
        theta_parts.append(panel.theta_deg)
        ddcs_parts.append(panel.ddcs_mb_sr)
        err_parts.append(panel.rel_err)
        ratio_parts.append(panel.point_finite)
        offsets.append(offsets[-1] + panel.theta_deg.size)
        integrals.append(angular_integral(panel.theta_deg, panel.ddcs_mb_sr))
        vendor_integrals.append(float(vendor.value[vendor_row[row], column]))
        finite_nucleus.append(panel.finite_nucleus)

    _check_complete(z, len(t1_grid), seen)
    order = np.lexsort((np.asarray(k_index), np.asarray(t1_index)))
    return {
        "t1_MeV": t1_grid,
        "k_over_t1": sdcs.k_over_t1,
        "sdcs_mb": sdcs.value[keep],
        "sdcs_rel_err": sdcs.middle[keep].astype(np.float32),
        "sdcs_point_finite": sdcs.point_finite[keep].astype(np.float32),
        "node_t1_index": np.asarray(t1_index, dtype=np.int32)[order],
        "node_k_index": np.asarray(k_index, dtype=np.int32)[order],
        "node_k_MeV": np.asarray(k_mev, dtype=np.float64)[order],
        "node_offset": _reordered_offsets(offsets, order),
        "node_angular_integral_mb": np.asarray(integrals, dtype=np.float64)[order],
        "node_vendor_integral_mb": np.asarray(vendor_integrals, dtype=np.float64)[order],
        "node_finite_nucleus": np.asarray(finite_nucleus, dtype=bool)[order],
        "theta_deg": _concatenate(theta_parts, order),
        "ddcs_mb_sr": _concatenate(ddcs_parts, order),
        "ddcs_rel_err": _concatenate(err_parts, order).astype(np.float32),
        "ddcs_point_finite": _concatenate(ratio_parts, order).astype(np.float32),
    }


def _matching_row(rows_by_energy: Mapping[float, int], t1_MeV: float) -> int | None:
    """Return the retained row for a node energy, or ``None``.

    File names carry two significant digits while the SDCS file writes its
    grid to full precision, so the match is relative.
    """
    for energy, row in rows_by_energy.items():
        if np.isclose(energy, t1_MeV, rtol=1e-6, atol=0.0):
            return row
    return None


def _check_complete(z: int, rows: int, seen: set[tuple[int, int]]) -> None:
    """Fail when a retained energy is missing any of its ``k/T1`` nodes.

    A hole would be indistinguishable from a vanishing cross section once
    interpolated, so the table refuses to be built rather than shipping one.
    """
    missing = [
        (row, column)
        for row in range(rows)
        for column in range(RATIO_COUNT)
        if (row, column) not in seen
    ]
    if missing:
        first = ", ".join(f"T1[{row}] k/T1[{column}]" for row, column in missing[:4])
        raise ValueError(
            f"BremsLib Z={int(z)} is missing {len(missing)} of {rows * RATIO_COUNT} "
            f"DDCS nodes: {first}{', ...' if len(missing) > 4 else ''}"
        )


def _concatenate(parts: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    """Concatenate per-node arrays in the table's node order."""
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([parts[index] for index in order])


def _reordered_offsets(offsets: list[int], order: np.ndarray) -> np.ndarray:
    """Return the offset index matching the reordered angular data."""
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    return np.concatenate(([0], np.cumsum(lengths[order]))).astype(np.int64)


def panel_of(arrays: Mapping[str, np.ndarray], node: int) -> DdcsPanel:
    """Return one stored node as a :class:`~pyrite.xsgen.bremslib.read.DdcsPanel`."""
    offsets = np.asarray(arrays["node_offset"])
    start, stop = int(offsets[node]), int(offsets[node + 1])
    return DdcsPanel(
        theta_deg=np.asarray(arrays["theta_deg"])[start:stop],
        ddcs_mb_sr=np.asarray(arrays["ddcs_mb_sr"])[start:stop],
        rel_err=np.asarray(arrays["ddcs_rel_err"])[start:stop],
        point_finite=np.asarray(arrays["ddcs_point_finite"])[start:stop],
        finite_nucleus=bool(np.asarray(arrays["node_finite_nucleus"])[node]),
    )


__all__ = [
    "QUANTITY",
    "angular_integral",
    "build_table",
    "panel_of",
    "shape_function",
]
