"""BremsLib bremsstrahlung: energy-differential and direction-resolved cross sections.

BremsLib v2.0 (Poškus, At. Data Nucl. Data Tables 166, 101734 (2025)) tabulates
the scaled single differential cross section (SDCS) ``chi = (k/Z**2) dsigma/dk``
and, at every ``(T1, k/T1)`` grid node, the scaled double differential cross
section ``(k/Z**2) d2sigma/(dk dOmega)`` over the photon emission angle
``theta`` measured from the incident electron direction. This module turns one
element's stored table into both cross sections at arbitrary ``(T, k, theta)``.

It holds no I/O. The physics core may not import :mod:`pyrite.xsgen`, so a
driver resolves the stored table and hands its arrays to
:func:`prepare_bremslib_table`; :mod:`pyrite.xsgen.bremslib.tables` does that
for the built-in catalogue and for locally generated tables.

The interpolation is PyRITE's own, not a port of upstream ``Interpolate_DCS``
(GPL-3). At each node the DDCS is renormalized so that the solid-angle integral
of its linear-in-``theta`` interpolant equals that node's SDCS exactly: the
angular shape is taken against its own parent SDCS, as the library defines it.
At load, each ``T1`` interval is refined with geometric sub-nodes (see
:func:`_refine_incident_grid`); at evaluation the scaled DDCS is interpolated
linearly in ``k/T`` and in ``ln T`` on that refined grid. Every such
combination is convex, so the angular integral of the
interpolated DDCS is exactly the identically interpolated SDCS -- integrating
the direction-resolved result over ``4 pi`` recovers the energy spectrum by
construction, not by a separate renormalization.

Source equations, assumptions, and limiting cases are documented in
``docs/physics/radiation-physics/bremsstrahlung.md``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._backend import REAL, xp

#: Identity of the BremsLib model and of this module's interpolation scheme.
#: Bump the suffix whenever the interpolation changes the numbers it serves.
BREMSSTRAHLUNG_BREMSLIB_MODEL = "bremslib-v2.0-sdcs-ddcs/lin-x-geo4-ln-t-lin-theta-v1"

#: Number of ``k/T1`` grid values per incident energy.
_RATIO_COUNT = 13

#: Sub-intervals per library ``T1`` interval in the staged grid.
_INCIDENT_REFINEMENT = 4

#: One millibarn in square centimetres.
_MB_CM2 = 1.0e-27


@dataclass(frozen=True, slots=True)
class BremsLibBremsstrahlungTable:
    """One element's BremsLib cross sections, staged for interpolation.

    Parameters
    ----------
    atomic_number
        Element ``Z``.
    key, digest
        Stored-table key and provenance-manifest digest, for run identity.
    incident_energy_keV
        Incident electron kinetic-energy grid ``T1``, ascending.
    nominal_reduced_energy
        The 13 nominal ``k/T1`` grid values; the last is spelled 1.
    top_reduced_energy
        The actual last ``k/T1`` value at each ``T1``, slightly below 1.
    scaled_sdcs_mb
        ``chi(T1, k/T1)`` in mb, shape ``(n_T1, 13)``.
    theta_rad
        Common photon emission-angle grid, a superset of every node's grid.
    scaled_ddcs_mb_sr
        Scaled DDCS in mb/sr, shape ``(n_T1, 13, n_theta)``, renormalized so
        that its linear-in-``theta`` solid-angle integral is ``scaled_sdcs_mb``.
    """

    atomic_number: int
    key: str
    digest: str
    incident_energy_keV: np.ndarray
    nominal_reduced_energy: np.ndarray
    top_reduced_energy: np.ndarray
    scaled_sdcs_mb: np.ndarray
    theta_rad: np.ndarray
    scaled_ddcs_mb_sr: np.ndarray

    @property
    def minimum_incident_energy_keV(self) -> float:
        return float(self.incident_energy_keV[0])

    @property
    def maximum_incident_energy_keV(self) -> float:
        return float(self.incident_energy_keV[-1])


def top_reduced_energy(t1_MeV) -> np.ndarray:
    """Return the actual last ``k/T1`` grid value, per the BremsLib manual.

    0.99 for ``T1 <= 5 keV``; ``1 - 50 eV / T1`` (a 50 eV outgoing electron) for
    ``5 keV < T1 <= 500 keV``; 0.9999 above. The node file names round ``k``
    to four digits, which is too coarse to recover this near ``k = T1``.
    """
    t1 = np.asarray(t1_MeV, dtype=np.float64)
    return np.where(
        t1 <= 5.0e-3,
        0.99,
        np.where(t1 <= 0.5, 1.0 - 50.0e-6 / t1, 0.9999),
    )


def solid_angle_integral(theta_rad: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Exact ``2 pi int f sin(theta) dtheta`` of the linear interpolant of ``values``.

    ``values`` carries ``theta`` on its last axis. Per interval ``[a, b]`` of
    width ``h`` with end values ``f_a, f_b``:
    ``int f sin = f_a (cos a - cos b) + (f_b - f_a)/h (sin b - sin a - h cos b)``.
    """
    theta = np.asarray(theta_rad, dtype=np.float64)
    f = np.asarray(values, dtype=np.float64)
    a, b = theta[:-1], theta[1:]
    h = b - a
    ca, cb = np.cos(a), np.cos(b)
    sa, sb = np.sin(a), np.sin(b)
    fa, fb = f[..., :-1], f[..., 1:]
    per_interval = fa * (ca - cb) + (fb - fa) / h * (sb - sa - h * cb)
    return 2.0 * np.pi * per_interval.sum(axis=-1)


def prepare_bremslib_table(
    arrays: Mapping[str, np.ndarray],
    *,
    atomic_number: int,
    key: str = "",
    digest: str = "",
) -> BremsLibBremsstrahlungTable:
    """Stage one stored BremsLib table (``pyrite.xsgen.bremslib`` layout).

    Validation: bremslib-angular-model

    Raises
    ------
    ValueError
        If the table is not a complete ``(T1, k/T1)`` node grid, or a node's
        DDCS has no positive solid-angle integral.
    """
    t1_MeV = np.asarray(arrays["t1_MeV"], dtype=np.float64)
    nominal = np.asarray(arrays["k_over_t1"], dtype=np.float64)
    sdcs = np.asarray(arrays["sdcs_mb"], dtype=np.float64)
    n_t1 = t1_MeV.size
    if nominal.shape != (_RATIO_COUNT,) or sdcs.shape != (n_t1, _RATIO_COUNT) or n_t1 < 2:
        raise ValueError("BremsLib table does not hold a (T1, 13 k/T1) SDCS grid")
    if np.any(np.diff(t1_MeV) <= 0.0) or t1_MeV[0] <= 0.0:
        raise ValueError("BremsLib T1 grid must be positive and ascending")
    if nominal[0] != 0.0 or np.any(np.diff(nominal) <= 0.0):
        raise ValueError("BremsLib k/T1 grid must ascend from zero")
    if np.any(sdcs < 0.0) or not np.all(np.isfinite(sdcs)):
        raise ValueError("BremsLib SDCS must be finite and non-negative")

    t1_index = np.asarray(arrays["node_t1_index"], dtype=np.int64)
    k_index = np.asarray(arrays["node_k_index"], dtype=np.int64)
    expected_t1 = np.repeat(np.arange(n_t1), _RATIO_COUNT)
    expected_k = np.tile(np.arange(_RATIO_COUNT), n_t1)
    if not (np.array_equal(t1_index, expected_t1) and np.array_equal(k_index, expected_k)):
        raise ValueError("BremsLib table does not hold every (T1, k/T1) DDCS node in order")

    offsets = np.asarray(arrays["node_offset"], dtype=np.int64)
    theta_deg = np.asarray(arrays["theta_deg"], dtype=np.float64)
    ddcs = np.asarray(arrays["ddcs_mb_sr"], dtype=np.float64)
    # Every node grid is a subset of the finest one, so linear resampling onto
    # the union reproduces each node's own linear interpolant exactly.
    theta_common_deg = np.unique(np.round(theta_deg, 9))
    if theta_common_deg[0] != 0.0 or theta_common_deg[-1] != 180.0:
        raise ValueError("BremsLib DDCS theta grids must span 0 to 180 degrees")
    theta_common = np.radians(theta_common_deg)

    resampled = np.empty((n_t1 * _RATIO_COUNT, theta_common.size), dtype=np.float64)
    for node in range(n_t1 * _RATIO_COUNT):
        start, stop = offsets[node], offsets[node + 1]
        resampled[node] = np.interp(theta_common_deg, theta_deg[start:stop], ddcs[start:stop])
    integral = solid_angle_integral(theta_common, resampled)
    if np.any(~np.isfinite(integral)) or np.any(integral <= 0.0):
        raise ValueError("a BremsLib DDCS node has no positive finite angular integral")
    shape = (resampled / integral[:, None]).reshape(n_t1, _RATIO_COUNT, theta_common.size)

    t1_fine, sdcs_fine, shape_fine = _refine_incident_grid(t1_MeV, sdcs, shape, theta_common)
    return BremsLibBremsstrahlungTable(
        atomic_number=int(atomic_number),
        key=str(key),
        digest=str(digest),
        incident_energy_keV=t1_fine * 1.0e3,
        nominal_reduced_energy=nominal,
        top_reduced_energy=top_reduced_energy(t1_fine),
        scaled_sdcs_mb=sdcs_fine,
        theta_rad=theta_common,
        scaled_ddcs_mb_sr=shape_fine * sdcs_fine[..., None],
    )


def _refine_incident_grid(t1_MeV, sdcs, shape, theta):
    """Insert geometric sub-nodes between adjacent ``T1`` grid values.

    The library's ``T1`` spacing (ratios up to 1.33) is coarse for the angular
    shape: backward emission falls steeply and convexly with ``T1``, so plain
    linear interpolation overshoots it by up to ~5 % near the tip. Between two
    nodes each inserted sub-node takes the shape as the weighted geometric mean
    of its neighbours' shapes (linear in ``ln S`` versus ``ln T1``),
    renormalized to unit solid-angle integral, and the SDCS log-log in ``T1``.
    Runtime interpolation on the refined grid is then linear, so the exact
    angular normalization of every convex combination is kept. Original nodes
    are reproduced unchanged.
    """
    refine = _INCIDENT_REFINEMENT
    fractions = np.arange(refine) / refine
    log_t1 = np.log(t1_MeV)
    t1_parts, sdcs_parts, shape_parts = [], [], []
    tiny = np.finfo(np.float64).tiny
    for row in range(t1_MeV.size - 1):
        w = fractions[:, None]
        t1_parts.append(np.exp(log_t1[row] + fractions * (log_t1[row + 1] - log_t1[row])))
        s0, s1 = sdcs[row], sdcs[row + 1]
        positive = (s0 > 0.0) & (s1 > 0.0)
        geometric = np.exp(
            (1.0 - w) * np.log(np.maximum(s0, tiny)) + w * np.log(np.maximum(s1, tiny))
        )
        sdcs_parts.append(np.where(positive, geometric, (1.0 - w) * s0 + w * s1))
        wt = fractions[:, None, None]
        log0 = np.log(np.maximum(shape[row], tiny))
        log1 = np.log(np.maximum(shape[row + 1], tiny))
        mixed = np.exp((1.0 - wt) * log0[None] + wt * log1[None])
        mixed /= solid_angle_integral(theta, mixed)[..., None]
        mixed[0] = shape[row]  # the original node itself, bit for bit
        shape_parts.append(mixed)
    t1_parts.append(t1_MeV[-1:])
    sdcs_parts.append(sdcs[-1:])
    shape_parts.append(shape[-1:])
    return np.concatenate(t1_parts), np.concatenate(sdcs_parts), np.concatenate(shape_parts)


@dataclass(frozen=True, slots=True)
class _StagedBremsLib:
    """A table's arrays on the active backend."""

    table: BremsLibBremsstrahlungTable
    log_incident_energy: Any
    inner_reduced_energy: Any
    top_reduced_energy: Any
    scaled_sdcs_mb: Any
    theta_rad: Any
    scaled_ddcs_mb_sr: Any


def stage_bremslib_table(table: BremsLibBremsstrahlungTable) -> _StagedBremsLib:
    """Upload one table's interpolation arrays once per spectrum call."""

    def staged(values):
        return xp.ascontiguousarray(xp.asarray(values, dtype=REAL))

    return _StagedBremsLib(
        table=table,
        log_incident_energy=staged(np.log(table.incident_energy_keV)),
        # Interior breakpoints searched for the lower node; node 0 is k/T = 0.
        inner_reduced_energy=staged(table.nominal_reduced_energy[1:-1]),
        top_reduced_energy=staged(table.top_reduced_energy),
        scaled_sdcs_mb=staged(table.scaled_sdcs_mb),
        theta_rad=staged(table.theta_rad),
        # ``(T1, theta, k/T1)`` on the device: the per-segment gather below
        # then uses adjacent advanced indices, which dpnp requires.
        scaled_ddcs_mb_sr=staged(np.swapaxes(table.scaled_ddcs_mb_sr, 1, 2)),
    )


@dataclass(frozen=True, slots=True)
class _BremsLibSegmentState:
    """Per-segment incident-energy bracket and node values, O(Nsegment * 13)."""

    incident_energy_keV: Any
    available: Any
    lower_row: Any
    energy_fraction: Any
    lower_values: Any
    upper_values: Any


def _incident_bracket(staged: _StagedBremsLib, T_keV):
    T = xp.asarray(T_keV, dtype=REAL)
    log_grid = staged.log_incident_energy
    table = staged.table
    available = (T >= REAL(table.minimum_incident_energy_keV)) & (
        T <= REAL(table.maximum_incident_energy_keV)
    )
    log_T = xp.clip(xp.log(xp.maximum(T, REAL(1.0e-30))), log_grid[0], log_grid[-1])
    row = xp.clip(xp.searchsorted(log_grid, log_T, side="right") - 1, 0, log_grid.size - 2)
    fraction = (log_T - log_grid[row]) / (log_grid[row + 1] - log_grid[row])
    return T, available, row, xp.clip(fraction, REAL(0.0), REAL(1.0))


def bremslib_segment_state(staged: _StagedBremsLib, T_keV, cos_theta=None):
    """Bracket each segment in ``ln T`` and gather its 2 x 13 node values.

    With ``cos_theta`` the node values are the scaled DDCS at each segment's
    emission angle, interpolated linearly in ``theta``; without it they are
    the scaled SDCS.
    """
    T, available, row, fraction = _incident_bracket(staged, T_keV)
    if cos_theta is None:
        lower = staged.scaled_sdcs_mb[row]
        upper = staged.scaled_sdcs_mb[row + 1]
    else:
        theta_grid = staged.theta_rad
        theta = xp.arccos(xp.clip(xp.asarray(cos_theta, dtype=REAL), REAL(-1.0), REAL(1.0)))
        cell = xp.clip(xp.searchsorted(theta_grid, theta, side="right") - 1, 0, theta_grid.size - 2)
        weight = (theta - theta_grid[cell]) / (theta_grid[cell + 1] - theta_grid[cell])
        weight = xp.clip(weight, REAL(0.0), REAL(1.0))[:, None]
        ddcs = staged.scaled_ddcs_mb_sr

        def at_angle(rows):
            left = ddcs[rows, cell]
            right = ddcs[rows, cell + 1]
            return left + weight * (right - left)

        lower = at_angle(row)
        upper = at_angle(row + 1)
    return _BremsLibSegmentState(
        incident_energy_keV=T,
        available=available,
        lower_row=row,
        energy_fraction=fraction,
        lower_values=lower,
        upper_values=upper,
    )


def _along_reduced_energy(staged, values, top, reduced):
    """Interpolate ``(M, 13)`` node values linearly in ``k/T`` at ``(M, NE)``.

    Above the top node (``k/T`` between it and 1) the top value is held: the
    library stops a 50 eV (or 1e-4 T) outgoing electron short of the tip.
    """
    inner = staged.inner_reduced_energy
    nominal = staged.table.nominal_reduced_energy
    lower = xp.searchsorted(inner, reduced, side="right")  # 0..11
    x0 = xp.asarray(nominal, dtype=REAL)[lower]
    x1 = xp.where(
        lower == _RATIO_COUNT - 2,
        top[:, None],
        xp.asarray(nominal, dtype=REAL)[xp.minimum(lower + 1, _RATIO_COUNT - 1)],
    )
    fraction = xp.clip((reduced - x0) / (x1 - x0), REAL(0.0), REAL(1.0))
    rows = xp.arange(values.shape[0])[:, None]
    v0 = values[rows, lower]
    v1 = values[rows, lower + 1]
    return v0 + fraction * (v1 - v0)


def evaluate_bremslib(staged: _StagedBremsLib, state: _BremsLibSegmentState, photon_energy_eV):
    """Return ``d sigma/dk`` [cm²/eV] or ``d2 sigma/(dk dOmega)`` [cm²/eV/sr].

    Which one depends on whether ``state`` was built with emission angles.
    Shape ``(Nsegment, NE)``; zero outside ``0 < k <= T``.

    Validation: bremslib-angular-model
    """
    photon_eV = xp.asarray(photon_energy_eV, dtype=REAL)
    T_eV = state.incident_energy_keV * REAL(1.0e3)
    reduced = photon_eV[None, :] / xp.maximum(T_eV, REAL(1.0e-30))[:, None]
    top = staged.top_reduced_energy
    lower = _along_reduced_energy(staged, state.lower_values, top[state.lower_row], reduced)
    upper = _along_reduced_energy(staged, state.upper_values, top[state.lower_row + 1], reduced)
    scaled = lower + state.energy_fraction[:, None] * (upper - lower)
    Z = REAL(staged.table.atomic_number)
    physical = (photon_eV[None, :] > REAL(0.0)) & (reduced <= REAL(1.0))
    return xp.where(
        physical,
        scaled * REAL(_MB_CM2) * Z * Z / xp.maximum(photon_eV[None, :], REAL(1.0e-30)),
        REAL(0.0),
    )


__all__ = [
    "BREMSSTRAHLUNG_BREMSLIB_MODEL",
    "BremsLibBremsstrahlungTable",
    "bremslib_segment_state",
    "evaluate_bremslib",
    "prepare_bremslib_table",
    "solid_angle_integral",
    "stage_bremslib_table",
    "top_reduced_energy",
]
