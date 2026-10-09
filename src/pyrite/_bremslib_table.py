"""Stage stored BremsLib tables for the bremsstrahlung cross sections.

A leaf below both :mod:`pyrite.montecarlo` and :mod:`pyrite.xsgen`: the
spectrum evaluates the staged table, and :mod:`pyrite.xsgen.bremslib.tables`
stages the tables it resolves, so neither imports the other for it. The
public names stay importable from
:mod:`pyrite.montecarlo.spectrum.brem_bremslib`.

At load, each ``T1`` interval is refined with geometric sub-nodes (see
:func:`_refine_incident_grid`) and every node's DDCS is renormalized so that
the solid-angle integral of its linear-in-``theta`` interpolant equals that
node's SDCS exactly. Source equations, assumptions, and limiting cases are
documented in ``docs/physics/radiation-physics/bremsstrahlung.md``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

#: Identity of the BremsLib model and of this module's interpolation scheme.
#: Bump the suffix whenever the interpolation changes the numbers it serves.
BREMSSTRAHLUNG_BREMSLIB_MODEL = "bremslib-v2.0-sdcs-ddcs/lin-x-geo4-ln-t-lin-theta-v1"

#: Number of ``k/T1`` grid values per incident energy.
_RATIO_COUNT = 13

#: Sub-intervals per library ``T1`` interval in the staged grid.
_INCIDENT_REFINEMENT = 4


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

    Validation: bremslib-angular-schiff
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

    Validation: bremslib-angular-model
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

    Validation: bremslib-angular-model
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
