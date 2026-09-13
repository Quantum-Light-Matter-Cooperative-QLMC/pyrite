"""Energy-grid semantics: the uniform-grid guard and full-grid cache identity.

A package-root leaf, like ``_energy_grid_encoding``: every physics-core package
(``detectors``, ``montecarlo``, ``results``) consumes it, so it must not sit
inside a domain package -- an edge from ``detectors`` into ``energy_grid`` would
put both inside the repository's driver import cycle.
:mod:`pyrite.energy_grid.semantics` is the public facade.

The project's photon-energy conventions are documented in
[Energy-grid semantics](../../docs/physics/radiation-physics/energy-grid-semantics.md).
This module carries the one piece of that contract the code has to enforce:
several consumers read a *single* spacing (``E[1] - E[0]``) and apply it across
the whole grid. That is correct only for a uniform grid, so those consumers call
:func:`require_uniform_grid` and refuse a nonuniform grid explicitly instead of
returning a silently wrong number.

Nothing here changes a uniform-grid result: :func:`require_uniform_grid` returns
exactly ``float(E[1] - E[0])``, the value its callers already used.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "NonuniformEnergyGridError",
    "grid_identity",
    "is_uniform_grid",
    "require_uniform_grid",
    "spacing_spread",
]

#: Relative spread of adjacent spacings still accepted as "uniform", measured
#: after the representation floor below is subtracted. Any deliberately graded
#: or logarithmic grid is off by orders of magnitude, so the exact value is not
#: delicate.
DEFAULT_UNIFORM_RTOL = 1.0e-9

#: Adjacent nodes are only stored to the backend's working precision, so a grid
#: that IS uniform by construction still carries a few ulp of rounding jitter in
#: its differences -- and that jitter is large relative to the step whenever the
#: step is small compared with the absolute energy (a 2.5e-5 eV step at 1600 eV
#: is only ~1e8 float64 ulp wide, giving ~1e-8 relative jitter). The spread is
#: therefore credited this many ulp of the largest node before being compared
#: against ``rtol``.
_REPRESENTATION_ULPS = 16.0


class NonuniformEnergyGridError(ValueError):
    """A uniform-only consumer was handed a nonuniform energy grid.

    Raised rather than silently applying one spacing everywhere. Issue #98
    pinned the semantics and audited the consumers; widening a consumer to real
    nonuniform support is tracked separately (issues #99-#101).
    """


def spacing_spread(E_grid_eV) -> float:
    """Relative spread of the grid spacings, net of the representation floor.

    ``ptp(diff(E))`` is credited ``_REPRESENTATION_ULPS`` ulp of the largest node
    (in the input array's own floating dtype) and then divided by the median
    spacing, so a grid that is uniform by construction reports exactly ``0.0``
    whatever its step-to-energy ratio. Returns ``inf`` for a grid with no
    positive median spacing.
    """
    raw = np.asarray(E_grid_eV)
    dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.dtype(float)
    E = raw.astype(float, copy=False)
    if E.ndim != 1 or E.size < 3:
        # Fewer than three nodes define at most one spacing, so nothing can
        # disagree with anything: treat it as uniform.
        return 0.0
    d = np.diff(E)
    reference = float(np.median(np.abs(d)))
    if not np.isfinite(reference) or reference <= 0.0:
        return float("inf")
    floor = _REPRESENTATION_ULPS * float(np.spacing(dtype.type(np.max(np.abs(E)))))
    return float(max(float(np.ptp(d)) - floor, 0.0) / reference)


def is_uniform_grid(E_grid_eV, *, rtol: float = DEFAULT_UNIFORM_RTOL) -> bool:
    """Whether every adjacent spacing agrees to within ``rtol``."""
    return spacing_spread(E_grid_eV) <= rtol


def require_uniform_grid(
    E_grid_eV,
    *,
    consumer: str,
    remedy: str | None = None,
    rtol: float = DEFAULT_UNIFORM_RTOL,
) -> float:
    """Return the single spacing ``float(E[1] - E[0])``, or refuse the grid.

    Parameters
    ----------
    E_grid_eV
        One-dimensional, strictly increasing photon-energy coordinate in eV.
    consumer
        Dotted name of the calling code, quoted in the error message.
    remedy
        Optional caller-specific sentence appended to the error.
    rtol
        Relative tolerance on the spread of adjacent spacings.

    Raises
    ------
    NonuniformEnergyGridError
        If the spacings disagree by more than ``rtol``.
    """
    if np.asarray(E_grid_eV).ndim != 1 or np.asarray(E_grid_eV).size < 2:
        raise ValueError(f"{consumer}: energy grid must be 1-D with at least two nodes")
    # spread is measured on the ORIGINAL dtype: the representation floor depends
    # on the precision the grid is actually stored (and indexed) in.
    spread = spacing_spread(E_grid_eV)
    E = np.asarray(E_grid_eV, dtype=float)
    if spread > rtol:
        d = np.diff(E)
        message = (
            f"{consumer} requires a UNIFORM energy grid: it applies the single "
            f"spacing E[1] - E[0] = {float(E[1] - E[0]):.6g} eV across the whole "
            f"grid, but the {d.size} spacings range over "
            f"[{float(d.min()):.6g}, {float(d.max()):.6g}] eV "
            f"(relative spread {spread:.3g} > rtol {rtol:.3g}). "
            "Nonuniform (for example logarithmic) photon grids are not yet "
            "supported by this consumer."
        )
        if remedy:
            message = f"{message} {remedy}"
        raise NonuniformEnergyGridError(message)
    return float(E[1] - E[0])


def grid_identity(E_grid_eV) -> tuple[int, float, float, str]:
    """Cache-key prefix that identifies the COMPLETE grid coordinates.

    ``(size, first, last)`` alone does not: two grids can share all three and
    still differ node by node (a linear and a logarithmic grid over the same
    interval, for instance), which silently returns one grid's cached detector
    response for the other. The trailing digest is over the exact float64 bytes,
    so distinct coordinates always produce distinct keys.
    """
    E = np.ascontiguousarray(np.asarray(E_grid_eV, dtype=float))
    digest = hashlib.blake2b(E.tobytes(), digest_size=16).hexdigest()
    return (E.size, round(float(E[0]), 6), round(float(E[-1]), 6), digest)
