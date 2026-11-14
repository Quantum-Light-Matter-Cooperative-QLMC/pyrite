"""Measured reciprocal-cutoff selection for incoherent CXR lines (#1173).

A fixed family count does not bound the omitted line yield: on the #642 scenes
``n_families=4`` misses 61--78 % of the 0.5--10 keV incoherent line yield. This
module chooses instead the reciprocal cutoff ``G`` from the case's own
transport. The non-forbidden reflections are grouped into shells
``|g| in (G - width, G]``. Each shell's in-band line yield is measured with the
production line kernels, and the shell sequence stops once the estimated
in-band yield beyond ``G`` is at most ``tolerance`` of the yield kept.

The scorer is a callable, so this module holds no runner state. The caller
(the #642 gate, and later the case driver) evaluates the incoherent line
spectrum of one shell on a ``bin-mean`` axis spanning the band, where the
integrated yield is exact at any bin width.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...materials.crystal import CRYSTALS, dominant_reflections, reciprocal_g_vector

__all__ = [
    "DEFAULT_SHELL_WIDTH_INVANG",
    "F0_RANGE_INVANG",
    "ReflectionCutoff",
    "band_bin_grid",
    "geometric_tail_fraction",
    "reflection_shells",
    "select_reflection_cutoff",
]

#: Shell width in 1/Angstrom. Above |g| ~ 20 the measured Si shell shares
#: halve per 2 /A, so a shell is one Debye--Waller e-fold of the tail.
DEFAULT_SHELL_WIDTH_INVANG = 2.0

#: Waasmaier--Kirfel f0 validity, s = |g| / 4 pi <= 6 /A.
F0_RANGE_INVANG = 4.0 * np.pi * 6.0

#: Shells after the first decaying pair before the tail estimate may stop.
_MIN_DECAYING_SHELLS = 2


def band_bin_grid(start_eV: float, stop_eV: float, bin_eV: float = 500.0) -> np.ndarray:
    """Node coordinates whose ``bin-mean`` bins tile ``[start_eV, stop_eV]`` exactly.

    The bin-mean routes take midpoints between nodes as inner edges and reflect
    the outer half-widths, so uniform nodes at ``start + h (i + 1/2)`` give
    bins ``[start + h i, start + h (i + 1)]``. The line yield in the band is
    then ``density.sum() * h``.
    """
    start, stop = float(start_eV), float(stop_eV)
    if not 0.0 < start < stop:
        raise ValueError("band must satisfy 0 < start < stop")
    count = max(4, int(np.ceil((stop - start) / float(bin_eV))))
    width = (stop - start) / count
    return start + width * (np.arange(count) + 0.5)


def reflection_shells(
    crystal: str,
    B_ang2: float,
    g_max_invang: float,
    width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
) -> list[tuple[float, list[tuple[int, int, int]]]]:
    """Non-forbidden reflections up to ``g_max_invang``, in ``(upper |g|, hkls)`` shells.

    The members are exactly those :func:`~pyrite.materials.crystal.dominant_reflections`
    keeps at ``g_max_invang`` with no family limit, so the union of the shells
    up to ``G`` is that function's set at cutoff ``G``. Empty shells are dropped.
    """
    if not 0.0 < g_max_invang <= F0_RANGE_INVANG:
        raise ValueError(f"g_max_invang must lie in (0, {F0_RANGE_INVANG:.1f}] 1/Angstrom")
    hkls = dominant_reflections(crystal, n_families=10**9, B_ang2=B_ang2, g_max_invang=g_max_invang)
    lattice = CRYSTALS[crystal]["lattice"]
    g = np.array([np.linalg.norm(reciprocal_g_vector(hkl, lattice)[0]) for hkl in hkls])
    edges = np.arange(width_invang, g_max_invang + width_invang, width_invang)
    edges[-1] = max(edges[-1], g_max_invang)
    shell = np.searchsorted(edges, g - 1e-9 * g_max_invang)
    out = []
    for index, upper in enumerate(edges):
        members = [hkl for hkl, s in zip(hkls, shell, strict=True) if s == index]
        if members:
            out.append((float(min(upper, g_max_invang)), members))
    return out


def geometric_tail_fraction(shell_yields: Sequence[float]) -> float:
    """Estimated yield beyond the last shell, as a fraction of the shells so far.

    With successive shell yields ``s_{k-1}, s_k`` and ratio ``r = s_k / s_{k-1}``,
    a geometric continuation leaves ``s_k r / (1 - r)``. The ratio is the
    Debye--Waller factor ``exp(-2 B Delta(g^2) / (16 pi^2))`` times the shell
    multiplicity growth, so it falls with ``G``. That makes the estimate
    conservative once the sequence decays.

    Returns ``inf`` until the last two shells decay (``0 < r < 1``).

    Limiting case: an exactly geometric sequence ``s_k = r^k`` returns its exact
    remaining sum over its partial sum.

    Validation: reflection-yield-tolerance-selector
    """
    yields = np.asarray(shell_yields, dtype=float)
    if yields.size < 2 or not np.all(np.isfinite(yields)) or np.any(yields < 0.0):
        return float("inf")
    kept = float(yields.sum())
    last, previous = float(yields[-1]), float(yields[-2])
    if kept <= 0.0:
        return float("inf")
    if last == 0.0:
        return 0.0
    if previous <= 0.0:
        return float("inf")
    ratio = last / previous
    if not 0.0 < ratio < 1.0:
        return float("inf")
    return last * ratio / (1.0 - ratio) / kept


@dataclass(frozen=True)
class ReflectionCutoff:
    """Outcome of :func:`select_reflection_cutoff`.

    ``cutoff_invang`` is the upper edge of the last kept shell. ``converged``
    is false when ``g_max_invang`` was reached before the tail met the
    tolerance; ``tail_fraction`` then reports the estimate at the cap
    (``inf`` when the sequence never decayed).
    """

    cutoff_invang: float
    hkl_list: list
    tolerance: float
    tail_fraction: float
    converged: bool
    shells: tuple


def select_reflection_cutoff(
    in_band_yield: Callable[[list], float],
    shells: Sequence[tuple[float, list]],
    tolerance: float,
) -> ReflectionCutoff:
    """Smallest shell cutoff whose estimated in-band tail is at most ``tolerance``.

    ``in_band_yield(hkls)`` returns one shell's incoherent line yield in the
    band. Incoherent line yields add over reflections, so shell yields add.
    Scoring stops at the first shell where :func:`geometric_tail_fraction` is
    at most ``tolerance`` and at least two shells have decayed in a row. That
    avoids stopping on a dip between strong low-order shells. If ``shells``
    runs out first, the result keeps every shell and reports
    ``converged=False``.

    Validation: reflection-yield-tolerance-selector
    """
    tol = float(tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    if not shells:
        raise ValueError("no reflection shells to select from")
    yields: list[float] = []
    table = []
    kept: list = []
    decaying = 0
    tail = float("inf")
    for upper, members in shells:
        value = float(in_band_yield(list(members)))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"shell |g| <= {upper:g} gave a nonfinite or negative yield")
        decaying = decaying + 1 if yields and value < yields[-1] else 0
        yields.append(value)
        kept.extend(members)
        tail = geometric_tail_fraction(yields)
        table.append((float(upper), len(members), value, tail))
        if decaying >= _MIN_DECAYING_SHELLS and tail <= tol:
            return ReflectionCutoff(float(upper), kept, tol, tail, True, tuple(table))
    return ReflectionCutoff(float(shells[-1][0]), kept, tol, tail, False, tuple(table))
