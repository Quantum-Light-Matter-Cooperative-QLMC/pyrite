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

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..._numerics import CXR_REFLECTION_SELECTION_MODEL
from ...materials.crystal import (
    CRYSTALS,
    _reflection_families,
    dominant_reflections,
    reciprocal_g_vector,
)

__all__ = [
    "DEFAULT_SHELL_WIDTH_INVANG",
    "F0_RANGE_INVANG",
    "REFLECTION_SELECTION_MODEL",
    "ReflectionCutoff",
    "band_bin_grid",
    "geometric_tail_fraction",
    "axis_yield_weights",
    "iter_reflection_shells",
    "reflection_selection_record",
    "prune_case_reflections",
    "reflection_shells",
    "select_case_reflections",
    "select_reflection_cutoff",
    "selected_line_spectrum",
    "validate_reflection_selection",
]

#: Shell width in 1/Angstrom. Near |g| ~ 30 the measured Si shell shares
#: halve per 2 /A (0.47--0.70 Debye--Waller e-folds per shell at |g| = 20--30).
DEFAULT_SHELL_WIDTH_INVANG = 2.0

#: Waasmaier--Kirfel f0 validity, s = |g| / 4 pi <= 6 /A.
F0_RANGE_INVANG = 4.0 * np.pi * 6.0

#: Identity of the measured shell-cutoff selection; bump on any change that
#: moves a selected set or its spectrum.
REFLECTION_SELECTION_MODEL = CXR_REFLECTION_SELECTION_MODEL

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


def iter_reflection_shells(
    crystal: str,
    B_ang2: float,
    g_max_invang: float = F0_RANGE_INVANG,
    width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
) -> Iterator[tuple[float, list[tuple[int, int, int]]]]:
    """The shells of :func:`reflection_shells`, enumerated one cutoff at a time.

    Shell ``k`` is the set at cutoff ``G_k`` less the set at ``G_{k-1}``, in
    :func:`~pyrite.materials.crystal.dominant_reflections` order, so a search
    that stops at ``G`` never enumerates past it. The shells and their union
    equal :func:`reflection_shells` at the same cap.
    """
    if not 0.0 < g_max_invang <= F0_RANGE_INVANG:
        raise ValueError(f"g_max_invang must lie in (0, {F0_RANGE_INVANG:.1f}] 1/Angstrom")
    # dominant_reflections drops families below 1e-9 of the strongest one it
    # enumerated. The strongest family is low order, so fix it once from the
    # production enumeration; a tiny first cutoff holding only forbidden
    # families would otherwise keep them.
    floor = 1e-9 * max(
        m for _, m in _reflection_families(crystal, 1000.0, B_ang2, False, min(g_max_invang, 8.0))
    )
    seen: set = set()
    upper = 0.0
    while upper < g_max_invang:
        upper = min(upper + width_invang, g_max_invang)
        if g_max_invang - upper < 1e-9 * g_max_invang:
            upper = g_max_invang
        families = _reflection_families(crystal, 1000.0, B_ang2, False, upper)
        members = [
            hkl
            for (_, m), hkls in sorted(families.items(), key=lambda kv: -kv[0][1])
            if m >= floor
            for hkl in sorted(hkls)
            if hkl not in seen
        ]
        seen.update(members)
        if members:
            yield float(upper), members


def geometric_tail_fraction(shell_yields: Sequence[float]) -> float:
    """Estimated yield beyond the last shell, as a fraction of the shells so far.

    A geometric continuation from the last shell ``s_k`` with ratio ``r`` leaves
    ``s_k r / (1 - r)``. The ratio is ``r = max(s_k / s_{k-1}, s_{k-1} / s_{k-2})``,
    the larger of the last two. One ratio alone under-predicts after a
    shell-to-shell dip. Real shells carry ``f0(s)**2`` and lattice-count
    fluctuations (10--40 % in a single ratio) on top of the smooth
    Debye--Waller times multiplicity decay. This is an extrapolation, not a
    bound.

    Returns ``inf`` until three shells exist and both ratios lie in ``(0, 1)``.
    A zero shell never counts as converged.

    Limiting case: an exactly geometric sequence ``s_k = r^k`` returns its exact
    remaining sum over its partial sum.

    Validation: reflection-yield-tolerance-selector
    """
    yields = np.asarray(shell_yields, dtype=float)
    if yields.size < 3 or not np.all(np.isfinite(yields)) or np.any(yields <= 0.0):
        return float("inf")
    ratio = max(float(yields[-1] / yields[-2]), float(yields[-2] / yields[-3]))
    if not ratio < 1.0:
        return float("inf")
    return float(yields[-1]) * ratio / (1.0 - ratio) / float(yields.sum())


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
    shells: Iterable[tuple[float, list]],
    tolerance: float,
) -> ReflectionCutoff:
    """Smallest shell cutoff whose estimated in-band tail is at most ``tolerance``.

    ``in_band_yield(hkls)`` returns one shell's incoherent line yield in the
    band. Incoherent line yields add over reflections, so shell yields add.
    Scoring stops at the first shell where :func:`geometric_tail_fraction` is
    at most ``tolerance`` and at least two shells have decayed in a row. Both
    conditions guard against stopping on a dip between strong low-order shells. If ``shells``
    runs out first, the result keeps every shell and reports
    ``converged=False``.

    Validation: reflection-yield-tolerance-selector
    """
    tol = float(tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
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
    if not table:
        raise ValueError("no reflection shells to select from")
    return ReflectionCutoff(table[-1][0], kept, tol, tail, False, tuple(table))


def _pilot_case(case: dict, layer: int | None, hkls: list, quadrature: str | None) -> dict:
    """``case`` radiating only ``hkls`` from radiator ``layer``.

    ``quadrature`` overrides ``line_quadrature`` when given.
    """
    extra = {} if quadrature is None else {"line_quadrature": quadrature}
    if layer is None:
        return dict(case, hkl_list=hkls, **extra)
    layers: list = [None] * len(case["layer_radiators"])
    layers[layer] = dict(case["layer_radiators"][layer], hkl_list=hkls)
    return dict(case, layer_radiators=layers, hkl_list=hkls, **extra)


def axis_yield_weights(grid: np.ndarray, quadrature: str) -> np.ndarray:
    """Weights ``w`` with ``density @ w`` the integrated yield on ``grid``.

    ``node``: trapezoid weights over the nodes. ``bin-mean``: the bin widths
    of the production bins (midpoint inner edges, reflected outer
    half-widths), on which the integral is exact.
    """
    grid = np.asarray(grid, dtype=float)
    if quadrature == "bin-mean":
        from ..spectrum.characteristic import _energy_bin_edges_and_widths

        return np.asarray(_energy_bin_edges_and_widths(grid)[1], dtype=float)
    if quadrature != "node":
        raise ValueError(f"unknown line quadrature {quadrature!r}")
    if grid.size < 2:
        raise ValueError("a node axis needs at least two nodes")
    step = np.diff(grid)
    weights = np.zeros(grid.size)
    weights[:-1] += 0.5 * step
    weights[1:] += 0.5 * step
    return weights


def select_case_reflections(
    case: dict,
    line_density: Callable[[dict, np.ndarray], np.ndarray],
    tolerance: float,
    grid: np.ndarray,
    weights: np.ndarray,
    *,
    quadrature: str | None = None,
    g_max_invang: float = F0_RANGE_INVANG,
    width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
    max_reflections: int = 512,
) -> tuple[dict, list[ReflectionCutoff], np.ndarray]:
    """Cut each radiator at its measured cutoff and keep the scored spectrum.

    Every crystalline radiator (the slab, or each non-``None`` entry of
    ``layer_radiators``) runs :func:`select_reflection_cutoff` on its own
    candidate shells up to ``g_max_invang``. A shell's yield is
    ``line_density(pilot, grid) @ weights``, where ``pilot`` radiates that
    shell alone from that radiator (``line_quadrature`` set to
    ``quadrature`` when given), in calls of at most ``max_reflections``
    reflections. Incoherent line densities add over reflections and
    radiators, so the sum of the scored shells is the line spectrum of the
    selected set on ``grid``, and a per-radiator tolerance bounds the
    estimated omission of the whole case by the same fraction.

    ``line_density`` is the caller's production line scorer on the case's
    transported segments. A layered case's ``hkl_list`` mirrors its first
    radiator. Returns the pruned case, the per-radiator outcomes in radiator
    order, and the spectrum.

    Validation: reflection-yield-tolerance-selector
    """
    weights = np.asarray(weights, dtype=float)
    chunk = max(1, int(max_reflections))
    layered = case.get("layer_radiators") is not None
    radiators = list(enumerate(case["layer_radiators"])) if layered else [(None, case)]
    out = dict(case)
    if layered:
        out["layer_radiators"] = list(case["layer_radiators"])
    outcomes = []
    total: np.ndarray | None = None
    for layer, radiator in radiators:
        if radiator is None:
            continue

        def shell_yield(hkls: list, layer: int | None = layer) -> float:
            nonlocal total
            shell = 0.0
            for i in range(0, len(hkls), chunk):
                pilot = _pilot_case(case, layer, hkls[i : i + chunk], quadrature)
                density = np.asarray(line_density(pilot, grid), dtype=float)
                total = density if total is None else total + density
                shell += float(density @ weights)
            return shell

        shells = iter_reflection_shells(
            radiator["crystal"], float(radiator["B_ang2"]), g_max_invang, width_invang
        )
        result = select_reflection_cutoff(shell_yield, shells, tolerance)
        outcomes.append(result)
        if layer is None:
            out["hkl_list"] = result.hkl_list
        else:
            out["layer_radiators"][layer] = dict(radiator, hkl_list=result.hkl_list)
    if layered:
        out["hkl_list"] = next(r for r in out["layer_radiators"] if r is not None)["hkl_list"]
    if total is None:
        total = np.zeros(np.shape(grid))
    return out, outcomes, total


def prune_case_reflections(
    case: dict,
    line_density: Callable[[dict, np.ndarray], np.ndarray],
    tolerance: float,
    band_eV: tuple[float, float],
    *,
    bin_eV: float = 500.0,
    g_max_invang: float = F0_RANGE_INVANG,
    width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
    max_reflections: int = 512,
) -> tuple[dict, list[ReflectionCutoff]]:
    """:func:`select_case_reflections` on coarse ``bin-mean`` bins tiling ``band_eV``.

    The bin-mean integral is exact at any bin width, so the pilot measures
    each shell's in-band yield on :func:`band_bin_grid` at ``bin_eV``.
    Returns the pruned case and the per-radiator outcomes.

    Validation: reflection-yield-tolerance-selector
    """
    grid = band_bin_grid(*band_eV, bin_eV)
    pruned, outcomes, _ = select_case_reflections(
        case,
        line_density,
        tolerance,
        grid,
        np.full(grid.size, float(grid[1] - grid[0])),
        quadrature="bin-mean",
        g_max_invang=g_max_invang,
        width_invang=width_invang,
        max_reflections=max_reflections,
    )
    return pruned, outcomes


def selected_line_spectrum(
    case: dict, evaluate: Callable[[dict], object], grid: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Incoherent line spectrum of ``case`` under its ``reflection_selection``.

    ``evaluate(pilot)`` is the spectrum phase's line evaluation of ``pilot``
    (the case radiating one shell chunk) on ``grid``. Shells are scored on the
    production axis in the case's own ``line_quadrature`` and summed, so the
    selection costs about one shell beyond its stop rather than a second pass.
    Returns the host spectrum and the selection record: model, tolerance,
    and per radiator its cutoff, reflection count, tail estimate, convergence
    and shell table.

    Validation: reflection-yield-tolerance-selector
    """
    from ..._backend import _to_cpu

    selection = case["reflection_selection"]
    quadrature = case.get("line_quadrature", "node")
    _, outcomes, spectrum = select_case_reflections(
        case,
        lambda pilot, _grid: np.asarray(_to_cpu(evaluate(pilot)), dtype=float),
        float(selection["tolerance"]),
        grid,
        axis_yield_weights(grid, quadrature),
        g_max_invang=float(selection["g_max_invang"]),
        width_invang=float(selection["shell_width_invang"]),
        max_reflections=int(selection.get("max_reflections", 512)),
    )
    layered = case.get("layer_radiators") is not None
    radiators = [
        (layer, radiator)
        for layer, radiator in (enumerate(case["layer_radiators"]) if layered else [(None, case)])
        if radiator is not None
    ]
    record = dict(
        model=REFLECTION_SELECTION_MODEL,
        tolerance=float(selection["tolerance"]),
        quadrature=quadrature,
        radiators=[
            dict(
                layer=layer,
                crystal=radiator["crystal"],
                cutoff_invang=outcome.cutoff_invang,
                reflections=len(outcome.hkl_list),
                tail_fraction=outcome.tail_fraction if np.isfinite(outcome.tail_fraction) else None,
                converged=outcome.converged,
                shells=[
                    dict(
                        upper_invang=u, reflections=n, yield_=y, tail=t if np.isfinite(t) else None
                    )
                    for u, n, y, t in outcome.shells
                ],
            )
            for (layer, radiator), outcome in zip(radiators, outcomes, strict=True)
        ],
    )
    return spectrum, record


_SELECTION_KEYS = frozenset({"model", "tolerance", "g_max_invang", "shell_width_invang"})


def reflection_selection_record(
    tolerance: float,
    g_max_invang: float = F0_RANGE_INVANG,
    shell_width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
) -> dict:
    """The case record of a measured selection at ``tolerance`` (validated)."""
    record = dict(
        model=REFLECTION_SELECTION_MODEL,
        tolerance=float(tolerance),
        g_max_invang=float(g_max_invang),
        shell_width_invang=float(shell_width_invang),
    )
    validate_reflection_selection(record)
    return record


def validate_reflection_selection(record, case=None) -> None:
    """Refuse a malformed record, or one ``case`` cannot score.

    The selection scores incoherent lines shell by shell, so it refuses
    coherent emission, line roulette, temporal profiles, channeling and the
    two-beam transition-radiation route, whose line paths it does not drive.

    Validation: reflection-yield-tolerance-selector
    """
    if not isinstance(record, dict) or set(record) not in (
        _SELECTION_KEYS,
        _SELECTION_KEYS | {"automatic"},
    ):
        raise ValueError(f"reflection_selection must have keys {sorted(_SELECTION_KEYS)}")
    if "automatic" in record and record["automatic"] is not True:
        raise ValueError("reflection_selection automatic must be True when present")
    if record["model"] != REFLECTION_SELECTION_MODEL:
        raise ValueError(f"unknown reflection selection model {record['model']!r}")
    if not 0.0 < float(record["tolerance"]) < 1.0:
        raise ValueError("reflection_selection tolerance must lie in (0, 1)")
    if not 0.0 < float(record["g_max_invang"]) <= F0_RANGE_INVANG:
        raise ValueError(
            f"reflection_selection g_max_invang must lie in (0, {F0_RANGE_INVANG:.1f}]"
        )
    if not 0.0 < float(record["shell_width_invang"]) < float(record["g_max_invang"]):
        raise ValueError("reflection_selection shell width must lie in (0, g_max_invang)")
    if case is None:
        return
    get = case.get
    refused = [
        name
        for name, active in (
            ("coherent emission", get("coherent_emission")),
            ("line roulette", get("line_roulette") is not None),
            ("temporal profiles", get("temporal_profile")),
            ("channeling", get("channeling") is not None),
            (
                "two-beam transition radiation",
                (get("transition_radiation") or {}).get("model") == "two-beam",
            ),
        )
        if active
    ]
    if refused:
        raise ValueError(
            "reflection_selection scores incoherent lines only; refused with " + ", ".join(refused)
        )
