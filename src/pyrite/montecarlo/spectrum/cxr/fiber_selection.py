"""Measured reflection selection over the full fiber domain (#1484).

The basal-domain selector (``reflection-yield-tolerance-selector``) stops on
``(00l)`` shells only. Here the candidate shells are the full reciprocal
lattice of the opt-in fiber model, ``G_{k-1} < |g| <= G_k``: a shell is a union
of complete families, closed under Friedel pairs and 60-degree rolls because
``|g|`` is invariant under both. Each shell's row is its roll-averaged in-band
line yield (the model's sector quadrature, same roll count as production),
less the in-band continuum its projected-tensor share removes (#1250 net
metric). The tail estimate and stop are
:func:`~pyrite.montecarlo.runner.reflection_cutoff.select_reflection_cutoff`
unchanged, on the case's electron prefix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from ...._backend import _to_cpu
from ....materials.crystal import CRYSTALS, _reflection_families
from ...runner.reflection_cutoff import (
    DEFAULT_SHELL_WIDTH_INVANG,
    REFLECTION_SELECTION_MODEL,
    SELECTION_HISTORIES,
    SELECTION_ROWS,
    _metric_record,
    axis_band_eV,
    axis_yield_weights,
    select_reflection_cutoff,
)
from .fiber import FIBER_CXR_MODEL, FiberCXRModel, FiberReflectionSet, check_correction_inputs

#: Candidate domain recorded with a fiber selection and hashed into its set identity.
FIBER_SELECTION_DOMAIN = "full-fiber"


def iter_fiber_shells(
    model: FiberCXRModel, width_invang: float = DEFAULT_SHELL_WIDTH_INVANG
) -> Iterator[tuple[float, list[tuple[int, int, int]]]]:
    """Full-lattice shells ``(G - width, G]`` up to ``model.cutoff_invang``, one at a time.

    Members are the non-extinct reflections :meth:`FiberCXRModel.prepare`
    keeps (zero-B enumeration, metric above 1e-9 of the strongest family), so
    the union of the shells up to ``G`` is ``prepare()`` at cutoff ``G``
    whenever ``G`` exceeds the strongest family's ``|g|`` (HOPG ``(002)``,
    1.87 1/Angstrom). Every shell is closed under Friedel pairs and 60-degree
    rolls. Empty shells are skipped; a search that stops at ``G`` never
    enumerates past it.

    Validation: reflection-yield-tolerance-selector
    """
    cap = float(model.cutoff_invang)
    width = float(width_invang)
    if not 0.0 < width:
        raise ValueError("shell width must be positive")
    first = _reflection_families(model.crystal, 1000.0, 0.0, False, min(cap, 8.0), full_fiber=True)
    floor = 1e-9 * max((metric for _, metric in first), default=0.0)
    seen: set = set()
    upper = 0.0
    while upper < cap:
        upper = min(upper + width, cap)
        if cap - upper < 1e-9 * cap:
            upper = cap
        families = _reflection_families(model.crystal, 1000.0, 0.0, False, upper, full_fiber=True)
        members = sorted(
            hkl
            for (_, metric), hkls in families.items()
            if metric > 0.0 and metric >= floor
            for hkl in hkls
            if hkl not in seen
        )
        seen.update(members)
        if members:
            yield float(upper), members


class _FiberShellContinuum:
    """Net-metric continuum of the fiber selection: background and tensor shares.

    A :class:`~pyrite.montecarlo.runner.lattice_share.SelectionContinuum` on
    the selection prefix supplies the uncorrected in-band background and the
    yield of any tables; a shell's share is the continuum its projected-tensor
    shell weights remove, scored from ``base - corrected`` tables (the same
    correction :meth:`FiberReflectionSet.corrected_bremslib_tables` applies
    to the selected set).

    Validation: reflection-yield-tolerance-selector
    """

    def __init__(self, model, case, segments, n_hat, band_eV, *, groove, Ne, floor_keV, max_keV):
        from ...runner.lattice_share import SelectionContinuum

        self._scorer = SelectionContinuum(
            case, segments, n_hat, case.get("abs_layers"), band_eV, groove=groove, Ne=Ne
        )
        if self._scorer._tables is None or "C" not in self._scorer._tables:
            raise ValueError("fiber selection requires a carbon BremsLib table")
        self._model = model
        self._composition = [(str(el), float(n)) for el, n in case["composition"]]
        self._limits = dict(E_cut_keV=float(floor_keV), max_energy_keV=float(max_keV))

    def background(self) -> float:
        return self._scorer.background(None)

    def share(self, hkls) -> float:
        from ...runner.lattice_share import removed_share_tables

        base = self._scorer._tables
        corrected, _ = self._model.domain(hkls).corrected_bremslib_tables(
            base, self._composition, **self._limits
        )
        tables = removed_share_tables(base, corrected, self._composition)
        return self._scorer._yield(None, tables, True)


def check_selection_arguments(grid: np.ndarray, tolerance: float, histories=None) -> None:
    """Refuse a fiber selection before any staging or scoring.

    The tolerance must lie in ``(0, 1)``. The continuum background is scored on
    a log grid over the ``bin-mean`` band, so the band's lower edge must be
    positive: the first bin's mirrored edge ``grid[0] - (grid[1] - grid[0])/2``
    is clamped to 0 eV when the first spacing is at least ``2 * grid[0]``.

    Validation: reflection-yield-tolerance-selector
    """
    tolerance = float(tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("line_yield_tolerance must lie in (0, 1)")
    if histories is not None and int(histories) < 1:
        raise ValueError("selection histories must be positive")
    lo, hi = axis_band_eV(grid, "bin-mean")
    if not 0.0 < lo < hi:
        raise ValueError(
            "fiber selection needs a positive lower band edge: make the first photon-grid "
            "spacing smaller than twice the first node"
        )


def selection_prefix(segments: dict, count: int, histories=None, rows=None):
    """``(k, rows, segments)``: the electron prefix the selection scores (#1281, #1368).

    ``k`` is the smallest electron count whose segment rows reach ``rows``
    (default ``SELECTION_ROWS``), at most ``histories`` (default
    ``SELECTION_HISTORIES``) and the population ``count``.
    """
    from ...runner.electron_blocks import electron_prefix, row_budget_prefix

    budget = int(SELECTION_HISTORIES if histories is None else histories)
    row_budget = int(SELECTION_ROWS if rows is None else rows)
    if budget < 1 or row_budget < 1:
        raise ValueError("selection histories and rows must be positive")
    k, prefix_rows = row_budget_prefix([segments], min(budget, int(count)), row_budget)
    if k >= int(count):
        return int(count), prefix_rows, segments
    return k, prefix_rows, electron_prefix(segments, k)


def select_fiber_reflections(
    model: FiberCXRModel,
    pilot: dict,
    tp: dict,
    grid: np.ndarray,
    tolerance: float,
    *,
    floor_keV: float,
    max_keV: float,
    histories: int | None = None,
    rows: int | None = None,
    width_invang: float = DEFAULT_SHELL_WIDTH_INVANG,
) -> tuple[FiberReflectionSet, np.ndarray | None, dict, object]:
    """Measured cutoff of the full fiber domain on ``pilot``'s transported scene.

    ``pilot`` is the validated single-slab HOPG case of
    :func:`~pyrite.montecarlo.spectrum.cxr.fiber_case.score_fiber_case` and
    ``grid`` its ``bin-mean`` photon axis; the band is that axis's outer bin
    edges. Shells of :func:`iter_fiber_shells` up to ``model.cutoff_invang``
    (the search cap) are scored on the electron prefix of
    :func:`selection_prefix`: each shell's line yield is its sector-quadrature
    roll average with ``model.azimuth_count`` nodes, less its tensor
    continuum share; the background is the uncorrected in-band continuum.
    The stop is ``select_reflection_cutoff``'s net form.

    Returns the selected set (prepared at the selected cutoff, so its
    ``record()`` names that cutoff), the shell-summed CXR density when the
    prefix is the whole population (else ``None``: rescore on every
    history), the selection record, and the staged population segments when
    the prefix is the whole population (else ``None``), so the caller stages
    them once. The arguments (:func:`check_selection_arguments`) and the
    continuum correction's carbon table and energy interval
    (:func:`~pyrite.montecarlo.spectrum.cxr.fiber.check_correction_inputs`)
    are checked before anything is staged.

    Validation: reflection-yield-tolerance-selector
    """
    from ... import runner
    from ...runner.case_tables import _case_bremslib_tables

    check_selection_arguments(grid, tolerance, histories)
    check_correction_inputs(
        _case_bremslib_tables(pilot),
        pilot["composition"],
        E_cut_keV=float(floor_keV),
        max_energy_keV=float(max_keV),
    )
    count = int(tp["Ne_lines"])
    k, prefix_rows, segments = selection_prefix(tp["segs"], count, histories, rows)
    staged = runner._segments_on_device(segments)
    n_hat, groove = tp["n_hat"], tp.get("groove")
    phi0 = float(pilot.get("azimuth_rad", 0.0))
    weights = axis_yield_weights(grid, "bin-mean")
    total = np.zeros(grid.shape, dtype=np.float64)

    def line_yield(hkls):
        nonlocal total
        domain = model.domain(hkls)
        density = domain._average(
            row_scorer(staged, grid, pilot, n_hat, groove, k, domain), grid.shape, phi0
        )
        total = total + density
        return float(density @ weights)

    continuum = _FiberShellContinuum(
        model,
        pilot,
        staged,
        n_hat,
        axis_band_eV(grid, "bin-mean"),
        groove=groove,
        Ne=k,
        floor_keV=floor_keV,
        max_keV=max_keV,
    )
    outcome = select_reflection_cutoff(
        line_yield,
        iter_fiber_shells(model, width_invang),
        tolerance,
        share_yield=continuum.share,
        background=continuum.background(),
    )
    selected = FiberCXRModel(
        replace(model.numerics, cutoff_invang=outcome.cutoff_invang),
        model.temperature_K,
        model.crystal,
    ).domain(sorted(outcome.hkl_list))
    record = fiber_selection_record(model, outcome, width_invang)
    if k < count:
        record["selection_histories"] = dict(
            lines=k, continuum=k, rows=prefix_rows, row_budget=int(rows or SELECTION_ROWS)
        )
    if k == count:
        return selected, total, record, staged
    return selected, None, record, None


def flag_clamped_share(record: dict, correction: dict) -> dict:
    """Mark a selection unconverged when the selected set's share correction clamped.

    Shell shares are scored per shell; production sums the selected set's
    share before clamping it at the base DDCS. At a clamped node the summed
    shell shares exceed the share production removes, so the net metric no
    longer matches the scored total and the stop is not trusted.

    Validation: reflection-yield-tolerance-selector
    """
    clamped = sum(int(row.get("clamped_nodes", 0)) for row in correction.get("elements", ()))
    record = dict(record, share_clamped_nodes=clamped)
    if clamped:
        record["converged"] = False
    return record


def row_scorer(segments, grid, pilot, n_hat, groove, count, domain):
    """``score(hkls, phi)`` of one roll node through the runner's incoherent line route.

    Validation: cxr-fiber-intensity-average
    """
    from ... import runner

    lookup = domain._thermal_lookup()

    def score(hkls, phi):
        return _to_cpu(
            runner._cxr_for_segments(
                segments,
                grid,
                dict(pilot, hkl_list=hkls, B_ang2=lookup.__getitem__, azimuth_rad=phi),
                n_hat,
                pilot.get("abs_layers"),
                groove,
                coherent=False,
                Ne=count,
                table_cache=None,
            )
        )

    return score


def fiber_selection_record(model: FiberCXRModel, outcome, width_invang: float) -> dict:
    """Selection record; the set identity hashes domain, axis and roll count with the hkls.

    Validation: reflection-yield-tolerance-selector
    """
    identity = dict(
        domain=FIBER_SELECTION_DOMAIN,
        crystal=model.crystal,
        texture_axis_hkl=list(CRYSTALS[model.crystal].get("texture_axis_hkl") or (0, 0, 1)),
        azimuth_count=int(model.azimuth_count),
        hkl_list=sorted(list(h) for h in outcome.hkl_list),
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    tail = outcome.tail_fraction
    return dict(
        model=REFLECTION_SELECTION_MODEL,
        fiber_model=FIBER_CXR_MODEL,
        domain=FIBER_SELECTION_DOMAIN,
        crystal=model.crystal,
        texture_axis_hkl=identity["texture_axis_hkl"],
        azimuth_count=int(model.azimuth_count),
        temperature_K=float(model.temperature_K),
        tolerance=float(outcome.tolerance),
        g_max_invang=float(model.cutoff_invang),
        shell_width_invang=float(width_invang),
        cutoff_invang=float(outcome.cutoff_invang),
        reflections=len(outcome.hkl_list),
        tail_fraction=float(tail) if np.isfinite(tail) else None,
        converged=bool(outcome.converged),
        selected_set_sha256=digest.hexdigest(),
        **_metric_record(outcome),
    )
