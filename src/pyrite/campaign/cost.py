"""Compute-cost proxy for sweep progress weighting (instrumentation, not physics).

Split from ``sweep.py``; ``pyrite.campaign.sweep`` re-exports every public name.
"""

import numpy as np

from .._energy_grid_encoding import decode_energy_grid

# ---- compute-cost proxy (progress weighting; instrumentation, not physics) ----
# The flat "N of M cases" progress readout misrepresents reality because per-case
# cost varies by multiples across a sweep: a 60 keV case runs several times the
# matmul work of a 20 keV one (deeper penetration -> more transport segments,
# wider per-E0 line grid). These helpers estimate the RELATIVE matmul-element
# count per case so bars/meters can denominate by compute instead of case count.
#
# This is instrumentation, NOT a physics claim -- no validation-ledger entry.
# Only the ORDERING of the weights matters (callers normalize by the sweep
# total), so every global constant (the electron mean free path, unit choices)
# cancels and the proxy is deliberately kept coarse: do not over-fit it.

_COST_E_CUT_KEV = 5.0  # matches transport.simulate_trajectories' default E_cut_keV


def _dEds_magnitude_keV_per_ang(composition, E_keV):
    """Transport stopping-power magnitude [keV/Angstrom].

    Delegates to ``montecarlo.transport.spliced_stopping_keV_per_ang`` rather
    than re-stating the constants, so the cost proxy's CSDA range cannot drift
    from the transport model whose runtime it predicts -- the copy that used to
    live here went stale the moment the stopping model changed. ``sweep.py``
    already imports ``montecarlo.case``, so this adds no import cost.
    Additive over elements with number densities ``n_i`` [1/Angstrom^3];
    ``E_keV`` may be a scalar or an array."""
    from ..montecarlo.transport import spliced_stopping_keV_per_ang

    # magnitude only (transport uses the negative)
    return -spliced_stopping_keV_per_ang(composition, E_keV)


def _csda_profile(composition, E0_keV, e_cut_keV=_COST_E_CUT_KEV, n_quad=64):
    """Cumulative continuous-slowing-down range [Angstrom] vs energy.

    Returns ``(E, cum)`` on ``n_quad`` points from ``e_cut_keV`` to ``E0_keV``,
    where ``cum[i]`` is the CSDA path length from ``e_cut`` up to ``E[i]``
    (trapezoid integral of ``1/|dE/ds|``). ``cum[-1]`` is the full range at
    ``E0``. Monotonic in ``E0``; used both for the range and to invert it (how
    far a slab lets an electron travel before it enters the next layer)."""
    E = np.linspace(e_cut_keV, E0_keV, n_quad)
    inv_dEds = 1.0 / _dEds_magnitude_keV_per_ang(composition, E)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv_dEds[1:] + inv_dEds[:-1]) * np.diff(E))])
    return E, cum


def _mean_segments_per_electron(case):
    """Relative transport-segment count per incident electron for a case.

    N_seg per electron ~ (path length) / (mean free path); taking the free path
    as a case-independent constant (it cancels in the normalized weights), the
    ordering is set by the CSDA path length through the stack. Electrons slow
    from ``E0`` and deposit over ``min(range-in-layer, layer thickness)`` before
    entering the next layer at the reduced energy, so a multilayer/stack case
    walks its ``abs_layers`` rather than assuming a single slab; a bare slab is
    the one-layer case over ``thickness_ang``."""
    abs_layers = case.get("abs_layers")
    if abs_layers:
        layers = [(float(z_bot) - float(z_top), comp) for (z_top, z_bot, comp) in abs_layers]
    else:
        layers = [(float(case["thickness_ang"]), case["composition"])]
    path = 0.0
    E = float(case["E0_keV"])
    for thickness_ang, comp in layers:
        if E <= _COST_E_CUT_KEV:
            break
        grid, cum = _csda_profile(comp, E)
        total_range = float(cum[-1])
        if total_range <= thickness_ang:
            path += total_range  # electron stops inside this layer
            break
        path += thickness_ang  # electron crosses this layer, enters the next slower
        E = float(np.interp(total_range - thickness_ang, cum, grid))
    return path


def case_cost(case):
    """Estimated RELATIVE matmul-element count for one ``run_case`` dict.

    The dominant runtime is the chunked ``mc_spectrum`` / ``mc_brem_spectrum``
    matmuls of shape ``(n_segments, nbins)``, so per case::

        cost ~ N_seg_line * nbins_line  +  N_seg_brem * nbins_brem

    with ``N_seg = electrons * segments-per-electron`` (see
    :func:`_mean_segments_per_electron`) and ``nbins_line`` / ``nbins_brem`` the
    evaluated grid widths (both vary by ``E0``). Pure function of the case dict;
    runs no transport. Callers normalize by the sweep total, so the absolute
    scale is meaningless -- only the ordering across cases is used."""
    nbins_line = decode_energy_grid(case.get("E_grid_line", case["E_grid"])).size
    brem = case.get("E_grid_brem")
    nbins_brem = decode_energy_grid(brem).size if brem is not None else 0
    steps = _mean_segments_per_electron(case)
    n_line = float(case.get("Ne", 0) or 0)
    n_brem = float(case.get("Ne_brem", 0) or 0)
    return steps * (n_line * nbins_line + n_brem * nbins_brem)


def sweep_cost_weights(cases):
    """Relative compute weight of every case: ``({(name, E0_keV): cost}, total)``.

    Deterministic from the case list alone, so a viewer can rebuild the same
    weights from a reconstructed sweep without the weights being shipped in a
    checkpoint. ``total`` is ``sum(weights.values())`` (``0.0`` for no cases)."""
    weights = {(c["name"], float(c["E0_keV"])): case_cost(c) for c in cases}
    return weights, float(sum(weights.values()))


def _grid_key(case):
    """(name, E0_keV) progress key for a case -- what checkpoints/weights index on."""
    return (case["name"], float(case["E0_keV"]))


def scan_grid_rows(cases, *, cached=(), excluded=(), running=(), done=(), weights=None):
    """Aggregate a case list into rows for the notebook ``scan_grid`` renderer.

    One row per distinct polar tilt, one cell per distinct beam energy; each cell
    reports the FRACTION of its ``(tilt, E0_keV)`` cases in each state and the
    SUMMED relative compute cost of that group (so the renderer can size the
    heavy corner). Pass every requested case (kept + penetration-excluded) as
    ``cases`` so fully-excluded or unrequested ``(tilt, E0)`` slots render empty.

    ``cached`` / ``running`` / ``done`` are iterables of ``(name, E0_keV)`` keys
    (see :func:`sweep_cost_weights`); ``excluded`` may be either such keys or the
    dropped case dicts (as returned by ``config.gate_cases_by_penetration``).
    ``weights`` is the ``{(name, E0): cost}`` map (default: recomputed via
    :func:`case_cost`). Returns ``(energies, rows)`` with ``energies`` sorted
    ascending. Instrumentation only -- no physics claim."""
    from collections import defaultdict

    cached, running, done = set(cached), set(running), set(done)
    excluded_keys = {item if isinstance(item, tuple) else _grid_key(item) for item in excluded}
    if weights is None:
        weights = {_grid_key(c): case_cost(c) for c in cases}

    energies = sorted({float(c["E0_keV"]) for c in cases})
    tilts = sorted({float(c["tilt_deg"]) for c in cases})
    agg: dict[tuple[float, float], dict[str, float]] = defaultdict(
        lambda: {"total": 0, "cached": 0, "done": 0, "running": 0, "excluded": 0, "weight": 0.0}
    )
    for c in cases:
        tilt, E0 = float(c["tilt_deg"]), float(c["E0_keV"])
        key = _grid_key(c)
        cell = agg[(tilt, E0)]
        cell["total"] += 1
        cell["weight"] += weights.get(key, 0.0)
        if key in excluded_keys:
            cell["excluded"] += 1
        elif key in done:
            cell["done"] += 1
        elif key in running:
            cell["running"] += 1
        elif key in cached:
            cell["cached"] += 1

    rows = []
    for tilt in tilts:
        cells = []
        for E0 in energies:
            cell = agg.get((tilt, E0))
            if not cell or cell["total"] == 0:
                cells.append(None)
                continue
            n = cell["total"]
            cells.append(
                {
                    "cached": cell["cached"] / n,
                    "done": cell["done"] / n,
                    "running": cell["running"] / n,
                    "excluded": cell["excluded"] / n,
                    "weight": cell["weight"],
                }
            )
        rows.append({"label": f"{tilt:g}°", "cells": cells})
    return energies, rows
