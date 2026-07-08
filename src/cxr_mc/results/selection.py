"""
results.selection
==================

Subsetting and reducing a ``results`` store: flat record lists
(:func:`records`, :func:`records_for_cases`), config-name / case-value
filters (:func:`filter_results`, :func:`select_results`, :func:`sweep_values`),
a transfer-size trimmer (:func:`slim_results`), and the azimuth-max reduction
(:func:`best_azimuth`) -- the key knob for big sweeps: for each fixed
(material, thickness, polar tilt, energy) it keeps only the azimuth whose
spectrum has the highest PEAK value ``max(spectrum)`` (not the integrated
flux), collapsing hundreds of azimuth runs to one row/curve each.
"""

import numpy as np


# ---- record selection --------------------------------------------------------
def records(results, names=None):
    """Flat list of every record in ``results`` (optionally restricted to
    ``names``)."""
    keys = list(results) if names is None else [n for n in names if n in results]
    return [results[n][E0] for n in keys for E0 in results[n]]


def records_for_cases(results, cases):
    """Flat record list restricted to the configs in ``cases`` (``cases=None``
    -> every record). The ``cases``-facing wrapper over :func:`records` that
    folds the ``{c["name"] for c in cases}`` prologue every sweep/detector plot
    repeats."""
    names = None if cases is None else {c["name"] for c in cases}
    return records(results, names)


def filter_results(results, cases):
    """Subset ``results`` to just the configs in ``cases`` (e.g. the current
    sweep from build_cases), dropping anything left in the checkpoint from
    earlier sweeps. Pass the result to the plot/table functions to get a clean,
    dense grid instead of the sparse UNION of every sweep ever run:

        res = filter_results(results, cases)
        plot_by_energy(res, settings, collapse_azimuth=True)
        plot_heatmaps(res, settings)
    """
    names = {c["name"] for c in cases}
    return {n: results[n] for n in results if n in names}


def sweep_values(results, fields=None):
    """The distinct swept values present in ``results``, per case field -- a quick
    "what's actually in this checkpoint" introspection so you know what to pass to
    :func:`select_results`. Returns ``{field: sorted list of values}`` for the
    given ``fields`` (default the common geometry/energy knobs), skipping fields
    absent from the records::

        sweep_values(results)
        # {'E0_keV': [30.0], 'tilt_deg': [-89.9, -80.4, ...], 'thickness_ang': [...]}
    """
    if fields is None:
        fields = (
            "crystal",
            "E0_keV",
            "tilt_deg",
            "tilt_azim_deg",
            "thickness_ang",
            "B_ang2",
        )
    out = {}
    for r in records(results):
        for f in fields:
            if f in r["case"]:
                out.setdefault(f, set()).add(r["case"][f])
    return {f: sorted(out[f]) for f in fields if f in out}


def select_results(results, **constraints):
    """Subset ``results`` to the records whose ``case`` matches EVERY constraint --
    a VALUE-based filter for slicing a big accumulated checkpoint down to the
    handful of cases you actually want to plot, WITHOUT building a matching
    build_cases list (cf. :func:`filter_results`, which filters by config NAME).

    This is the fix for "I pulled hopg.pkl to plot a few polar tilts and got every
    hopg case ever run, all overplotted." Each ``field=spec`` constraint is matched
    against ``record['case'][field]``:

      * scalar          -> equals (floats within a tolerance, so an exact swept
                           value like ``tilt_deg=-89.9`` works)
      * list/tuple/set  -> membership, any-of (floats matched within tolerance)
      * callable        -> ``spec(value)`` is truthy -- use for RANGES / fuzzy
                           matches, e.g. ``tilt_deg=lambda t: -60 <= t <= -20`` or
                           ``thickness_ang=lambda x: x <= 5e4``

    A record is kept only if it carries every named field and all constraints pass.
    Returns a new ``{name: {E0: record}}`` store (same nesting), so it drops
    straight into any plotter::

        res = select_results(results, tilt_deg=[-89.9, -50.0, -20.0],
                             thickness_ang=lambda x: x <= 5e4)
        plot_metric_vs(res, settings, x="thickness_ang", hue="tilt_deg")

    Call :func:`sweep_values` first to see which values are available to ask for.
    """

    def _match(val, spec):
        if callable(spec):
            return bool(spec(val))
        if isinstance(spec, (list, tuple, set, np.ndarray)):
            return any(_match(val, s) for s in spec)
        if isinstance(val, (int, float)) and isinstance(spec, (int, float)):
            return bool(np.isclose(val, spec, rtol=1e-9, atol=1e-6))
        return val == spec

    out = {}
    for name, by_E in results.items():
        kept = {
            E0: r
            for E0, r in by_E.items()
            if all(f in r["case"] and _match(r["case"][f], spec) for f, spec in constraints.items())
        }
        if kept:
            out[name] = kept
    return out


# record array fields, by size; the wide-brem pair is the largest (full-range grid)
_RECORD_ARRAY_FIELDS = ("E_grid", "spec", "brem", "E_grid_brem", "brem_wide")
_WIDE_BREM_FIELDS = ("brem_wide", "E_grid_brem")


def _grid_names(material):
    """Config names in the CURRENT grid for ``material`` -- exactly the set
    ``config.material_sweep(material)`` -> ``sweep.build_cases`` produces now.
    A stale config is any name NOT in this set.

    The ``config`` / ``sweep`` imports are function-local on purpose: ``config``
    imports ``results`` at module load (``from .results import Settings``), so a
    module-level ``config`` import here would close an import cycle. Deferring it
    to call time breaks the cycle.
    """
    from ..config import default_settings, material_sweep
    from ..sweep import build_cases

    settings = default_settings()
    sweep = material_sweep(material)
    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    return {c["name"] for c in cases}


def slim_results(
    results, *, grid=None, drop_wide_brem=False, downcast=False, fields=None, **constraints
):
    """Return a NEW results store carrying only what a viz session needs, to cut a
    checkpoint's on-disk / transfer size (TODO P2 #5). Does NOT mutate ``results``;
    the output keeps the ``{name: {E0: record}}`` shape, so it loads and plots
    exactly like a full checkpoint.

    The single-pickle-per-material checkpoint stores the full union of every swept
    config at full resolution, so transferring it from the GPU box is gigabyte-
    scale and mostly stale for any one plot. This trims it four independent ways:

    grid : a material key. Keep only the configs in that material's CURRENT grid
        (:func:`_grid_names`), dropping configs left over from earlier grids --
        the ``filter_results`` narrowing computed from ``config.py`` alone, so the
        GPU box can shrink its accumulated union to just the live run before the
        wire transfer. Lossless per record (only whole stale configs drop). This
        is applied FIRST, before ``constraints``.
    constraints : case-field filters, identical to :func:`select_results` (scalar /
        list / callable), e.g. ``tilt_deg=0.0, E0_keV=[25, 30]`` -- keep only those
        records. None given keeps every record.
    drop_wide_brem : drop the full-range bremsstrahlung arrays (``brem_wide`` +
        ``E_grid_brem``), the largest fields, when the line-grid ``brem`` is enough
        for the plots you'll make remotely (``detected_background`` /
        ``summary_table`` fall back to it; the wide-range log curve is what needs
        them).
    downcast : store the spectral arrays as float32 -- half the bytes, ample for
        viz. Scalars and ``case`` are untouched.
    fields : explicit allow-list of record keys to keep besides ``case`` (always
        kept); overrides ``drop_wide_brem``. Unknown keys are ignored.

    Round-trips through ``pickle`` and ``run.load_checkpoint`` unchanged. For the
    on-disk wrapper that reports the size saved, see ``run.slim_checkpoint`` /
    ``cxr slim``.
    """
    if grid is not None:
        names = _grid_names(grid)
        results = {n: by_E for n, by_E in results.items() if n in names}
    base = select_results(results, **constraints) if constraints else results
    drop = set(_WIDE_BREM_FIELDS) if (fields is None and drop_wide_brem) else set()
    keep_keys = (set(fields) | {"case"}) if fields is not None else None
    out = {}
    for name, by_E in base.items():
        new_by_E = {}
        for E0, r in by_E.items():
            if keep_keys is not None:
                nr = {k: v for k, v in r.items() if k in keep_keys}
            else:
                nr = {k: v for k, v in r.items() if k not in drop}
            if downcast:
                nr = {
                    k: (
                        np.asarray(v, dtype=np.float32)
                        if (k in _RECORD_ARRAY_FIELDS and v is not None)
                        else v
                    )
                    for k, v in nr.items()
                }
            new_by_E[E0] = nr
        if new_by_E:
            out[name] = new_by_E
    return out


def _peak(r):
    """The selection metric: the highest spectral flux value, max(spectrum)."""
    return float(np.max(r["spec"]))


def best_azimuth(recs):
    """Collapse an azimuth sweep. Group the records by
    (material, thickness, polar tilt, energy) and, within each group, keep only
    the one whose spectrum has the largest peak ``max(spectrum)``. Returns the
    selected records, sorted by (polar tilt, energy). A no-op shape-wise when
    azimuth is not swept (each group already has one member)."""
    groups = {}
    for r in recs:
        c = r["case"]
        key = (c["crystal"], c["thickness_ang"], c["tilt_deg"], c["E0_keV"])
        groups.setdefault(key, []).append(r)
    best = [max(g, key=_peak) for g in groups.values()]
    return sorted(best, key=lambda r: (r["case"]["tilt_deg"], r["case"]["E0_keV"]))
