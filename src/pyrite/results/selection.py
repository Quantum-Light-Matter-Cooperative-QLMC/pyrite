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

from ..campaign.sweep import fmt_thickness


# ---- record selection --------------------------------------------------------
def records(results, names=None):
    """Flatten records from a nested checkpoint store.

    Parameters
    ----------
    results
        ``{configuration_name: {electron_energy: record}}`` mapping.
    names
        Optional configuration names to include, in requested order.

    Returns
    -------
    list
        Record mappings in configuration and nested energy insertion order.
    """
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

    Parameters
    ----------
    results
        Nested checkpoint result store.
    cases
        Cases whose exact ``name`` values define the retained configurations.

    Returns
    -------
    dict
        New nested store containing matching configuration names.
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

    Parameters
    ----------
    results
        Nested checkpoint result store.
    **constraints
        Case field predicates. A scalar tests equality, a collection tests
        membership, and a callable receives the stored field value.

    Returns
    -------
    dict
        New nested store containing records that satisfy every constraint.
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


def thicknesses_by_energy(results):
    """``{E0_keV: sorted[thickness_ang]}`` -- every COMPUTED slab per beam energy.

    The penetration watchdog (:func:`config.gate_cases_by_penetration`) stops
    computing an energy past the depth where its beam is dead, so a low beam
    energy carries FEWER thicknesses than a high one. This inventory is what
    :func:`select_thickness` consults to decide, per energy, whether a requested
    thickness was actually computed or must fall back to that energy's thickest
    computed slab.
    """
    out: dict[float, set] = {}
    for by_E in results.values():
        for E0, r in by_E.items():
            t = r["case"].get("thickness_ang")
            if t is not None:
                out.setdefault(E0, set()).add(t)
    return {E0: sorted(ts) for E0, ts in out.items()}


def select_thickness(results, thickness_ang):
    """Pin one crystal thickness across a multi-beam-energy overlay WITHOUT ever
    dropping a beam energy the penetration watchdog stopped computing early.

    A plain ``select_results(results, thickness_ang=T)`` silently loses every
    energy whose beam is already dead before ``T`` -- exactly the low beams
    (30/50 keV) at a bulk thickness -- because the watchdog never computed those
    (energy, T) cases. But past its penetration cutoff an energy's emitted
    spectrum is SATURATED (further thickness adds negligible emission -- the very
    invariant the watchdog encodes), so that energy's THICKEST computed slab is
    the physically-correct stand-in for any thicker request.

    So: for each energy that WAS computed at ``thickness_ang`` keep exactly those
    records; for each energy that was NOT, substitute its thickest computed slab
    and stamp ``case["thickness_fallback"] = (thickness_ang, actual_thickness)``
    on a COPY of the case so the UI can annotate the substitution. Returns a new
    ``{name: {E0: record}}`` store; never mutates the input.
    """
    inventory = thicknesses_by_energy(results)

    def _has_exact(E0):
        return any(np.isclose(thickness_ang, t, rtol=1e-9, atol=1e-6) for t in inventory[E0])

    out: dict = {}
    for name, by_E in results.items():
        for E0, r in by_E.items():
            case = r["case"]
            record_thickness = case.get("thickness_ang")
            if record_thickness is None:
                continue
            exact = _has_exact(E0)
            # Past its penetration cutoff an energy is saturated, so any thicker
            # request collapses onto that energy's thickest computed slab.
            target = thickness_ang if exact else inventory[E0][-1]
            keep = bool(np.isclose(record_thickness, target, rtol=1e-9, atol=1e-6))
            if not keep:
                continue
            if not exact:
                # copy, don't mutate: other views share these records
                case = {**case, "thickness_fallback": (thickness_ang, target)}
            out.setdefault(name, {})[E0] = {**r, "case": case}
    return out


# record array fields, by size; the wide-brem pair is the largest (full-range grid)
_RECORD_ARRAY_FIELDS = (
    "E_grid",
    "spec",
    "brem",
    "E_grid_brem",
    "brem_wide",
    "spec_coherent",
    "spec_characteristic",
)
_WIDE_BREM_FIELDS = ("brem_wide", "E_grid_brem")

# The coherent total and characteristic audit component use the line grid, so a
# --line-only projection/merge must carry them with the combined ``spec``.
LINE_RECORD_KEYS = ("spec", "E_grid", "spec_coherent", "spec_characteristic")
BREM_RECORD_KEYS = ("brem_wide", "brem", "E_grid_brem")


def project_dataset(results, dataset):
    """Return a NEW results store carrying only ``dataset``'s record arrays
    (plus ``case``) per record -- the wire payload for a dataset-partial pull
    (``pyrite slim --brem-only/--line-only``). ``dataset`` is ``"line"`` (keeps
    :data:`LINE_RECORD_KEYS`) or ``"brem"`` (:data:`BREM_RECORD_KEYS`).
    Delegates to :func:`slim_results`' ``fields`` allow-list so the drop logic
    lives in one place; ``case`` is always kept."""
    keys = {"line": LINE_RECORD_KEYS, "brem": BREM_RECORD_KEYS}.get(dataset)
    if keys is None:
        raise ValueError(f"dataset must be 'line' or 'brem', got {dataset!r}")
    return slim_results(results, fields=list(keys))


def merge_dataset(local, incoming, dataset, force=False):
    """Overwrite ONLY ``dataset``'s record keys in ``local`` from ``incoming``,
    matched by (config name, E0), leaving the other dataset's arrays intact.
    ``dataset`` is ``"line"`` or ``"brem"``. After copying, ``brem`` is always
    re-derived as ``interp(E_grid, E_grid_brem, brem_wide)`` so it stays
    consistent with whichever grid/brem_wide now holds (a line merge that
    changed ``E_grid`` re-interps the retained local brem_wide; a brem merge
    re-interps the fresh brem_wide onto the local line grid). Records in
    ``incoming`` absent from ``local`` are SKIPPED (reported) unless ``force``,
    which inserts them whole. Mutates ``local``; returns (n_merged, n_skipped)."""
    import numpy as np

    keys = {"line": LINE_RECORD_KEYS, "brem": BREM_RECORD_KEYS}.get(dataset)
    if keys is None:
        raise ValueError(f"dataset must be 'line' or 'brem', got {dataset!r}")
    n_merged = n_skipped = 0
    for name, by_E in incoming.items():
        for E0, inc in by_E.items():
            local_by_E = local.get(name)
            if local_by_E is None or E0 not in local_by_E:
                if force:
                    local.setdefault(name, {})[E0] = dict(inc)
                    n_merged += 1
                else:
                    n_skipped += 1
                continue
            r = local_by_E[E0]
            for k in keys:
                if k in inc:
                    r[k] = inc[k]
            # A line payload that lacks an optional companion must remove the old
            # one: otherwise a stale, possibly wrong-length array is paired with
            # the incoming line grid.
            if dataset == "line" and "spec" in inc and "spec_coherent" not in inc:
                r.pop("spec_coherent", None)
            if dataset == "line" and "spec" in inc and "spec_characteristic" not in inc:
                r.pop("spec_characteristic", None)
            bw, egb, eg = r.get("brem_wide"), r.get("E_grid_brem"), r.get("E_grid")
            if bw is not None and egb is not None and eg is not None:
                r["brem"] = np.interp(
                    np.asarray(eg, float), np.asarray(egb, float), np.asarray(bw, float)
                )
            n_merged += 1
    if n_skipped:
        print(
            f"merge_dataset: skipped {n_skipped} record(s) not present locally "
            f"(pass force=True to insert them)"
        )
    return n_merged, n_skipped


def _grid_names(material, fidelity="full", catalog_profile="standard"):
    """Config names in the CURRENT grid for ``material`` -- exactly the set
    ``config.material_sweep(material, catalog_profile=catalog_profile)`` ->
    ``sweep.build_cases`` produces now. A stale config is any name NOT in this
    set. ``catalog_profile`` selects the named materials.toml profile whose
    grid a profile-variant checkpoint was swept on.

    The ``config`` / ``sweep`` imports are function-local on purpose: ``config``
    imports ``results`` at module load (``from .results import Settings``), so a
    module-level ``config`` import here would close an import cycle. Deferring it
    to call time breaks the cycle.
    """
    from ..campaign.config import default_settings, material_sweep
    from ..campaign.sweep import build_cases

    settings = default_settings(fidelity)
    sweep = material_sweep(material, fidelity=fidelity, catalog_profile=catalog_profile)
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

    grid : a material key, or a ``(material, fidelity[, catalog_profile])``
        tuple. Keep only the configs in that material's CURRENT grid
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
    ``pyrite slim``.
    """
    if grid is not None:
        if isinstance(grid, tuple):
            names = _grid_names(*grid)
        else:
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
    azimuth is not swept (each group already has one member).

    Parameters
    ----------
    recs
        Iterable of result records carrying ``case`` and ``spec`` fields.

    Returns
    -------
    list
        Peak-maximizing record per material/thickness/polar-tilt/energy group.
    """
    groups = {}
    for r in recs:
        c = r["case"]
        key = (c["crystal"], c["thickness_ang"], c["tilt_deg"], c["E0_keV"])
        groups.setdefault(key, []).append(r)
    best = [max(g, key=_peak) for g in groups.values()]
    return sorted(best, key=lambda r: (r["case"]["tilt_deg"], r["case"]["E0_keV"]))


# ---- cross-material case basket ----------------------------------------------
# Case fields case_label() may show/elide (E0_keV is always shown, so it's not
# in this set); the same fields case_table_rows() surfaces for the picker.
_LABEL_FIELDS = ("thickness_ang", "tilt_deg", "tilt_azim_deg")

# Record keys a basket entry needs to be plottable by
# ``plots.altair.spectra._record_frame`` / ``metrics.line_metrics`` -- the same
# spectral-array set ``slim_results`` trims to, minus ``E_pk``/``hit_frac``/
# ``eta`` which neither consumer reads.
_BASKET_RECORD_FIELDS = (
    "E_grid",
    "spec",
    "spec_characteristic",
    "brem",
    "E_grid_brem",
    "brem_wide",
    "fwhm",
    "scale",
)


def case_label(case, *, material_label=None, face=None, varying=None):
    """Human-readable case-identity label for the cross-material case basket,
    e.g. ``"HOPG - 100 keV - 5um - tilt -80deg - az 120deg"``.

    ``material_label`` and the beam energy are always shown when present;
    ``varying`` (a set/container of field names, typically the keys of
    :func:`sweep_values` run over just the basket's cases) restricts the rest
    of :data:`_LABEL_FIELDS` to only those that actually differ across the
    basket, so a basket that's all one thickness doesn't repeat it on every
    line. ``varying=None`` (the default) shows every field -- the right choice
    for a single, standalone label. ``face="blazed"`` appends a trailing
    ``(blazed)`` marker (mirrors the checkpoint-stem convention in
    :func:`pyrite.apps.analyze.face_stem`); any other face is unmarked.
    """
    parts = []
    if material_label:
        parts.append(str(material_label))
    if case.get("E0_keV") is not None:
        parts.append(f"{case['E0_keV']:g} keV")

    def _shown(field):
        return varying is None or field in varying

    if _shown("thickness_ang") and case.get("thickness_ang") is not None:
        parts.append(fmt_thickness(case["thickness_ang"]))
    if _shown("tilt_deg") and case.get("tilt_deg") is not None:
        parts.append(f"tilt {case['tilt_deg']:g}deg")
    if _shown("tilt_azim_deg") and case.get("tilt_azim_deg") is not None:
        parts.append(f"az {case['tilt_azim_deg']:g}deg")
    label = " - ".join(parts)
    return f"{label} (blazed)" if face == "blazed" else label


def case_table_rows(results):
    """One plain dict per record in ``results``, for the case-picker table that
    feeds the "Add to comparison" basket flow. Carries the record's own
    ``(name, E0_keV)`` primary key (so a selected row maps straight back to
    ``results[name][E0]``) plus the case-varying columns and a cheap peak-flux
    metric -- no heavy per-row analysis. Row order matches :func:`records`."""
    rows = []
    for r in records(results):
        case = r["case"]
        rows.append(
            {
                "name": case["name"],
                "E0_keV": case["E0_keV"],
                "thickness": fmt_thickness(case["thickness_ang"]),
                "thickness_ang": case["thickness_ang"],
                "tilt_deg": case["tilt_deg"],
                "tilt_azim_deg": case["tilt_azim_deg"],
                "peak_flux": _peak(r),
            }
        )
    return rows


def slim_case_record(record, *, material, label, face="flat"):
    """Standalone, basket-sized copy of one checkpoint ``record``: just the
    spectral arrays :func:`pyrite.plots.altair.spectra._record_frame` and
    :func:`pyrite.results.metrics.line_metrics` need
    (:data:`_BASKET_RECORD_FIELDS`), plus a ``case`` dict tagged with the
    basket-only identity fields (``material``, ``face``, ``label``) a bare
    checkpoint record has no slot for. Everything else (``eta``, ``hit_frac``,
    ``E_pk``, ...) is dropped, so a dozen basket entries from a dozen
    checkpoints stay kB-sized rather than carrying each source record's full
    footprint. Never mutates ``record``."""
    slim = {k: record[k] for k in _BASKET_RECORD_FIELDS if k in record}
    slim["case"] = {**record["case"], "material": material, "face": face, "label": label}
    return slim
