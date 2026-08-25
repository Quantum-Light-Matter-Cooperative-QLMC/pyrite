"""Checkpoint recompute drivers for the brem and line datasets.

Both rewrite one dataset of existing per-material checkpoints in place and
leave the other alone, so they are orders of magnitude cheaper than re-running
the sweep. :func:`rebrem_checkpoints` rewrites the brem background
(``brem_wide``, ``brem``, ``E_grid_brem`` and the case's ``Ne_brem``);
:func:`reline_checkpoints` rewrites the line ``spec`` (and ``E_grid``),
re-interpolating the stored brem onto the new line grid.

Typical use is bumping an electron count (brem noise falls as 1/sqrt(Ne_brem))
or refining grid spacing without invalidating the other dataset. Records
already at the target parameters are skipped, so a crashed or OOM'd run
resumes on re-invocation; ``redo_all`` forces a full recompute.

``progress_file`` (single material only) atomically maintains the same compact
JSON progress record a remote sweep writes (see
``scan._write_progress_record``), so the remote recompute queues feed the
``pyrite job status``/``attach`` case-progress dashboard. ``max_minutes``
bounds the whole run across the material list against one monotonic deadline;
if any material stops short of complete the driver raises ``SystemExit(75)``
so the chunked remote queue self-resubmits.

Click wiring for these lives in :mod:`pyrite.cli.commands.recompute`.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..energy_grid.encoding import decode_energy_grid
from ..montecarlo import runner
from . import _checkpoint_store
from .persistence import (
    _checkpoint_exists,
    _checkpoint_load,
    _save_recomputed_checkpoint,
    checkpoint_path_for,
)


def _progress_callback(progress_file, stem, latest, progress_timer, write_progress_record):
    """Build the shared recompute progress callback for one checkpoint stem."""

    def _on_progress(done, todo_total, skipped):
        latest.update(
            total_cases=todo_total + skipped,
            cached_cases=skipped,
            completed_new_cases=done,
        )
        write_progress_record(
            progress_file,
            material=stem,
            state="running",
            **latest,
            **progress_timer.snapshot(completed_new_cases=done),
        )

    return _on_progress


def rebrem_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    ne_brem=None,
    brem_start_eV=None,
    brem_stop_eV=None,
    brem_step_eV=None,
    fidelity=None,
    catalog_profile=None,
    require_identity=False,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
    summary_status=None,
):
    """Run :func:`repair_checkpoint` with new brem parameters over one
    or more materials. ``materials=None`` sweeps every ``*.pkl`` in
    ``checkpoint_dir`` (grooved checkpoints included; ``*.slim.pkl`` transfer
    copies excluded). Returns ``{stem: results}``.

    ``progress_file`` (single material only) atomically maintains the same
    compact JSON progress record a remote run writes (see
    ``scan._write_progress_record``), so the remote rebrem queue feeds the
    ``pyrite job status``/``attach`` case-progress dashboard: repaired records
    count as new cases, already-at-target records as cached.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    ckpt_dir = Path(checkpoint_dir)
    if materials:
        paths = [checkpoint_path_for(m, str(ckpt_dir)) for m in materials]
    else:
        paths = [
            checkpoint_path_for(stem, str(ckpt_dir))
            for stem in _checkpoint_store.discover(ckpt_dir)
        ]
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    if progress_file is not None and len(paths) != 1:
        raise SystemExit("progress_file needs exactly one material")
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60
    incomplete = False
    out = {}
    for path in paths:
        stem = Path(path).stem if str(path).endswith(".pkl") else Path(path).name
        print(f"== {stem} ==")
        from .recompute_defaults import dataset_context

        context = dataset_context(
            path,
            catalog_profile=catalog_profile,
            fidelity=fidelity,
            require_identity=require_identity,
        )
        resolved_ne = ne_brem
        resolved_start = brem_start_eV
        resolved_step = brem_step_eV
        if context.identified or fidelity is not None or require_identity:
            from .recompute_defaults import settings, sweep, uniform_bounds

            profile_settings = settings(context.fidelity)
            if resolved_ne is None:
                resolved_ne = profile_settings.n_electrons_brem
            try:
                profile_sweep = sweep(
                    context.material,
                    context.fidelity,
                    catalog_profile=context.catalog_profile,
                )
                profile_start, _profile_stop, profile_step = uniform_bounds(
                    profile_sweep.detector.energy_bins.brem
                )
            except (KeyError, TypeError, ValueError):
                # Derived stems (for example ``*_blazed``) have no catalog row.
                # Keep their stored grid while still applying profile Ne/provenance.
                pass
            else:
                if resolved_start is None:
                    resolved_start = profile_start
                if resolved_step is None:
                    resolved_step = profile_step
        kw = {}
        if progress_file is not None:
            from ..runs.scan import _ProgressTimer, _write_progress_record

            latest = {"total_cases": 0, "cached_cases": 0, "completed_new_cases": 0}
            progress_timer = _ProgressTimer(progress_file)

            kw["on_progress"] = _progress_callback(
                progress_file, stem, latest, progress_timer, _write_progress_record
            )
            _write_progress_record(
                progress_file,
                material=stem,
                state="running",
                **latest,
                **progress_timer.snapshot(),
            )
        max_seconds = None if deadline is None else max(0.0, deadline - time.monotonic())
        status = {}
        repair_options = {}
        if context.identified or fidelity is not None or require_identity:
            repair_options["fidelity"] = context.fidelity
        try:
            if progress_file is not None:
                progress_timer.start()
            out[stem] = repair_checkpoint(
                path,
                save_every=save_every,
                only_nonfinite=not redo_all,
                ne_brem=resolved_ne,
                brem_start_eV=resolved_start,
                brem_stop_eV=brem_stop_eV,
                brem_step_eV=resolved_step,
                max_seconds=max_seconds,
                status=status,
                **repair_options,
                **kw,
            )
        except BaseException as exc:
            status["complete"] = False
            status["error"] = str(exc) or type(exc).__name__
            if progress_file is not None:
                _write_progress_record(
                    progress_file,
                    material=stem,
                    state="failed",
                    **latest,
                    **progress_timer.snapshot(completed_new_cases=latest["completed_new_cases"]),
                )
            raise
        finally:
            if summary_status is not None:
                summary_status[stem] = dict(status)
        if progress_file is not None:
            _write_progress_record(
                progress_file,
                material=stem,
                state="done" if status.get("complete", True) else "paused",
                **latest,
                **progress_timer.snapshot(completed_new_cases=latest["completed_new_cases"]),
            )
        if not status.get("complete", True):
            incomplete = True
    if incomplete:
        raise SystemExit(75)
    return out


def reline_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    line_ne=None,
    line_start_eV=None,
    line_stop_eV=None,
    line_step_eV=None,
    fidelity=None,
    catalog_profile=None,
    require_identity=False,
    from_config=True,
    redo_all=False,
    save_every=100,
    progress_file=None,
    max_minutes=None,
    summary_status=None,
):
    """Run :func:`reline_checkpoint` over one or more materials.
    ``materials=None`` sweeps every ``*.pkl`` (excluding ``*.slim.pkl``).
    Returns ``{stem: results}``. ``progress_file`` (single material only) writes
    the same compact JSON progress record ``pyrite rebrem`` does, feeding the remote
    plain/attached status dashboard.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    ckpt_dir = Path(checkpoint_dir)
    if materials:
        paths = [checkpoint_path_for(m, str(ckpt_dir)) for m in materials]
    else:
        paths = [
            checkpoint_path_for(stem, str(ckpt_dir))
            for stem in _checkpoint_store.discover(ckpt_dir)
        ]
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    if progress_file is not None and len(paths) != 1:
        raise SystemExit("progress_file needs exactly one material")
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60
    incomplete = False
    out = {}
    for path in paths:
        stem = Path(path).stem if str(path).endswith(".pkl") else Path(path).name
        print(f"== {stem} ==")
        from .recompute_defaults import dataset_context

        context = dataset_context(
            path,
            catalog_profile=catalog_profile,
            fidelity=fidelity,
            require_identity=require_identity,
        )
        resolved_ne = line_ne
        use_profile_defaults = context.identified or fidelity is not None or require_identity
        if use_profile_defaults and resolved_ne is None:
            from .recompute_defaults import settings

            resolved_ne = settings(context.fidelity).n_electrons
        kw = {}
        if progress_file is not None:
            from ..runs.scan import _ProgressTimer, _write_progress_record

            latest = {"total_cases": 0, "cached_cases": 0, "completed_new_cases": 0}
            progress_timer = _ProgressTimer(progress_file)

            kw["on_progress"] = _progress_callback(
                progress_file, stem, latest, progress_timer, _write_progress_record
            )
            _write_progress_record(
                progress_file,
                material=stem,
                state="running",
                **latest,
                **progress_timer.snapshot(),
            )
        max_seconds = None if deadline is None else max(0.0, deadline - time.monotonic())
        status = {}
        recompute_options = {
            "line_ne": resolved_ne,
            "line_step_eV": line_step_eV,
            "from_config": from_config,
            "redo_all": redo_all,
        }
        if line_start_eV is not None:
            recompute_options["line_start_eV"] = line_start_eV
        if line_stop_eV is not None:
            recompute_options["line_stop_eV"] = line_stop_eV
        if use_profile_defaults:
            recompute_options["fidelity"] = context.fidelity
            recompute_options["catalog_profile"] = context.catalog_profile
        try:
            if progress_file is not None:
                progress_timer.start()
            out[stem] = reline_checkpoint(
                path,
                material=context.material,
                save_every=save_every,
                max_seconds=max_seconds,
                status=status,
                **recompute_options,
                **kw,
            )
        except BaseException as exc:
            status["complete"] = False
            status["error"] = str(exc) or type(exc).__name__
            if progress_file is not None:
                _write_progress_record(
                    progress_file,
                    material=stem,
                    state="failed",
                    **latest,
                    **progress_timer.snapshot(completed_new_cases=latest["completed_new_cases"]),
                )
            raise
        finally:
            if summary_status is not None:
                summary_status[stem] = dict(status)
        if progress_file is not None:
            _write_progress_record(
                progress_file,
                material=stem,
                state="done" if status.get("complete", True) else "paused",
                **latest,
                **progress_timer.snapshot(completed_new_cases=latest["completed_new_cases"]),
            )
        if not status.get("complete", True):
            incomplete = True
    if incomplete:
        raise SystemExit(75)
    return out


# ---- brem-only repair --------------------------------------------------------
def _stored_brem_grid(r):
    """The brem grid a cached record was computed on: the case's encoded grid,
    the record's stored array, or the legacy single-grid fallback."""
    import numpy as np

    c = r["case"]
    if "E_grid_brem" in c:
        return decode_energy_grid(c["E_grid_brem"])
    if r.get("E_grid_brem") is not None and np.asarray(r["E_grid_brem"]).size:
        return np.asarray(r["E_grid_brem"], float)
    # legacy single-grid record
    step_b = c.get("brem_step_eV", 10.0)
    eg = np.asarray(r["E_grid"], float)
    return np.arange(eg[0], eg[-1] + step_b, step_b)


def repair_brem_wide(
    results,
    only_nonfinite=True,
    progress=True,
    save_every=0,
    save_cb=None,
    ne_brem=None,
    brem_start_eV=None,
    brem_stop_eV=None,
    brem_step_eV=None,
    fidelity=None,
    on_progress=None,
    max_seconds=None,
    status=None,
):
    """Regenerate ``brem_wide`` (and the line-grid ``brem``) for cached records
    using the CURRENT ``mc_brem_spectrum`` -- WITHOUT re-running the expensive
    line spectrum.

    Records computed before the brem-absorption fix have NaN ``brem_wide`` above
    the Henke ceiling (~30 keV) and at E=0, which the log plots drop (the
    full-range curve clips at 30 keV). All-zero arrays are stale placeholders,
    too: accepting them as merely finite makes resume/cache paths suppress the
    physical pedestal indefinitely. This re-runs only the cheap bremsstrahlung
    transport (``Ne_brem`` electrons, same seed as the original case) per record
    and rewrites ``r["brem_wide"]`` / ``r["brem"]`` in place; the line ``spec`` is
    left untouched. Everything needed is already stored in ``r["case"]``.

    The transport + per-layer brem sum is delegated to the runner's
    ``_brem_for_case`` -- the SAME path a live sweep takes -- so a film-on-
    substrate / multilayer record is repaired with its full stacked brem
    (``layers=abs_layers``, per-layer Z^2 sum, ``brem_chunk`` honored) rather
    than being silently rewritten as single-slab brem.

    ne_brem / brem_step_eV : NEW brem parameters to recompute with (``pyrite
        rebrem``). ``ne_brem`` overrides the electron count (noise goes down as
        1/sqrt(Ne_brem)); ``brem_step_eV`` rebuilds a uniform grid at that
        spacing from the record's existing low cutoff up to the sweep-convention
        stop ``E0_keV*1e3 + step``. Both are persisted into ``r["case"]``, so a
        record already at the target parameters is SKIPPED on a re-run (crash/
        OOM resumable) -- and downstream re-repairs keep the new parameters.
        With either set, selection is by parameter mismatch (plus nonfinite
        ``brem_wide``) instead of nonfinite alone.
    only_nonfinite : skip records whose ``brem_wide`` is already finite and
        contains a nonzero sample (default), so only stale missing/nonfinite/
        all-zero placeholders are touched -- or, with new
        parameters given, skip records already at those parameters. Set False
        to redo all.
    save_every / save_cb : if both set, call ``save_cb(results)`` every
        ``save_every`` repaired records (and once at the end) -- used by
        :func:`repair_checkpoint` to checkpoint progress so a crash/OOM doesn't
        lose everything; a re-run then resumes (only_nonfinite skips the saved ones).
    on_progress : optional callback ``on_progress(done, todo_total, skipped)``,
        fired once before the loop and after every repaired record -- same shape
        as run_sweep's ``on_progress(completed_new_cases, total, cached_cases)``,
        so ``pyrite rebrem --progress-file`` feeds the remote progress dashboard.
    max_seconds / status : chunked resubmission (mirror of ``repair_line_spec``).
        When ``max_seconds`` is not ``None`` a ``time.monotonic``-based deadline is
        checked BEFORE each record; on expiry the loop breaks with unprocessed
        records left. When ``status`` is a dict, ``status["complete"]`` is set to
        ``False`` iff the deadline cut the pass short (``True`` otherwise). Both
        default to a no-op so the local path is byte-identical to today.
    Returns the number of records actually repaired (0 on an immediate deadline).
    Mutates ``results`` in place; re-pickle the checkpoint afterwards (or use
    :func:`repair_checkpoint`).
    """
    import numpy as np

    from ..montecarlo import _brem_for_case

    retune = any(
        value is not None
        for value in (ne_brem, brem_start_eV, brem_stop_eV, brem_step_eV, fidelity)
    )
    todo = []
    n_skipped = 0
    for name in results:
        for r in results[name].values():
            bw = r.get("brem_wide")
            bw_array = np.asarray(bw) if bw is not None else np.asarray([])
            finite = bw_array.size > 0 and np.isfinite(bw_array).all() and np.any(bw_array != 0)
            if only_nonfinite:
                at_target = finite
                if retune:
                    c = r["case"]
                    if ne_brem is not None and int(c.get("Ne_brem", -1)) != int(ne_brem):
                        at_target = False
                    if fidelity is not None and c.get("brem_profile") != fidelity:
                        at_target = False
                    if any(
                        value is not None for value in (brem_start_eV, brem_stop_eV, brem_step_eV)
                    ):
                        g = _stored_brem_grid(r)
                        if (
                            brem_start_eV is None
                            and brem_stop_eV is None
                            and brem_step_eV is not None
                        ):
                            if g.size < 2 or not np.isclose(g[1] - g[0], float(brem_step_eV)):
                                at_target = False
                        else:
                            stored_step = float(g[1] - g[0]) if g.size >= 2 else None
                            step = float(brem_step_eV) if brem_step_eV is not None else stored_step
                            start = (
                                float(brem_start_eV)
                                if brem_start_eV is not None
                                else (float(g[0]) if g.size else None)
                            )
                            stop = (
                                float(brem_stop_eV)
                                if brem_stop_eV is not None
                                else (float(c["E0_keV"]) * 1e3 + step if step is not None else None)
                            )
                            if start is None or stop is None or step is None:
                                at_target = False
                            else:
                                target = np.arange(start, stop, step)
                                if g.shape != target.shape or not np.allclose(g, target):
                                    at_target = False
                if at_target:
                    n_skipped += 1
                    continue
            todo.append(r)
    if on_progress is not None:
        on_progress(0, len(todo), n_skipped)
    if not todo:
        print(
            "nothing to redo (all records already at target brem parameters)"
            if retune
            else "nothing to repair (all brem_wide already finite)"
        )
        if status is not None:
            status["complete"] = True
        return 0
    print(f"repairing brem for {len(todo)} record(s)...")
    t0 = time.perf_counter()
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    done = 0
    broke = False
    for k, r in enumerate(todo, 1):
        if deadline is not None and time.monotonic() >= deadline:
            broke = True
            break
        c = r["case"]
        if ne_brem is not None:
            c["Ne_brem"] = int(ne_brem)
        if fidelity is not None:
            c["brem_profile"] = fidelity
        if any(value is not None for value in (brem_start_eV, brem_stop_eV, brem_step_eV)):
            stored = _stored_brem_grid(r)
            step_b = (
                float(brem_step_eV) if brem_step_eV is not None else float(stored[1] - stored[0])
            )
            start = float(brem_start_eV) if brem_start_eV is not None else float(stored[0])
            stop = (
                float(brem_stop_eV)
                if brem_stop_eV is not None
                else float(c["E0_keV"]) * 1e3 + step_b
            )
            E_brem = np.arange(start, stop, step_b)
            c["E_grid_brem"] = (start, stop, step_b)
        else:
            E_brem = _stored_brem_grid(r)
        brem_wide = _brem_for_case(c, E_brem)
        r["brem_wide"] = brem_wide
        r["E_grid_brem"] = E_brem
        r["brem"] = np.interp(np.asarray(r["E_grid"], float), E_brem, brem_wide)
        # Return this record's GPU scratch on the SAME A2 cadence the live sweep
        # uses in _spectrum_case. _brem_for_case runs the full transport+spectrum
        # path in-process, so without this the CuPy reserved pool grows and
        # fragments record-over-record until a long `pyrite rebrem` fills the card
        # (Task 8). Guarded like the live path; skipped cases never reach here, so
        # resumability (deadline break, save_cb, skip-at-target) is untouched.
        if runner._GPU:
            runner._maybe_free_pool()
        done = k
        if on_progress is not None:
            on_progress(k, len(todo), n_skipped)
        if save_cb is not None and save_every and (k % save_every == 0):
            save_cb(results)
            if progress:
                print(f"    ...saved progress ({k}/{len(todo)})")
        if progress and (k % 50 == 0 or k == len(todo)):
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
    if save_cb is not None:
        save_cb(results)  # final save
    if status is not None:
        status["complete"] = not broke
    return done


def repair_checkpoint(checkpoint_path, save_every=100, max_seconds=None, status=None, **kw):
    """Load a per-material checkpoint, repair its ``brem_wide`` (see
    :func:`repair_brem_wide`), and re-pickle it in place -- saving progress every
    ``save_every`` records (atomic temp+replace) so a crash/OOM is RESUMABLE: just
    call again and only_nonfinite picks up where it left off. Extra ``**kw``
    (e.g. ``ne_brem=``, ``brem_step_eV=`` -- ``pyrite rebrem``) pass through to
    :func:`repair_brem_wide`. ``max_seconds``/``status`` thread the
    chunked-resubmission deadline through. Returns the repaired ``results`` dict
    (also usable directly in the notebook)."""
    if not _checkpoint_exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    results = _checkpoint_load(checkpoint_path)

    def save_cb(results):  # atomic; resumable on crash; keeps the sidecar manifest in step
        _save_recomputed_checkpoint(checkpoint_path, results, components=("brem",))

    n = repair_brem_wide(
        results,
        save_every=save_every,
        save_cb=save_cb,
        max_seconds=max_seconds,
        status=status,
        **kw,
    )
    print(f"re-saved {checkpoint_path}" if n else "checkpoint unchanged")
    return results


def _target_line_grid(
    sweep,
    r,
    line_step_eV,
    from_config,
    line_start_eV=None,
    line_stop_eV=None,
):
    """The line grid a reline should land on for record ``r``.

    ``line_step_eV`` -> uniform grid at that spacing over the record's current
    [E_grid[0], E_grid[-1]]. Else ``from_config`` -> the material's CURRENT
    E_grid_line_by_energy at this record's E0 (the bespoke-grid workflow), via
    the same lookup build_cases uses. Else the record's existing grid."""
    import numpy as np

    eg = np.asarray(r["E_grid"], float)
    if from_config and sweep is not None:
        from ..campaign.sweep import _line_grid_for_energy

        base = np.asarray(_line_grid_for_energy(sweep, eg, float(r["case"]["E0_keV"])), float)
    else:
        base = eg
    if any(value is not None for value in (line_start_eV, line_stop_eV, line_step_eV)):
        from ..checkpoints.recompute_defaults import uniform_bounds

        start, stop, step = uniform_bounds(base)
        return np.arange(
            start if line_start_eV is None else float(line_start_eV),
            stop if line_stop_eV is None else float(line_stop_eV),
            step if line_step_eV is None else float(line_step_eV),
        )
    return base


def repair_line_spec(
    results,
    material,
    only_stale=True,
    line_ne=None,
    line_start_eV=None,
    line_stop_eV=None,
    line_step_eV=None,
    fidelity=None,
    catalog_profile="standard",
    from_config=True,
    redo_all=False,
    save_every=0,
    save_cb=None,
    on_progress=None,
    max_seconds=None,
    status=None,
):
    """Regenerate the incoherent line ``spec`` for cached records with the
    CURRENT line path (:func:`pyrite.montecarlo._line_pair_for_case`) -- the
    mirror of :func:`repair_brem_wide`, WITHOUT recomputing the brem background.
    A ``coherent``/``both`` checkpoint (one carrying ``spec_coherent``) has BOTH
    its incoherent ``spec`` and its ``spec_coherent`` refreshed onto the new grid
    from a single re-transport, so the coherent array never drifts off-grid.

    Target line grid per record (see :func:`_target_line_grid`): ``line_step_eV``
    (explicit uniform spacing), else ``from_config`` rebuilds it from
    ``material``'s current ``E_grid_line_by_energy`` at the record's E0 (edit
    materials.toml, ``pyrite reline`` re-runs lines on the new grid), else the
    record's existing grid (pure ``line_ne`` bump). ``line_ne`` overrides
    ``case["Ne"]``.

    After relining, brem stays consistent: ``r["brem"]`` is re-interpolated from
    the RETAINED ``brem_wide`` onto the new ``E_grid`` (no brem transport). A
    record already at the target grid + Ne (and finite spec) is SKIPPED
    (``only_stale``), so a crashed/OOM run resumes; ``redo_all`` forces all.
    ``save_every``/``save_cb``/``on_progress`` have the SAME contract as
    :func:`repair_brem_wide`.

    ``max_seconds`` (chunked resubmission, mirror of ``scan.py --max-minutes``):
    when not ``None`` a ``time.monotonic``-based deadline is checked BEFORE each
    record; on expiry the loop breaks with unprocessed records left. When
    ``status`` is a dict, ``status["complete"]`` is set to ``False`` iff the
    deadline cut the pass short (``True`` otherwise). Both default to a no-op so
    the local path is byte-identical to today. Returns the number of records
    actually relined; mutates in place."""
    import time

    import numpy as np

    sweep = None
    if from_config:
        try:
            if fidelity is None:
                from ..campaign.config import material_sweep

                sweep = material_sweep(material, catalog_profile=catalog_profile)
            else:
                from ..checkpoints.recompute_defaults import sweep as fidelity_sweep

                sweep = fidelity_sweep(
                    material,
                    fidelity,
                    catalog_profile=catalog_profile,
                )
        except Exception:
            sweep = None  # unknown/derived stem -> fall back to each record's grid

    todo = []
    n_skipped = 0
    for name in results:
        for r in results[name].values():
            target = _target_line_grid(
                sweep,
                r,
                line_step_eV,
                from_config,
                line_start_eV,
                line_stop_eV,
            )
            eg = np.asarray(r["E_grid"], float)
            spec = r.get("spec")
            finite = spec is not None and np.isfinite(np.asarray(spec)).all()
            grid_ok = eg.shape == target.shape and np.allclose(eg, target)
            ne_ok = line_ne is None or int(r["case"].get("Ne", -1)) == int(line_ne)
            fidelity_ok = fidelity is None or r["case"].get("line_profile") == fidelity
            at_target = finite and grid_ok and ne_ok and fidelity_ok
            if only_stale and not redo_all and at_target:
                n_skipped += 1
                continue
            todo.append((r, target))
    if on_progress is not None:
        on_progress(0, len(todo), n_skipped)
    if not todo:
        print("nothing to reline (all records already at target line grid / Ne)")
        if status is not None:
            status["complete"] = True
        return 0
    print(f"relining {len(todo)} record(s)...")
    t0 = time.perf_counter()
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    done = 0
    broke = False
    for k, (r, target) in enumerate(todo, 1):
        if deadline is not None and time.monotonic() >= deadline:
            broke = True
            break
        c = r["case"]
        if line_ne is not None:
            c["Ne"] = int(line_ne)
        if fidelity is not None:
            c["line_profile"] = fidelity
        # `spec` is always the incoherent line sum; a coherent/both checkpoint
        # (detected by an existing `spec_coherent`) gets BOTH arrays refreshed
        # onto the new grid from ONE re-transport, so `spec_coherent` never goes
        # stale relative to `spec`/`E_grid`. Old pre-dual coherent checkpoints
        # (no `spec_coherent` key) are orphaned by design -- reline only ever
        # re-derives the incoherent `spec` for those.
        want_coherent = "spec_coherent" in r
        spec, spec_coherent = runner._line_pair_for_case(c, target, want_coherent=want_coherent)
        r["spec"] = spec
        if want_coherent:
            r["spec_coherent"] = spec_coherent
        r["E_grid"] = target
        c["E_grid_line"] = (float(target[0]), float(target[-1]), len(target))
        bw = r.get("brem_wide")
        egb = r.get("E_grid_brem")
        if bw is not None and egb is not None:
            r["brem"] = np.interp(target, np.asarray(egb, float), np.asarray(bw, float))
        done = k
        if on_progress is not None:
            on_progress(k, len(todo), n_skipped)
        if save_cb is not None and save_every and (k % save_every == 0):
            save_cb(results)
        if k % 50 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
    if save_cb is not None:
        save_cb(results)
    if status is not None:
        status["complete"] = not broke
    return done


def reline_checkpoint(
    checkpoint_path, material, save_every=100, max_seconds=None, status=None, **kw
):
    """Load a per-material checkpoint, reline its line ``spec`` (see
    :func:`repair_line_spec`), and re-pickle it in place (atomic, resumable),
    saving every ``save_every`` records. Mirror of :func:`repair_checkpoint`.
    ``max_seconds``/``status`` thread the chunked-resubmission deadline through
    to :func:`repair_line_spec`."""
    if not _checkpoint_exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    results = _checkpoint_load(checkpoint_path)

    def save_cb(results):
        _save_recomputed_checkpoint(checkpoint_path, results, components=("line",))

    n = repair_line_spec(
        results,
        material=material,
        save_every=save_every,
        save_cb=save_cb,
        max_seconds=max_seconds,
        status=status,
        **kw,
    )
    print(f"re-saved {checkpoint_path}" if n else "checkpoint unchanged")
    return results
