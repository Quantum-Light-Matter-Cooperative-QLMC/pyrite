"""Checkpoint recompute drivers for the brem and line datasets.

Both rewrite one dataset of existing per-material checkpoints in place and
leave the other alone, so they are orders of magnitude cheaper than re-running
the sweep. :func:`rebrem_checkpoints` rewrites the brem background
(``brem_wide``, ``brem``, ``E_grid_brem`` and the case's ``Ne_brem``);
:func:`recxr_checkpoints` rewrites the line ``spec`` (and ``E_grid``),
re-interpolating the stored brem onto the new line grid.

Typical use is bumping an electron count (brem noise falls as 1/sqrt(Ne_brem))
or refining grid spacing without invalidating the other dataset. Records
already at the target parameters are skipped, so a crashed or OOM'd run
resumes on re-invocation; ``redo_all`` forces a full recompute.

``progress_file`` (single material only) atomically maintains the same compact
JSON progress record a remote sweep writes (see
``pyrite._progress._write_progress_record``), so the remote recompute queues feed the
``pyrite remote job status``/``attach`` case-progress dashboard. ``max_minutes``
bounds the whole run across the material list against one monotonic deadline;
if any material stops short of complete the driver raises ``SystemExit(75)``
so the chunked remote queue self-resubmits.

Click wiring for these lives in :mod:`pyrite.cli.commands.recompute`.
"""

import time
from pathlib import Path

from .._energy_grid_encoding import encode_energy_grid
from .._grid_semantics import is_uniform_grid
from .._progress import _ProgressTimer, _write_progress_record
from ..console.outputs import output_dir
from ..montecarlo import runner
from . import _checkpoint_store
from ._repair_planning import _BremRequest, _plan_brem, _plan_cxr
from ._repair_progress import _RepairProgress
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
    checkpoint_dir=None,
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
    ``pyrite remote job status``/``attach`` case-progress dashboard: repaired records
    count as new cases, already-at-target records as cached.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    ckpt_dir = output_dir("checkpoints") if checkpoint_dir is None else Path(checkpoint_dir)
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
        resolved_nodes = None
        if context.identified or fidelity is not None or require_identity:
            import numpy as np

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
            except (KeyError, TypeError, ValueError):
                # Derived stems (for example ``*_blazed``) have no catalog row.
                # Keep their stored grid while still applying profile Ne/provenance.
                pass
            else:
                profile_brem = profile_sweep.energy_bins.brem
                profile_grid = None if profile_brem is None else np.asarray(profile_brem, float)
                if (
                    profile_grid is not None
                    and profile_grid.size >= 2
                    and not is_uniform_grid(profile_grid)
                ):
                    # A graded profile continuum has no (start, stop, step) to
                    # retune through. ``uniform_bounds`` raises on one, and that
                    # raise used to land in the except branch above -- so the
                    # profile grid was silently ignored and the record kept its
                    # own. Retune onto the coordinates instead.
                    if not any(
                        value is not None for value in (brem_start_eV, brem_stop_eV, brem_step_eV)
                    ):
                        resolved_nodes = profile_grid
                else:
                    try:
                        profile_start, _profile_stop, profile_step = uniform_bounds(profile_grid)
                    except (TypeError, ValueError):
                        pass
                    else:
                        if resolved_start is None:
                            resolved_start = profile_start
                        if resolved_step is None:
                            resolved_step = profile_step
        kw = {}
        if progress_file is not None:
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
                brem_nodes=resolved_nodes,
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


def recxr_checkpoints(
    materials=None,
    checkpoint_dir=None,
    cxr_ne=None,
    cxr_start_eV=None,
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
    """Run :func:`recxr_checkpoint` over one or more materials.
    ``materials=None`` sweeps every ``*.pkl`` (excluding ``*.slim.pkl``).
    Returns ``{stem: results}``. ``progress_file`` (single material only) writes
    the same compact JSON progress record ``pyrite rebrem`` does, feeding the remote
    plain/attached status dashboard.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    ckpt_dir = output_dir("checkpoints") if checkpoint_dir is None else Path(checkpoint_dir)
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
        resolved_ne = cxr_ne
        use_profile_defaults = context.identified or fidelity is not None or require_identity
        if use_profile_defaults and resolved_ne is None:
            from .recompute_defaults import settings

            resolved_ne = settings(context.fidelity).n_electrons
        kw = {}
        if progress_file is not None:
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
            "cxr_ne": resolved_ne,
            "line_step_eV": line_step_eV,
            "from_config": from_config,
            "redo_all": redo_all,
        }
        if cxr_start_eV is not None:
            recompute_options["cxr_start_eV"] = cxr_start_eV
        if line_stop_eV is not None:
            recompute_options["line_stop_eV"] = line_stop_eV
        if use_profile_defaults:
            recompute_options["fidelity"] = context.fidelity
            recompute_options["catalog_profile"] = context.catalog_profile
        try:
            if progress_file is not None:
                progress_timer.start()
            out[stem] = recxr_checkpoint(
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
    brem_nodes=None,
    fidelity=None,
    on_progress=None,
    max_seconds=None,
    status=None,
):
    """Regenerate ``brem_wide`` (and the line-grid ``brem``) for cached records
    using the CURRENT ``mc_brem_spectrum`` -- WITHOUT re-running the expensive
    CXR spectrum.

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
    max_seconds / status : chunked resubmission (mirror of ``repair_cxr_spec``).
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

    if brem_nodes is not None and any(
        value is not None for value in (brem_start_eV, brem_stop_eV, brem_step_eV)
    ):
        raise ValueError(
            "brem_nodes names the target coordinates outright; it cannot be combined "
            "with brem_start_eV/brem_stop_eV/brem_step_eV"
        )
    target_nodes = None if brem_nodes is None else np.asarray(brem_nodes, dtype=float)
    if target_nodes is not None and (target_nodes.ndim != 1 or target_nodes.size < 2):
        raise ValueError("brem_nodes must be a one-dimensional grid of at least two nodes")
    request = _BremRequest(
        ne_brem,
        brem_start_eV,
        brem_stop_eV,
        brem_step_eV,
        target_nodes,
        fidelity,
    )
    todo, n_skipped = _plan_brem(results, request, only_nonfinite)
    cadence = _RepairProgress(
        results,
        len(todo),
        n_skipped,
        save_every,
        save_cb,
        on_progress,
        max_seconds,
        status,
        progress=progress,
        saved_message=True,
    )
    if not todo:
        print(
            "nothing to redo (all records already at target brem parameters)"
            if request.retune
            else "nothing to repair (all brem_wide already finite)"
        )
        return cadence.finish()
    print(f"repairing brem for {len(todo)} record(s)...")
    cadence.start()
    for r, E_brem, updates in todo:
        if cadence.expired():
            break
        r["case"].update(updates)
        _repair_brem_record(r, E_brem, _brem_for_case)
        cadence.completed()
    return cadence.finish()


def _repair_brem_record(r, E_brem, _brem_for_case):
    """Score one brem record through the live runner and release GPU scratch."""
    import numpy as np

    brem_wide = _brem_for_case(r["case"], E_brem)
    r["brem_wide"] = brem_wide
    r["E_grid_brem"] = E_brem
    r["brem"] = np.interp(np.asarray(r["E_grid"], float), E_brem, brem_wide)
    # Return this record's GPU scratch on the SAME A2 cadence the live sweep
    # uses in _spectrum_case. _brem_for_case runs the full transport+spectrum
    # path in-process, so without this the CuPy reserved pool grows and
    # fragments record-over-record until a long `pyrite rebrem` fills the card
    # (Task 8). Guarded like the live path; skipped cases never reach here, so
    # resumability (deadline break, save_cb, skip-at-target) is untouched.
    if runner._RESOURCE_POLICY.gpu:
        runner._maybe_free_pool()


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


def repair_cxr_spec(
    results,
    material,
    only_stale=True,
    cxr_ne=None,
    cxr_start_eV=None,
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
    CURRENT line path (:func:`pyrite.montecarlo._cxr_pair_for_case`) -- the
    mirror of :func:`repair_brem_wide`, WITHOUT recomputing the brem background.
    A ``coherent``/``both`` checkpoint (one carrying ``spec_coherent``) has BOTH
    its incoherent ``spec`` and its ``spec_coherent`` refreshed onto the new grid
    from a single re-transport, so the coherent array never drifts off-grid.
    Planar ``spec_channeling`` is independently rescored from the case on the
    target grid; brem repair preserves this separate component.

    Target line grid per record: ``line_step_eV``
    (explicit uniform spacing), else ``from_config`` rebuilds it from
    ``material``'s current ``E_grid_line_by_energy`` at the record's E0 (edit
    the catalog, ``pyrite reline`` re-runs lines on the new grid), else the
    record's existing grid (pure ``cxr_ne`` bump). ``cxr_ne`` overrides
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

    todo, n_skipped = _plan_cxr(
        results,
        sweep,
        only_stale,
        redo_all,
        cxr_ne,
        fidelity,
        line_step_eV,
        from_config,
        cxr_start_eV,
        line_stop_eV,
    )
    cadence = _RepairProgress(
        results,
        len(todo),
        n_skipped,
        save_every,
        save_cb,
        on_progress,
        max_seconds,
        status,
    )
    if not todo:
        print("nothing to reline (all records already at target line grid / Ne)")
        return cadence.finish()
    print(f"relining {len(todo)} record(s)...")
    cadence.start()
    for r, target, updates in todo:
        if cadence.expired():
            break
        r["case"].update(updates)
        _repair_cxr_record(r, target)
        cadence.completed()
    return cadence.finish()


def _repair_cxr_record(r, target):
    """Refresh the line companions together, retaining the wide brem source."""
    import numpy as np

    c = r["case"]
    # `spec` is always the incoherent line sum; a coherent/both checkpoint
    # (detected by an existing `spec_coherent`) gets BOTH arrays refreshed
    # onto the new grid from ONE re-transport, so `spec_coherent` never goes
    # stale relative to `spec`/`E_grid`. Old pre-dual coherent checkpoints
    # (no `spec_coherent` key) are orphaned by design -- reline only ever
    # re-derives the incoherent `spec` for those.
    want_coherent = "spec_coherent" in r
    spec, spec_coherent, spec_characteristic = runner._cxr_pair_for_case(
        c, target, want_coherent=want_coherent, return_characteristic=True
    )
    if c.get("channeling") is not None:
        from ..montecarlo.runner.channeling_emission import rescore_channeling_spectrum

        r["spec_channeling"] = rescore_channeling_spectrum(c, target)
    r["spec"] = spec
    r["spec_characteristic"] = spec_characteristic
    if want_coherent:
        r["spec_coherent"] = spec_coherent
    r["E_grid"] = target
    # A windowed (#101) target has no triple form; keep its exact coordinates.
    c["E_grid_line"] = (
        (float(target[0]), float(target[-1]), len(target))
        if is_uniform_grid(target)
        else encode_energy_grid(target)
    )
    # Recompute pins explicit coordinates, which outrank any automatic
    # case-local policy the record was built under (issue #101). Drop the
    # policy so the stored case does not claim a resolution it no longer has.
    c.pop("line_grid_policy", None)
    bw = r.get("brem_wide")
    egb = r.get("E_grid_brem")
    if bw is not None and egb is not None:
        r["brem"] = np.interp(target, np.asarray(egb, float), np.asarray(bw, float))


def recxr_checkpoint(
    checkpoint_path, material, save_every=100, max_seconds=None, status=None, **kw
):
    """Load a per-material checkpoint, reline its line ``spec`` (see
    :func:`repair_cxr_spec`), and re-pickle it in place (atomic, resumable),
    saving every ``save_every`` records. Mirror of :func:`repair_checkpoint`.
    ``max_seconds``/``status`` thread the chunked-resubmission deadline through
    to :func:`repair_cxr_spec`."""
    if not _checkpoint_exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    results = _checkpoint_load(checkpoint_path)

    def save_cb(results):
        _save_recomputed_checkpoint(
            checkpoint_path,
            results,
            components=("characteristic", "channeling", "line"),
        )

    n = repair_cxr_spec(
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
