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

from . import _checkpoint_store


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
    """Run :func:`pyrite.runs.run.repair_checkpoint` with new brem parameters over one
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
    from ..runs.run import checkpoint_path_for, repair_checkpoint

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

            def _on_progress(
                done,
                todo_total,
                skipped,
                _latest=latest,
                _stem=stem,
                _timer=progress_timer,
            ):
                _latest.update(
                    total_cases=todo_total + skipped,
                    cached_cases=skipped,
                    completed_new_cases=done,
                )
                _write_progress_record(
                    progress_file,
                    material=_stem,
                    state="running",
                    **_latest,
                    **_timer.snapshot(completed_new_cases=done),
                )

            kw["on_progress"] = _on_progress
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
    """Run :func:`pyrite.runs.run.reline_checkpoint` over one or more materials.
    ``materials=None`` sweeps every ``*.pkl`` (excluding ``*.slim.pkl``).
    Returns ``{stem: results}``. ``progress_file`` (single material only) writes
    the same compact JSON progress record ``pyrite rebrem`` does, feeding the remote
    plain/attached status dashboard.

    ``max_minutes`` bounds the whole run across the material list against one
    monotonic deadline; each material gets the remaining budget as ``max_seconds``
    and a fresh ``status`` dict. If any material stops short of complete, the
    sweep raises ``SystemExit(75)`` after the loop so the chunked remote queue
    self-resubmits. ``max_minutes=None`` (local default) never raises."""
    from ..runs.run import checkpoint_path_for, reline_checkpoint

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

            def _on_progress(
                done,
                todo_total,
                skipped,
                _latest=latest,
                _stem=stem,
                _timer=progress_timer,
            ):
                _latest.update(
                    total_cases=todo_total + skipped,
                    cached_cases=skipped,
                    completed_new_cases=done,
                )
                _write_progress_record(
                    progress_file,
                    material=_stem,
                    state="running",
                    **_latest,
                    **_timer.snapshot(completed_new_cases=done),
                )

            kw["on_progress"] = _on_progress
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
