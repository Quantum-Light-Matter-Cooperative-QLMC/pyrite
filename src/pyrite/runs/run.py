"""
run.py
==========

Drive a sweep through ``montecarlo.run_cases`` with on-disk checkpointing
and per-group streaming feedback.

:func:`run_sweep` resumes from a per-material component checkpoint
(``checkpoints/<material>/``, skipping cached cases), stores each finished
case into ``results``, publishes immutable shards at config
granularity, and invokes ``on_chunk(group_config_names)`` once a
whole **group**
has finished. By default a group is everything that shares
(material, thickness, polar tilt, finite footprint) -- i.e. the full azimuth
sweep at one tilt and footprint --
so the streamed plot/table waits until every azimuth is in and can collapse to
the best azimuth per energy (see plots.stream_chunk).

run_cases shows a tqdm bar over completed cases; it renders once at the top of
the cell, so as streamed plots/tables pile up below it scrolls out of view --
hence the explicit "tilt N/M" progress line printed with each chunk here, which
stays next to the latest output. With a GPU present, CPU transport is pipelined
through workers behind one main-process CUDA spectrum context; see
montecarlo.run_cases.
"""

import os
import time

from ..checkpoints.persistence import (  # noqa: F401 -- compatibility exports
    _CAS_OPTIONAL_PAYLOAD_KEYS,
    _CAS_PAYLOAD_KEYS,
    _cas_payload_from_record,
    _case_manifest_path,
    _case_set_proof,
    _case_set_proof_from_cases,
    _case_set_proof_from_results,
    _checkpoint_components_save,
    _checkpoint_exists,
    _checkpoint_load,
    _checkpoint_save,
    _checkpoint_signature,
    _IncrementalManifest,
    _load_checkpoint_cached,
    _manifest_for,
    _manifest_path_for,
    _manifest_save,
    _manifest_write,
    _material_analysis_cache,
    _material_analysis_cache_path,
    _save_recomputed_checkpoint,
    _SweepCheckpoint,
    _valid_cas_payload,
    _write_case_manifest,
    cached_material_analysis,
    cases_from_results,
    checkpoint_manifest,
    checkpoint_path_for,
    default_checkpoint_dir,
    load_checkpoint,
    sweep_checkpoint,
)
from ..checkpoints.recompute import (  # noqa: F401 -- compatibility exports
    recxr_checkpoint,
    repair_brem_wide,
    repair_checkpoint,
    repair_cxr_spec,
)
from ..montecarlo import run_cases, runner
from ..montecarlo.case import refuse_axial_channeling_cases
from ..results import store_result
from ._sweep_execution import (
    SweepBudget,
    SweepCompletion,
    resolve_worker_output,
    terminal_activity,
)
from ._sweep_plan import SweepResume, SweepWork, validate_sweep_request


def __getattr__(name):
    # Compatibility export resolved on access, like its persistence original.
    if name == "DEFAULT_CHECKPOINT_DIR":
        return default_checkpoint_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _default_group_key(case):
    """Everything but azimuth and energy: the sweep at one tilt and footprint.

    Missing footprint fields represent the legacy laterally infinite slab, so
    older case dictionaries continue to group together.
    """
    return (
        case["crystal"],
        case["thickness_ang"],
        case["tilt_deg"],
        case.get("crystal_width_mm"),
        case.get("crystal_height_mm"),
    )


def _runtime_plan(executor, cases, max_workers):
    """Describe where ``cases`` run; never the local topology for a foreign executor."""
    if executor is None:
        return runner.runtime_plan(cases, max_workers)
    planner = getattr(executor, "runtime_plan", None)
    if planner is not None:
        return planner(cases, max_workers)
    return {
        "engine": "external",
        "executor": type(executor).__qualname__,
        "requested_workers": max_workers,
        "effective_workers": None,
    }


def _run_options(
    max_workers, progress, should_stop, transport_only, trajectory_capture, on_timing, on_activity
):
    """Options common to ``run_cases`` and ``executor.execute``; optional hooks stay absent."""
    options = {
        "max_workers": max_workers,
        "progress": progress,
        "should_stop": should_stop,
        "keep_results": False,  # the case callback owns storage; don't pin spectra in RAM
        "transport_only": transport_only,
    }
    if trajectory_capture is not None:
        options["trajectory_capture"] = trajectory_capture
    if on_timing is not None:
        options["on_timing"] = on_timing
    if on_activity is not None:
        options["on_activity"] = on_activity
    return options


def _capture_size_report(trajectory_capture, todo):
    if trajectory_capture is None or os.environ.get("PYRITE_TRAJECTORY_SIZE_REPORT") != "1":
        return None
    from ._trajectory_capture import CaptureSizeReport

    return CaptureSizeReport(trajectory_capture, todo)


def _case_callback(
    results, preparation, checkpoint, work, completion, artifact_store, size_report, on_case
):
    """Store one finished case: artifact, observation, scalar, then notifications.

    Durable state precedes each notification that promises it. ``on_case`` and
    ``on_progress`` report the in-memory record; ``on_chunk`` follows persistence.
    """

    def callback(_index, case, out):
        out = resolve_worker_output(
            case,
            out,
            artifact_store=artifact_store,
            content_key=preparation.content_key,
            observation_plans=work.observation_plans,
            material=checkpoint.material,
        )
        if size_report is not None:
            size_report.completed(case)

        if work.transport_only:
            work.transport_finished(case)
            completion.case_finished(case)
            return

        if work.accept_observation(case, out):
            completion.case_finished(case)
            return

        store_result(results, case, out)

        preparation.store_transport(case, out)
        checkpoint.record_finished(case)
        if on_case is not None:
            on_case(case)
        completion.scalar_finished(case, checkpoint)
        completion.case_finished(case)

    return callback


def run_sweep(
    cases,
    results,
    *,
    checkpoint_dir=None,
    checkpoint_path=None,
    resume=True,
    max_workers=None,
    progress=True,
    group_key=None,
    on_chunk=None,
    on_progress=None,
    on_case=None,
    on_runtime=None,
    on_timing=None,
    on_activity=None,
    max_seconds=None,
    deadline=None,
    time_fn=None,
    dataset_identity=None,
    case_cost_fn=None,
    on_cost=None,
    content_key_fn=None,
    cache_read=True,
    cache_write=True,
    transport_only=False,
    metadata_only_complete=False,
    trajectory_capture=None,
    observation=None,
    retain_results=True,
    executor=None,
    artifact_store=None,
):
    """Run ``cases`` into ``results`` (mutated in place).

    Parameters
    ----------
    cases
        list of typed cases from sweep.build_cases. All cases must name one
        ``crystal`` and have unique ``(name, E0_keV)`` pairs; a violation raises
        ``ValueError`` before any checkpoint, cache, file or ``results`` effect.
    results
        the dict to fill ({name: {E0: record}}). It may hold other materials
        under other config names, but a requested config already holding another
        material's records, or a config mixing this material with another, raises
        ``ValueError``. So does a restored checkpoint holding another material's
        records, before ``results`` changes and leaving the file untouched.
    retain_results
        defaults to True, preserving in-memory spectra. False
        requires an empty results mapping and a component checkpoint path;
        completed records become {"case": case, "checkpoint_path": path} after
        successful persistence. Use checkpoint loading for spectra, including
        in on_chunk callbacks. Resume and consolidation keep only case metadata
        plus one config's arrays resident (legacy input files may fully decode).
    executor
        optional object implementing ``execute(cases, callback=..., **options)``.
        When omitted, use ``montecarlo.run_cases`` exactly as before. When
        supplied, callbacks must execute in this coordinator process so the
        existing sweep remains sole owner of manifests and consolidation. An
        optional ``validate_run_options`` rejects unsupported options before any
        checkpoint, cache or file effect, whether or not any case turns out to
        need execution; an executor without it can only reject inside ``execute``,
        which a fully cached sweep never calls. See
        :class:`~pyrite.execution.protocol.Executor`.
    artifact_store
        optional pyrite.execution.CaseArtifactStore.
        A shared-filesystem inbox for independently published worker results.
        A worker may return a CaseArtifactRef instead of bulk result arrays;
        the coordinator verifies its physics key and digest, then performs the
        established checkpoint writes. With resume and cache_read enabled,
        completed but uncommitted worker artifacts are recovered on restart.
    checkpoint_dir
        directory for the per-material checkpoints (default
        ``pyrite-output/checkpoints`` in the workspace). A sweep is
        single-material, so each writes ``<checkpoint_dir>/<material>/``
        holding only that material's configs. ``results`` can hold several
        materials in one kernel without them clobbering each other on disk.
        Completed configs are sharded until the sweep completes; consolidation
        then publishes component files and a manifest before removing shards.
    checkpoint_path
        explicit override for the checkpoint path; None (default)
        derives the per-material path above. An explicit ``.pkl`` path retains
        whole-store saves. Resume skips stored ``(name, E0_keV)`` pairs.
    max_workers
        forwarded to run_cases. None sizes the GPU transport pool or
        the full-case CPU pool automatically; 0 forces serial execution. CPU
        full-case pools are also capped by available memory.
    progress
        forwarded to run_cases (the per-case tqdm bar).
    group_key
        case -> hashable. Configs sharing a key form one group;
        on_chunk fires once the WHOLE group has finished. Default groups by
        (material, thickness, polar tilt, finite footprint), so the azimuth
        sweep at one tilt and footprint is one group -- on_chunk gets every
        azimuth at once. Every energy of a config must map to the same key
        (``ValueError`` otherwise); an unhashable key raises ``TypeError``. Keys
        need no ordering: fully cached groups replay in request order.
    on_chunk
        optional callback(list_of_config_names) fired once per completed
        group, with all of that group's config names (cached + freshly run). It
        runs after the group's configs are persisted and before ``on_progress``
        reports the finishing case. Only scalar records complete a group, so
        observation-only and transport-only execution never fire it; groups
        whose scalars were already cached replay once before new work starts.
    on_progress
        optional callback(completed_new_cases, total_cases, cached_cases)
        fired once after resume filtering and after every newly completed case,
        including observation-only and transport-only cases. A case that raises
        is not reported. ``cached_cases`` excludes observation-only cases, which
        are still to run. Progress does not imply durability: only ``on_chunk`` and
        the CAS (with ``cache_write``) promise persisted output, and a default-mode
        config is persisted only once all its energies finish.
    on_case
        optional callback(case) fired when a case newly stores its scalar
        record, before ``on_chunk`` and ``on_progress`` -- the frontier the sweep
        is working through. Lets a live viewer surface which crystal case
        (energy, tilts, thickness) is currently under test without threading it
        through ``on_progress``'s fixed count signature. Observation-only and
        transport-only cases store no new scalar, so they advance ``on_progress``
        without firing it.
    on_runtime
        optional callback(dict) fired once after resume filtering. Without an
        executor it holds the local resolved engine, effective worker count,
        memory policy, representative energy-grid widths, and spectrum/brem chunk
        sizing. With one it holds that executor's ``runtime_plan``, or
        ``engine="external"`` and ``effective_workers=None`` if it has none.
    on_timing
        optional callback(dict) receiving rolling per-case transport,
        spectrum, GPU feed-wait, retry, CuPy-pool, and checkpoint timings.
    on_activity
        optional callback(dict) receiving current driver phase,
        case index, and in-flight work counts. A multi-batch in-process
        transport also reports ``transport_electrons_done``/
        ``transport_electrons_total`` for the active case (see ``run_cases``).
        The last event is always ``{"phase": "handoff", "case": None,
        "in_flight_case_count": 0}``, on success, on a pause and on an exception
        once the sweep has begun loading. If this callback itself raises during a
        failure, the original exception propagates with a note.
    max_seconds
        optional soft wall-clock budget, measured from just before dispatch,
        after resume, cache replay and cached-group callbacks. None (default) means
        unbounded. When set, a deadline of ``time_fn() + max_seconds`` is checked
        (via ``run_cases``'s ``should_stop`` hook) before each new case starts;
        once it passes, no new case begins but in-flight work still drains and is
        still stored. Unstarted configs are simply left for the next call to pick
        up -- resume filtering is per ``(name, E0_keV)`` (see checkpoint_path).
    deadline
        optional absolute stop time on the ``time_fn`` clock, for one budget shared
        across calls; resume and replay spend it. Takes precedence over
        ``max_seconds``, which is then ignored. Either makes the sweep bounded, so
        an executor without bounded support rejects it.
    time_fn
        clock used for both budgets; defaults to ``time.monotonic``.
        Override for deterministic tests.
    dataset_identity
        optional JSON-serializable resolved-profile identity,
        persisted in ``meta.json``. Variant-aware callers use this to prove
        exactly which settings and sweep produced the component artifacts.
    case_cost_fn
        optional callable(case) -> float (e.g. sweep.case_cost).
        When given, enables compute-weighted progress: relative cost is summed
        over ``cases`` for the total and over the exact cached/completed case
        set for what's done, so a live viewer can show a compute-aware percent
        instead of a flat case count. None (default) disables ``on_cost``.
    on_cost
        optional callback(done_cost, total_cost), fired alongside
        ``on_progress`` (after resume filtering and after every newly completed
        case) whenever ``case_cost_fn`` is set; otherwise never called.
    content_key_fn
        optional callable(case) -> hex str (``profiles.case_content_key``).
        When given, enables the shared per-material content-addressable case store
        (CAS): each finished case's transport ``out`` is written once under its
        content key (``cache_write``), and on resume any case still to run whose
        content key already has a blob is REPLAYED from it instead of recomputed
        (``cache_read``). The content key excludes the profile name, so a case
        computed by one profile is reused by any other profile requesting the same
        physics. None (default) leaves the CAS entirely inert -- byte-identical to
        the pre-CAS per-stem-only path. Reuse replays through ``store_result`` with
        the REQUESTED case, so the reconstructed record is bit-identical to a fresh
        compute (including the requester's own detector/flux scalars).
    resume
        load this checkpoint's records before planning (default True). Independent
        of ``cache_read``: ``resume=False`` still replays the CAS unless that is
        also False. The ``run`` CLI deliberately passes one value for both.
        Recovering worker artifacts from ``artifact_store`` requires both.
    cache_read
        replay CAS blobs for cases lacking a record, with or without ``resume``, for
        cross-profile reuse (default True; ignored when ``content_key_fn`` is None).
        Set False by ``--no-cache`` / ``--recompute`` / ``-p`` so a run recomputes
        rather than reuses.
    cache_write
        write each finished case into the CAS (default True; ignored
        when ``content_key_fn`` is None). Set False by ``--no-cache`` / ``-p`` so
        an ephemeral or measurement run never populates the shared store.
    trajectory_capture
        optional ``montecarlo.trajectories.TrajectoryCapture``.
        When given, every case this call transports writes its transport result
        to one HDF5 artifact before its spectrum runs. Resumed or cache-replayed
        cases are not re-transported and get no new artifact. Existing artifacts
        are checked before any transport (see ``preflight_capture``), and the
        metadata-only completion shortcut is skipped. None (default) captures
        nothing and leaves the run unchanged.
    observation
        optional :class:`~pyrite.observations.SweepObservation`. Every
        case then also needs its counting observation in ``observation.store``.
        A stored observation with matching true-spatial factors is reused and
        only rescored; otherwise the case's physical-detector directions are
        evaluated on the same transport as its scalar record. A case whose
        scalar record is already cached but whose observation is missing runs
        again for the observation only; its cached record is kept. Directional
        arrays never enter the scalar record or the CAS blob. Output lacking a
        required ``directional`` payload raises ``ValueError`` before that case
        stores or reports anything; previously completed configs stay resumable,
        and a cached scalar record is kept. Ignored with ``transport_only``.
    transport_only
        run transport without spectra or observations. Cases that already have
        a scalar record are not re-transported. Outputs are not scalar records,
        so execution adds nothing to ``results``, checkpoints or the CAS; only
        ``trajectory_capture`` retains transport data. Execution advances
        ``on_progress`` and ``on_cost`` but not ``on_case`` or ``on_chunk``.
    metadata_only_complete
        allow a fresh manifest whose case-set proof and ``dataset_identity``
        match this request to complete without loading any record. Skipped with
        ``trajectory_capture``, ``observation``, ``on_chunk`` or
        ``retain_results=False``.

    Returns
    -------
    bool
        True iff the requested work finished, budget or not; False if
        ``max_seconds`` or an executor left work undone. By default that means
        every requested ``(name, E0_keV)`` pair is in ``results`` and every
        required observation is stored. Exceptions: on a metadata-proof completion
        (``metadata_only_complete``) ``results`` is left unloaded, so True rests on
        the persisted proof; with ``retain_results=False`` records are
        ``{"case", "checkpoint_path"}`` references; with ``transport_only`` True
        means only that every case lacking a scalar record was executed in this
        call, says nothing about persistence, and ``results`` may stay empty.
        Incomplete component checkpoints retain their completed-config shards
        for resume. Legacy ``.pkl`` paths receive a final whole-store save on an
        incomplete return, including partial configs. An exception skips that
        final save; configs published before it remain resumable.
    """
    refuse_axial_channeling_cases(cases)
    if artifact_store is not None and content_key_fn is None:
        raise ValueError("artifact_store requires content_key_fn for physics identity validation")
    if group_key is None:
        group_key = _default_group_key
    validate_sweep_request(cases, results, group_key)
    budget = SweepBudget(deadline, max_seconds, time.monotonic if time_fn is None else time_fn)
    run_options = _run_options(
        max_workers,
        progress,
        budget.should_stop,
        transport_only,
        trajectory_capture,
        on_timing,
        on_activity,
    )
    validate_options = getattr(executor, "validate_run_options", None)
    if validate_options is not None:
        validate_options(run_options)
    if dataset_identity is not None:
        from ..campaign.profiles import normalize_dataset_identity

        dataset_identity = normalize_dataset_identity(dataset_identity)

    # per-material checkpoint: a sweep is single-material, so name the pickle for
    # the crystal and keep them together in their own subdir.
    material = cases[0]["crystal"] if cases else "mixed"
    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_for(material, checkpoint_dir)
    checkpoint = sweep_checkpoint(
        checkpoint_path,
        results,
        material,
        dataset_identity,
        on_timing,
        on_activity,
        retain_results=retain_results,
    )
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    preparation = SweepResume(cases, results, checkpoint, content_key_fn, cache_read, cache_write)
    metadata_eligible = (
        metadata_only_complete
        and trajectory_capture is None
        and observation is None
        and on_chunk is None
    )
    with terminal_activity(on_activity):
        if preparation.load(resume, metadata_eligible):
            if on_runtime is not None:
                on_runtime(_runtime_plan(executor, [], max_workers))
            SweepCompletion(cases, [], group_key, case_cost_fn, on_cost, on_progress, None).report()
            print(f"0 of {len(cases)} cases to run ({len(cases)} cached; metadata proof)")
            return True

        if preparation.loaded:
            preparation.seed_cache()
        preparation.replay_cache()

        # Resume / --no-cache semantics control whether existing artifacts may be
        # replayed. Do not use a scalar-only artifact in place of a missing
        # directional observation.
        if (
            resume
            and cache_read
            and artifact_store is not None
            and not transport_only
            and observation is None
        ):
            preparation.recover_artifacts(artifact_store)

        work = SweepWork(cases, results, observation, transport_only)
        todo = work.todo
        print(f"{len(todo)} of {len(cases)} cases to run ({len(cases) - len(todo)} cached)")
        if trajectory_capture is not None:
            from ._trajectory_capture import prepare_sweep_capture

            trajectory_capture = prepare_sweep_capture(trajectory_capture, observation, cases, todo)
            run_options["trajectory_capture"] = trajectory_capture
        if on_runtime is not None:
            on_runtime(_runtime_plan(executor, todo, max_workers))
        size_report = _capture_size_report(trajectory_capture, todo)
        completion = SweepCompletion(
            cases, todo, group_key, case_cost_fn, on_cost, on_progress, on_chunk
        )
        completion.report()
        completion.prepare_groups(todo, work.observation_only)
        callback = _case_callback(
            results, preparation, checkpoint, work, completion, artifact_store, size_report, on_case
        )

        # Replay cached groups before launching new work, including observation-only
        # work whose scalar group is already complete.
        completion.replay_cached_groups(results)

        directions = work.observation_directions()
        if directions is not None:
            run_options["observation_directions"] = directions

        budget.start()
        t0 = time.perf_counter()
        # A fully cached sweep has no work for an external executor. Preserve the
        # historical run_cases([]) call when no executor was explicitly supplied.
        if todo or executor is None:
            dispatch = run_cases if executor is None else executor.execute
            dispatch(todo, callback=callback, **run_options)
        print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")
        # Publication follows what is stored; transport-only execution stores nothing.
        checkpoint.finish(work.stored_complete())
        return work.requested_complete()
