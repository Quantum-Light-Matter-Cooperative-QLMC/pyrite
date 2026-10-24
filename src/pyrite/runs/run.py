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
from ..execution.artifacts import (
    CaseArtifactRef,
    DirectionalArtifactRef,
    load_directional_result,
)
from ..montecarlo import run_cases, runner
from ..results import store_result
from ._sweep_execution import SweepCompletion
from ._sweep_plan import SweepResume, SweepWork, has_record


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

    cases : list of typed cases from sweep.build_cases.
    results : the dict to fill ({name: {E0: record}}).
    retain_results : defaults to True, preserving in-memory spectra. False
        requires an empty results mapping and a component checkpoint path;
        completed records become {"case": case, "checkpoint_path": path} after
        successful persistence. Use checkpoint loading for spectra, including
        in on_chunk callbacks. Resume and consolidation keep only case metadata
        plus one config's arrays resident (legacy input files may fully decode).
    executor : optional object implementing ``execute(cases, callback=..., **options)``.
        When omitted, use ``montecarlo.run_cases`` exactly as before. When
        supplied, callbacks must execute in this coordinator process so the
        existing sweep remains sole owner of manifests and consolidation.
    artifact_store : optional pyrite.execution.CaseArtifactStore.
        A shared-filesystem inbox for independently published worker results.
        A worker may return a CaseArtifactRef instead of bulk result arrays;
        the coordinator verifies its physics key and digest, then performs the
        established checkpoint writes. With resume and cache_read enabled,
        completed but uncommitted worker artifacts are recovered on restart.
    checkpoint_dir : directory for the per-material checkpoints (default
        ``pyrite-output/checkpoints`` in the workspace). A sweep is
        single-material, so each writes ``<checkpoint_dir>/<material>/``
        holding only that material's configs. ``results`` can hold several
        materials in one kernel without them clobbering each other on disk.
        Completed configs are sharded until the sweep completes; consolidation
        then publishes component files and a manifest before removing shards.
    checkpoint_path : explicit override for the checkpoint path; None (default)
        derives the per-material path above. An explicit ``.pkl`` path retains
        whole-store saves. Resume skips stored ``(name, E0_keV)`` pairs.
    max_workers : forwarded to run_cases. None sizes the GPU transport pool or
        the full-case CPU pool automatically; 0 forces serial execution. CPU
        full-case pools are also capped by available memory.
    progress : forwarded to run_cases (the per-case tqdm bar).
    group_key : case -> hashable. Configs sharing a key form one group;
        on_chunk fires once the WHOLE group has finished. Default groups by
        (material, thickness, polar tilt, finite footprint), so the azimuth
        sweep at one tilt and footprint is one group -- on_chunk gets every
        azimuth at once.
    on_chunk : optional callback(list_of_config_names) fired once per completed
        group, with all of that group's config names (cached + freshly run).
    on_progress : optional callback(completed_new_cases, total_cases, cached_cases)
        fired once after resume filtering and after every newly completed case.
    on_case : optional callback(case) fired with each case as it finishes,
        just before ``on_progress`` -- the frontier the sweep is working through.
        Lets a live viewer surface which crystal case (energy, tilts, thickness)
        is currently under test without threading it through ``on_progress``'s
        fixed count signature.
    on_runtime : optional callback(dict) fired once after resume filtering with
        resolved engine, effective worker count, memory policy, representative
        energy-grid widths, and spectrum/brem chunk sizing.
    on_timing : optional callback(dict) receiving rolling per-case transport,
        spectrum, GPU feed-wait, retry, CuPy-pool, and checkpoint timings.
    on_activity : optional callback(dict) receiving current driver phase,
        case index, and in-flight work counts. A multi-batch in-process
        transport also reports ``transport_electrons_done``/
        ``transport_electrons_total`` for the active case (see ``run_cases``).
    max_seconds : optional soft wall-clock budget, measured from just before
        ``run_cases`` starts. None (default) means unbounded. When set, a
        deadline of ``time_fn() + max_seconds`` is checked (via ``run_cases``'s
        ``should_stop`` hook) before each new case starts; once it passes, no
        new case begins but in-flight work still drains and is still stored.
        Unstarted configs are simply left for the next call to pick up --
        resume filtering is per ``(name, E0_keV)`` (see checkpoint_path).
    time_fn : clock used for the deadline; defaults to ``time.monotonic``.
        Override for deterministic tests.
    dataset_identity : optional JSON-serializable resolved-profile identity,
        persisted in ``meta.json``. Variant-aware callers use this to prove
        exactly which settings and sweep produced the component artifacts.
    case_cost_fn : optional callable(case) -> float (e.g. sweep.case_cost).
        When given, enables compute-weighted progress: relative cost is summed
        over ``cases`` for the total and over the exact cached/completed case
        set for what's done, so a live viewer can show a compute-aware percent
        instead of a flat case count. None (default) disables ``on_cost``.
    on_cost : optional callback(done_cost, total_cost), fired alongside
        ``on_progress`` (after resume filtering and after every newly completed
        case) whenever ``case_cost_fn`` is set; otherwise never called.
    content_key_fn : optional callable(case) -> hex str (``profiles.case_content_key``).
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
    cache_read : consult the CAS on resume for cross-profile reuse (default True;
        ignored when ``content_key_fn`` is None). Set False by ``--no-cache`` /
        ``--recompute`` / ``-p`` so a run recomputes rather than reuses.
    cache_write : write each finished case into the CAS (default True; ignored
        when ``content_key_fn`` is None). Set False by ``--no-cache`` / ``-p`` so
        an ephemeral or measurement run never populates the shared store.
    trajectory_capture : optional ``montecarlo.trajectories.TrajectoryCapture``.
        When given, every case this call transports writes its transport result
        to one HDF5 artifact before its spectrum runs. Resumed or cache-replayed
        cases are not re-transported and get no new artifact. Existing artifacts
        are checked before any transport (see ``preflight_capture``), and the
        metadata-only completion shortcut is skipped. None (default) captures
        nothing and leaves the run unchanged.
    observation : optional :class:`~pyrite.observations.SweepObservation`. Every
        case then also needs its counting observation in ``observation.store``.
        A stored observation with matching true-spatial factors is reused and
        only rescored; otherwise the case's physical-detector directions are
        evaluated on the same transport as its scalar record. A case whose
        scalar record is already cached but whose observation is missing runs
        again for the observation only; its cached record is kept. Directional
        arrays never enter the scalar record or the CAS blob. Ignored with
        ``transport_only``.

    Returns True iff every requested ``(name, E0_keV)`` pair ended up in
    ``results`` (i.e. the sweep ran to completion, budget or not); False if
    ``max_seconds`` cut it short. Incomplete component checkpoints retain their
    completed-config shards for resume. Legacy ``.pkl`` paths receive a final
    whole-store save on an incomplete return, including partial configs.
    """
    if artifact_store is not None and content_key_fn is None:
        raise ValueError("artifact_store requires content_key_fn for physics identity validation")
    if time_fn is None:
        time_fn = time.monotonic
    if deadline is None and max_seconds is not None:
        deadline = time_fn() + max_seconds
    if group_key is None:
        group_key = _default_group_key
    if dataset_identity is not None:
        from ..campaign.profiles import normalize_dataset_identity

        dataset_identity = normalize_dataset_identity(dataset_identity)

    # per-material checkpoint: a sweep is single-material, so name the pickle for
    # the crystal and keep them together in their own subdir.
    material = cases[0]["crystal"] if cases else "mixed"
    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_for(material, checkpoint_dir)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    checkpoint = sweep_checkpoint(
        checkpoint_path,
        results,
        material,
        dataset_identity,
        on_timing,
        on_activity,
        retain_results=retain_results,
    )
    preparation = SweepResume(cases, results, checkpoint, content_key_fn, cache_read, cache_write)
    metadata_eligible = (
        metadata_only_complete
        and trajectory_capture is None
        and observation is None
        and on_chunk is None
    )
    if preparation.load(resume, metadata_eligible):
        if on_runtime is not None:
            on_runtime(runner.runtime_plan([], max_workers))
        total_cost = sum(case_cost_fn(case) for case in cases) if case_cost_fn is not None else None
        if on_cost is not None and total_cost is not None:
            on_cost(total_cost, total_cost)
        if on_progress is not None:
            on_progress(0, len(cases), len(cases))
        if on_activity is not None:
            on_activity({"phase": "handoff", "case": None, "in_flight_case_count": 0})
        print(f"0 of {len(cases)} cases to run ({len(cases)} cached; metadata proof)")
        return True

    if preparation.loaded:
        preparation.seed_cache()
    preparation.replay_cache()

    # A worker may finish publishing its immutable artifact immediately before
    # the coordinator crashes. Recover those outputs before planning new work,
    # so successful transports are not recomputed. Resume / --no-cache semantics
    # still control whether existing artifacts may be replayed. Do not use a
    # scalar-only artifact in place of a missing directional observation.
    if (
        resume
        and cache_read
        and artifact_store is not None
        and not transport_only
        and observation is None
    ):
        recovered = 0
        for case in cases:
            if has_record(results, case):
                continue
            key = preparation.content_key(case)
            if key is None:
                continue
            found = artifact_store.recover(material, key)
            if found is None:
                continue
            _, output = found
            store_result(results, case, output)
            preparation.store_transport(case, output)
            checkpoint.record_finished(case)
            recovered += 1
        if recovered:
            checkpoint.cache_replayed()
            print(f"recovered {recovered} {material} case artifacts from distributed inbox")

    work = SweepWork(cases, results, observation, transport_only)
    todo = work.todo
    print(f"{len(todo)} of {len(cases)} cases to run ({len(cases) - len(todo)} cached)")
    if trajectory_capture is not None:
        if trajectory_capture.scene is None and observation is not None:
            from dataclasses import replace

            from ..instrument.scene import scene_payload

            trajectory_capture = replace(
                trajectory_capture,
                scene=scene_payload(
                    observation.observation.filters, observation.observation.detector
                ),
            )
        from ..montecarlo.trajectories import preflight_capture

        plan = preflight_capture(trajectory_capture, cases, todo)
        print(
            f"trajectories: writing {plan.to_write} case(s) to {trajectory_capture.root}"
            + (f" (replacing {plan.replaced})" if plan.replaced else "")
            + (f"; {plan.kept} already captured" if plan.kept else "")
        )
        if plan.missing_cached:
            print(
                f"trajectories: {plan.missing_cached} cached case(s) have no artifact "
                "and are not re-transported; rerun with --recompute to capture them"
            )
    if on_runtime is not None:
        on_runtime(runner.runtime_plan(todo, max_workers))
    capture_size_report = None
    if trajectory_capture is not None and os.environ.get("PYRITE_TRAJECTORY_SIZE_REPORT") == "1":
        from ._trajectory_capture import CaptureSizeReport

        capture_size_report = CaptureSizeReport(trajectory_capture, todo)
    completion = SweepCompletion(
        cases, todo, group_key, case_cost_fn, on_cost, on_progress, on_chunk
    )
    completion.report()
    completion.prepare_groups(todo, work.observation_only)

    def _cb(i, case, out):
        if isinstance(out, DirectionalArtifactRef):
            if artifact_store is None:
                raise ValueError("executor returned a directional artifact without artifact_store")
            key = preparation.content_key(case)
            if key is None:
                raise ValueError("cannot resolve key for worker-published observation")
            plan = work.observation_plans.get((case["name"], case["E0_keV"]))
            if plan is None:
                raise ValueError("executor returned unexpected directional artifact")
            out = load_directional_result(
                artifact_store,
                out,
                material=material,
                case_key=key,
                directions=plan.directions_sample,
            )
        if isinstance(out, CaseArtifactRef):
            if artifact_store is None:
                raise ValueError("executor returned a case artifact without artifact_store")
            key = preparation.content_key(case)
            if key is None:
                raise ValueError("cannot resolve content key for worker-published case")
            out = artifact_store.load(
                out,
                expected_material=material,
                expected_content_key=key,
            )
        if capture_size_report is not None:
            capture_size_report.completed(case)

        if transport_only:
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

    # Replay cached groups before launching new work, including observation-only
    # work whose scalar group is already complete.
    completion.replay_cached_groups(results)

    should_stop = None if deadline is None else (lambda: time_fn() >= deadline)
    run_cases_options = {}
    if trajectory_capture is not None:
        run_cases_options["trajectory_capture"] = trajectory_capture
    if work.observation_plans:
        run_cases_options["observation_directions"] = [
            (
                None
                if (plan := work.observation_plans.get((c["name"], c["E0_keV"]))) is None
                else plan.directions_sample
            )
            for c in todo
        ]
    if on_timing is not None:
        run_cases_options["on_timing"] = on_timing
    if on_activity is not None:
        run_cases_options["on_activity"] = on_activity

    t0 = time.perf_counter()
    # A fully cached sweep has no work for an external executor. Preserve the
    # historical run_cases([]) call when no executor was explicitly supplied.
    if todo or executor is None:
        dispatch = run_cases if executor is None else executor.execute
        dispatch(
            todo,
            max_workers=max_workers,
            progress=progress,
            callback=_cb,
            should_stop=should_stop,
            keep_results=False,  # _cb owns storage; don't pin every spectrum in RAM
            transport_only=transport_only,
            **run_cases_options,
        )
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")
    complete = all(has_record(results, c) for c in cases) and not work.observation_plans
    checkpoint.finish(complete)
    if on_activity is not None:
        on_activity({"phase": "handoff", "case": None, "in_flight_case_count": 0})
    return complete
