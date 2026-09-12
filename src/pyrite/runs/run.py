"""
run.py
==========

Drive a sweep through ``montecarlo.run_cases`` with on-disk checkpointing
and per-group streaming feedback.

:func:`run_sweep` resumes from a per-material checkpoint
(``checkpoints/<material>.pkl``, skipping cached cases), stores each finished
case into ``results``, re-pickles that material's checkpoint at config
granularity (crash-safe), and invokes ``on_chunk(group_config_names)`` once a
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

import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

from ..checkpoints import _checkpoint_store
from ..checkpoints.persistence import (  # noqa: F401 -- compatibility exports
    _CAS_OPTIONAL_PAYLOAD_KEYS,
    _CAS_PAYLOAD_KEYS,
    DEFAULT_CHECKPOINT_DIR,
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
    _valid_cas_payload,
    _write_case_manifest,
    cached_material_analysis,
    cases_from_results,
    checkpoint_manifest,
    checkpoint_path_for,
    load_checkpoint,
)
from ..checkpoints.recompute import (  # noqa: F401 -- compatibility exports
    reline_checkpoint,
    repair_brem_wide,
    repair_checkpoint,
    repair_line_spec,
)
from ..montecarlo import run_cases, runner
from ..results import store_result


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
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
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
):
    """Run ``cases`` into ``results`` (mutated in place).

    cases : list of typed cases from sweep.build_cases.
    results : the dict to fill ({name: {E0: record}}).
    checkpoint_dir : directory for the per-material checkpoints. A sweep is
        single-material, so each writes ``<checkpoint_dir>/<material>.pkl``
        holding ONLY that material's configs -- different materials never share a
        pickle, and ``results`` can hold several materials in one kernel without
        them clobbering each other on disk. Created if missing; seeded once from
        the legacy combined ``cxr_run_checkpoint.pkl`` if that's still around.
    checkpoint_path : explicit override for the pickle path; None (default)
        derives the per-material path above. A rerun with resume=True skips every
        config already in the pickle.
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
        case index, and in-flight work counts.
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

    Returns True iff every requested ``(name, E0_keV)`` pair ended up in
    ``results`` (i.e. the sweep ran to completion, budget or not); False if
    ``max_seconds`` cut it short. On an incomplete return, one final
    ``_save()`` runs (beyond the per-config saves already done in ``_cb``) so
    whatever finished before the deadline is persisted -- harmless and
    idempotent even when it turns out nothing new needed saving.
    """
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
    # Shared per-material content-addressable store lives under the checkpoint
    # directory root (the parent of this stem's component dir), so every profile
    # of this material addresses the SAME blob pool -- the cross-profile share.
    cas_root = os.path.dirname(checkpoint_path) or "."
    cache_read = cache_read and content_key_fn is not None
    cache_write = cache_write and content_key_fn is not None

    def _content_key(case):
        """Content key for ``case`` or None when it is not canonically hashable
        (an exotic/legacy case simply falls back to normal recompute + per-stem
        store rather than aborting the run)."""
        if content_key_fn is None:
            return None
        try:
            key = content_key_fn(case)
            _checkpoint_store.cas_blob_path(material, key, cas_root)
            return key
        except (TypeError, ValueError):
            return None

    def _crystal_of(rec_map):
        return next(iter(rec_map.values()))["case"]["crystal"]

    # Sharded checkpointing applies to the component-directory layout only; a
    # caller pinning a legacy ``<stem>.pkl`` keeps the whole-file save path.
    _sharded = Path(checkpoint_path).suffix != ".pkl"

    def _material_subset():
        return {n: results[n] for n in results if _crystal_of(results[n]) == material}

    def _save():
        """Pickle just THIS material's configs -- ``results`` may also hold other
        materials run earlier in the same kernel, which belong in their own pkl.
        Refreshes the sidecar manifest alongside (see :func:`_manifest_save`) so
        :func:`checkpoint_manifest` never serves a stale energy/record-count
        summary after a fresh save."""
        subset = _material_subset()
        _checkpoint_components_save(checkpoint_path, subset)
        _manifest_save(checkpoint_path, subset, dataset_identity)

    def _timed_save():
        started = time.perf_counter()
        _save()
        if on_timing is not None:
            elapsed = time.perf_counter() - started
            on_timing({"checkpoint_seconds": elapsed, "consolidation_seconds": elapsed})

    incremental_manifest = None

    def _save_part(name):
        """Persist one just-finished config as an immutable shard -- O(1) per
        config, versus ``_save``'s whole-store re-serialization (O(N^2) over a
        sweep). The manifest accumulator scans existing/resumed results once,
        then visits only each newly completed config so live viewers retain fresh
        counts without an O(N^2) sweep rescan."""
        nonlocal incremental_manifest

        started = time.perf_counter()
        path = Path(checkpoint_path)
        _checkpoint_store.save_part(path.name, path.parent, name, results[name])
        if incremental_manifest is None:
            incremental_manifest = _IncrementalManifest(dataset_identity)
            incremental_manifest.add(_material_subset())
        else:
            incremental_manifest.add({name: results[name]})
        _manifest_write(checkpoint_path, incremental_manifest.manifest())
        if on_timing is not None:
            elapsed = time.perf_counter() - started
            on_timing({"checkpoint_seconds": elapsed, "shard_write_seconds": elapsed})

    def _consolidate():
        """Fold shards into authoritative ``{line,brem,characteristic}.h5`` files the
        rest of the toolchain expects, then drop the shard directory."""
        path = Path(checkpoint_path)
        if not _checkpoint_store.has_parts(path.name, path.parent):
            return
        if on_activity is not None:
            on_activity({"phase": "saving", "case": None, "in_flight_case_count": 0})
        print("checkpoint: saving completed material")
        _timed_save()
        _checkpoint_store.clear_parts(path.name, path.parent)

    if resume and _checkpoint_exists(checkpoint_path):
        if on_activity is not None:
            on_activity({"phase": "loading", "case": None, "in_flight_case_count": 0})
        print(f"checkpoint: loading {checkpoint_path}")
        resume_started = time.perf_counter()
        manifest_path = _manifest_path_for(checkpoint_path)
        path = Path(checkpoint_path)
        existing_identity = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as handle:
                    existing_identity = json.load(handle).get("dataset_identity")
            except (OSError, ValueError, TypeError):
                pass
        if isinstance(existing_identity, dict):
            from ..campaign.profiles import normalize_dataset_identity

            existing_identity = normalize_dataset_identity(existing_identity)
        if (
            dataset_identity is not None
            and existing_identity is not None
            and dataset_identity.get("parameter_sha256")
            != existing_identity.get("parameter_sha256")
        ):
            raise ValueError(
                "checkpoint dataset identity mismatch: "
                f"requested {dataset_identity.get('profile')}:"
                f"{dataset_identity.get('parameter_sha256', '')[:12]}, existing "
                f"{existing_identity.get('profile')}:"
                f"{existing_identity.get('parameter_sha256', '')[:12]}; "
                "archive or select a different variant before resuming"
            )
        requested_proof = _case_set_proof_from_cases(cases)
        signature = _checkpoint_signature(checkpoint_path)
        manifest_fresh = (
            bool(signature)
            and os.path.isfile(manifest_path)
            and (os.stat(manifest_path).st_mtime_ns >= max(item[1] for item in signature))
        )
        if (
            metadata_only_complete
            and on_chunk is None
            and not _checkpoint_store.has_parts(path.name, path.parent)
            and manifest_fresh
            and isinstance(existing_identity, dict)
            and dataset_identity is not None
            and existing_identity.get("parameter_sha256")
            == dataset_identity.get("parameter_sha256")
        ):
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            if manifest.get("completed_case_set") == requested_proof:
                if on_timing is not None:
                    on_timing({"resume_seconds": time.perf_counter() - resume_started})
                if on_runtime is not None:
                    on_runtime(runner.runtime_plan([], max_workers))
                total_cost = (
                    sum(case_cost_fn(case) for case in cases) if case_cost_fn is not None else None
                )
                if on_cost is not None and total_cost is not None:
                    on_cost(total_cost, total_cost)
                if on_progress is not None:
                    on_progress(0, len(cases), len(cases))
                if on_activity is not None:
                    on_activity({"phase": "handoff", "case": None, "in_flight_case_count": 0})
                print(f"0 of {len(cases)} cases to run ({len(cases)} cached; metadata proof)")
                return True
        loaded = _checkpoint_load(checkpoint_path)
        print(f"checkpoint: loaded in {time.perf_counter() - resume_started:.3f} s")
        if on_timing is not None:
            on_timing({"resume_seconds": time.perf_counter() - resume_started})
        results.update(loaded)
        print(
            f"resumed {sum(len(v) for v in loaded.values())} {material} cases from {checkpoint_path}"
        )

        # Migration: seed missing blobs from a compatible existing checkpoint.
        # Compare the stored case's key with the currently requested case before
        # writing, so a legacy checkpoint without identity provenance can never
        # poison the shared store with a mismatched record.
        if cache_write:
            for case in cases:
                record = results.get(case["name"], {}).get(case["E0_keV"])
                if record is None:
                    continue
                key = _content_key(case)
                stored_key = _content_key(record.get("case", {}))
                payload = _cas_payload_from_record(record)
                if (
                    key is not None
                    and key == stored_key
                    and payload is not None
                    and not _checkpoint_store.cas_contains(material, key, cas_root)
                ):
                    _checkpoint_store.cas_save(material, key, cas_root, payload)

    # Cross-profile reuse: any case not already resumed from this profile's own
    # checkpoint whose content key already has a CAS blob (populated by ANY
    # profile of this material) is replayed through store_result instead of
    # recomputed. Replaying with the REQUESTED case makes the reconstructed
    # record bit-identical to a fresh compute (the requester's own detector/flux
    # scalars, its own case payload); only the profile-invariant transport arrays
    # come from the shared blob.
    if cache_read:
        reused = 0
        for c in cases:
            if c["name"] in results and c["E0_keV"] in results[c["name"]]:
                continue
            key = _content_key(c)
            if key is None or not _checkpoint_store.cas_contains(material, key, cas_root):
                continue
            try:
                out = _checkpoint_store.cas_load(material, key, cas_root)
            except (OSError, EOFError, pickle.UnpicklingError):
                continue
            if not _valid_cas_payload(out):
                continue
            store_result(results, c, out)
            reused += 1
        if reused:
            print(f"reused {reused} {material} cases from shared content cache")
            _save()  # persist reused cases into this profile's checkpoint + manifest

    # Thin per-profile pointer table: (name, E0_keV) -> content_key, plus label
    # provenance. Enables a future GC / `pyrite clear` to know which shared blobs a
    # profile references. Written on any run that populates the store.
    if cache_write:
        _write_case_manifest(checkpoint_path, cases, _content_key, dataset_identity)

    todo = [c for c in cases if not (c["name"] in results and c["E0_keV"] in results[c["name"]])]
    cached_cases = len(cases) - len(todo)
    print(f"{len(todo)} of {len(cases)} cases to run ({cached_cases} cached)")
    if on_runtime is not None:
        on_runtime(runner.runtime_plan(todo, max_workers))
    completed_new_cases = 0
    total_cost = sum(case_cost_fn(c) for c in cases) if case_cost_fn is not None else None
    done_cost = (
        total_cost - sum(case_cost_fn(c) for c in todo) if case_cost_fn is not None else None
    )
    if on_cost is not None and case_cost_fn is not None:
        on_cost(done_cost, total_cost)
    if on_progress is not None:
        on_progress(completed_new_cases, len(cases), cached_cases)

    # all config names in each group (cached + to-run), in first-seen order
    group_names = defaultdict(list)
    seen = set()
    for c in cases:
        if c["name"] not in seen:
            seen.add(c["name"])
            group_names[group_key(c)].append(c["name"])

    # per group: how many of its configs still have unfinished energies;
    # per config: how many beam energies it still owes
    group_remaining = defaultdict(int)
    energies_remaining = {}
    for c in todo:
        if c["name"] not in energies_remaining:
            group_remaining[group_key(c)] += 1
        energies_remaining[c["name"]] = energies_remaining.get(c["name"], 0) + 1
    n_groups = len(group_remaining)
    progress_state = {"done": 0}

    def _cb(i, case, out):
        nonlocal completed_new_cases, done_cost

        if transport_only:
            completed_new_cases += 1

            if case_cost_fn is not None:
                done_cost += case_cost_fn(case)

            if on_cost is not None and case_cost_fn is not None:
                on_cost(done_cost, total_cost)

            if on_progress is not None:
                on_progress(completed_new_cases, len(cases), cached_cases)

            return

        store_result(results, case, out)

        if cache_write:
            key = _content_key(case)
            if key is not None:
                _checkpoint_store.cas_save(material, key, cas_root, out)
        if on_case is not None:
            on_case(case)
        name = case["name"]
        energies_remaining[name] -= 1
        if energies_remaining[name] == 0:  # this config (all energies) is done
            if _sharded:
                _save_part(name)  # crash-safe at config granularity, O(1) write
            else:
                _timed_save()  # legacy monolith: whole-file rewrite
            g = group_key(case)
            group_remaining[g] -= 1
            if group_remaining[g] == 0 and on_chunk is not None:
                # the whole azimuth sweep at this tilt is in -> stream it, with a
                # progress line that stays next to the streamed plot/table (the
                # tqdm bar is up top and scrolls away)
                progress_state["done"] += 1
                print(
                    f"\n=== {case['thickness_ang'] / 1e4:g} um, tilt "
                    f"{case['tilt_deg']:g} deg done "
                    f"-- {progress_state['done']}/{n_groups} tilt-groups ==="
                )
                on_chunk(group_names[g])
        completed_new_cases += 1
        if case_cost_fn is not None:
            done_cost += case_cost_fn(case)
        if on_cost is not None and case_cost_fn is not None:
            on_cost(done_cost, total_cost)
        if on_progress is not None:
            on_progress(completed_new_cases, len(cases), cached_cases)

    # Fully-cached tilt-groups have no cases left to run, so the callback above
    # never fires for them. On resume, replay them through on_chunk first (in
    # tilt order) so their best-azimuth plots/tables are redrawn too.
    if on_chunk is not None:
        for g in sorted(g for g in group_names if group_remaining.get(g, 0) == 0):
            rep = next(iter(results[group_names[g][0]].values()))["case"]
            print(
                f"\n=== cached: {rep['thickness_ang'] / 1e4:g} um, "
                f"tilt {rep['tilt_deg']:g} deg (already computed) ==="
            )
            on_chunk(group_names[g])

    should_stop = None if deadline is None else (lambda: time_fn() >= deadline)
    profile_callbacks = {}
    if on_timing is not None:
        profile_callbacks["on_timing"] = on_timing
    if on_activity is not None:
        profile_callbacks["on_activity"] = on_activity

    t0 = time.perf_counter()
    run_cases(
        todo,
        max_workers=max_workers,
        progress=progress,
        callback=_cb,
        should_stop=should_stop,
        keep_results=False,  # _cb owns storage; don't pin every spectrum in RAM
        transport_only=transport_only,
        **profile_callbacks,
    )
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")
    complete = all(c["name"] in results and c["E0_keV"] in results[c["name"]] for c in cases)
    if _sharded:
        if complete:
            # Completion publishes one authoritative component pair. Budget pauses
            # retain immutable shards so the next slice does no full rewrite.
            _consolidate()
    elif not complete:
        _timed_save()  # persist whatever finished before the budget ran out
    if on_activity is not None:
        on_activity({"phase": "handoff", "case": None, "in_flight_case_count": 0})
    return complete
