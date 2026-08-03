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

import functools
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

from . import _checkpoint_io, _checkpoint_store
from .energy_grid.encoding import decode_energy_grid
from .montecarlo import run_cases, runner
from .results import records, store_result, sweep_values

# Anchored to the repo root (src/cxr_mc/run.py -> parents[2] = repo root) so
# checkpoint lookup works regardless of the notebook's kernel cwd.
_DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parents[2] / "checkpoints")


def checkpoint_path_for(material, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Path to the per-material component checkpoint directory."""
    return os.path.join(checkpoint_dir, material)


def _manifest_path_for(checkpoint_path):
    """Manifest path for a component directory or legacy checkpoint pickle."""
    if not str(checkpoint_path).endswith(".pkl"):
        legacy = Path(checkpoint_path).with_suffix(".pkl")
        if legacy.is_file() and not (Path(checkpoint_path) / "line.pkl").is_file():
            return str(legacy.with_suffix(".meta.json"))
        return os.path.join(checkpoint_path, "meta.json")
    base, _ = os.path.splitext(checkpoint_path)
    return f"{base}.meta.json"


def _case_manifest_path(checkpoint_path):
    """Path to a stem's per-profile case manifest (``cases.json``)."""
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return path.with_suffix(".cases.json")
    return path / "cases.json"


def _write_case_manifest(checkpoint_path, cases, content_key_fn, dataset_identity):
    """Atomically write the thin per-profile pointer table mapping each case's
    ``(name, E0_keV)`` to its shared-store ``content_key`` (see
    :func:`cxr_mc.profiles.case_content_key`), plus label provenance
    (``catalog_profile``/``variant``/``parameter_sha256``). This is the reference
    list a future GC / ``cxr clear`` walks to decide which shared blobs a profile
    still needs. Atomic temp-file + ``os.replace`` mirrors :func:`_manifest_save`."""
    entries = []
    for case in cases:
        key = content_key_fn(case)
        if key is None:
            continue
        entries.append({"name": case["name"], "E0_keV": float(case["E0_keV"]), "content_key": key})
    manifest = {
        "schema": "cxr.case-manifest.v1",
        "material": cases[0]["crystal"] if cases else None,
        "cases": entries,
    }
    if dataset_identity is not None:
        for field in ("catalog_profile", "variant", "parameter_sha256"):
            value = dataset_identity.get(field)
            if value is not None:
                manifest[field] = value
    manifest_path = _case_manifest_path(checkpoint_path)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        json.dump(manifest, handle)
    os.replace(tmp, manifest_path)
    return manifest


_CAS_PAYLOAD_KEYS = frozenset({"E_grid", "spec", "brem", "eta"})
_CAS_OPTIONAL_PAYLOAD_KEYS = frozenset({"E_grid_brem", "brem_wide", "hit_frac", "spec_coherent"})


def _cas_payload_from_record(record):
    """Recover the raw ``run_case`` payload retained by :func:`store_result`.

    Used only to seed the CAS from a compatible pre-CAS profile checkpoint.
    Derived/reporting fields (``scale``, current, FWHM, peak energy, case) stay
    requester-owned and are rebuilt by :func:`store_result` on reuse.
    """
    if not _CAS_PAYLOAD_KEYS <= record.keys():
        return None
    keys = _CAS_PAYLOAD_KEYS | _CAS_OPTIONAL_PAYLOAD_KEYS
    return {key: record[key] for key in keys if key in record}


def _valid_cas_payload(payload):
    return isinstance(payload, dict) and _CAS_PAYLOAD_KEYS <= payload.keys()


def _checkpoint_save(checkpoint_path, results):
    """Atomically gzip-pickle ``results`` to ``checkpoint_path`` (see
    :mod:`cxr_mc._checkpoint_io`, TODO P2 #8): write a sibling ``.<pid>.tmp``
    then ``os.replace`` it into place. The replace is atomic on a single
    filesystem, so a crash/OOM mid-write never leaves a half-written ``.pkl``
    -- the old checkpoint survives intact and the run stays resumable.

    The pid in the temp name keeps two processes writing the *same* checkpoint
    from sharing one ``.tmp``: with a fixed name they clobber each other's temp
    and race on the rename -- the first ``os.replace`` consumes it, the second
    dies with ``FileNotFoundError``. (The remote runner also guards against
    concurrent same-material jobs, but this makes the save safe on its own.)
    Shared by :func:`run_sweep` (per-config crash-safe saves) and
    :func:`repair_checkpoint`."""
    tmp = f"{checkpoint_path}.{os.getpid()}.tmp"
    _checkpoint_io.dump(results, tmp)
    os.replace(tmp, checkpoint_path)


def _checkpoint_exists(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return path.is_file()
    return (
        (path / "line.pkl").is_file()
        or path.with_suffix(".pkl").is_file()
        or _checkpoint_store.has_parts(path.name, path.parent)
    )


def _checkpoint_load(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return _checkpoint_io.load(str(path))
    if (path / "line.pkl").is_file():
        base = _checkpoint_store.load(path.name, path.parent)
    elif path.with_suffix(".pkl").is_file():
        base = _checkpoint_io.load(str(path.with_suffix(".pkl")))
    else:
        base = {}
    # Unconsolidated crash-safety shards (a sweep interrupted before its final
    # consolidation) hold the newest per-config records; union them over any
    # stale monolith, per (name, E0).
    for name, by_energy in _checkpoint_store.load_parts(path.name, path.parent).items():
        base.setdefault(name, {}).update(by_energy)
    return base


def _checkpoint_components_save(checkpoint_path, results, *, components=("line", "brem")):
    """Save component directory, retaining explicit legacy-file compatibility."""
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        _checkpoint_save(str(path), results)
        return
    _checkpoint_store.save(path.name, path.parent, results, components=components)


def _checkpoint_signature(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        stat = path.stat()
        return ((str(path.resolve()), stat.st_mtime_ns, stat.st_size),)
    if not (path / "line.pkl").is_file() and path.with_suffix(".pkl").is_file():
        legacy = path.with_suffix(".pkl")
        stat = legacy.stat()
        return ((str(legacy.resolve()), stat.st_mtime_ns, stat.st_size),)
    sig = _checkpoint_store.signature(path.name, path.parent)
    return sig or _checkpoint_store.parts_signature(path.name, path.parent)


@functools.lru_cache(maxsize=4)
def _load_checkpoint_cached(path, signature):
    """The actual gunzip+unpickle behind :func:`load_checkpoint`, memoized
    module-globally on ``(path, mtime)`` (``functools.lru_cache(maxsize=4)``).

    Checkpoints are 140-225 MB gzip-pickles and the marimo analysis app calls
    ``load_checkpoint`` repeatedly -- once per material switch, plus once per
    catalog material just to enumerate beam energies -- so a naive per-call
    load dominates wall time. Keying on mtime means a re-run scan (which
    rewrites the ``.pkl`` and bumps its mtime) invalidates the cache
    automatically; no manual invalidation, no staleness. The "loaded N
    records" print lives here rather than in ``load_checkpoint`` so cache hits
    stay silent.
    """
    results = _checkpoint_load(path)
    n = sum(len(v) for v in results.values())
    material = Path(path).stem if str(path).endswith(".pkl") else Path(path).name
    print(f"loaded {n} {material} records from {path}")
    return results


def load_checkpoint(material, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Load a per-material results checkpoint (``checkpoints/<material>.pkl``)
    written by :func:`run_sweep`, WITHOUT re-running anything -- this is how the
    visualization app (``notebooks/analysis_app.py``) gets its ``results`` after the
    scan-runner app (``notebooks/scan_app.py``) has produced them. Returns the
    ``{name: {E0: record}}`` store (empty dict if the checkpoint is missing).

    Reconstruct the sweep's case list straight from it with
    ``cases = [r["case"] for r in results.records(results)]`` -- the records
    carry their own cases, so the visualization app needs no Sweep to filter/plot.

    Cached module-globally keyed on the checkpoint's mtime (see
    :func:`_load_checkpoint_cached`): repeated calls for the same material are
    free until the checkpoint file next changes. Prefer :func:`checkpoint_manifest`
    when only the checkpoint's energies/record count/sweep values are needed --
    it avoids the unpickle entirely on the common path."""
    path = checkpoint_path_for(material, checkpoint_dir)
    if not _checkpoint_exists(path):
        print(f"no checkpoint at {path} -- run `cxr run standard -m {material}` first")
        return {}
    return _load_checkpoint_cached(path, _checkpoint_signature(path))


_material_analysis_cache = {}
_MATERIAL_ANALYSIS_CACHE_VERSION = 1


def _material_analysis_cache_path(path, key):
    """Stable disk-cache path for one checkpoint analysis."""
    payload = pickle.dumps((_MATERIAL_ANALYSIS_CACHE_VERSION, key), protocol=5)
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return Path(path).parent / ".analysis-cache" / f"{Path(path).stem}-{digest}.pkl"


def cached_material_analysis(material, analyze, key, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Persist ``analyze(load_checkpoint(material))`` per material, keyed on the
    checkpoint's ``(resolved path, mtime_ns, size)`` plus caller-supplied ``key``
    (e.g. selection parameters distinguishing what ``analyze`` computed).

    Distinct from :func:`_load_checkpoint_cached`, which memoizes the raw
    unpickle itself with ``maxsize=4`` -- fine for a user paging through 1-2
    materials at a time, but a cross-material comparison that touches every
    catalog material thrashes it, forcing a full 140-225 MB gzip re-unpickle
    of every material on each re-render. This cache instead stores
    ``analyze``'s small return value both in memory and under
    ``checkpoints/.analysis-cache/``. Thus app restarts and eviction from the
    small unpickle cache do not touch the large checkpoint again. Rewriting or
    replacing the checkpoint invalidates the persistent entry automatically.

    Returns ``None`` if no checkpoint exists for ``material``, without
    calling ``analyze``."""
    path = checkpoint_path_for(material, checkpoint_dir)
    if not _checkpoint_exists(path):
        return None
    checkpoint_key = _checkpoint_signature(path)
    cache_key = (*checkpoint_key, key)
    if cache_key in _material_analysis_cache:
        return _material_analysis_cache[cache_key]

    disk_path = _material_analysis_cache_path(path, key)
    try:
        cached = _checkpoint_io.load(disk_path)
        if cached["checkpoint"] == checkpoint_key:
            value = cached["value"]
            _material_analysis_cache[cache_key] = value
            return value
    except (FileNotFoundError, EOFError, OSError, KeyError, TypeError, pickle.UnpicklingError):
        pass

    value = analyze(load_checkpoint(material, checkpoint_dir))
    _material_analysis_cache[cache_key] = value
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = disk_path.with_name(f".{disk_path.name}.{os.getpid()}.tmp")
    _checkpoint_io.dump({"checkpoint": checkpoint_key, "value": value}, tmp)
    os.replace(tmp, disk_path)
    return value


def _manifest_for(results, dataset_identity=None):
    """Build the sidecar manifest dict for a ``results`` store: distinct beam
    energies, total record count, and the swept case fields (see
    :func:`cxr_mc.results.sweep_values`). Numpy scalars are coerced to plain
    Python via ``.item()`` so the result is JSON-serializable. Shared by
    :func:`_manifest_save` (write path) and :func:`checkpoint_manifest`'s
    backfill (read path)."""
    sweep = sweep_values(results)
    sweep_json = {
        field: [v.item() if hasattr(v, "item") else v for v in values]
        for field, values in sweep.items()
    }
    energies = sorted(float(e) for e in sweep_json.get("E0_keV", []))
    manifest = {
        "schema": "cxr.checkpoint-manifest.v2",
        "energies_keV": energies,
        "n_records": len(records(results)),
        "sweep": sweep_json,
    }
    if dataset_identity is not None:
        manifest["dataset_identity"] = dataset_identity
    return manifest


def _manifest_save(checkpoint_path, results, dataset_identity=None):
    """Atomically write/refresh the sidecar checkpoint manifest
    (``<material>.meta.json``) alongside a ``_checkpoint_save`` -- lets
    :func:`checkpoint_manifest` enumerate a checkpoint's energies/record
    count/sweep values without unpickling the (140-225 MB) checkpoint itself.
    JSON is tiny, so this runs on every save. Atomic write mirrors
    ``_checkpoint_save``: a sibling ``.<pid>.tmp`` then ``os.replace``, so a
    crash never leaves a half-written ``meta.json``. Returns the manifest dict
    written."""
    manifest_path = _manifest_path_for(checkpoint_path)
    if dataset_identity is None and os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as existing:
                dataset_identity = json.load(existing).get("dataset_identity")
        except (OSError, ValueError, TypeError):
            pass
    manifest = _manifest_for(results, dataset_identity)
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, manifest_path)
    return manifest


def _save_recomputed_checkpoint(checkpoint_path, results, *, components):
    """Atomically refresh a recomputed component and its CAS reachability data.

    CAS blobs land first, component and summary manifests next, and ``cases.json``
    last. A crash therefore never publishes a content-key reference before its
    blob or component exists.
    """
    from .profiles import case_content_key

    path = Path(checkpoint_path)
    root = path.parent
    cases = []
    keys_by_case_id = {}
    for by_energy in results.values():
        for record in by_energy.values():
            case = record.get("case")
            payload = _cas_payload_from_record(record)
            if not isinstance(case, dict) or not _valid_cas_payload(payload):
                continue
            key = case_content_key(case)
            _checkpoint_store.cas_save(str(case["crystal"]), key, root, payload)
            cases.append(case)
            keys_by_case_id[id(case)] = key

    manifest_path = _manifest_path_for(checkpoint_path)
    dataset_identity = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as handle:
                dataset_identity = json.load(handle).get("dataset_identity")
        except (OSError, ValueError, TypeError):
            dataset_identity = None

    _checkpoint_components_save(checkpoint_path, results, components=components)
    _manifest_save(checkpoint_path, results, dataset_identity)
    _write_case_manifest(
        checkpoint_path,
        cases,
        lambda case: keys_by_case_id.get(id(case)),
        dataset_identity,
    )


def checkpoint_manifest(material, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Summary of a checkpoint's contents -- distinct beam energies, record
    count, and swept case fields -- WITHOUT unpickling the checkpoint itself.

    The analysis app enumerates beam energies across every catalog material
    just to populate its selectors; doing that through :func:`load_checkpoint`
    (even cached) means a real gunzip+unpickle of a 140-225 MB pickle the
    first time each material is touched. This reads the sidecar
    ``checkpoints/<material>.meta.json`` instead when it is at least as fresh
    as the ``.pkl`` (mtime comparison) -- a re-run scan bumps the ``.pkl``'s
    mtime past the last written manifest via :func:`_manifest_save`, so a
    stale manifest is detected automatically.

    Returns ``None`` if no checkpoint exists for ``material``. If the manifest
    is missing or stale, backfills it: loads the checkpoint (through the
    :func:`load_checkpoint` cache), computes the manifest, atomically writes
    it, and returns it. Schema::

        {"energies_keV": [float, ...], "n_records": int,
         "sweep": {field: [values, ...]}}
    """
    path = checkpoint_path_for(material, checkpoint_dir)
    if not _checkpoint_exists(path):
        return None
    manifest_path = _manifest_path_for(path)
    newest_component = max(item[1] for item in _checkpoint_signature(path))
    if os.path.exists(manifest_path) and os.stat(manifest_path).st_mtime_ns >= newest_component:
        with open(manifest_path) as f:
            return json.load(f)
    results = load_checkpoint(material, checkpoint_dir)
    return _manifest_save(path, results)


def cases_from_results(results):
    """The flat case list backing a loaded ``results`` store (each record carries
    its own ``case``) -- pass to filter_results / plot_heatmaps / the trajectory
    grid so the visualization app never has to rebuild the Sweep."""
    return [rec["case"] for recs in results.values() for rec in recs.values()]


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
    checkpoint_dir=_DEFAULT_CHECKPOINT_DIR,
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
    time_fn=None,
    dataset_identity=None,
    case_cost_fn=None,
    on_cost=None,
    content_key_fn=None,
    cache_read=True,
    cache_write=True,
):
    """Run ``cases`` into ``results`` (mutated in place).

    cases : list of case dicts from sweep.build_cases.
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
    on_case : optional callback(case) fired with each case dict as it finishes,
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
    if group_key is None:
        group_key = _default_group_key

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
            on_timing({"checkpoint_seconds": time.perf_counter() - started})

    def _save_part(name):
        """Persist one just-finished config as an immutable shard -- O(1) per
        config, versus ``_save``'s whole-store re-serialization (O(N^2) over a
        sweep). The manifest is still refreshed each time (tiny JSON) so live
        viewers see fresh energy/record counts before consolidation."""
        started = time.perf_counter()
        path = Path(checkpoint_path)
        _checkpoint_store.save_part(path.name, path.parent, name, results[name])
        _manifest_save(checkpoint_path, _material_subset(), dataset_identity)
        if on_timing is not None:
            on_timing({"checkpoint_seconds": time.perf_counter() - started})

    def _consolidate():
        """Fold shards into the authoritative ``{line,brem}.pkl`` monolith the
        rest of the toolchain expects, then drop the shard directory."""
        _timed_save()
        path = Path(checkpoint_path)
        _checkpoint_store.clear_parts(path.name, path.parent)

    if resume and _checkpoint_exists(checkpoint_path):
        manifest_path = _manifest_path_for(checkpoint_path)
        existing_identity = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as handle:
                    existing_identity = json.load(handle).get("dataset_identity")
            except (OSError, ValueError, TypeError):
                pass
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
        loaded = _checkpoint_load(checkpoint_path)
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
    # provenance. Enables a future GC / `cxr clear` to know which shared blobs a
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

    if time_fn is None:
        time_fn = time.monotonic
    deadline = None if max_seconds is None else time_fn() + max_seconds
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
        **profile_callbacks,
    )
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")
    complete = all(c["name"] in results and c["E0_keV"] in results[c["name"]] for c in cases)
    if _sharded:
        # Shards carry the live records; fold them into the monolith once, at the
        # end -- whether the sweep finished or the budget cut it short.
        _consolidate()
    elif not complete:
        _timed_save()  # persist whatever finished before the budget ran out
    return complete


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
    profile=None,
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

    ne_brem / brem_step_eV : NEW brem parameters to recompute with (``cxr
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
        so ``cxr rebrem --progress-file`` feeds the remote progress dashboard.
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

    from .montecarlo import _brem_for_case

    retune = any(
        value is not None for value in (ne_brem, brem_start_eV, brem_stop_eV, brem_step_eV, profile)
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
                    if profile is not None and c.get("brem_profile") != profile:
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
        if profile is not None:
            c["brem_profile"] = profile
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
        # fragments record-over-record until a long `cxr rebrem` fills the card
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
    (e.g. ``ne_brem=``, ``brem_step_eV=`` -- ``cxr rebrem``) pass through to
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
        from .sweep import _line_grid_for_energy

        base = np.asarray(_line_grid_for_energy(sweep, eg, float(r["case"]["E0_keV"])), float)
    else:
        base = eg
    if any(value is not None for value in (line_start_eV, line_stop_eV, line_step_eV)):
        from .recompute_defaults import uniform_bounds

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
    profile=None,
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
    CURRENT line path (:func:`cxr_mc.montecarlo._line_pair_for_case`) -- the
    mirror of :func:`repair_brem_wide`, WITHOUT recomputing the brem background.
    A ``coherent``/``both`` checkpoint (one carrying ``spec_coherent``) has BOTH
    its incoherent ``spec`` and its ``spec_coherent`` refreshed onto the new grid
    from a single re-transport, so the coherent array never drifts off-grid.

    Target line grid per record (see :func:`_target_line_grid`): ``line_step_eV``
    (explicit uniform spacing), else ``from_config`` rebuilds it from
    ``material``'s current ``E_grid_line_by_energy`` at the record's E0 (edit
    materials.toml, ``cxr reline`` re-runs lines on the new grid), else the
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
            if profile is None:
                from .config import material_sweep

                sweep = material_sweep(material, catalog_profile=catalog_profile)
            else:
                from .recompute_defaults import sweep as profile_sweep

                sweep = profile_sweep(
                    material,
                    profile,
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
            profile_ok = profile is None or r["case"].get("line_profile") == profile
            at_target = finite and grid_ok and ne_ok and profile_ok
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
        if profile is not None:
            c["line_profile"] = profile
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
