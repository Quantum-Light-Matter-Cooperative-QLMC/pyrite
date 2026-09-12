"""Checkpoint paths, serialization, manifests, and cached analysis."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

from ..paths import workspace_root
from ..results import records, sweep_values
from . import _checkpoint_io, _checkpoint_store

DEFAULT_CHECKPOINT_DIR = str(workspace_root() / "checkpoints")


def checkpoint_path_for(material, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Path to the per-material component checkpoint directory."""
    return os.path.join(checkpoint_dir, material)


def _manifest_path_for(checkpoint_path):
    """Manifest path for a component directory or legacy checkpoint pickle."""
    if not str(checkpoint_path).endswith(".pkl"):
        legacy = Path(checkpoint_path).with_suffix(".pkl")
        path = Path(checkpoint_path)
        migrated = _checkpoint_store.component_path(path.name, "line", path.parent).is_file()
        if legacy.is_file() and not migrated:
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
    :func:`pyrite.campaign.profiles.case_content_key`), plus label provenance
    (``catalog_profile``/``variant``/``parameter_sha256``). This is the reference
    list a future GC / ``pyrite clear`` walks to decide which shared blobs a profile
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
        from ..campaign.profiles import normalize_dataset_identity

        dataset_identity = normalize_dataset_identity(dataset_identity)
        manifest["identity_version"] = dataset_identity["identity_version"]
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
_CAS_OPTIONAL_PAYLOAD_KEYS = frozenset(
    {"E_grid_brem", "brem_wide", "hit_frac", "spec_coherent", "spec_characteristic"}
)


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
    :mod:`pyrite.checkpoints._checkpoint_io`, TODO P2 #8): write a sibling ``.<pid>.tmp``
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
    return _checkpoint_store.checkpoint_exists(path.name, path.parent)


def _checkpoint_load(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return _checkpoint_io.load(str(path))
    return _checkpoint_store.load(path.name, path.parent)


def _checkpoint_components_save(
    checkpoint_path,
    results,
    *,
    components=("line", "brem", "characteristic"),
):
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
    return _checkpoint_store.signature(path.name, path.parent)


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


def load_checkpoint(material, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Load a per-material results checkpoint (``checkpoints/<material>.pkl``)
    written by :func:`run_sweep`, WITHOUT re-running anything -- this is how the
    visualization app (``src/pyrite/apps/analysis_app.py``) gets its ``results`` after the
    scan-runner app (``src/pyrite/apps/scan_app.py``) has produced them. Returns the
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
        print(f"no checkpoint at {path} -- run `pyrite run standard -m {material}` first")
        return {}
    return _load_checkpoint_cached(path, _checkpoint_signature(path))


_material_analysis_cache = {}
_MATERIAL_ANALYSIS_CACHE_VERSION = 1


def _material_analysis_cache_path(path, key):
    """Stable disk-cache path for one checkpoint analysis."""
    payload = pickle.dumps((_MATERIAL_ANALYSIS_CACHE_VERSION, key), protocol=5)
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return Path(path).parent / ".analysis-cache" / f"{Path(path).stem}-{digest}.pkl"


def cached_material_analysis(material, analyze, key, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
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
    :func:`pyrite.results.sweep_values`). Numpy scalars are coerced to plain
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
        # v2 readers ignore additive fields; keep the public schema label while
        # completed_case_set carries its own independently versioned schema.
        "schema": "cxr.checkpoint-manifest.v2",
        "energies_keV": energies,
        "n_records": len(records(results)),
        "sweep": sweep_json,
        "completed_case_set": _case_set_proof_from_results(results),
    }
    if dataset_identity is not None:
        from ..campaign.profiles import normalize_dataset_identity

        identity = normalize_dataset_identity(dataset_identity)
        manifest["identity_version"] = identity["identity_version"]
        manifest["dataset_identity"] = identity
    return manifest


class _IncrementalManifest:
    """Accumulate a live shard manifest without rescanning prior records.

    The exact completed-case proof is intentionally published only by the final
    consolidated manifest. While shards exist, consumers need current counts and
    sweep values; the completion fast path already refuses metadata-only proofs
    when ``parts/`` is present.
    """

    _FIELDS = ("crystal", "E0_keV", "tilt_deg", "tilt_azim_deg", "thickness_ang", "B_ang2")

    def __init__(self, dataset_identity=None):
        self._keys = set()
        self._values = defaultdict(set)
        self._identity = dataset_identity

    def add(self, results):
        for name, by_energy in results.items():
            for energy, record in by_energy.items():
                key = (str(name), float(energy))
                if key in self._keys:
                    continue
                self._keys.add(key)
                case = record["case"]
                for field in self._FIELDS:
                    if field in case:
                        self._values[field].add(case[field])

    def manifest(self):
        sweep = {
            field: [value.item() if hasattr(value, "item") else value for value in sorted(values)]
            for field, values in self._values.items()
        }
        manifest = {
            "schema": "cxr.checkpoint-manifest.v2",
            "energies_keV": sorted(float(value) for value in sweep.get("E0_keV", [])),
            "n_records": len(self._keys),
            "sweep": sweep,
        }
        if self._identity is not None:
            from ..campaign.profiles import normalize_dataset_identity

            identity = normalize_dataset_identity(self._identity)
            manifest["identity_version"] = identity["identity_version"]
            manifest["dataset_identity"] = identity
        return manifest


def _case_set_proof(keys):
    """Compact exact proof for a set of checkpoint record identities."""
    canonical = sorted((str(name), float(energy).hex()) for name, energy in keys)
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "schema": "cxr.completed-case-set.v1",
        "count": len(canonical),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _case_set_proof_from_results(results):
    return _case_set_proof(
        (name, energy) for name, by_energy in results.items() for energy in by_energy
    )


def _case_set_proof_from_cases(cases):
    return _case_set_proof((case["name"], case["E0_keV"]) for case in cases)


def _manifest_write(checkpoint_path, manifest):
    """Atomically publish one already-built sidecar manifest."""
    manifest_path = _manifest_path_for(checkpoint_path)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, manifest_path)
    return manifest


def _manifest_save(checkpoint_path, results, dataset_identity=None):
    """Build and atomically refresh the sidecar checkpoint manifest.

    Full construction is used for authoritative component saves and backfills.
    Per-config shards use :class:`_IncrementalManifest` so a sweep does not
    repeatedly traverse every accumulated result.
    """
    manifest_path = _manifest_path_for(checkpoint_path)
    if dataset_identity is None and os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as existing:
                dataset_identity = json.load(existing).get("dataset_identity")
        except (OSError, ValueError, TypeError):
            pass
    manifest = _manifest_for(results, dataset_identity)
    return _manifest_write(checkpoint_path, manifest)


def _save_recomputed_checkpoint(checkpoint_path, results, *, components):
    """Atomically refresh a recomputed component and its CAS reachability data.

    CAS blobs land first, component and summary manifests next, and ``cases.json``
    last. A crash therefore never publishes a content-key reference before its
    blob or component exists.
    """
    from ..campaign.profiles import case_content_key

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


def checkpoint_manifest(material, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
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
            manifest = json.load(f)
        identity = manifest.get("dataset_identity")
        if identity is not None:
            from ..campaign.profiles import normalize_dataset_identity

            if not isinstance(identity, dict):
                raise ValueError(f"invalid checkpoint metadata {manifest_path}: dataset_identity")
            identity = normalize_dataset_identity(identity)
            recorded_version = manifest.get("identity_version", identity["identity_version"])
            if recorded_version != identity["identity_version"]:
                raise ValueError(f"invalid checkpoint metadata {manifest_path}: identity versions")
            manifest["identity_version"] = identity["identity_version"]
            manifest["dataset_identity"] = identity
        return manifest
    results = load_checkpoint(material, checkpoint_dir)
    return _manifest_save(path, results)


def cases_from_results(results):
    """The flat case list backing a loaded ``results`` store (each record carries
    its own ``case``) -- pass to filter_results / plot_heatmaps / the trajectory
    grid so the visualization app never has to rebuild the Sweep."""
    return [rec["case"] for recs in results.values() for rec in recs.values()]
