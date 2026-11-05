"""Resume and cache eligibility, independent of execution accounting."""

import json
import os
import pickle
import time
from pathlib import Path

from ..checkpoints import _checkpoint_store
from ..checkpoints.persistence import (
    _cas_payload_from_record,
    _case_set_proof_from_cases,
    _checkpoint_exists,
    _checkpoint_signature,
    _CheckpointIdentityError,
    _manifest_path_for,
    _valid_cas_payload,
    _write_case_manifest,
)
from ..results import store_result


def validate_sweep_request(cases, results, group_key):
    """Reject an ambiguous request before any checkpoint, cache or file effect.

    A sweep is single-material, ``(name, E0_keV)`` identifies one record and one
    observation, and every energy of a config belongs to one ``group_key`` group
    because configs complete and publish by name. An empty request has no
    material or identity to contradict and passes.
    """
    if not cases:
        return
    material = cases[0]["crystal"]
    identities = set()
    group_by_name = {}
    for case in cases:
        if case["crystal"] != material:
            raise ValueError(
                f"run_sweep is single-material: case {case['name']!r} is "
                f"{case['crystal']!r} but the sweep is {material!r}; "
                "run one sweep per material"
            )
        identity = (case["name"], case["E0_keV"])
        if identity in identities:
            raise ValueError(
                f"duplicate sweep case (name={identity[0]!r}, E0_keV={identity[1]!r}); "
                "records and observations are keyed by this pair"
            )
        identities.add(identity)
        group = _hashable_group(group_key, case)
        if group_by_name.setdefault(case["name"], group) != group:
            raise ValueError(
                f"config {case['name']!r} spans group_key groups {group_by_name[case['name']]!r} "
                f"and {group!r}; all energies of a config must share one group"
            )
    for name, values in results.items():
        crystals = _record_crystals(values)
        ambiguous = material in crystals and len(crystals) > 1
        if ambiguous or (name in group_by_name and crystals - {material}):
            raise ValueError(
                f"results config {name!r} holds {sorted(crystals)} records, which conflicts "
                f"with this {material!r} sweep; use a distinct results mapping or config name"
            )


def _hashable_group(group_key, case):
    group = group_key(case)
    try:
        hash(group)
    except TypeError as error:
        raise TypeError(
            f"group_key must return a hashable value; got {type(group).__name__} "
            f"for case {case['name']!r}"
        ) from error
    return group


def _record_crystals(values):
    """Crystals named by one config's records; legacy case-less records name none."""
    cases = (record.get("case") for record in values.values())
    return {case.get("crystal") for case in cases if case is not None} - {None}


class SweepResume:
    """Prepare scalar records before deciding which cases need execution."""

    def __init__(self, cases, results, checkpoint, content_key_fn, cache_read, cache_write):
        self.loaded = False
        self.cases = cases
        self.results = results
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint.checkpoint_path
        self.material = checkpoint.material
        self.dataset_identity = checkpoint.dataset_identity
        self.on_timing = checkpoint.on_timing
        self.on_activity = checkpoint.on_activity
        self.cas_root = os.path.dirname(self.checkpoint_path) or "."
        self.content_key_fn = content_key_fn
        self.cache_read = cache_read and content_key_fn is not None
        self.cache_write = cache_write and content_key_fn is not None

    def content_key(self, case):
        """Content key for ``case`` or None when it is not canonically hashable
        (an exotic/legacy case simply falls back to normal recompute + per-stem
        store rather than aborting the run)."""
        if self.content_key_fn is None:
            return None
        try:
            key = self.content_key_fn(case)
            _checkpoint_store.cas_blob_path(self.material, key, self.cas_root)
            return key
        except (TypeError, ValueError):
            return None

    def load(self, resume, metadata_only_complete, *, transport_only=False):
        if not resume and not transport_only and _checkpoint_exists(self.checkpoint_path):
            self.checkpoint.restart()
        if resume and _checkpoint_exists(self.checkpoint_path):
            if self.on_activity is not None:
                self.on_activity({"phase": "loading", "case": None, "in_flight_case_count": 0})
            print(f"checkpoint: loading {self.checkpoint_path}")
            resume_started = time.perf_counter()
            manifest_path = _manifest_path_for(self.checkpoint_path)
            path = Path(self.checkpoint_path)
            existing_identity = self.validate_identity(manifest_path)
            if (
                metadata_only_complete
                and self.checkpoint.metadata_completion_supported
                and self.metadata_complete(manifest_path, path, existing_identity)
            ):
                if self.on_timing is not None:
                    self.on_timing({"resume_seconds": time.perf_counter() - resume_started})
                return True
            loaded = self.checkpoint.load()
            print(f"checkpoint: loaded in {time.perf_counter() - resume_started:.3f} s")
            if self.on_timing is not None:
                self.on_timing({"resume_seconds": time.perf_counter() - resume_started})
            self.validate_restored(loaded)
            self.results.update(loaded)
            self.loaded = True
            print(
                f"resumed {sum(len(v) for v in loaded.values())} {self.material} cases from {self.checkpoint_path}"
            )

        return False

    def validate_restored(self, loaded):
        """Refuse before merging: a later save keeps only this material's records.

        Merging a foreign or same-name mixed store would therefore drop or
        misfile the others. The file stays untouched for the user to inspect.
        """
        if not self.cases:
            return
        for name, values in loaded.items():
            restored = _record_crystals(values)
            live = _record_crystals(self.results.get(name, {}))
            foreign = (restored | live) - {self.material}
            if foreign:
                raise ValueError(
                    f"checkpoint {self.checkpoint_path} config {name!r} conflicts with this "
                    f"{self.material!r} sweep: found {sorted(foreign)} records; "
                    "archive it or select that material's checkpoint before resuming"
                )

    def validate_identity(self, manifest_path):
        existing_identity = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as handle:
                    manifest = json.load(handle)
                if isinstance(manifest, dict):
                    existing_identity = manifest.get("dataset_identity")
            except (OSError, ValueError, TypeError):
                pass
        if isinstance(existing_identity, dict):
            from ..campaign.profiles import normalize_dataset_identity

            existing_identity = normalize_dataset_identity(existing_identity)
        if self.dataset_identity is not None and not isinstance(existing_identity, dict):
            raise _CheckpointIdentityError(
                f"checkpoint {self.checkpoint_path} dataset identity cannot be verified; "
                "rerun with --recompute or archive the checkpoint before resuming"
            )
        if (
            self.dataset_identity is not None
            and existing_identity is not None
            and self.dataset_identity.get("parameter_sha256")
            != existing_identity.get("parameter_sha256")
        ):
            raise _CheckpointIdentityError(
                "checkpoint dataset identity mismatch: "
                f"requested {self.dataset_identity.get('profile')}:"
                f"{self.dataset_identity.get('parameter_sha256', '')[:12]}, existing "
                f"{existing_identity.get('profile')}:"
                f"{existing_identity.get('parameter_sha256', '')[:12]}; "
                "rerun with --recompute, archive the checkpoint, or select a different variant "
                "before resuming"
            )
        return existing_identity

    def metadata_complete(self, manifest_path, path, existing_identity):
        requested_proof = _case_set_proof_from_cases(self.cases)
        signature = _checkpoint_signature(self.checkpoint_path)
        manifest_fresh = (
            bool(signature)
            and os.path.isfile(manifest_path)
            and (os.stat(manifest_path).st_mtime_ns >= max(item[1] for item in signature))
        )
        if (
            not _checkpoint_store.has_parts(path.name, path.parent)
            and manifest_fresh
            and isinstance(existing_identity, dict)
            and self.dataset_identity is not None
            and existing_identity.get("parameter_sha256")
            == self.dataset_identity.get("parameter_sha256")
        ):
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            if manifest.get("completed_case_set") == requested_proof:
                return True
        return False

    def seed_cache(self):
        # Migration: seed missing blobs from a compatible existing checkpoint.
        # Compare the stored case's key with the currently requested case before
        # writing, so a legacy checkpoint without identity provenance can never
        # poison the shared store with a mismatched record.
        if self.cache_write:
            for case in self.cases:
                key = self.content_key(case)
                if key is None or _checkpoint_store.cas_contains(self.material, key, self.cas_root):
                    continue
                record = self.checkpoint.record_for(case)
                if record is None:
                    continue
                stored_key = self.content_key(record.get("case", {}))
                payload = _cas_payload_from_record(record)
                if (
                    key is not None
                    and key == stored_key
                    and payload is not None
                    and not _checkpoint_store.cas_contains(self.material, key, self.cas_root)
                ):
                    _checkpoint_store.cas_save(self.material, key, self.cas_root, payload)

    def replay_cache(self):
        # Cross-profile reuse: any case not already resumed from this profile's own
        # checkpoint whose content key already has a CAS blob (populated by ANY
        # profile of this material) is replayed through store_result instead of
        # recomputed. Replaying with the REQUESTED case makes the reconstructed
        # record bit-identical to a fresh compute (the requester's own detector/flux
        # scalars, its own case payload); only the profile-invariant transport arrays
        # come from the shared blob.
        if self.cache_read:
            reused = 0
            for c in self.cases:
                if c["name"] in self.results and c["E0_keV"] in self.results[c["name"]]:
                    continue
                key = self.content_key(c)
                if key is None or not _checkpoint_store.cas_contains(
                    self.material, key, self.cas_root
                ):
                    continue
                try:
                    out = _checkpoint_store.cas_load(self.material, key, self.cas_root)
                except (OSError, EOFError, pickle.UnpicklingError):
                    continue
                if not _valid_cas_payload(out):
                    continue
                store_result(self.results, c, out)
                self.checkpoint.record_finished(c)
                reused += 1
            if reused:
                print(f"reused {reused} {self.material} cases from shared content cache")
                self.checkpoint.cache_replayed()  # publish reused cases for this profile

        # Thin per-profile pointer table: (name, E0_keV) -> content_key, plus label
        # provenance. Enables a future GC / `pyrite clear` to know which shared blobs a
        # profile references. Written on any run that populates the store.
        if self.cache_write:
            _write_case_manifest(
                self.checkpoint_path, self.cases, self.content_key, self.dataset_identity
            )

    def recover_artifacts(self, artifact_store):
        """Commit outputs a worker published before the coordinator stopped.

        A worker may finish publishing its immutable artifact immediately before
        the coordinator crashes; recover those outputs before planning new work,
        so successful transports are not recomputed.
        """
        recovered = 0
        for case in self.cases:
            if has_record(self.results, case):
                continue
            key = self.content_key(case)
            if key is None:
                continue
            found = artifact_store.recover(self.material, key)
            if found is None:
                continue
            _, output = found
            store_result(self.results, case, output)
            self.store_transport(case, output)
            self.checkpoint.record_finished(case)
            recovered += 1
        if recovered:
            self.checkpoint.cache_replayed()
            print(f"recovered {recovered} {self.material} case artifacts from distributed inbox")

    def store_transport(self, case, out):
        if self.cache_write:
            key = self.content_key(case)
            if key is not None:
                _checkpoint_store.cas_save(self.material, key, self.cas_root, out)


def has_record(results, case):
    return case["name"] in results and case["E0_keV"] in results[case["name"]]


class SweepWork:
    """Scalar eligibility is fixed before observation production starts.

    Observation plans remain pending until their directional payload is stored;
    they participate in completion even when every scalar record is cached.
    """

    def __init__(self, cases, results, observation, transport_only):
        self.cases = cases
        self.results = results
        self.transport_only = transport_only
        self.observation = observation
        self.observation_plans = {}
        self.observation_only = set()
        if observation is not None and not transport_only:
            reused = 0
            for case in cases:
                plan = observation.plan(case)
                stored = plan.find_reusable(observation.store)
                if stored is not None:
                    observation.store.put(stored)
                    reused += 1
                    continue
                key = (case["name"], case["E0_keV"])
                self.observation_plans[key] = plan
                if has_record(results, case):
                    self.observation_only.add(key)
            print(
                f"observations: {len(self.observation_plans)} to produce, {reused} reused "
                f"({observation.store.path})"
            )
        self.todo = [
            case
            for case in cases
            if not has_record(results, case)
            or (case["name"], case["E0_keV"]) in self.observation_plans
        ]

        self.transport_pending = {(case["name"], case["E0_keV"]) for case in self.todo}

    def observation_directions(self):
        """Per-``todo`` directions, or None when nothing needs an observation."""
        if not self.observation_plans:
            return None
        plans = (self.observation_plans.get((c["name"], c["E0_keV"])) for c in self.todo)
        return [None if plan is None else plan.directions_sample for plan in plans]

    def transport_finished(self, case):
        """Count execution only; a transport-only output is never a stored record."""
        self.transport_pending.discard((case["name"], case["E0_keV"]))

    def stored_complete(self):
        """Every scalar record and required observation is available."""
        return not self.observation_plans and all(has_record(self.results, c) for c in self.cases)

    def requested_complete(self):
        """Mode-specific meaning of the ``run_sweep`` return value."""
        return not self.transport_pending if self.transport_only else self.stored_complete()

    def accept_observation(self, case, out):
        """Store directional work; return whether the scalar must be preserved.

        A required directional payload that is absent raises before ``out`` or
        any sweep state changes, so the case stays unreported and reruns on
        resume. Otherwise ``directional`` is removed from ``out`` because it
        never enters the scalar record or the CAS blob.
        """
        key = (case["name"], case["E0_keV"])
        plan = self.observation_plans.get(key)
        if plan is not None and out.get("directional") is None:
            raise ValueError(
                f"executor output for case (name={key[0]!r}, E0_keV={key[1]!r}) lacks the "
                "'directional' payload its observation requires; completed configs remain "
                "resumable, so fix the executor and rerun"
            )
        directional = out.pop("directional", None)
        if plan is not None:
            assert self.observation is not None
            self.observation.store.put(
                plan.assemble(directional, self.observation.provenance(case))
            )
            del self.observation_plans[key]
        return key in self.observation_only
