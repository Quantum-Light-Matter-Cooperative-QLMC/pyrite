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
    _checkpoint_load,
    _checkpoint_signature,
    _manifest_path_for,
    _valid_cas_payload,
    _write_case_manifest,
)
from ..results import store_result


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

    def load(self, resume, metadata_only_complete):
        if resume and _checkpoint_exists(self.checkpoint_path):
            if self.on_activity is not None:
                self.on_activity({"phase": "loading", "case": None, "in_flight_case_count": 0})
            print(f"checkpoint: loading {self.checkpoint_path}")
            resume_started = time.perf_counter()
            manifest_path = _manifest_path_for(self.checkpoint_path)
            path = Path(self.checkpoint_path)
            existing_identity = self.validate_identity(manifest_path)
            if metadata_only_complete and self.metadata_complete(
                manifest_path, path, existing_identity
            ):
                if self.on_timing is not None:
                    self.on_timing({"resume_seconds": time.perf_counter() - resume_started})
                return True
            loaded = _checkpoint_load(self.checkpoint_path)
            print(f"checkpoint: loaded in {time.perf_counter() - resume_started:.3f} s")
            if self.on_timing is not None:
                self.on_timing({"resume_seconds": time.perf_counter() - resume_started})
            self.results.update(loaded)
            self.loaded = True
            print(
                f"resumed {sum(len(v) for v in loaded.values())} {self.material} cases from {self.checkpoint_path}"
            )

        return False

    def validate_identity(self, manifest_path):
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
            self.dataset_identity is not None
            and existing_identity is not None
            and self.dataset_identity.get("parameter_sha256")
            != existing_identity.get("parameter_sha256")
        ):
            raise ValueError(
                "checkpoint dataset identity mismatch: "
                f"requested {self.dataset_identity.get('profile')}:"
                f"{self.dataset_identity.get('parameter_sha256', '')[:12]}, existing "
                f"{existing_identity.get('profile')}:"
                f"{existing_identity.get('parameter_sha256', '')[:12]}; "
                "archive or select a different variant before resuming"
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
                record = self.results.get(case["name"], {}).get(case["E0_keV"])
                if record is None:
                    continue
                key = self.content_key(case)
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
                reused += 1
            if reused:
                print(f"reused {reused} {self.material} cases from shared content cache")
                self.checkpoint.save()  # persist reused cases into this profile's checkpoint + manifest

        # Thin per-profile pointer table: (name, E0_keV) -> content_key, plus label
        # provenance. Enables a future GC / `pyrite clear` to know which shared blobs a
        # profile references. Written on any run that populates the store.
        if self.cache_write:
            _write_case_manifest(
                self.checkpoint_path, self.cases, self.content_key, self.dataset_identity
            )

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

    def accept_observation(self, case, out):
        """Store directional work; return whether the scalar must be preserved."""
        directional = out.pop("directional", None)
        key = (case["name"], case["E0_keV"])
        plan = self.observation_plans.get(key)
        if plan is not None and directional is not None:
            assert self.observation is not None
            self.observation.store.put(
                plan.assemble(directional, self.observation.provenance(case))
            )
            del self.observation_plans[key]
        return key in self.observation_only
