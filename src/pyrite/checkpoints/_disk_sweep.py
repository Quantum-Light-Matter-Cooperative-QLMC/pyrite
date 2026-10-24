"""Bounded-memory sweep publication over the existing component/shard store."""

import time
from pathlib import Path

from . import _checkpoint_store
from .persistence import (
    _IncrementalManifest,
    _manifest_save,
    _manifest_write,
    _SweepCheckpoint,
)


class DiskSweepCheckpoint(_SweepCheckpoint):
    """Keep only metadata after each successful case-level shard write."""

    metadata_completion_supported = False  # low-retention callers receive the case index

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.sharded:
            raise ValueError("retain_results=False requires a component checkpoint path")
        if self.results:
            raise ValueError("retain_results=False requires an empty results mapping")
        self.path = Path(self.checkpoint_path)

    def reference(self, record):
        return {"case": record["case"], "checkpoint_path": str(self.path.resolve())}

    def load(self):
        records = _checkpoint_store.load_metadata(self.path.name, self.path.parent)
        return {
            name: {energy: self.reference(record) for energy, record in values.items()}
            for name, values in records.items()
        }

    def record_for(self, case):
        key = (case["name"], case["E0_keV"])
        if key[1] not in self.results.get(key[0], {}):
            return None
        return _checkpoint_store.load_selected(self.path.name, self.path.parent, {key})[key[0]][
            key[1]
        ]

    def record_finished(self, case):
        name, energy = case["name"], case["E0_keV"]
        started = time.perf_counter()
        # Existing energies may have been released already. Rewrite one config,
        # never the growing campaign, preserving its prior durable records.
        existing = set(self.results[name]) - {energy}
        records = {}
        if existing:
            keys = {(name, value) for value in existing}
            records = _checkpoint_store.load_selected(self.path.name, self.path.parent, keys).get(
                name, {}
            )
        records[energy] = self.results[name][energy]
        _checkpoint_store.save_part(self.path.name, self.path.parent, name, records)
        if self.incremental_manifest is None:
            self.incremental_manifest = _IncrementalManifest(self.dataset_identity)
            self.incremental_manifest.add(self.material_subset())
        else:
            self.incremental_manifest.add({name: {energy: records[energy]}})
        _manifest_write(self.checkpoint_path, self.incremental_manifest.manifest())
        # A failed shard or manifest write leaves this live result intact.
        self.results[name][energy] = self.reference(records[energy])
        if self.on_timing is not None:
            elapsed = time.perf_counter() - started
            self.on_timing({"checkpoint_seconds": elapsed, "shard_write_seconds": elapsed})

    def cache_replayed(self):
        # Each reused case is durable already; finish performs one consolidation.
        pass

    def config_finished(self, name):
        # Every energy has already been persisted by record_finished.
        pass

    def save(self):
        subset = self.material_subset()  # metadata only
        _checkpoint_store.save_streaming(self.path.name, self.path.parent, subset)
        _manifest_save(self.checkpoint_path, subset, self.dataset_identity)
