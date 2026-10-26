"""Issue #707: canonical storage access does not change scientific identities."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from pyrite import results
from pyrite.checkpoints import _checkpoint_store
from pyrite.checkpoints.persistence import load_checkpoint as historical_load_checkpoint
from pyrite.execution.artifacts import CaseArtifactStore as HistoricalCaseArtifactStore
from pyrite.observations.store import ObservationStore as HistoricalObservationStore
from pyrite.storage import (
    CaseArtifactStore,
    ObservationStore,
    checkpoint_manifest,
    load_checkpoint,
)


def test_storage_aliases_preserve_implementations_and_missing_checkpoint(tmp_path):
    assert CaseArtifactStore is HistoricalCaseArtifactStore
    assert ObservationStore is HistoricalObservationStore
    assert load_checkpoint is historical_load_checkpoint
    assert load_checkpoint("missing", checkpoint_dir=tmp_path) == {}
    assert checkpoint_manifest("missing", checkpoint_dir=tmp_path) is None
    assert ObservationStore("missing", root=tmp_path).sources() == ()
    assert results.Result.__module__ == "pyrite.results.model"
    from pyrite.results.model import CxrSpectra, PixelMetadata, PixelRayMap, SpatialResult
    from pyrite.results.model import SpectralDensity, SpectralFactors

    assert results.SpectralDensity is SpectralDensity
    assert results.CxrSpectra is CxrSpectra
    assert results.SpatialResult is SpatialResult
    assert results.SpectralFactors is SpectralFactors
    assert results.PixelRayMap is PixelRayMap
    assert results.PixelMetadata is PixelMetadata
    assert results.FLUX_UNITS["electron"] == "photons / (electron eV sr)"


def test_campaign_checkpoint_round_trip_through_public_reader(tmp_path):
    original = {
        "sweep-a": {
            30.0: {
                "case": {"name": "sweep-a", "crystal": "hopg", "E0_keV": 30.0},
                "E_grid": np.array([200.0, 300.0]),
                "spec": np.array([1.0, 2.0]),
                "brem": np.array([0.25, 0.5]),
                "eta": 0.75,
                "scale": 1.25,
            },
        }
    }
    # Campaign owns the checkpoint writer; the facade exposes its public reader.
    _checkpoint_store.save("hopg", tmp_path, original)
    saved = load_checkpoint("hopg", checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(saved["sweep-a"][30.0]["spec"], [1.0, 2.0])
    np.testing.assert_array_equal(saved["sweep-a"][30.0]["brem"], [0.25, 0.5])
    assert saved["sweep-a"][30.0]["case"] == original["sweep-a"][30.0]["case"]
    meta = checkpoint_manifest("hopg", checkpoint_dir=tmp_path)
    assert meta is not None
    assert meta["n_records"] == 1
    assert meta["energies_keV"] == [30.0]
    assert meta["completed_case_set"]["count"] == 1
    assert checkpoint_manifest("hopg", checkpoint_dir=tmp_path) == meta


def test_worker_artifact_round_trip_uses_existing_identity(tmp_path):
    store = CaseArtifactStore(tmp_path)
    key = "a" * 64
    ref = store.publish(
        "hopg", key, {"E_grid": np.array([100.0, 200.0]), "spec": np.array([1.0, 4.0])}
    )
    assert ref.content_key == key
    recovered = store.recover("hopg", key)
    assert recovered is not None
    reopened_ref, output = recovered
    assert reopened_ref == ref
    np.testing.assert_array_equal(output["spec"], [1.0, 4.0])
    with pytest.raises(ValueError, match="content key"):
        store.load(ref, expected_content_key="b" * 64)


def test_facade_import_does_not_initialize_scheduler_modules():
    script = """
import sys
import pyrite.storage as storage
from pyrite.storage import CaseArtifactStore
from pyrite.execution.artifacts import CaseArtifactStore as original
assert CaseArtifactStore is original
assert 'pyrite.execution.dask' not in sys.modules
assert 'pyrite.execution.slurm' not in sys.modules
assert 'pyrite.execution.local' not in sys.modules
assert 'pyrite.execution.profiles' not in sys.modules
assert 'pyrite.cli' not in sys.modules
assert not hasattr(storage, 'NotAnExport')
"""
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
