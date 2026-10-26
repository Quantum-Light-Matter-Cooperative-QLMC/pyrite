import json
import subprocess
import sys
import textwrap
from dataclasses import replace

import numpy as np

from pyrite.observations import ObservationStore, observation_from_result, observation_inventory
from tests.observations.test_store import _acquisition, _simulate, fake_directional  # noqa: F401


def test_inventory_imports_in_spawn_worker_without_checkpoint_driver(tmp_path) -> None:
    script = tmp_path / "inventory_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import multiprocessing
            import os
            import sys

            def worker():
                import pyrite.observations as observations
                assert "pyrite.observations.inventory" not in sys.modules
                inventory = observations.observation_inventory
                assert inventory is observations.observation_inventory
                assert inventory("unwritten").entries == ()
                assert not any(
                    name == "pyrite.checkpoints" or name.startswith("pyrite.checkpoints.")
                    for name in sys.modules
                )

            if __name__ == "__main__":
                os.environ["PYRITE_HOME"] = sys.argv[1]
                process = multiprocessing.get_context("spawn").Process(target=worker)
                process.start()
                process.join(30)
                if process.is_alive():
                    process.terminate()
                    process.join()
                assert process.exitcode == 0, process.exitcode
            """
        )
    )
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, timeout=40
    )
    assert result.returncode == 0, result.stderr


def _checkpoint(root, stem, cases) -> None:
    directory = root / "checkpoints" / stem
    directory.mkdir(parents=True)
    (directory / "cases.json").write_text(
        json.dumps({"schema": "cxr.case-manifest.v1", "material": "hopg", "cases": cases})
    )


def _stored(tmp_path):
    result = _simulate()
    store = ObservationStore("hopg", root=tmp_path / "observations")
    stored = observation_from_result(result)
    digest = store.put(stored)
    longer = stored.rescore(acquisition=_acquisition(exposure_s=5.0))
    store.put(longer)
    return result.provenance["identity_digest"], digest, longer.digest


def test_inventory_names_observations_by_checkpoint_case(tmp_path) -> None:
    source, digest, longer = _stored(tmp_path)
    _checkpoint(
        tmp_path,
        "hopg",
        [
            {"name": "unobserved", "E0_keV": 10.0, "content_key": "f" * 64},
            {"name": "hopg_t1000", "E0_keV": 30.0, "content_key": source},
        ],
    )

    inventory = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")

    assert inventory.problems == ()
    assert {entry.observation_digest for entry in inventory.entries} == {digest, longer}
    entry = next(entry for entry in inventory.entries if entry.observation_digest == digest)
    assert (entry.case_name, entry.E0_keV, entry.source_identity_digest) == (
        "hopg_t1000",
        30.0,
        source,
    )
    assert entry.response_type == "IdealPhotonCounter"
    assert entry.event_semantics == "ideal-energy-preserving-photon-event"
    assert (entry.exposure_s, entry.mode) == (1.0, "expected")
    assert entry.label.startswith("hopg_t1000 @ 30 keV -- IdealPhotonCounter, 1 s, expected")
    loaded = inventory.load(digest)
    assert loaded.digest == digest


def test_inventory_default_checkpoint_root_matches_persistence(tmp_path, monkeypatch) -> None:
    from pyrite.checkpoints.persistence import default_checkpoint_dir
    from pyrite.console.outputs import output_dir

    monkeypatch.setenv("PYRITE_HOME", str(tmp_path))
    inventory = observation_inventory("unwritten")

    assert default_checkpoint_dir() == str(output_dir("checkpoints"))
    assert inventory.entries == ()
    assert inventory.store.index_path == output_dir("observations") / "unwritten" / "index.json"


def test_inventory_without_a_store_is_empty(tmp_path) -> None:
    assert observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints").entries == ()
    _checkpoint(tmp_path, "hopg", [{"name": "c", "E0_keV": 1.0, "content_key": "a" * 64}])
    empty = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")
    assert (empty.entries, empty.problems) == ((), ())


def test_inventory_reports_an_unreadable_record_and_keeps_the_rest(tmp_path) -> None:
    source, digest, longer = _stored(tmp_path)
    _checkpoint(tmp_path, "hopg", [{"name": "c", "E0_keV": 30.0, "content_key": source}])
    store = ObservationStore("hopg", root=tmp_path / "observations")
    store.record_path(longer).write_text("{not json")

    inventory = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")

    assert [entry.observation_digest for entry in inventory.entries] == [digest]
    assert len(inventory.problems) == 1 and longer in inventory.problems[0]


def test_unknown_manifest_schema_is_a_problem_and_entries_fall_back_to_keys(tmp_path) -> None:
    source, digest, longer = _stored(tmp_path)
    directory = tmp_path / "checkpoints" / "hopg"
    directory.mkdir(parents=True)
    (directory / "cases.json").write_text(json.dumps({"schema": "other", "cases": []}))

    inventory = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")

    assert "unknown case manifest schema" in inventory.problems[0]
    assert [entry.case_name for entry in inventory.entries] == [f"source {source[:12]}"] * 2
    assert inventory.entries[0].label.startswith(f"source {source[:12]} -- IdealPhotonCounter")


def test_recorded_case_names_an_observation_without_a_manifest(tmp_path) -> None:
    result = _simulate()
    stored = observation_from_result(result)
    stored = replace(
        stored, provenance={**stored.provenance, "case": {"name": "hopg_a", "E0_keV": 30}}
    )
    store = ObservationStore("hopg", root=tmp_path / "observations")
    digest = store.put(stored)

    inventory = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")

    assert inventory.problems == ()
    [entry] = inventory.entries
    assert (entry.case_name, entry.E0_keV, entry.observation_digest) == ("hopg_a", 30.0, digest)


def test_realized_and_expected_views_never_substitute_for_each_other(tmp_path) -> None:
    import pytest

    from pyrite.plots._pixel_frames import counting_observation

    poisson = observation_from_result(
        _simulate(acquisition=_acquisition(exposure_s=1.0e-3, mode="poisson", seed=3))
    )
    realized = counting_observation(poisson, "realized")
    expected = counting_observation(poisson, "expected")

    assert realized is poisson
    assert expected.acquisition == replace(poisson.acquisition, mode="expected", seed=None)
    assert expected.spatial is poisson.spatial
    assert realized.acquisition_image().dtype == np.int64
    assert expected.acquisition_image().dtype == float
    with pytest.raises(ValueError, match="no Poisson realization"):
        counting_observation(expected, "realized")
