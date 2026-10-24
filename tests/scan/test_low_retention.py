"""Disk-backed sweeps retain metadata and release only persisted spectra."""

import numpy as np
import pytest

from pyrite.checkpoints import _checkpoint_store as store
from pyrite.runs import run
from tests.helpers import stub_run_cases
from tests.scan.test_run import _fake_case


def test_low_retention_sweep_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "run_cases", stub_run_cases)
    cases = [_fake_case(f"cfg_{i}", energy) for i in range(3) for energy in (20.0, 30.0)]
    results = {}
    assert run.run_sweep(
        cases, results, checkpoint_dir=tmp_path, retain_results=False, progress=False
    )
    assert all(
        set(record) == {"case", "checkpoint_path"}
        for values in results.values()
        for record in values.values()
    )
    loaded = store.load("hopg", tmp_path)
    assert sum(map(len, loaded.values())) == 6
    assert not store.has_parts("hopg", tmp_path)
    monkeypatch.setattr(
        run, "run_cases", lambda cases, **kw: pytest.fail("resume recomputed") if cases else None
    )
    resumed = {}
    assert run.run_sweep(
        cases, resumed, checkpoint_dir=tmp_path, retain_results=False, progress=False
    )
    assert resumed.keys() == results.keys()
    np.testing.assert_array_equal(
        store.load("hopg", tmp_path)["cfg_0"][20.0]["spec"], loaded["cfg_0"][20.0]["spec"]
    )


def test_failed_persistence_keeps_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "run_cases", stub_run_cases)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_part", fail)
    results = {}
    with pytest.raises(OSError, match="disk full"):
        run.run_sweep(
            [_fake_case("cfg", 20.0)], results, checkpoint_dir=tmp_path, retain_results=False
        )
    assert isinstance(results["cfg"][20.0]["spec"], np.ndarray)


def test_low_retention_rejects_legacy_output(tmp_path):
    with pytest.raises(ValueError, match="component checkpoint"):
        run.run_sweep([], {}, checkpoint_path=str(tmp_path / "old.pkl"), retain_results=False)


def test_arrays_are_released_between_cases(tmp_path, monkeypatch):
    import gc
    import weakref

    from tests.helpers import fake_out

    references = []

    def checked_runner(cases, callback, **kwargs):
        for i, case in enumerate(cases):
            gc.collect()
            assert all(reference() is None for reference in references)
            out = fake_out(case)
            references.append(weakref.ref(out["spec"]))
            callback(i, case, out)
            del out

    monkeypatch.setattr(run, "run_cases", checked_runner)
    run.run_sweep(
        [_fake_case(f"cfg_{i}", 30.0) for i in range(4)],
        {},
        checkpoint_dir=tmp_path,
        retain_results=False,
    )
    assert all(reference() is None for reference in references)


def test_partial_config_is_durable_and_resume_uses_metadata(tmp_path, monkeypatch):
    from pyrite.checkpoints import _checkpoint_io, _result_v2
    from tests.helpers import fake_out

    cases = [_fake_case("cfg", energy) for energy in (20.0, 30.0)]

    def first_only(cases, callback, **kwargs):
        callback(0, cases[0], fake_out(cases[0]))

    monkeypatch.setattr(run, "run_cases", first_only)
    assert not run.run_sweep(cases, {}, checkpoint_dir=tmp_path, retain_results=False)
    assert store.has_parts("hopg", tmp_path)
    assert list(store.load("hopg", tmp_path)["cfg"]) == [20.0]

    def no_arrays(self, index):
        pytest.fail("metadata resume decoded spectral blob")

    with monkeypatch.context() as patch:
        patch.setattr(_result_v2._Reader, "blob", no_arrays)
        metadata = store.load_metadata("hopg", tmp_path)
        assert list(metadata["cfg"]) == [20.0]

    def no_full_load(path):
        pytest.fail("v2 resume/consolidation used full checkpoint load")

    monkeypatch.setattr(_checkpoint_io, "load", no_full_load)
    ran = []

    def remaining(cases, **kwargs):
        ran.extend(case["E0_keV"] for case in cases)
        return stub_run_cases(cases, **kwargs)

    monkeypatch.setattr(run, "run_cases", remaining)
    assert run.run_sweep(cases, {}, checkpoint_dir=tmp_path, retain_results=False)
    assert ran == [30.0]


@pytest.mark.parametrize("failure", ["stage", "publish", "manifest"])
def test_consolidation_failure_keeps_recoverable_shards(tmp_path, monkeypatch, failure):
    from pyrite.checkpoints import _checkpoint_io, _disk_sweep

    monkeypatch.setattr(run, "run_cases", stub_run_cases)
    cases = [_fake_case("cfg", 30.0)]

    def fail(*args, **kwargs):
        raise OSError("publication failed")

    with monkeypatch.context() as patch:
        if failure == "stage":
            patch.setattr(_checkpoint_io, "config_writer", fail)
        elif failure == "publish":
            original = store.os.replace

            def fail_line(source, target):
                if str(target).endswith("line.h5"):
                    fail()
                return original(source, target)

            patch.setattr(store.os, "replace", fail_line)
        else:
            patch.setattr(_disk_sweep, "_manifest_save", fail)
        with pytest.raises(OSError, match="publication failed"):
            run.run_sweep(cases, {}, checkpoint_dir=tmp_path, retain_results=False)
    assert store.has_parts("hopg", tmp_path)
    assert "spec" in store.load("hopg", tmp_path)["cfg"][30.0]
    assert not list((tmp_path / "hopg").glob("*.tmp"))
    assert run.run_sweep(cases, {}, checkpoint_dir=tmp_path, retain_results=False)
    assert not store.has_parts("hopg", tmp_path)


def test_cache_replay_and_seeding_preserve_metadata_results(tmp_path, monkeypatch):
    from pyrite.campaign.profiles import case_content_key

    monkeypatch.setattr(run, "run_cases", stub_run_cases)
    cases = [_fake_case("cfg", energy) for energy in (20.0, 30.0)]
    first = tmp_path / "first"
    assert run.run_sweep(
        cases, {}, checkpoint_path=str(first), retain_results=False, content_key_fn=case_content_key
    )
    second = tmp_path / "second"

    def no_compute(cases, **kwargs):
        assert not cases

    monkeypatch.setattr(run, "run_cases", no_compute)
    results = {}
    assert run.run_sweep(
        cases,
        results,
        checkpoint_path=str(second),
        retain_results=False,
        content_key_fn=case_content_key,
    )
    assert set(results["cfg"][20.0]) == {"case", "checkpoint_path"}
    for case in cases:
        store.cas_blob_path("hopg", case_content_key(case), tmp_path).unlink()
    assert run.run_sweep(
        cases,
        {},
        checkpoint_path=str(second),
        retain_results=False,
        content_key_fn=case_content_key,
    )
    assert all(store.cas_contains("hopg", case_content_key(case), tmp_path) for case in cases)


def test_low_retention_matches_default_components_and_manifest(tmp_path, monkeypatch):
    import json

    from tests.helpers import fake_out

    cases = [_fake_case(f"cfg_{i}", energy) for i in range(2) for energy in (20.0, 30.0)]

    def components_runner(cases, callback, **kwargs):
        for index, case in enumerate(cases):
            out = fake_out(case)
            for component in ("characteristic", "channeling", "transition", "coherent"):
                out[f"spec_{component}"] = np.full_like(out["spec"], (index + 1) * 0.001)
            callback(index, case, out)

    monkeypatch.setattr(run, "run_cases", components_runner)
    for retain, directory in [(True, tmp_path / "memory"), (False, tmp_path / "disk")]:
        assert run.run_sweep(cases, {}, checkpoint_dir=directory, retain_results=retain)
    expected = store.load("hopg", tmp_path / "memory")
    actual = store.load("hopg", tmp_path / "disk")
    for name, values in expected.items():
        for energy, record in values.items():
            assert actual[name][energy].keys() == record.keys()
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(actual[name][energy][key], value)
                elif isinstance(value, float) and np.isnan(value):
                    assert np.isnan(actual[name][energy][key])
                else:
                    assert actual[name][energy][key] == value
    manifests = [
        json.loads((tmp_path / directory / "hopg" / "meta.json").read_text())
        for directory in ("memory", "disk")
    ]
    assert manifests[0] == manifests[1]


def test_low_retention_rejects_prefilled_results(tmp_path):
    results = {"caller_owned": {30.0: {"case": {"crystal": "hopg"}}}}
    with pytest.raises(ValueError, match="empty results"):
        run.run_sweep([], results, checkpoint_dir=tmp_path, retain_results=False)
    assert "caller_owned" in results


def test_failed_shard_manifest_keeps_arrays_and_recoverable_data(tmp_path, monkeypatch):
    from pyrite.checkpoints import _disk_sweep

    monkeypatch.setattr(run, "run_cases", stub_run_cases)

    def fail(*args, **kwargs):
        raise OSError("manifest failed")

    monkeypatch.setattr(_disk_sweep, "_manifest_write", fail)
    results = {}
    with pytest.raises(OSError, match="manifest failed"):
        run.run_sweep(
            [_fake_case("cfg", 30.0)], results, checkpoint_dir=tmp_path, retain_results=False
        )
    assert "spec" in results["cfg"][30.0]
    assert "spec" in store.load("hopg", tmp_path)["cfg"][30.0]


def test_group_callbacks_can_selectively_load_persisted_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "run_cases", stub_run_cases)
    cases = [_fake_case(f"cfg_{i}", energy) for i in range(2) for energy in (20.0, 30.0)]
    seen = []

    def on_chunk(names):
        selected = store.load_selected(
            "hopg", tmp_path, {(name, energy) for name in names for energy in (20.0, 30.0)}
        )
        assert all("spec" in record for values in selected.values() for record in values.values())
        seen.extend(selected)

    assert run.run_sweep(
        cases, {}, checkpoint_dir=tmp_path, retain_results=False, on_chunk=on_chunk
    )
    assert sorted(seen) == ["cfg_0", "cfg_1"]
