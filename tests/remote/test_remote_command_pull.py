"""Remote pull tests."""

import shutil

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _PARTIAL_SECTIONS,
    _remote_main,
    lifecycle_pull,
)


def test_pull_performance_profile_fetches_each_matching_job(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: "job-1\thopg.ndjson\njob-2\tmos2.ndjson\n",
    )
    remote_commands = []

    def fake_download(command, destination, **_kwargs):
        remote_commands.append(command)
        destination.write_text('{"schema":"cxr.performance.v1"}\n')

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    pulled = lifecycle.pull_performance_profile("baseline")

    assert pulled == [
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-1" / "hopg.ndjson",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-2" / "mos2.ndjson",
    ]
    assert all(path.is_file() for path in pulled)
    assert all("/pyrite-output/performance/job-" in command for command in remote_commands)


def test_pull_performance_profile_fetches_nsys_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: (
            "job-3\tmos2.ndjson\n"
            "job-3\tmos2.nsys-rep\n"
            "job-3\tmos2.sqlite\n"
            "job-3\tmos2.nsys-stats.txt\n"
            "job-3\tmos2.cpu.prof\n"
            "job-3\tmos2.cpu.txt\n"
            "job-3\tunsafe.report\n"
        ),
    )

    def fake_download(_command, destination, **_kwargs):
        destination.write_bytes(b"artifact")

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    pulled = lifecycle.pull_performance_profile("baseline")

    assert pulled == [
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.ndjson",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.nsys-rep",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.sqlite",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.nsys-stats.txt",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.cpu.prof",
        tmp_path / "pyrite-output" / "performance" / "baseline" / "job-3" / "mos2.cpu.txt",
    ]


def test_pull_attached_progress_uses_remote_reserved_stems_not_local_catalog(monkeypatch):
    pulls = []
    monkeypatch.setattr(
        scripts, "_stems", lambda *a, **k: pytest.fail("must not predict stems from local catalog")
    )
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: (
            "written hbn@high_energy-1ca2a03dbfd5\n"
            "unwritten hopg@high_energy-aaaaaaaaaaaa\n"
            "written hbn@other-ffffffffffff\n"
            "written hbnx@high_energy-bbbbbbbbbbbb\n"
        ),
    )
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kwargs: pulls.append((stems, kwargs)))

    viewer._pull_attached_progress("j", _PARTIAL_SECTIONS)

    # Only the progress-reporting material (hbn) is pulled; unwritten hopg is skipped.
    assert pulls == [
        (["hbn@high_energy-1ca2a03dbfd5", "hbn@other-ffffffffffff"], {"no_sync": True})
    ]


def test_pull_attached_progress_unwritten_checkpoint_is_actionable(monkeypatch):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda _cmd: "unwritten hbn@high_energy-1ca2a03dbfd5\n"
    )
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **k: pytest.fail("nothing to pull"))

    with pytest.raises(
        SystemExit, match="no written checkpoints.*not yet written.*hbn@high_energy"
    ):
        viewer._pull_attached_progress("j", _PARTIAL_SECTIONS)


def test_pull_defaults_to_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    _remote_main(["pull", "hopg"])
    assert calls[0]["grid"] is True


def test_pull_full_flag_disables_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    _remote_main(["pull", "hopg", "--full"])
    assert calls[0]["grid"] is False


def test_pull_short_full_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    _remote_main(["pull", "hopg", "-f"])
    assert calls[0]["grid"] is False


def test_pull_level9_flag_defaults_off_and_wires_through(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    _remote_main(["pull", "hopg"])
    assert calls[0]["level9"] is False
    _remote_main(["pull", "hopg", "--level9"])
    assert calls[1]["level9"] is True


def test_pull_hash_requires_exactly_one_qualified_selector(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="--hash requires exactly one MATERIAL@PROFILE"):
        remote.pull(["hopg@standard", "wse2@standard"], grid=True, hash_prefix="a" * 12)


def test_pull_resolves_profile_selector_to_the_predicted_stem(monkeypatch, tmp_path):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io, _checkpoint_store

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    resolved = []

    def fake_resolve(material, profile, **kwargs):
        resolved.append((material, profile, kwargs.get("hash_prefix")))
        return "hopg"

    monkeypatch.setattr(lifecycle_pull, "resolve_profile_stem", fake_resolve)
    monkeypatch.setattr(transport, "sync_code", lambda: None)
    # resolves to bare "hopg" -- also triggers _resolve_survey_stems' listing;
    # empty means no sibling survey checkpoint to also pull.
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    payload = {"cfg": {30.0: {"case": {}, "E_grid": np.array([1.0]), "spec": np.array([1.0])}}}
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    def fake_download(_command, destination, **_kwargs):
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg@sub_100keV"], grid=True, no_sync=True)

    assert resolved == [("hopg", "sub_100keV", None)]
    assert _checkpoint_store.checkpoint_exists("hopg", tmp_path / "pyrite-output" / "checkpoints")


def test_pull_bare_material_also_pulls_matching_survey_checkpoint(monkeypatch, tmp_path, capsys):
    import numpy as np

    from pyrite.campaign.profiles import named_profile_stem
    from pyrite.checkpoints import _checkpoint_io, _checkpoint_store

    survey_stem = named_profile_stem("hopg", "survey")
    payload = {
        "cfg": {
            30.0: {
                "case": {"crystal": "hopg", "Ne_brem": 10},
                "E_grid": np.array([1.0, 2.0]),
                "spec": np.array([3.0, 4.0]),
                "E_grid_brem": np.array([1.0, 2.0]),
                "brem_wide": np.array([5.0, 6.0]),
                "brem": np.array([5.0, 6.0]),
            }
        }
    }
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: f"{survey_stem}\n")
    transfers = []

    def fake_download(command, destination, **_kwargs):
        transfers.append((command, destination))
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg"], grid=True, no_sync=True)

    assert len(transfers) == 2  # canonical stem + discovered survey variant
    assert _checkpoint_store.checkpoint_exists("hopg", tmp_path / "pyrite-output" / "checkpoints")
    assert _checkpoint_store.checkpoint_exists(
        survey_stem, tmp_path / "pyrite-output" / "checkpoints"
    )
    assert "also pulling identity-qualified survey checkpoint" in capsys.readouterr().out


def test_pull_quick_stem_does_not_resolve_survey_siblings(monkeypatch, tmp_path):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *_a: pytest.fail("--quick pull must not list checkpoints/"),
    )
    payload = {"cfg": {30.0: {"case": {}, "E_grid": np.array([1.0]), "spec": np.array([1.0])}}}
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    def fake_download(_command, destination, **_kwargs):
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg_quick"], grid=False, no_sync=True)  # must not raise


def test_pull_dataset_merge_skips_survey_discovery(monkeypatch, tmp_path):
    def _boom(*_a, **_kw):
        raise RuntimeError("stub -- dataset merge test does not need a real download")

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *_a: pytest.fail("--brem-only pull must not list checkpoints/"),
    )
    monkeypatch.setattr(transport, "_ssh_download", _boom)

    with pytest.raises(SystemExit, match="remote pull failed"):
        lifecycle.pull(["hopg"], dataset="brem", no_sync=True)


def test_pull_rejects_unsafe_stem_before_sync_or_local_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="invalid remote shell token"):
        remote.pull(["bad;stem"], grid=True)

    assert not (tmp_path / "pyrite-output" / "checkpoints").exists()


def test_pull_zhai_cache_fetches_every_listed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *a: (
            "/r/pyrite-output/cache/zhai_reproduction/zhai-a.pkl\n"
            "/r/pyrite-output/cache/zhai_reproduction/zhai-b.pkl\n"
        ),
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.pull_zhai_cache()

    assert len(runs) == 2
    assert runs[0][0] == "scp" and runs[0][1].endswith("zhai-a.pkl")
    assert (tmp_path / "pyrite-output" / "cache" / "zhai_reproduction").is_dir()


def test_pull_zhai_cache_reports_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: pytest.fail("nothing to pull"))

    remote.pull_zhai_cache()

    assert "no zhai cache files" in capsys.readouterr().out
    assert "find" in commands[0]


def test_pull_dataset_merges_and_archives(monkeypatch, tmp_path):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io
    from pyrite.remote import lifecycle

    # local checkpoint with a line record
    ckpt = tmp_path / "pyrite-output" / "checkpoints" / "mos2.pkl"
    ckpt.parent.mkdir(parents=True)
    local = {
        "n": {
            30.0: {
                "case": {},
                "spec": np.zeros(3),
                "E_grid": np.array([1.0, 2.0, 3.0]),
                "brem_wide": np.array([1.0, 1.0, 1.0]),
                "E_grid_brem": np.array([1.0, 2.0, 3.0]),
                "brem": np.zeros(3),
            }
        }
    }
    _checkpoint_io.dump(local, str(ckpt))
    # the "remote" sub-pickle (what scp would land): new spec only
    remote_tmp = tmp_path / "remote.pkl"
    _checkpoint_io.dump(
        {"n": {30.0: {"case": {}, "spec": np.full(3, 7.0), "E_grid": np.array([1.0, 2.0, 3.0])}}},
        str(remote_tmp),
    )

    monkeypatch.setattr(lifecycle.config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle.transport, "sync_code", lambda: None)
    archived = []
    monkeypatch.setattr(
        lifecycle.archive, "archive_checkpoint", lambda stem, **kw: archived.append(stem)
    )

    # Stub the one-session box round-trip into the incoming transfer file.
    def fake_download(_command, destination, **_kwargs):
        shutil.copy(remote_tmp, destination)

    monkeypatch.setattr(lifecycle.transport, "_ssh_download", fake_download)

    lifecycle.pull(["mos2"], dataset="line")

    from pyrite.checkpoints import _checkpoint_store

    merged = _checkpoint_store.load("mos2", ckpt.parent)["n"][30.0]
    assert np.all(merged["spec"] == 7.0)  # line overwritten
    assert archived == ["mos2"]  # archived before merge


@pytest.mark.parametrize("dataset", ["spec", "", "BREM", 1])
def test_pull_rejects_invalid_public_dataset_before_side_effects(monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(lifecycle.config, "LOCAL_ROOT", tmp_path / "must-not-exist")
    monkeypatch.setattr(
        lifecycle.transport,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid dataset must not reach transport"),
    )

    with pytest.raises(ValueError, match="dataset must be"):
        lifecycle.pull(["mos2"], dataset=dataset)

    assert not lifecycle.config.LOCAL_ROOT.exists()
